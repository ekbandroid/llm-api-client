"""Хранилище пользователей на SQLite."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "app.db"))

PENDING, APPROVED, BLOCKED = "pending", "approved", "blocked"
STATUSES = (PENDING, APPROVED, BLOCKED)

# login COLLATE NOCASE — «Ivan» и «ivan» это один и тот же пользователь.
# password_hash пуст у входа через Яндекс, yandex_id — у входа по паролю.
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    login         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT,
    yandex_id     TEXT    UNIQUE,
    email         TEXT,
    name          TEXT,
    status        TEXT    NOT NULL DEFAULT 'pending',
    is_admin      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL,
    last_login_at TEXT
);

-- Профили — долговременная память. Их несколько: у одного человека бывает
-- несколько ролей, и диалог ссылается на ту, в которой сейчас работают.
CREATE TABLE IF NOT EXISTS profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT    NOT NULL,
    -- about — кто и над чем работает, style — как отвечать,
    -- constraints — чего держаться и чего избегать.
    about           TEXT,
    style           TEXT,
    constraints     TEXT,
    response_format TEXT    NOT NULL DEFAULT 'text',
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id, id);

-- Проекты — рабочая память: несколько диалогов об одной задаче делят бриф
-- и накопленные факты. Диалог заводится внутри проекта и остаётся в нём.
CREATE TABLE IF NOT EXISTS projects (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title         TEXT    NOT NULL,
    -- brief пишет пользователь, facts накапливаются из карточек диалогов.
    brief         TEXT,
    facts         TEXT,
    collect_facts INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id, created_at);

-- Инварианты проекта: правила, которые ассистент не имеет права нарушать.
-- Лежат отдельно от переписки намеренно: стратегии контекста режут историю,
-- и правило, сказанное в начале длинного диалога, рано или поздно выпало бы
-- из запроса. Отсюда оно уходит в каждый запрос целиком.
CREATE TABLE IF NOT EXISTS invariants (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- Номер INV-<number> не переиспользуется после удаления: в истории
    -- остаются ответы со ссылкой «нарушает INV-2», и они не должны начать
    -- указывать на другое правило.
    number      INTEGER NOT NULL,
    category    TEXT    NOT NULL,
    rule        TEXT    NOT NULL,
    -- Причина: ею ассистент объясняет отказ, и по замерам именно она
    -- удерживает модель под давлением «я разрешаю нарушить».
    rationale   TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    -- Удаление мягкое: строка остаётся и держит свой номер занятым.
    deleted     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_invariants_project ON invariants(project_id, number);

-- Диалоги и их сообщения. Настройки (модель, thinking) живут на диалоге:
-- вернувшись к старой переписке, возвращаемся и к условиям, при которых она шла.
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT    NOT NULL,
    model      TEXT,
    thinking   INTEGER NOT NULL DEFAULT 0,
    max_tokens INTEGER,
    -- Стратегия сборки запроса: full, window, facts, summary.
    strategy   TEXT    NOT NULL DEFAULT 'full',
    context_n  INTEGER NOT NULL DEFAULT 10,
    facts      TEXT,
    summary    TEXT,
    summary_upto INTEGER,
    -- Ветка: откуда отпочковалась и от какого сообщения.
    parent_id  INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    branched_from INTEGER,
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);

-- Журнал состояния задачи. Текущее состояние лежит в conversations и при
-- каждом нажатии перезаписывается; здесь остаётся хронология — в том числе
-- единственный след того, что задача стояла на паузе.
CREATE TABLE IF NOT EXISTS task_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    -- Вид события отдельной колонкой: пауза не меняет этап, и класть 'paused'
    -- в колонку этапов значило бы хранить в ней не этап.
    kind            TEXT    NOT NULL,   -- 'stage' | 'pause' | 'resume'
    from_stage      TEXT,               -- только у kind='stage'
    stage           TEXT    NOT NULL,   -- куда перешли; для паузы — где она случилась
    note            TEXT,
    author          TEXT    NOT NULL DEFAULT 'user',   -- 'user' | 'model'
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_events_conv ON task_events(conversation_id, id);

-- Серверы MCP, добавленные пользователем. Список свой у каждого: адрес —
-- это то, куда приложение пойдёт с его запросом, и общий перечень означал бы
-- общий выбор. Схемы инструментов кэшируются здесь же: ходить за ними на
-- сервер в каждом запросе к модели значило бы добавлять задержку к каждому
-- сообщению.
CREATE TABLE IF NOT EXISTS mcp_servers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT    NOT NULL,
    url        TEXT    NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 0,
    tools      TEXT,               -- JSON: кэш схем, полученных от сервера
    checked_at TEXT,               -- когда последний раз сходили успешно
    error      TEXT,               -- текст последней ошибки, если она была
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mcp_user ON mcp_servers(user_id, id);

-- tokens_* вынесены в колонки: по ним будет считаться месячный расход.
-- Остальная телеметрия ответа лежит в meta — она только показывается.
CREATE TABLE IF NOT EXISTS messages (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id   INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role              TEXT    NOT NULL,
    content           TEXT    NOT NULL,
    reasoning         TEXT,
    tokens_completion INTEGER NOT NULL DEFAULT 0,
    tokens_total      INTEGER NOT NULL DEFAULT 0,
    meta              TEXT,
    created_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
"""


class SchemaError(RuntimeError):
    """База создана прежней версией приложения."""


def connect() -> sqlite3.Connection:
    """Открывает соединение с включённым WAL и доступом к колонкам по имени."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# Колонки, добавленные после первого выпуска. CREATE TABLE IF NOT EXISTS
# их не создаст в уже существующей таблице, поэтому добавляем отдельно.
MIGRATIONS = {
    "invariants": {
        "deleted": "INTEGER NOT NULL DEFAULT 0",
    },
    "users": {
        # Историческая колонка: одна долговременная память на пользователя.
        # Содержимое перенесено в profiles, отсюда больше не читается.
        "profile_memory": "TEXT",
        # Готовые профили заводятся один раз; флаг не даёт завести их снова
        # после того, как пользователь их поправил или удалил.
        "profiles_seeded": "INTEGER NOT NULL DEFAULT 0",
        # Готовые MCP-серверы заводятся один раз — тем же правилом, что профили.
        "mcp_seeded": "INTEGER NOT NULL DEFAULT 0",
    },
    "conversations": {
        "max_tokens": "INTEGER",
        "compress": "INTEGER NOT NULL DEFAULT 0",
        "summary": "TEXT",
        # id последнего сообщения, вошедшего в конспект
        "summary_upto": "INTEGER",
        "strategy": "TEXT NOT NULL DEFAULT 'full'",
        "context_n": "INTEGER NOT NULL DEFAULT 10",
        "facts": "TEXT",
        "parent_id": "INTEGER",
        "branched_from": "INTEGER",
        # Слои памяти: к какому проекту относится диалог и какие слои
        # подключать к его запросам.
        "project_id": "INTEGER",
        "use_profile": "INTEGER NOT NULL DEFAULT 1",
        "use_project": "INTEGER NOT NULL DEFAULT 1",
        # Какой профиль долговременной памяти подключён к диалогу.
        "profile_id": "INTEGER",
        # Состояние задачи: этап, текущий шаг, ожидаемое действие и чей ход.
        # Пауза — отдельный флаг, а не этап: приостановить можно на любом.
        "task_mode": "TEXT NOT NULL DEFAULT 'chat'",
        "task_stage": "TEXT NOT NULL DEFAULT 'planning'",
        "task_step": "TEXT",
        "task_expected": "TEXT",
        "task_actor": "TEXT NOT NULL DEFAULT 'user'",
        "task_paused": "INTEGER NOT NULL DEFAULT 0",
        "task_auto": "INTEGER NOT NULL DEFAULT 0",
        "task_updated_at": "TEXT",
        # Автопилот: модель отвечает и за пользователя, задача проходит
        # этапы сама. Лимит ходов — предохранитель от бесконечного цикла.
        "task_autopilot": "INTEGER NOT NULL DEFAULT 0",
        "task_max_turns": "INTEGER NOT NULL DEFAULT 8",
        # Условия входа в этап: без отметки вперёд не пускают никого.
        "guard_plan_approved": "INTEGER NOT NULL DEFAULT 0",
        "guard_result_ready": "INTEGER NOT NULL DEFAULT 0",
        "guard_validation_passed": "INTEGER NOT NULL DEFAULT 0",
    },
}

# Что выполнить один раз сразу после появления колонки.
BACKFILL = {
    # Прежний переключатель «сжимать историю» стал одной из стратегий.
    ("conversations", "strategy"):
        "UPDATE conversations SET strategy = 'summary' WHERE compress = 1",
}


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for table, columns in MIGRATIONS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if name in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
            backfill = BACKFILL.get((table, name))
            if backfill:
                conn.execute(backfill)


def init() -> None:
    """Создаёт таблицы, если их ещё нет, и проверяет совместимость схемы."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
        columns = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
        pending = [r[0] for r in conn.execute(
            "SELECT id FROM users WHERE COALESCE(profiles_seeded, 0) = 0")]
        pending_mcp = [r[0] for r in conn.execute(
            "SELECT id FROM users WHERE COALESCE(mcp_seeded, 0) < ?",
            (MCP_PRESETS_VERSION,))]
    # Готовые профили заводим и тем, кто зарегистрировался до их появления.
    for user_id in pending:
        seed_profiles(user_id)
    for user_id in pending_mcp:
        seed_mcp_servers(user_id)
    missing = {"login", "password_hash", "status", "is_admin"} - columns
    if missing:
        raise SchemaError(
            f"В таблице users нет колонок: {', '.join(sorted(missing))}. "
            f"База {DB_PATH} создана прежней версией — удалите её и запустите заново."
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fetch(conn: sqlite3.Connection, where: str, value) -> dict | None:
    row = conn.execute(f"SELECT * FROM users WHERE {where}", (value,)).fetchone()
    return dict(row) if row else None


# ---------- вход по логину и паролю ----------

def create_user(
    login: str, password_hash: str, *, email: str = "", name: str = "", is_admin: bool = False
) -> dict:
    """Заводит пользователя. Возвращает None, если логин уже занят."""
    with connect() as conn:
        if _fetch(conn, "login = ?", login):
            return None
        conn.execute(
            "INSERT INTO users (login, password_hash, email, name, status, is_admin,"
            " created_at, last_login_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                login,
                password_hash,
                email or None,
                name or login,
                APPROVED if is_admin else PENDING,
                int(is_admin),
                _now(),
                _now(),
            ),
        )
        user = _fetch(conn, "login = ?", login)
    # Заводим профили отдельным подключением: внутри открытой транзакции
    # второй писатель в WAL упёрся бы в блокировку.
    if user:
        seed_profiles(user["id"])
        seed_mcp_servers(user["id"])
        user = get(user["id"])
    return user


def get_by_login(login: str) -> dict | None:
    """Ищет пользователя по логину без учёта регистра."""
    with connect() as conn:
        return _fetch(conn, "login = ?", login)


def set_password(user_id: int, password_hash: str) -> bool:
    """Меняет пароль. Возвращает False, если пользователь не найден."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
        )
    return cur.rowcount > 0


def touch_login(user_id: int) -> None:
    """Отмечает момент успешного входа."""
    with connect() as conn:
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (_now(), user_id))


# ---------- вход через Яндекс ----------

def upsert_from_yandex(profile: dict, *, admin_logins: set[str]) -> dict:
    """Создаёт или обновляет пользователя по профилю Яндекса.

    Новый пользователь получает статус pending — доступ подтверждает админ.
    Логины из admin_logins сразу становятся админами с доступом.
    """
    yandex_id = str(profile["id"])
    login = profile.get("login") or yandex_id
    email = profile.get("default_email")
    name = profile.get("real_name") or profile.get("display_name") or login
    is_admin = 1 if login.lower() in admin_logins else 0

    with connect() as conn:
        existing = _fetch(conn, "yandex_id = ?", yandex_id)

        if existing is None:
            # Логин мог быть занят аккаунтом с паролем — тогда разводим суффиксом.
            if _fetch(conn, "login = ?", login):
                login = f"{login}@yandex"
            conn.execute(
                "INSERT INTO users (login, yandex_id, email, name, status, is_admin,"
                " created_at, last_login_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    login,
                    yandex_id,
                    email,
                    name,
                    APPROVED if is_admin else PENDING,
                    is_admin,
                    _now(),
                    _now(),
                ),
            )
        else:
            # Профиль мог измениться; статус трогаем только для админов из конфига.
            status = APPROVED if is_admin else existing["status"]
            conn.execute(
                "UPDATE users SET email = ?, name = ?, is_admin = ?, status = ?,"
                " last_login_at = ? WHERE yandex_id = ?",
                (email, name, is_admin, status, _now(), yandex_id),
            )

        user = _fetch(conn, "yandex_id = ?", yandex_id)
    if user:
        seed_profiles(user["id"])
        seed_mcp_servers(user["id"])
        user = get(user["id"])
    return user


# ---------- общее ----------

def get(user_id: int) -> dict | None:
    """Возвращает пользователя по внутреннему id."""
    with connect() as conn:
        return _fetch(conn, "id = ?", user_id)


def list_all() -> list[dict]:
    """Все пользователи: сначала ожидающие подтверждения, затем по дате входа."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END,"
            " COALESCE(last_login_at, created_at) DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def set_status(user_id: int, status: str) -> bool:
    """Меняет статус доступа. Возвращает False, если пользователь не найден."""
    if status not in STATUSES:
        raise ValueError(f"Недопустимый статус: {status}")
    with connect() as conn:
        cur = conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
    return cur.rowcount > 0


PROFILE_LIMIT = 4000


def set_profile_memory(user_id: int, text: str) -> None:
    """Сохраняет долговременную память пользователя — текст о нём самом.

    Потолок нужен не ради места в базе: этот текст уходит в каждый запрос
    всех диалогов, где слой подключён, и оплачивается заново каждый раз.
    """
    with connect() as conn:
        conn.execute(
            "UPDATE users SET profile_memory = ? WHERE id = ?",
            (text.strip()[:PROFILE_LIMIT], user_id),
        )


def count_admins() -> int:
    """Число админов — чтобы не остаться без единого администратора."""
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1").fetchone()[0]


# ---------- профили ----------
#
# Долговременная память. У человека несколько ролей — по профилю на каждую:
# в одном он пишет банковское приложение на Kotlin, в другом разбирает выгрузки
# на Python, и ответы должны отличаться не только содержанием, но и стилем.

TEXT_FORMAT, JSON_FORMAT = "text", "json_object"
# Проверено на живом API: json_schema и regex провайдер пока отклоняет
# («This response_format type is unavailable now»), поэтому их здесь нет.
RESPONSE_FORMATS = (TEXT_FORMAT, JSON_FORMAT)

NEW_PROFILE_TITLE = "Новый профиль"
PROFILE_TITLE_LIMIT = 60
LEGACY_PROFILE_TITLE = "Мой профиль"

# Готовые профили: две роли на Android и две на Python. Это обычные строки —
# их правят и удаляют, как любые другие.
PROFILE_PRESETS = (
    {
        "title": "Android: мобильный банк",
        "about": "Роль: Senior Android Developer.\n"
                 "Проект: мобильный банк, команда 4 человека.",
        "style": "Краткие ответы.\nФормальный тон.\nС примерами кода на Kotlin.",
        "constraints": "Стек: Kotlin, Ktor client, Coroutines.\n"
                       "Архитектура: MVVM.\n"
                       "Минимум зависимостей, только бесплатные API.\n"
                       "minSdk 26.",
        "response_format": TEXT_FORMAT,
    },
    {
        "title": "Android: инди-приложение",
        "about": "Роль: Android-разработчик-одиночка.\n"
                 "Проект: трекер привычек в Google Play, всё делаю сам.",
        "style": "Подробные ответы с пошаговыми объяснениями.\n"
                 "Разговорный тон.\nВсегда с примерами.",
        "constraints": "Стек: Kotlin, Jetpack Compose, Room, Hilt.\n"
                       "Без платных SDK и подписок.\n"
                       "Нужны готовые сниппеты, а не общие советы.",
        "response_format": TEXT_FORMAT,
    },
    {
        "title": "Python: бэкенд на FastAPI",
        "about": "Роль: Python backend developer.\n"
                 "Проект: API сервиса на FastAPI и PostgreSQL, в команде двое.",
        "style": "Кратко, формально, без вводных фраз.\n"
                 "С примерами кода и аннотациями типов.",
        "constraints": "Python 3.13, FastAPI, Pydantic v2, SQLAlchemy 2.x, pytest.\n"
                       "Где хватает стандартной библиотеки — обходимся ею.\n"
                       "Синхронный и асинхронный код не смешивать.",
        "response_format": TEXT_FORMAT,
    },
    {
        "title": "Python: данные и отчёты",
        "about": "Роль: Python-разработчик по данным.\n"
                 "Задачи: разбор выгрузок, отчёты, автоматизация рутины.",
        "style": "Предельно сжато, без вводных фраз и извинений.\n"
                 "Ответ — машиночитаемый json.",
        "constraints": "pandas, polars, matplotlib.\n"
                       "Код запускается как обычный скрипт, без Jupyter-магии.\n"
                       "Большие выгрузки читать по частям.",
        "response_format": JSON_FORMAT,
    },
)


def _profile_fields(title, about, style, constraints, response_format):
    """Приводит поля профиля к допустимым значениям."""
    if response_format not in RESPONSE_FORMATS:
        raise ValueError(f"Неизвестный формат ответа: {response_format}")
    return (
        title.strip()[:PROFILE_TITLE_LIMIT] or NEW_PROFILE_TITLE,
        (about or "").strip()[:PROFILE_LIMIT],
        (style or "").strip()[:PROFILE_LIMIT],
        (constraints or "").strip()[:PROFILE_LIMIT],
        response_format,
    )


def create_profile(
    user_id: int, *, title: str = "", about: str = "", style: str = "",
    constraints: str = "", response_format: str = TEXT_FORMAT,
) -> dict:
    """Заводит профиль и возвращает его."""
    values = _profile_fields(title, about, style, constraints, response_format)
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO profiles (user_id, title, about, style, constraints,"
            " response_format, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, *values, _now(), _now()),
        )
        row = conn.execute("SELECT * FROM profiles WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_profile(profile_id: int, user_id: int) -> dict | None:
    """Профиль по id, только если он принадлежит этому пользователю."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM profiles WHERE id = ? AND user_id = ?", (profile_id, user_id)
        ).fetchone()
    return dict(row) if row else None


def list_profiles(user_id: int) -> list[dict]:
    """Профили пользователя в порядке создания, с числом диалогов у каждого."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT p.*, (SELECT COUNT(*) FROM conversations c WHERE c.profile_id = p.id)"
            " AS conversation_count FROM profiles p WHERE p.user_id = ? ORDER BY p.id",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_profile(
    profile_id: int, user_id: int, *, title: str | None = None, about: str | None = None,
    style: str | None = None, constraints: str | None = None,
    response_format: str | None = None,
) -> dict | None:
    """Меняет поля профиля. None — профиль чужой или его нет."""
    sets, values = [], []
    if title is not None:
        sets.append("title = ?")
        values.append(title.strip()[:PROFILE_TITLE_LIMIT] or NEW_PROFILE_TITLE)
    for name, value in (("about", about), ("style", style), ("constraints", constraints)):
        if value is not None:
            sets.append(f"{name} = ?")
            values.append(value.strip()[:PROFILE_LIMIT])
    if response_format is not None:
        if response_format not in RESPONSE_FORMATS:
            raise ValueError(f"Неизвестный формат ответа: {response_format}")
        sets.append("response_format = ?")
        values.append(response_format)
    if not sets:
        return get_profile(profile_id, user_id)

    sets.append("updated_at = ?")
    values.extend([_now(), profile_id, user_id])
    with connect() as conn:
        cur = conn.execute(
            f"UPDATE profiles SET {', '.join(sets)} WHERE id = ? AND user_id = ?", values
        )
    return get_profile(profile_id, user_id) if cur.rowcount else None


def delete_profile(profile_id: int, user_id: int) -> int | None:
    """Удаляет профиль. Диалоги остаются — просто теряют долговременный слой.

    Возвращает число осиротевших диалогов или None, если профиля нет.
    """
    with connect() as conn:
        if conn.execute(
            "SELECT 1 FROM profiles WHERE id = ? AND user_id = ?", (profile_id, user_id)
        ).fetchone() is None:
            return None
        orphaned = conn.execute(
            "UPDATE conversations SET profile_id = NULL WHERE profile_id = ? AND user_id = ?",
            (profile_id, user_id),
        ).rowcount
        conn.execute("DELETE FROM profiles WHERE id = ? AND user_id = ?", (profile_id, user_id))
    return orphaned


def seed_profiles(user_id: int) -> int:
    """Заводит готовые профили и переносит прежнюю долговременную память.

    Вызывается один раз на пользователя: и при регистрации, и при старте для
    тех, кто завёлся раньше. Флаг `profiles_seeded` важнее, чем «профилей нет»:
    иначе удалённые заготовки возвращались бы после каждого перезапуска.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT profile_memory, COALESCE(profiles_seeded, 0) AS seeded"
            " FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None or row["seeded"]:
            return 0

        created = 0
        for preset in PROFILE_PRESETS:
            conn.execute(
                "INSERT INTO profiles (user_id, title, about, style, constraints,"
                " response_format, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, preset["title"], preset["about"], preset["style"],
                 preset["constraints"], preset["response_format"], _now(), _now()),
            )
            created += 1

        legacy = (row["profile_memory"] or "").strip()
        if legacy:
            # Прежний единственный профиль не теряем и оставляем подключённым:
            # иначе старые диалоги молча лишились бы долговременного слоя.
            cur = conn.execute(
                "INSERT INTO profiles (user_id, title, about, style, constraints,"
                " response_format, created_at, updated_at) VALUES (?, ?, ?, '', '', ?, ?, ?)",
                (user_id, LEGACY_PROFILE_TITLE, legacy[:PROFILE_LIMIT],
                 TEXT_FORMAT, _now(), _now()),
            )
            conn.execute(
                "UPDATE conversations SET profile_id = ?"
                " WHERE user_id = ? AND profile_id IS NULL",
                (cur.lastrowid, user_id),
            )
            created += 1

        conn.execute("UPDATE users SET profiles_seeded = 1 WHERE id = ?", (user_id,))
    return created


# ---------- проекты ----------
#
# Проект — рабочая память: общий бриф и общая карточка фактов для нескольких
# диалогов об одной задаче. Владелец проверяется в каждом запросе, как и у диалогов.

NEW_PROJECT_TITLE = "Новый проект"
PROJECT_TITLE_LIMIT = 60
BRIEF_LIMIT = 4000


def create_project(user_id: int, *, title: str = "", brief: str = "") -> dict:
    """Заводит проект и возвращает его."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO projects (user_id, title, brief, collect_facts, created_at, updated_at)"
            " VALUES (?, ?, ?, 1, ?, ?)",
            (user_id, title.strip()[:PROJECT_TITLE_LIMIT] or NEW_PROJECT_TITLE,
             brief.strip()[:BRIEF_LIMIT], _now(), _now()),
        )
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_project(project_id: int, user_id: int) -> dict | None:
    """Проект по id, только если он принадлежит этому пользователю."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)
        ).fetchone()
    return dict(row) if row else None


def list_projects(user_id: int) -> list[dict]:
    """Проекты пользователя в порядке создания, с числом диалогов в каждом.

    Порядок постоянный, а не по свежести: список проектов — это оглавление,
    и переставлять его после каждого сообщения значило бы терять место глазами.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT p.*, (SELECT COUNT(*) FROM conversations c WHERE c.project_id = p.id)"
            " AS conversation_count FROM projects p WHERE p.user_id = ?"
            " ORDER BY p.created_at, p.id",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_project(
    project_id: int, user_id: int, *, title: str | None = None,
    brief: str | None = None, collect_facts: bool | None = None,
) -> dict | None:
    """Меняет название, бриф или сбор фактов. None — проект чужой или его нет."""
    sets, values = [], []
    if title is not None:
        sets.append("title = ?")
        values.append(title.strip()[:PROJECT_TITLE_LIMIT] or NEW_PROJECT_TITLE)
    if brief is not None:
        sets.append("brief = ?")
        values.append(brief.strip()[:BRIEF_LIMIT])
    if collect_facts is not None:
        sets.append("collect_facts = ?")
        values.append(int(collect_facts))
    if not sets:
        return get_project(project_id, user_id)

    sets.append("updated_at = ?")
    values.extend([_now(), project_id, user_id])
    with connect() as conn:
        cur = conn.execute(
            f"UPDATE projects SET {', '.join(sets)} WHERE id = ? AND user_id = ?", values
        )
    return get_project(project_id, user_id) if cur.rowcount else None


def set_project_facts(project_id: int, facts_json: str) -> None:
    """Сохраняет накопленную карточку фактов проекта."""
    with connect() as conn:
        conn.execute(
            "UPDATE projects SET facts = ?, updated_at = ? WHERE id = ?",
            (facts_json, _now(), project_id),
        )


def delete_project(project_id: int, user_id: int) -> int | None:
    """Удаляет проект вместе с его диалогами. Возвращает число удалённых диалогов.

    Диалоги удаляются явно, а не каскадом по внешнему ключу: колонка project_id
    добавлена миграцией, и полагаться на её ограничение в базах, созданных
    прежними версиями, нельзя. Сообщения уходят следом — этот каскад объявлен
    при создании таблицы messages и работает везде.
    """
    with connect() as conn:
        if conn.execute(
            "SELECT 1 FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id)
        ).fetchone() is None:
            return None
        killed = conn.execute(
            "DELETE FROM conversations WHERE project_id = ? AND user_id = ?",
            (project_id, user_id),
        ).rowcount
        conn.execute("DELETE FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id))
    return killed


# ---------- инварианты ----------

ARCHITECTURE, DECISION, STACK, BUSINESS = "architecture", "decision", "stack", "business"
INVARIANT_CATEGORIES = (ARCHITECTURE, DECISION, STACK, BUSINESS)
INVARIANT_LIMIT = 1000


def _check_category(category: str) -> str:
    if category not in INVARIANT_CATEGORIES:
        raise ValueError(f"Неизвестная категория инварианта: {category}")
    return category


def list_invariants(project_id: int, user_id: int, *, only_active: bool = False) -> list[dict]:
    """Инварианты проекта по порядку номеров."""
    sql = "SELECT * FROM invariants WHERE project_id = ? AND user_id = ? AND deleted = 0"
    if only_active:
        sql += " AND active = 1"
    with connect() as conn:
        rows = conn.execute(sql + " ORDER BY number", (project_id, user_id)).fetchall()
    return [dict(r) for r in rows]


def get_invariant(invariant_id: int, user_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM invariants WHERE id = ? AND user_id = ? AND deleted = 0",
            (invariant_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def create_invariant(
    project_id: int, user_id: int, *, category: str, rule: str, rationale: str = "",
) -> dict:
    """Заводит инвариант со следующим свободным номером проекта."""
    rule = rule.strip()[:INVARIANT_LIMIT]
    if not rule:
        raise ValueError("Пустое правило")
    with connect() as conn:
        # Считаем и удалённые: удаление мягкое, поэтому номер удалённого
        # правила остаётся занятым и ссылки в истории не начнут указывать
        # на новое правило.
        number = conn.execute(
            "SELECT COALESCE(MAX(number), 0) + 1 FROM invariants WHERE project_id = ?",
            (project_id,),
        ).fetchone()[0]
        cur = conn.execute(
            "INSERT INTO invariants (project_id, user_id, number, category, rule, rationale,"
            " active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
            (project_id, user_id, number, _check_category(category), rule,
             rationale.strip()[:INVARIANT_LIMIT] or None, _now(), _now()),
        )
        row = conn.execute("SELECT * FROM invariants WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def update_invariant(
    invariant_id: int, user_id: int, *, category: str | None = None, rule: str | None = None,
    rationale: str | None = None, active: bool | None = None,
) -> dict | None:
    sets, values = [], []
    if category is not None:
        sets.append("category = ?")
        values.append(_check_category(category))
    if rule is not None:
        if not rule.strip():
            raise ValueError("Пустое правило")
        sets.append("rule = ?")
        values.append(rule.strip()[:INVARIANT_LIMIT])
    if rationale is not None:
        sets.append("rationale = ?")
        values.append(rationale.strip()[:INVARIANT_LIMIT] or None)
    if active is not None:
        sets.append("active = ?")
        values.append(int(active))
    if not sets:
        return get_invariant(invariant_id, user_id)
    sets.append("updated_at = ?")
    values.extend([_now(), invariant_id, user_id])
    with connect() as conn:
        cur = conn.execute(
            f"UPDATE invariants SET {', '.join(sets)} WHERE id = ? AND user_id = ? AND deleted = 0",
            values,
        )
    return get_invariant(invariant_id, user_id) if cur.rowcount else None


def delete_invariant(invariant_id: int, user_id: int) -> bool:
    """Убирает инвариант из проекта, не освобождая его номер."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE invariants SET deleted = 1, active = 0, updated_at = ?"
            " WHERE id = ? AND user_id = ? AND deleted = 0",
            (_now(), invariant_id, user_id),
        )
    return cur.rowcount > 0


# ---------- серверы MCP ----------
#
# Каждый сервер объявляет инструменты со схемами аргументов; включённые
# уходят в запрос к модели, и она сама решает, что вызвать. Владелец, как и
# везде, проверяется прямо в запросе.

MCP_TITLE_LIMIT = 80
MCP_URL_LIMIT = 500

# Свои серверы — соседние процессы на том же хосте. Адреса переменными: пока
# они смотрят только внутрь, но публичный вариант включится одной строкой
# в .env, без правок кода.
WEATHER_MCP_URL = os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8001/mcp")
SCHEDULER_MCP_URL = os.getenv("SCHEDULER_MCP_URL", "http://127.0.0.1:8002/mcp")
OWN_MCP_URLS = (WEATHER_MCP_URL, SCHEDULER_MCP_URL)

# Готовые серверы — все проверены живым запросом: отвечают без ключа и
# регистрации. Включены те, что знают меняющиеся данные: курсы и погоду.
# Схемы выключенных не занимают токены в запросе.
#
# since — версия, в которой сервер появился в списке. Колонка users.mcp_seeded
# хранит не «да/нет», а номер версии: иначе добавить пресет тем, кто уже вошёл,
# было бы нечем — сброс флага вернул бы им и удалённые заготовки.
MCP_PRESETS_VERSION = 3

MCP_PRESETS = (
    {"title": "Курсы валют", "since": 1,
     "url": "https://currency-mcp.wesbos.com/mcp", "enabled": 1},
    {"title": "DeepWiki — документация репозиториев", "since": 1,
     "url": "https://mcp.deepwiki.com/mcp", "enabled": 0},
    {"title": "Context7 — документация библиотек", "since": 1,
     "url": "https://mcp.context7.com/mcp", "enabled": 0},
    {"title": "GitMCP — документация по ссылке", "since": 1,
     "url": "https://gitmcp.io/docs", "enabled": 0},
    {"title": "Погода, прогноз и качество воздуха", "since": 2,
     "url": WEATHER_MCP_URL, "enabled": 1},
    {"title": "Планировщик поручений", "since": 3,
     "url": SCHEDULER_MCP_URL, "enabled": 1},
)


def list_mcp_servers(user_id: int, *, only_enabled: bool = False) -> list[dict]:
    """Серверы пользователя. only_enabled — те, что уходят в запрос."""
    where = " AND enabled = 1" if only_enabled else ""
    with connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM mcp_servers WHERE user_id = ?{where} ORDER BY id",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_mcp_server(server_id: int, user_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM mcp_servers WHERE id = ? AND user_id = ?",
            (server_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def create_mcp_server(user_id: int, *, title: str, url: str,
                      enabled: bool = False) -> dict:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO mcp_servers (user_id, title, url, enabled, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, title.strip()[:MCP_TITLE_LIMIT] or url[:MCP_TITLE_LIMIT],
             url.strip()[:MCP_URL_LIMIT], int(enabled), _now()),
        )
        row = conn.execute(
            "SELECT * FROM mcp_servers WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def set_mcp_enabled(server_id: int, user_id: int, enabled: bool) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "UPDATE mcp_servers SET enabled = ? WHERE id = ? AND user_id = ?",
            (int(enabled), server_id, user_id),
        )
    return cur.rowcount > 0


def set_mcp_tools(server_id: int, user_id: int, *,
                  tools_json: str | None, error: str | None) -> bool:
    """Запоминает результат похода на сервер: схемы или текст ошибки.

    Прежний кэш при ошибке не стирается: сервер мог отвалиться на минуту, а
    диалог с его инструментами продолжается.
    """
    with connect() as conn:
        if error:
            cur = conn.execute(
                "UPDATE mcp_servers SET error = ? WHERE id = ? AND user_id = ?",
                (error[:500], server_id, user_id),
            )
        else:
            cur = conn.execute(
                "UPDATE mcp_servers SET tools = ?, checked_at = ?, error = NULL"
                " WHERE id = ? AND user_id = ?",
                (tools_json, _now(), server_id, user_id),
            )
    return cur.rowcount > 0


def delete_mcp_server(server_id: int, user_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM mcp_servers WHERE id = ? AND user_id = ?",
            (server_id, user_id),
        )
    return cur.rowcount > 0


def seed_mcp_servers(user_id: int) -> int:
    """Доводит список готовых серверов до текущей версии пресетов.

    Заводятся только те, что появились позже отметки пользователя: удалённые
    им заготовки прежних версий не возвращаются — ровно как с готовыми
    профилями, где тем же занят флаг profiles_seeded.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(mcp_seeded, 0) AS seeded FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None or row["seeded"] >= MCP_PRESETS_VERSION:
            return 0
        added = 0
        for preset in MCP_PRESETS:
            if preset["since"] <= row["seeded"]:
                continue
            conn.execute(
                "INSERT INTO mcp_servers (user_id, title, url, enabled, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (user_id, preset["title"], preset["url"], preset["enabled"], _now()),
            )
            added += 1
        conn.execute("UPDATE users SET mcp_seeded = ? WHERE id = ?",
                     (MCP_PRESETS_VERSION, user_id))
    return added


# ---------- диалоги ----------
#
# Владелец проверяется прямо в запросе: во всех функциях есть условие
# user_id = ?. Иначе чужую переписку можно было бы открыть, подставив чужой id.

NEW_TITLE = "Новый диалог"
TITLE_LIMIT = 60

# Способы собрать запрос из переписки.
FULL, WINDOW, FACTS, SUMMARY = "full", "window", "facts", "summary"
STRATEGIES = (FULL, WINDOW, FACTS, SUMMARY)
DEFAULT_CONTEXT_N = 10

# Новые диалоги начинают с фактов: карточка ключ-значение переживает обрезку
# хвоста и, если диалог в проекте, перетекает в рабочую память проекта.
DEFAULT_STRATEGY = FACTS


def create_conversation(
    user_id: int, *, title: str = NEW_TITLE, model: str | None = None,
    thinking: bool = False, max_tokens: int | None = None,
    strategy: str = DEFAULT_STRATEGY, context_n: int = DEFAULT_CONTEXT_N,
    project_id: int | None = None, profile_id: int | None = None,
) -> dict:
    """Заводит пустой диалог и возвращает его."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Неизвестная стратегия: {strategy}")
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, model, thinking, max_tokens,"
            " strategy, context_n, project_id, profile_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, title.strip()[:TITLE_LIMIT] or NEW_TITLE, model, int(thinking),
             max_tokens, strategy, max(1, min(context_n, 200)), project_id, profile_id,
             _now(), _now()),
        )
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_conversation(conversation_id: int, user_id: int) -> dict | None:
    """Диалог по id, только если он принадлежит этому пользователю."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
        ).fetchone()
    return dict(row) if row else None


def conversation_owner(conversation_id: int) -> dict | None:
    """Владелец диалога. Исполнителю заданий нужен пользователь, а не только чат:
    задание приходит из планировщика, где пользователей нет вовсе."""
    with connect() as conn:
        row = conn.execute(
            "SELECT user_id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return get(row["user_id"]) if row else None


def list_conversations(user_id: int) -> list[dict]:
    """Диалоги пользователя, свежие сверху, с числом сообщений в каждом."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id)"
            " AS message_count FROM conversations c WHERE c.user_id = ?"
            " ORDER BY c.updated_at DESC, c.id DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_conversation(
    conversation_id: int, user_id: int, *, title: str | None = None,
    model: str | None = None, thinking: bool | None = None,
    max_tokens: int | None = None, clear_max_tokens: bool = False,
    compress: bool | None = None, strategy: str | None = None,
    context_n: int | None = None,
    use_profile: bool | None = None, use_project: bool | None = None,
    profile_id: int | None = None, clear_profile: bool = False,
) -> dict | None:
    """Меняет название или настройки. Возвращает None, если диалог чужой или его нет."""
    sets, values = [], []
    if title is not None:
        sets.append("title = ?")
        values.append(title.strip()[:TITLE_LIMIT] or NEW_TITLE)
    if model is not None:
        sets.append("model = ?")
        values.append(model)
    if thinking is not None:
        sets.append("thinking = ?")
        values.append(int(thinking))
    # Снять лимит и не трогать его — разные намерения, поэтому отдельный флаг:
    # по одному лишь max_tokens=None их не различить.
    if compress is not None:
        sets.append("compress = ?")
        values.append(int(compress))
    if strategy is not None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Неизвестная стратегия: {strategy}")
        sets.append("strategy = ?")
        values.append(strategy)
    if context_n is not None:
        sets.append("context_n = ?")
        values.append(max(1, min(context_n, 200)))
    if use_profile is not None:
        sets.append("use_profile = ?")
        values.append(int(use_profile))
    if use_project is not None:
        sets.append("use_project = ?")
        values.append(int(use_project))
    # «Снять профиль» и «не трогать профиль» — разные намерения, по одному
    # profile_id=None их не различить, отсюда отдельный флаг.
    if clear_profile:
        sets.append("profile_id = NULL")
    elif profile_id is not None:
        sets.append("profile_id = ?")
        values.append(profile_id)
    if clear_max_tokens:
        sets.append("max_tokens = NULL")
    elif max_tokens is not None:
        sets.append("max_tokens = ?")
        values.append(max_tokens)
    if not sets:
        return get_conversation(conversation_id, user_id)

    sets.append("updated_at = ?")
    values.extend([_now(), conversation_id, user_id])
    with connect() as conn:
        cur = conn.execute(
            f"UPDATE conversations SET {', '.join(sets)} WHERE id = ? AND user_id = ?", values
        )
    return get_conversation(conversation_id, user_id) if cur.rowcount else None


def delete_conversation(conversation_id: int, user_id: int) -> bool:
    """Удаляет диалог вместе с сообщениями (каскадом по внешнему ключу)."""
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
        )
    return cur.rowcount > 0


def touch_conversation(conversation_id: int) -> None:
    """Отмечает диалог как недавно изменённый — он всплывает в списке."""
    with connect() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id)
        )


# ---------- состояние задачи ----------
#
# Хранение состояния и журнал. Правила переходов живут в task.py: здесь только
# запись, иначе модуль базы начал бы знать, как устроена работа над задачей.

CHAT_MODE, TASK_MODE = "chat", "task"
TASK_MODES = (CHAT_MODE, TASK_MODE)

PLANNING, EXECUTION, VALIDATION, DONE = "planning", "execution", "validation", "done"
STAGES = (PLANNING, EXECUTION, VALIDATION, DONE)
STAGE_LABELS = {
    PLANNING: "планирование",
    EXECUTION: "выполнение",
    VALIDATION: "проверка",
    DONE: "готово",
}

USER_ACTOR, ASSISTANT_ACTOR = "user", "assistant"
ACTORS = (USER_ACTOR, ASSISTANT_ACTOR)

# Условие входа в этап. Ключ — название условия, значение — этап, который оно
# открывает. Сама таблица переходов живёт в task.py, здесь только хранение.
PLAN_APPROVED, RESULT_READY, VALIDATION_PASSED = (
    "plan_approved", "result_ready", "validation_passed")
GUARD_STAGE = {
    PLAN_APPROVED: EXECUTION,
    RESULT_READY: VALIDATION,
    VALIDATION_PASSED: DONE,
}
GUARD_COLUMN = {name: f"guard_{name}" for name in GUARD_STAGE}
GUARD_LABELS = {
    PLAN_APPROVED: "план утверждён",
    RESULT_READY: "результат предъявлен",
    VALIDATION_PASSED: "проверка пройдена",
}

STEP_LIMIT = 500

# Потолок лимита ходов автопилота: каждый ход — четыре вызова к модели.
MAX_TURNS_LIMIT = 20


def _add_task_event(conn, conversation_id: int, kind: str, stage: str, *,
                    from_stage: str | None = None, note: str | None = None,
                    author: str = "user") -> None:
    conn.execute(
        "INSERT INTO task_events (conversation_id, kind, from_stage, stage, note,"
        " author, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (conversation_id, kind, from_stage, stage, (note or "").strip()[:STEP_LIMIT] or None,
         author, _now()),
    )


def update_task(
    conversation_id: int, user_id: int, *, mode: str | None = None,
    step: str | None = None, expected: str | None = None, actor: str | None = None,
    auto: bool | None = None, autopilot: bool | None = None, max_turns: int | None = None,
) -> dict | None:
    """Меняет настройки задачи. Этап и пауза идут отдельными функциями —
    у них есть правила и журнал."""
    sets, values = [], []
    if mode is not None:
        if mode not in TASK_MODES:
            raise ValueError(f"Неизвестный режим диалога: {mode}")
        sets.append("task_mode = ?")
        values.append(mode)
    for name, value in (("task_step", step), ("task_expected", expected)):
        if value is not None:
            sets.append(f"{name} = ?")
            values.append(value.strip()[:STEP_LIMIT])
    if actor is not None:
        if actor not in ACTORS:
            raise ValueError(f"Неизвестный участник: {actor}")
        sets.append("task_actor = ?")
        values.append(actor)
    if auto is not None:
        sets.append("task_auto = ?")
        values.append(int(auto))
    if autopilot is not None:
        sets.append("task_autopilot = ?")
        values.append(int(autopilot))
        # Автопилот без автоматического переключения крутился бы на месте:
        # модель отвечает за пользователя, а этап не двигается никогда.
        if autopilot:
            sets.append("task_auto = 1")
    if max_turns is not None:
        sets.append("task_max_turns = ?")
        values.append(max(1, min(int(max_turns), MAX_TURNS_LIMIT)))
    if not sets:
        return get_conversation(conversation_id, user_id)

    sets.append("task_updated_at = ?")
    values.extend([_now(), conversation_id, user_id])
    with connect() as conn:
        cur = conn.execute(
            f"UPDATE conversations SET {', '.join(sets)} WHERE id = ? AND user_id = ?", values
        )
    return get_conversation(conversation_id, user_id) if cur.rowcount else None


def set_task_guard(
    conversation_id: int, user_id: int, guard: str, value: bool, *,
    note: str | None = None, author: str = "user",
) -> dict | None:
    """Отмечает или снимает условие перехода. Отдельное действие, не побочный
    эффект переключения этапа: отметка — это и есть акт утверждения."""
    if guard not in GUARD_STAGE:
        raise ValueError(f"Неизвестное условие: {guard}")
    conversation = get_conversation(conversation_id, user_id)
    if conversation is None:
        return None

    with connect() as conn:
        conn.execute(
            f"UPDATE conversations SET {GUARD_COLUMN[guard]} = ?, task_updated_at = ?"
            " WHERE id = ? AND user_id = ?",
            (int(value), _now(), conversation_id, user_id),
        )
        _add_task_event(
            conn, conversation_id, "guard", GUARD_STAGE[guard],
            note=note or (GUARD_LABELS[guard] if value else f"снято: {GUARD_LABELS[guard]}"),
            author=author,
        )
    return get_conversation(conversation_id, user_id)


def set_task_stage(
    conversation_id: int, user_id: int, stage: str, *,
    note: str | None = None, author: str = "user",
) -> dict | None:
    """Ставит этап и записывает переход в журнал.

    Допустимость перехода проверяет вызывающий (task.py): здесь нет знания
    о том, какой этап за каким следует.
    """
    if stage not in STAGES:
        raise ValueError(f"Неизвестный этап: {stage}")
    conversation = get_conversation(conversation_id, user_id)
    if conversation is None:
        return None

    # Возврат назад снимает условия своего этапа и всех последующих: иначе
    # можно было бы вернуться к планированию и тут же прыгнуть вперёд по
    # старой отметке, хотя план уже переделывают.
    back_to = STAGES.index(stage)
    cleared = [
        name for name, opens in GUARD_STAGE.items()
        if STAGES.index(opens) > back_to and conversation[GUARD_COLUMN[name]]
    ] if back_to < STAGES.index(conversation["task_stage"]) else []

    with connect() as conn:
        conn.execute(
            "UPDATE conversations SET task_stage = ?, task_updated_at = ?"
            " WHERE id = ? AND user_id = ?",
            (stage, _now(), conversation_id, user_id),
        )
        # Отметка в самой переписке. Блока состояния мало: в истории остаются
        # реплики, сказанные на прежнем этапе, и модель повторяла «сейчас этап
        # планирования», когда задача уже была в выполнении.
        if conversation["task_stage"] != stage:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, reasoning,"
                " tokens_completion, tokens_total, meta, created_at)"
                " VALUES (?, 'system', ?, NULL, 0, 0, ?, ?)",
                (conversation_id,
                 f"Этап задачи изменён: {STAGE_LABELS[conversation['task_stage']]} → "
                 f"{STAGE_LABELS[stage]}.",
                 json.dumps({"stage_change": {"from": conversation["task_stage"], "to": stage,
                                              "author": author}}, ensure_ascii=False),
                 _now()),
            )
        for name in cleared:
            conn.execute(
                f"UPDATE conversations SET {GUARD_COLUMN[name]} = 0 WHERE id = ?",
                (conversation_id,),
            )
            _add_task_event(conn, conversation_id, "guard", GUARD_STAGE[name],
                            note=f"снято возвратом: {GUARD_LABELS[name]}", author=author)
        # Заведение задачи — это не переход «этап → тот же этап»: у первого
        # события предыдущего этапа нет, и журнал показывает его как начало.
        previous = conversation["task_stage"]
        _add_task_event(conn, conversation_id, "stage", stage,
                        from_stage=previous if previous != stage else None,
                        note=note, author=author)
    return get_conversation(conversation_id, user_id)


def set_task_pause(conversation_id: int, user_id: int, paused: bool,
                   *, note: str | None = None) -> dict | None:
    """Ставит задачу на паузу или снимает её. Этап при этом не меняется."""
    conversation = get_conversation(conversation_id, user_id)
    if conversation is None:
        return None

    with connect() as conn:
        conn.execute(
            "UPDATE conversations SET task_paused = ?, task_updated_at = ?"
            " WHERE id = ? AND user_id = ?",
            (int(paused), _now(), conversation_id, user_id),
        )
        _add_task_event(conn, conversation_id, "pause" if paused else "resume",
                        conversation["task_stage"], note=note)
    return get_conversation(conversation_id, user_id)


def undo_task_stage(conversation_id: int, user_id: int) -> dict | None:
    """Отменяет последний переход между этапами.

    Отмена — не новый переход, а возврат к прежнему состоянию, поэтому правила
    TRANSITIONS здесь не применяются: иначе откатить «проверка → готово» было бы
    нельзя, ведь обратного перехода в цепочке нет.
    """
    conversation = get_conversation(conversation_id, user_id)
    if conversation is None:
        return None

    with connect() as conn:
        last = conn.execute(
            "SELECT * FROM task_events WHERE conversation_id = ? AND kind = 'stage'"
            " ORDER BY id DESC LIMIT 1", (conversation_id,),
        ).fetchone()
        if last is None or not last["from_stage"]:
            return None
        conn.execute(
            "UPDATE conversations SET task_stage = ?, task_updated_at = ?"
            " WHERE id = ? AND user_id = ?",
            (last["from_stage"], _now(), conversation_id, user_id),
        )
        _add_task_event(conn, conversation_id, "stage", last["from_stage"],
                        from_stage=last["stage"], note="отмена перехода")
    return get_conversation(conversation_id, user_id)


def list_task_events(conversation_id: int, limit: int = 50) -> list[dict]:
    """Хронология состояния: переходы, паузы и возобновления вперемешку."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM task_events WHERE conversation_id = ?"
            " ORDER BY id DESC LIMIT ?", (conversation_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ---------- сообщения ----------

def add_message(
    conversation_id: int, role: str, content: str, *, reasoning: str | None = None,
    tokens_completion: int = 0, tokens_total: int = 0, meta: str | None = None,
) -> dict:
    """Добавляет сообщение в диалог и поднимает диалог в списке."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, reasoning,"
            " tokens_completion, tokens_total, meta, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, reasoning, tokens_completion,
             tokens_total, meta, _now()),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id)
        )
        row = conn.execute("SELECT * FROM messages WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def set_facts(conversation_id: int, facts_json: str) -> None:
    """Сохраняет блок фактов диалога (JSON-объект ключ-значение)."""
    with connect() as conn:
        conn.execute(
            "UPDATE conversations SET facts = ? WHERE id = ?", (facts_json, conversation_id)
        )


def create_branch(conversation_id: int, user_id: int, from_message_id: int) -> dict | None:
    """Создаёт ветку: копию диалога по сообщение from_message_id включительно.

    Ветка — отдельный диалог с копией сообщений, а не общее дерево. Так две
    ветки развиваются совершенно независимо, удаление одной не задевает
    другую, и весь остальной код работает с веткой как с обычным диалогом.
    Цена — дублирование уже сказанного; на здешних объёмах это дешевле,
    чем усложнять выборку сообщений во всех запросах.
    """
    source = get_conversation(conversation_id, user_id)
    if source is None:
        return None

    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? AND id <= ? ORDER BY id",
            (conversation_id, from_message_id),
        ).fetchall()
        if not rows:
            return None

        base = source["title"].removeprefix("↳ ")
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, model, thinking, max_tokens,"
            " strategy, context_n, facts, summary, summary_upto, parent_id, branched_from,"
            " project_id, use_profile, use_project, profile_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, f"↳ {base}"[:TITLE_LIMIT], source["model"], source["thinking"],
             source["max_tokens"], source["strategy"], source["context_n"],
             source["facts"], source["summary"], source["summary_upto"],
             conversation_id, from_message_id,
             source["project_id"], source["use_profile"], source["use_project"],
             source["profile_id"], _now(), _now()),
        )
        branch_id = cur.lastrowid
        for r in rows:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, reasoning,"
                " tokens_completion, tokens_total, meta, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (branch_id, r["role"], r["content"], r["reasoning"], r["tokens_completion"],
                 r["tokens_total"], r["meta"], r["created_at"]),
            )
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (branch_id,)).fetchone()
    return dict(row)


def set_summary(conversation_id: int, summary: str, upto_message_id: int) -> None:
    """Сохраняет конспект и границу, до которой он покрывает переписку."""
    with connect() as conn:
        conn.execute(
            "UPDATE conversations SET summary = ?, summary_upto = ? WHERE id = ?",
            (summary, upto_message_id, conversation_id),
        )


def update_message_meta(message_id: int, meta_json: str) -> bool:
    """Заменяет телеметрию уже сохранённого сообщения.

    Служебные вызовы — карточка фактов, конспект, переключатель этапов — идут
    после ответа, когда сообщение уже в базе. Другого способа сохранить их
    рядом с ответом, кроме правки meta, нет.
    """
    with connect() as conn:
        cur = conn.execute("UPDATE messages SET meta = ? WHERE id = ?", (meta_json, message_id))
    return cur.rowcount > 0


def delete_message(message_id: int) -> bool:
    """Удаляет сообщение. Нужно, чтобы откатить неудавшийся обмен."""
    with connect() as conn:
        cur = conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    return cur.rowcount > 0


def list_messages(conversation_id: int) -> list[dict]:
    """Все сообщения диалога в порядке добавления."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def list_messages_after(conversation_id: int, after: int) -> list[dict]:
    """Только сообщения новее указанного.

    Отдельный запрос, а не фильтр по списку: открытая страница спрашивает об
    этом раз в несколько секунд, и перечитывать ради этого всю переписку —
    работа, растущая вместе с диалогом.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? AND id > ? ORDER BY id",
            (conversation_id, after),
        ).fetchall()
    return [dict(r) for r in rows]


def history_for_api(conversation_id: int) -> list[dict]:
    """Полная история: только роль и текст, без сжатия."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]
