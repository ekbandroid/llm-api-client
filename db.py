"""Хранилище пользователей на SQLite."""

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
    "users": {
        # Долговременная память: что помнить о пользователе во всех диалогах.
        "profile_memory": "TEXT",
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
        return _fetch(conn, "login = ?", login)


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

        return _fetch(conn, "yandex_id = ?", yandex_id)


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
    project_id: int | None = None,
) -> dict:
    """Заводит пустой диалог и возвращает его."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Неизвестная стратегия: {strategy}")
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, model, thinking, max_tokens,"
            " strategy, context_n, project_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, title.strip()[:TITLE_LIMIT] or NEW_TITLE, model, int(thinking),
             max_tokens, strategy, max(1, min(context_n, 200)), project_id, _now(), _now()),
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
            " project_id, use_profile, use_project, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, f"↳ {base}"[:TITLE_LIMIT], source["model"], source["thinking"],
             source["max_tokens"], source["strategy"], source["context_n"],
             source["facts"], source["summary"], source["summary_upto"],
             conversation_id, from_message_id,
             source["project_id"], source["use_profile"], source["use_project"],
             _now(), _now()),
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


def history_for_api(conversation_id: int) -> list[dict]:
    """Полная история: только роль и текст, без сжатия."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]
