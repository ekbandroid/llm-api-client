"""Хранилище пользователей на SQLite."""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

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


def init() -> None:
    """Создаёт таблицы, если их ещё нет, и проверяет совместимость схемы."""
    with connect() as conn:
        conn.executescript(SCHEMA)
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


def count_admins() -> int:
    """Число админов — чтобы не остаться без единого администратора."""
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1").fetchone()[0]
