"""Хранилище пользователей на SQLite."""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "app.db"))

PENDING, APPROVED, BLOCKED = "pending", "approved", "blocked"
STATUSES = (PENDING, APPROVED, BLOCKED)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    yandex_id     TEXT    NOT NULL UNIQUE,
    login         TEXT    NOT NULL,
    email         TEXT,
    name          TEXT,
    status        TEXT    NOT NULL DEFAULT 'pending',
    is_admin      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL,
    last_login_at TEXT
);
"""


def connect() -> sqlite3.Connection:
    """Открывает соединение с включённым WAL и доступом к колонкам по имени."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init() -> None:
    """Создаёт таблицы, если их ещё нет."""
    with connect() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_from_yandex(profile: dict, *, admin_logins: set[str]) -> dict:
    """Создаёт или обновляет пользователя по профилю Яндекса, возвращает его запись.

    Новый пользователь получает статус pending — доступ подтверждает админ.
    Логины из admin_logins сразу становятся админами с доступом.
    """
    yandex_id = str(profile["id"])
    login = profile.get("login") or yandex_id
    email = profile.get("default_email")
    name = profile.get("real_name") or profile.get("display_name") or login
    is_admin = 1 if login.lower() in admin_logins else 0

    with connect() as conn:
        existing = conn.execute(
            "SELECT * FROM users WHERE yandex_id = ?", (yandex_id,)
        ).fetchone()

        if existing is None:
            conn.execute(
                "INSERT INTO users (yandex_id, login, email, name, status, is_admin,"
                " created_at, last_login_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    yandex_id,
                    login,
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
                "UPDATE users SET login = ?, email = ?, name = ?, is_admin = ?,"
                " status = ?, last_login_at = ? WHERE yandex_id = ?",
                (login, email, name, is_admin, status, _now(), yandex_id),
            )

        return dict(conn.execute(
            "SELECT * FROM users WHERE yandex_id = ?", (yandex_id,)
        ).fetchone())


def get(user_id: int) -> dict | None:
    """Возвращает пользователя по внутреннему id."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


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
