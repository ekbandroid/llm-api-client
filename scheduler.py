"""Очередь отложенных и повторяющихся поручений агенту.

Хранилище общее для двух процессов: MCP-сервер заводит задания (их создаёт
модель инструментом), приложение забирает созревшие и выполняет. Оба работают
на одной машине от одного пользователя — SQLite в режиме WAL для такого и
сделан, а отдельный файл от app.db выбран намеренно: очередь не должна ездить
в бэкапах переписки и не должна страдать от блокировок чата.

Ключевое поле — next_run_at. Оно лежит в базе, а не в памяти процесса, поэтому
перезапуск расписание не теряет. Пропущенное за время простоя не навёрстывается
пачкой: время следующего запуска считается от «сейчас», иначе после часа
простоя задание «раз в минуту» выстрелило бы шестьдесят раз подряд.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("SCHEDULER_DB_PATH", "scheduler.db"))

# Виды заданий. summary отличается от repeat только тем, что перед выполнением
# приложение подкладывает в поручение результаты других заданий.
ONCE, REPEAT, SUMMARY = "once", "repeat", "summary"
KINDS = (ONCE, REPEAT, SUMMARY)

ACTIVE, PAUSED, DONE = "active", "paused", "done"
STATUSES = (ACTIVE, PAUSED, DONE)

# Нижняя граница периода. Минута — чтобы проверять не полчаса; для реальных
# наблюдений разумнее десятки минут: каждое выполнение стоит токенов.
MIN_PERIOD = 1
MAX_PERIOD = 7 * 24 * 60

TITLE_LIMIT = 80
PROMPT_LIMIT = 2000

# Сколько держим историю запусков: наблюдение раз в пять минут — это 288 строк
# в сутки, без уборки база растёт вечно.
KEEP_RUNS_DAYS = 7

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER NOT NULL,
    title         TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    -- Что поручено агенту: обычный текст, он станет репликой в диалоге.
    prompt        TEXT    NOT NULL,
    -- Для сводки: id заданий, чьи результаты обобщаются. Пусто — все задания
    -- этого чата.
    sources       TEXT,
    every_minutes INTEGER,
    next_run_at   TEXT,
    status        TEXT    NOT NULL DEFAULT 'active',
    runs          INTEGER NOT NULL DEFAULT 0,
    last_run_at   TEXT,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_chat ON jobs(chat_id, id);
CREATE INDEX IF NOT EXISTS idx_jobs_due ON jobs(status, next_run_at);

CREATE TABLE IF NOT EXISTS runs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id    INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ran_at    TEXT    NOT NULL,
    ok        INTEGER NOT NULL DEFAULT 1,
    answer    TEXT,
    tokens    INTEGER NOT NULL DEFAULT 0,
    error     TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_job ON runs(job_id, id);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return None


def _period(minutes) -> int:
    value = int(minutes)
    if not MIN_PERIOD <= value <= MAX_PERIOD:
        raise ValueError(
            f"период должен быть от {MIN_PERIOD} до {MAX_PERIOD} минут")
    return value


# ---------- заведение ----------

def create(
    chat_id: int, *, kind: str, prompt: str, title: str = "",
    in_minutes: int | None = None, every_minutes: int | None = None,
    sources: list[int] | None = None,
) -> dict:
    """Заводит задание. Первый запуск — через in_minutes или через период."""
    if kind not in KINDS:
        raise ValueError(f"неизвестный вид задания: {kind}")
    text = (prompt or "").strip()
    if not text:
        raise ValueError("пустое поручение")

    if kind == ONCE:
        delay = _period(in_minutes if in_minutes is not None else 0) if in_minutes else 0
        if delay <= 0:
            raise ValueError("для отложенного задания нужен in_minutes")
        every = None
    else:
        every = _period(every_minutes)
        # Первый запуск — через период, а не сразу: «каждые два часа» не должно
        # означать «прямо сейчас и потом каждые два часа».
        delay = every

    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (chat_id, title, kind, prompt, sources, every_minutes,"
            " next_run_at, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, (title or text)[:TITLE_LIMIT], kind, text[:PROMPT_LIMIT],
             json.dumps(sources) if sources else None, every,
             _stamp(_now() + timedelta(minutes=delay)), ACTIVE, _stamp(_now())),
        )
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


# ---------- чтение ----------

def get(job_id: int, chat_id: int | None = None) -> dict | None:
    where = "id = ?" + (" AND chat_id = ?" if chat_id is not None else "")
    args = (job_id,) if chat_id is None else (job_id, chat_id)
    with connect() as conn:
        row = conn.execute(f"SELECT * FROM jobs WHERE {where}", args).fetchone()
    return dict(row) if row else None


def list_jobs(chat_id: int, *, only_active: bool = False) -> list[dict]:
    where = "chat_id = ?" + (" AND status = 'active'" if only_active else "")
    with connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM jobs WHERE {where} ORDER BY id DESC", (chat_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def last_run(job_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE job_id = ? ORDER BY id DESC LIMIT 1", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def recent_runs(job_ids: list[int], *, hours: int = 24, limit: int = 40) -> list[dict]:
    """Результаты заданий за последние часы — материал для сводки."""
    if not job_ids:
        return []
    since = _stamp(_now() - timedelta(hours=hours))
    marks = ",".join("?" * len(job_ids))
    with connect() as conn:
        rows = conn.execute(
            f"SELECT runs.*, jobs.title FROM runs JOIN jobs ON jobs.id = runs.job_id"
            f" WHERE runs.job_id IN ({marks}) AND runs.ran_at >= ? AND runs.ok = 1"
            f" ORDER BY runs.id DESC LIMIT ?",
            (*job_ids, since, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ---------- выполнение ----------

def take_due(limit: int = 5) -> list[dict]:
    """Забирает созревшие задания и сразу переставляет расписание.

    Расписание переставляется до выполнения, а не после: выполнение идёт через
    модель и занимает секунды, за которые следующий тик успел бы взять то же
    задание второй раз.
    """
    now = _now()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status = 'active' AND next_run_at <= ?"
            " ORDER BY next_run_at LIMIT ?",
            (_stamp(now), limit),
        ).fetchall()
        taken = []
        for row in rows:
            job = dict(row)
            if job["kind"] == ONCE:
                conn.execute(
                    "UPDATE jobs SET status = ?, next_run_at = NULL WHERE id = ?",
                    (DONE, job["id"]),
                )
            else:
                # От «сейчас», а не от прежнего срока: догоняем один раз.
                nxt = now + timedelta(minutes=job["every_minutes"] or MIN_PERIOD)
                conn.execute("UPDATE jobs SET next_run_at = ? WHERE id = ?",
                             (_stamp(nxt), job["id"]))
            taken.append(job)
    return taken


def postpone(job_id: int, minutes: int = 1) -> None:
    """Возвращает задание в очередь: диалог был занят человеком.

    Нужна потому, что take_due переставляет расписание сразу — иначе задание
    взяли бы дважды. Пропущенный ход надо вернуть явно.
    """
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, next_run_at = ? WHERE id = ?",
            (ACTIVE, _stamp(_now() + timedelta(minutes=minutes)), job_id),
        )


def record(job_id: int, *, ok: bool, answer: str = "", tokens: int = 0,
           error: str = "") -> None:
    """Запоминает результат выполнения."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO runs (job_id, ran_at, ok, answer, tokens, error)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, _stamp(_now()), int(ok), (answer or "")[:4000], tokens,
             (error or "")[:500]),
        )
        conn.execute(
            "UPDATE jobs SET runs = runs + 1, last_run_at = ? WHERE id = ?",
            (_stamp(_now()), job_id),
        )
        conn.execute(
            "DELETE FROM runs WHERE ran_at < ?",
            (_stamp(_now() - timedelta(days=KEEP_RUNS_DAYS)),),
        )


# ---------- управление ----------

def set_status(job_id: int, status: str, chat_id: int | None = None) -> dict | None:
    """Пауза, возобновление или отмена.

    При возобновлении следующий запуск назначается от «сейчас»: задание,
    простоявшее на паузе неделю, не должно срабатывать сразу и потом ещё раз.
    """
    if status not in STATUSES:
        raise ValueError(f"неизвестный статус: {status}")
    job = get(job_id, chat_id)
    if job is None:
        return None

    next_run = job["next_run_at"]
    if status == ACTIVE:
        minutes = job["every_minutes"] or MIN_PERIOD
        next_run = _stamp(_now() + timedelta(minutes=minutes))
    elif status == DONE:
        next_run = None

    with connect() as conn:
        conn.execute("UPDATE jobs SET status = ?, next_run_at = ? WHERE id = ?",
                     (status, next_run, job_id))
    return get(job_id)


def delete(job_id: int, chat_id: int | None = None) -> bool:
    if get(job_id, chat_id) is None:
        return False
    with connect() as conn:
        conn.execute("DELETE FROM runs WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return True


def delete_chat_jobs(chat_id: int) -> int:
    """Убирает задания вместе с диалогом: без диалога их некуда выполнять."""
    with connect() as conn:
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM jobs WHERE chat_id = ?", (chat_id,))]
        for job_id in ids:
            conn.execute("DELETE FROM runs WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE chat_id = ?", (chat_id,))
    return len(ids)


# ---------- описание для человека и для модели ----------

def describe(job: dict, *, with_last: bool = True) -> dict:
    """Задание в виде, пригодном и для интерфейса, и для ответа инструмента."""
    view = {
        "id": job["id"],
        "title": job["title"],
        "kind": job["kind"],
        "prompt": job["prompt"],
        "status": job["status"],
        "every_minutes": job["every_minutes"],
        "next_run_at": job["next_run_at"],
        "runs": job["runs"],
        "last_run_at": job["last_run_at"],
    }
    if with_last and (run := last_run(job["id"])):
        view["last_answer"] = (run["answer"] or "")[:500]
        view["last_ok"] = bool(run["ok"])
    return view


def human_delay(stamp: str | None) -> str:
    """Сколько осталось до запуска, словами."""
    moment = _parse(stamp)
    if moment is None:
        return "не запланирован"
    minutes = round((moment - _now()).total_seconds() / 60)
    if minutes <= 0:
        return "вот-вот"
    if minutes < 60:
        return f"через {minutes} мин"
    hours = minutes / 60
    if hours < 24:
        return f"через {hours:.1f} ч".replace(".0", "")
    return f"через {hours / 24:.1f} сут".replace(".0", "")
