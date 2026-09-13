"""Сжатие истории диалога: хвост «как есть», остальное — конспектом.

Модуль называется history, а не compression: в Python 3.14 появился
стандартный пакет compression, и файл с таким именем в корне проекта
перекрывает его — ломается импорт bz2, а за ним shutil, tempfile и httpx.

Задача сжатия — уменьшить запрос, не потеряв смысл разговора. Отсюда правило,
которое здесь соблюдается строго:

    конспект покрывает всё, кроме хвоста; сообщения, которые уже вышли из
    хвоста, но ещё не попали в конспект, отправляются как есть.

Без этого на стыке терялась бы часть переписки: хвост уехал вперёд, а конспект
за ним ещё не пересобрали.

Конспект наращивается: новая версия делается из прежнего конспекта и только
тех сообщений, что добавились с прошлого раза. Пересобирать по всей переписке
было бы тем дороже, чем длиннее разговор, — то есть ровно там, где сжатие и
должно экономить.
"""

from dataclasses import dataclass

import db
import llm

# Сколько последних сообщений уходит в запрос дословно.
KEEP_LAST = 6

# Сколько сообщений должно накопиться вне конспекта, чтобы пересобрать его.
COMPRESS_EVERY = 10

SUMMARY_SYSTEM = (
    "Ты ведёшь конспект диалога. Тебе дают прежний конспект и новые реплики. "
    "Верни обновлённый конспект, в котором сохранено всё, что понадобится для "
    "продолжения разговора: факты о собеседнике, имена, числа, принятые решения, "
    "договорённости и нерешённые вопросы. Пиши сжато, по пунктам, без вводных "
    "фраз и без пересказа вежливых оборотов. Не выдумывай того, чего не было."
)

SUMMARY_PREFIX = "Конспект предыдущей части разговора:\n"


@dataclass
class Plan:
    """Что уйдёт в запрос и что из истории при этом свернулось."""

    messages: list[dict]        # готовый список для API, включая системный промпт
    summary: str                # действующий конспект, пустая строка если его нет
    verbatim: int               # сколько сообщений идёт дословно
    folded: int                 # сколько сообщений заменено конспектом
    stale: int                  # вышли из хвоста, но ещё не в конспекте


def split(rows: list[dict], summary_upto: int | None) -> tuple[list[dict], list[dict], list[dict]]:
    """Делит переписку на свёрнутую, ещё не свёрнутую и хвост."""
    tail = rows[-KEEP_LAST:] if KEEP_LAST else []
    head = rows[: len(rows) - len(tail)]
    upto = summary_upto or 0
    folded = [m for m in head if m["id"] <= upto]
    stale = [m for m in head if m["id"] > upto]
    return folded, stale, tail


def plan_request(conversation: dict, system_prompt: str) -> Plan:
    """Собирает запрос к API с учётом настройки сжатия."""
    rows = db.list_messages(conversation["id"])
    as_api = lambda items: [{"role": m["role"], "content": m["content"]} for m in items]

    if not conversation.get("compress"):
        return Plan(
            messages=[{"role": "system", "content": system_prompt}] + as_api(rows),
            summary="", verbatim=len(rows), folded=0, stale=0,
        )

    summary = conversation.get("summary") or ""
    folded, stale, tail = split(rows, conversation.get("summary_upto"))

    messages = [{"role": "system", "content": system_prompt}]
    if summary:
        messages.append({"role": "system", "content": SUMMARY_PREFIX + summary})
    messages += as_api(stale) + as_api(tail)

    return Plan(
        messages=messages, summary=summary,
        verbatim=len(stale) + len(tail), folded=len(folded), stale=len(stale),
    )


def needs_refresh(conversation: dict) -> bool:
    """Пора ли пересобирать конспект."""
    if not conversation.get("compress"):
        return False
    rows = db.list_messages(conversation["id"])
    _, stale, _ = split(rows, conversation.get("summary_upto"))
    return len(stale) >= COMPRESS_EVERY


def refresh(conversation: dict, *, model: str | None = None) -> dict | None:
    """Сворачивает накопившиеся сообщения в конспект.

    Возвращает сведения о пересборке или None, если сворачивать нечего.
    """
    rows = db.list_messages(conversation["id"])
    _, stale, _ = split(rows, conversation.get("summary_upto"))
    if not stale:
        return None

    previous = conversation.get("summary") or ""
    transcript = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content']}"
        for m in stale
    )
    user_part = (
        (f"Прежний конспект:\n{previous}\n\n" if previous else "")
        + f"Новые реплики:\n{transcript}"
    )

    result = llm.complete(
        [{"role": "system", "content": SUMMARY_SYSTEM},
         {"role": "user", "content": user_part}],
        model=model, thinking=False,
    )
    summary = result.content.strip()
    db.set_summary(conversation["id"], summary, stale[-1]["id"])

    return {
        "folded_messages": len(stale),
        "summary_chars": len(summary),
        "cost_tokens": result.total_tokens,
        "upto_message_id": stale[-1]["id"],
    }
