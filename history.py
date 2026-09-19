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

import json
from dataclasses import dataclass, field

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
    "фраз и без пересказа вежливых оборотов. Не выдумывай того, чего не было. "
    "Реплика «[просьба отклонена по INV-…]» — просьба, которую ассистент "
    "отклонил по правилам проекта. То, что отклонено, — не цель, не решение, "
    "не предпочтение и не договорённость; можно лишь отметить, что такая "
    "просьба была отклонена."
)

SUMMARY_PREFIX = "Конспект предыдущей части разговора:\n"

FACTS_PREFIX = "Известные факты о задаче и собеседнике:\n"

FACTS_SYSTEM = (
    "Ты ведёшь карточку фактов о диалоге. Тебе дают текущую карточку и новые "
    "реплики. Верни обновлённую карточку — ТОЛЬКО JSON-объект вида "
    '{"ключ": "значение"}, без пояснений и без markdown-ограды. '
    "Храни то, без чего нельзя продолжить работу: цель, ограничения, "
    "предпочтения, принятые решения, договорённости, открытые вопросы. "
    "Ключи — короткие существительные на русском. Старые факты сохраняй, если "
    "они не отменены; отменённые заменяй новыми. Не выдумывай того, чего не было. "
    "Карточка описывает задачу и собеседника, а не состояние ассистента: никогда "
    "не записывай, чего ассистент не знает или не смог ответить. "
    "Реплика «[просьба отклонена по INV-…]» — просьба, которую ассистент "
    "отклонил по правилам проекта. То, что отклонено, — не цель, не решение, "
    "не предпочтение и не договорённость; можно лишь отметить, что такая "
    "просьба была отклонена."
)

# Сколько фактов держим: больше — и блок сам становится длинной историей.
FACTS_LIMIT = 25


@dataclass
class Plan:
    """Что уйдёт в запрос и что из истории при этом свернулось."""

    messages: list[dict]        # готовый список для API, включая системный промпт
    summary: str                # действующий конспект, пустая строка если его нет
    verbatim: int               # сколько сообщений идёт дословно
    folded: int                 # сколько сообщений заменено конспектом
    stale: int                  # вышли из хвоста, но ещё не в конспекте
    strategy: str = db.FULL
    dropped: int = 0            # сколько сообщений просто отброшено
    facts: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)   # что дали слои памяти поверх диалога


def rejected_codes(message: dict) -> list[str]:
    """По каким инвариантам отклонена просьба. Пусто — не отклонена."""
    if message["role"] != "user":
        return []
    rejected = message.get("rejected_by")
    if rejected is None and message.get("meta"):
        meta = message["meta"]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        rejected = (meta or {}).get("rejected_by")
    return list(rejected or [])


def transcript_line(message: dict) -> str:
    """Строка реплики для служебных вызовов: карточки, конспекта, переключателя.

    Текст отклонённой по инварианту просьбы сюда не попадает — только
    пометка. Одной пометки с правилом «не записывай как решение» не хватило:
    на первой же реплике диалога, пока карточка пуста, служебный вызов
    записал «цель: переписать хранение заказов на MongoDB», хотя ассистент
    отказал, а дальше это перелилось в память проекта — рядом с инвариантом,
    противореча ему. Что именно просили, остаётся понятным из ответа
    ассистента: отказ называет и просьбу, и альтернативу, и он идёт целиком.
    """
    if message["role"] != "user":
        return f"Ассистент: {message['content']}"
    if codes := rejected_codes(message):
        return (f"Пользователь: [просьба отклонена по {', '.join(codes)}; "
                "её текст в память не переносится]")
    return f"Пользователь: {message['content']}"


def split(rows: list[dict], summary_upto: int | None) -> tuple[list[dict], list[dict], list[dict]]:
    """Делит переписку на свёрнутую, ещё не свёрнутую и хвост."""
    tail = rows[-KEEP_LAST:] if KEEP_LAST else []
    head = rows[: len(rows) - len(tail)]
    upto = summary_upto or 0
    folded = [m for m in head if m["id"] <= upto]
    stale = [m for m in head if m["id"] > upto]
    return folded, stale, tail


def as_api(items: list[dict]) -> list[dict]:
    """Оставляет от сообщений только то, что понимает API."""
    return [{"role": m["role"], "content": m["content"]} for m in items]


def load_facts(conversation: dict) -> dict:
    """Разбирает карточку фактов. Испорченный JSON трактуем как пустую."""
    raw = conversation.get("facts")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def facts_block(facts: dict) -> str:
    """Текст блока фактов для системного промпта."""
    lines = [f"- {k}: {v}" for k, v in list(facts.items())[:FACTS_LIMIT]]
    return FACTS_PREFIX + "\n".join(lines)


def plan_request(
    conversation: dict, system_prompt: str, *,
    memory_blocks: list[dict] | None = None, memory_info: dict | None = None,
) -> Plan:
    """Собирает запрос к API по выбранной стратегии.

    Слои памяти, если они подключены, идут сразу за системным промптом —
    до всего, что относится к самому диалогу. Собирает их memory.py: этому
    модулю принадлежит только краткосрочный слой.
    """
    rows = db.list_messages(conversation["id"])
    strategy = conversation.get("strategy") or db.FULL
    keep = conversation.get("context_n") or db.DEFAULT_CONTEXT_N
    head = [{"role": "system", "content": system_prompt}] + list(memory_blocks or [])
    remembered = dict(memory_info or {})

    if strategy == db.WINDOW:
        # Всё, что не попало в окно, просто отбрасывается — это и есть
        # суть стратегии: дёшево, но старые договорённости теряются.
        tail = rows[-keep:]
        return Plan(
            messages=head + as_api(tail), summary="", verbatim=len(tail),
            folded=0, stale=0, strategy=strategy, dropped=len(rows) - len(tail),
            memory=remembered,
        )

    if strategy == db.FACTS:
        facts = load_facts(conversation)
        tail = rows[-keep:]
        messages = head + ([{"role": "system", "content": facts_block(facts)}] if facts else [])
        return Plan(
            messages=messages + as_api(tail), summary="", verbatim=len(tail),
            folded=0, stale=0, strategy=strategy,
            dropped=len(rows) - len(tail), facts=facts, memory=remembered,
        )

    if strategy == db.SUMMARY:
        summary = conversation.get("summary") or ""
        folded, stale, tail = split(rows, conversation.get("summary_upto"))
        messages = head + ([{"role": "system", "content": SUMMARY_PREFIX + summary}] if summary else [])
        return Plan(
            messages=messages + as_api(stale) + as_api(tail), summary=summary,
            verbatim=len(stale) + len(tail), folded=len(folded), stale=len(stale),
            strategy=strategy, memory=remembered,
        )

    return Plan(
        messages=head + as_api(rows), summary="", verbatim=len(rows),
        folded=0, stale=0, strategy=db.FULL, memory=remembered,
    )


def refresh_facts(conversation: dict, exchange: list[dict], *, model: str | None = None) -> dict | None:
    """Обновляет карточку фактов по последнему обмену репликами.

    В запрос уходит только текущая карточка и новые реплики, а не вся
    переписка: обновление должно стоить одинаково дёшево на любой длине
    разговора, иначе стратегия перестаёт экономить.
    """
    if not exchange:
        return None

    current = load_facts(conversation)
    transcript = "\n".join(transcript_line(m) for m in exchange)
    user_part = (
        (f"Текущая карточка:\n{json.dumps(current, ensure_ascii=False, indent=2)}\n\n"
         if current else "")
        + f"Новые реплики:\n{transcript}"
    )

    # keep_text=True: ответ этого вызова — сама карточка, и больше её нигде
    # не покажут; вырезать текст значило бы показать пустой блок.
    result = llm.complete(
        [{"role": "system", "content": FACTS_SYSTEM},
         {"role": "user", "content": user_part}],
        model=model, thinking=False, keep_text=True,
    )
    telemetry = {"kind": "facts", "request": result.request, "response": result.response}

    text = result.content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        updated = json.loads(text)
    except json.JSONDecodeError:
        # Модель ответила не JSON — прежнюю карточку не портим.
        return {"ok": False, "cost_tokens": result.total_tokens,
                "reason": "ответ не разобран как JSON", **telemetry}
    if not isinstance(updated, dict):
        return {"ok": False, "cost_tokens": result.total_tokens,
                "reason": "ожидался объект", **telemetry}

    updated = dict(list(updated.items())[:FACTS_LIMIT])
    db.set_facts(conversation["id"], json.dumps(updated, ensure_ascii=False))
    return {
        "ok": True,
        "facts": updated,
        "count": len(updated),
        "added": [k for k in updated if k not in current],
        "cost_tokens": result.total_tokens,
        **telemetry,
    }


def needs_refresh(conversation: dict) -> bool:
    """Пора ли пересобирать конспект."""
    if (conversation.get("strategy") or db.FULL) != db.SUMMARY:
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
    transcript = "\n".join(transcript_line(m) for m in stale)
    user_part = (
        (f"Прежний конспект:\n{previous}\n\n" if previous else "")
        + f"Новые реплики:\n{transcript}"
    )

    result = llm.complete(
        [{"role": "system", "content": SUMMARY_SYSTEM},
         {"role": "user", "content": user_part}],
        model=model, thinking=False, keep_text=True,
    )
    summary = result.content.strip()
    db.set_summary(conversation["id"], summary, stale[-1]["id"])

    return {
        "kind": "summary",
        "folded_messages": len(stale),
        "summary_chars": len(summary),
        "cost_tokens": result.total_tokens,
        "upto_message_id": stale[-1]["id"],
        "request": result.request,
        "response": result.response,
    }
