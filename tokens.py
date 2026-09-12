"""Подсчёт токенов: текущий запрос, вся история диалога, ответ модели.

Точные числа даёт только API — в поле usage после вызова. До вызова доступна
лишь оценка, поэтому здесь два разных механизма, и их не стоит путать:

  * факт    — из usage, сохраняется в messages при каждом ответе;
  * оценка  — по числу символов, для запроса, который ещё не отправлен.

Коэффициенты оценки измерены на этом же API (см. RATIOS): у русского текста
на токен приходится втрое меньше символов, чем кажется по английским меркам,
поэтому единый коэффициент врал бы почти вдвое.
"""

import re
from dataclasses import dataclass

import db
from benchmark import DEFAULT_PRICES

# Символов на токен, измерено запросами к deepseek-flash.
RATIOS = {"ru": 3.33, "en": 5.85, "code": 2.41}

# Каждое сообщение несёт служебную обвязку роли и разделителей.
MESSAGE_OVERHEAD = 4

# Заявленный предел контекста обеих моделей DeepSeek. Порог мягкий: запрос
# на 1 028 577 токенов прошёл успешно, поэтому число — ориентир, не гарантия.
CONTEXT_LIMIT = 1_000_000

_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_CODEY = re.compile(r"[{}()\[\];=<>]|\bdef\b|\bfunction\b|\bimport\b")


def ratio_for(text: str) -> float:
    """Подбирает коэффициент по составу текста."""
    if not text:
        return RATIOS["en"]
    if len(_CODEY.findall(text)) > len(text) / 120:
        return RATIOS["code"]
    cyrillic = len(_CYRILLIC.findall(text))
    letters = sum(1 for c in text if c.isalpha()) or 1
    # Смешанный текст оцениваем пропорционально доле кириллицы.
    share = cyrillic / letters
    return RATIOS["ru"] * share + RATIOS["en"] * (1 - share)


def estimate_tokens(text: str) -> int:
    """Оценка числа токенов в тексте. Точную цифру даёт только API."""
    if not text:
        return 0
    return max(1, round(len(text) / ratio_for(text)))


def estimate_messages(messages: list[dict]) -> int:
    """Оценка размера запроса: все сообщения плюс служебная обвязка."""
    return sum(
        estimate_tokens(m.get("content", "")) + MESSAGE_OVERHEAD for m in messages
    )


def cost(prompt_tokens: int, completion_tokens: int, model: str) -> float | None:
    """Стоимость в долларах. None — тариф для модели неизвестен."""
    price = DEFAULT_PRICES.get(model)
    if not price:
        return None
    return prompt_tokens / 1_000_000 * price["in"] + completion_tokens / 1_000_000 * price["out"]


@dataclass
class Turn:
    """Один обмен репликами со всеми числами."""

    index: int
    question: str
    answer: str
    prompt_tokens: int          # сколько стоила отправленная история
    completion_tokens: int      # сколько стоил ответ
    total_tokens: int
    turn_cost: float | None
    cumulative_tokens: int
    cumulative_cost: float | None


def dialog_growth(conversation_id: int, model: str) -> list[Turn]:
    """Разбирает диалог по обменам и считает накопление токенов и денег.

    prompt_tokens каждого обмена — это вся история на тот момент. Поэтому
    сумма по диалогу растёт не линейно: каждая новая реплика заново
    оплачивает всё сказанное раньше.
    """
    rows = db.list_messages(conversation_id)
    turns: list[Turn] = []
    pending_question = ""
    cum_tokens = 0
    cum_cost = 0.0
    cost_known = True

    for row in rows:
        if row["role"] == "user":
            pending_question = row["content"]
            continue

        prompt = max(row["tokens_total"] - row["tokens_completion"], 0)
        completion = row["tokens_completion"]
        turn_cost = cost(prompt, completion, model)
        cum_tokens += row["tokens_total"]
        if turn_cost is None:
            cost_known = False
        else:
            cum_cost += turn_cost

        turns.append(Turn(
            index=len(turns) + 1,
            question=pending_question,
            answer=row["content"],
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=row["tokens_total"],
            turn_cost=turn_cost,
            cumulative_tokens=cum_tokens,
            cumulative_cost=cum_cost if cost_known else None,
        ))
        pending_question = ""
    return turns


def next_request_estimate(conversation_id: int, draft: str = "") -> dict:
    """Во что обойдётся следующий запрос: история плюс черновик реплики."""
    history = db.history_for_api(conversation_id)
    history_tokens = estimate_messages(history)
    draft_tokens = estimate_tokens(draft) + (MESSAGE_OVERHEAD if draft else 0)
    total = history_tokens + draft_tokens
    return {
        "history_tokens": history_tokens,
        "draft_tokens": draft_tokens,
        "request_tokens": total,
        "context_limit": CONTEXT_LIMIT,
        "context_used": round(total / CONTEXT_LIMIT * 100, 3),
        "messages": len(history),
    }
