"""Клиент к LLM по OpenAI-совместимому API."""

import os
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
THINKING = os.getenv("LLM_THINKING", "true").lower() == "true"
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "high")
SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful assistant.")


class LLMError(RuntimeError):
    """Ошибка вызова API — сеть или ненулевой HTTP-статус."""


@dataclass
class Completion:
    """Ответ модели вместе с телеметрией вызова."""

    content: str
    reasoning: str | None
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed: float

    @property
    def truncated(self) -> bool:
        """True, если генерацию оборвал лимит max_tokens, а не сама модель."""
        return self.finish_reason == "length"


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    thinking: bool | None = None,
    temperature: float | None = None,
    timeout: int = 120,
) -> Completion:
    """Отправляет историю сообщений в API и возвращает ответ с телеметрией.

    max_tokens — жёсткий потолок генерации на стороне API.
    stop — стоп-последовательности: API обрывает генерацию, не включая их в ответ.
    """
    if not API_KEY:
        raise LLMError("Не задан LLM_API_KEY. Скопируйте .env.example в .env и впишите ключ.")

    use_thinking = THINKING if thinking is None else thinking
    payload: dict = {"model": model or MODEL, "messages": messages}
    if use_thinking:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = REASONING_EFFORT
    else:
        # Модели DeepSeek v4 рассуждают по умолчанию: пропущенное поле
        # их не выключает, а reasoning-токены расходуют бюджет max_tokens.
        payload["thinking"] = {"type": "disabled"}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stop:
        payload["stop"] = stop
    if temperature is not None:
        payload["temperature"] = temperature

    started = time.monotonic()
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.HTTPError as err:
        raise LLMError(f"Ошибка API {err.response.status_code}: {err.response.text}") from err
    except requests.RequestException as err:
        raise LLMError(f"Ошибка сети: {err}") from err
    elapsed = time.monotonic() - started

    body = response.json()
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    return Completion(
        content=choice["message"]["content"],
        reasoning=choice["message"].get("reasoning_content"),
        finish_reason=choice.get("finish_reason", "unknown"),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        elapsed=elapsed,
    )
