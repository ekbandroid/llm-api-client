"""Клиент к LLM по OpenAI-совместимому API: синхронный вызов и потоковый."""

import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-flash")
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
    reasoning_tokens: int
    elapsed: float
    request: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True, если генерацию оборвал лимит max_tokens, а не сама модель."""
        return self.finish_reason == "length"


def build_payload(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    thinking: bool | None = None,
    temperature: float | None = None,
) -> dict:
    """Собирает тело запроса к /chat/completions."""
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
    return payload


def describe_request(payload: dict) -> dict:
    """Как выглядит запрос к API — для показа в интерфейсе.

    Ключ подменяется звёздочками: он передаётся заголовком и наружу
    попадать не должен ни при каких обстоятельствах.
    """
    return {
        "method": "POST",
        "url": f"{BASE_URL}/chat/completions",
        "headers": {"Authorization": "Bearer ***", "Content-Type": "application/json"},
        "body": payload,
    }


def _headers() -> dict:
    if not API_KEY:
        raise LLMError("Не задан LLM_API_KEY. Скопируйте .env.example в .env и впишите ключ.")
    return {"Authorization": f"Bearer {API_KEY}"}


def complete(messages: list[dict], *, timeout: int = 120, **options) -> Completion:
    """Синхронный вызов: ждёт ответ целиком и возвращает его с телеметрией."""
    payload = build_payload(messages, **options)

    started = time.monotonic()
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions", headers=_headers(), json=payload, timeout=timeout
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
        # Сколько из выходных токенов ушло в рассуждение, а не в сам ответ.
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        elapsed=elapsed,
        request=describe_request(payload),
    )


async def stream(
    messages: list[dict], *, timeout: int = 300, **options
) -> AsyncIterator[dict]:
    """Потоковый вызов: отдаёт куски ответа по мере генерации.

    Генерирует события:
      {"type": "request", "request": {...}} — что именно уходит в API;
      {"type": "reasoning", "text": ...} — кусок рассуждения (если thinking включён);
      {"type": "content",   "text": ...} — кусок ответа;
      {"type": "done", "finish_reason": ..., "usage": {...}, "elapsed": ...}.
    """
    payload = build_payload(messages, **options)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    yield {"type": "request", "request": describe_request(payload)}

    started = time.monotonic()
    finish_reason = "unknown"
    usage: dict = {}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{BASE_URL}/chat/completions", headers=_headers(), json=payload
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise LLMError(f"Ошибка API {response.status_code}: {detail}")

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        if reasoning := delta.get("reasoning_content"):
                            yield {"type": "reasoning", "text": reasoning}
                        if content := delta.get("content"):
                            yield {"type": "content", "text": content}
    except httpx.HTTPError as err:
        raise LLMError(f"Ошибка сети: {err}") from err

    yield {
        "type": "done",
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed": round(time.monotonic() - started, 2),
    }
