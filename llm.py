"""Клиент к LLM по OpenAI-совместимому API: синхронный вызов и потоковый."""

import copy
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
    """Ошибка вызова API — сеть или ненулевой HTTP-статус.

    Тело ответа хранится разобранным, а не вклеенным в текст сообщения:
    иначе его нельзя показать пользователю тем же способом, что и обычный
    ответ. У сетевых сбоев ответа нет вовсе — тогда response остаётся None.
    """

    def __init__(self, message: str, *, status: int | None = None, body=None,
                 diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        # Для сетевых сбоев ответа нет, но сказать о них есть что.
        self.diagnostics = diagnostics

    @property
    def response(self) -> dict | None:
        """Ответ сервера для показа в интерфейсе. None — ответа не было."""
        if self.status is None:
            return None
        return {
            "url": f"{BASE_URL}/chat/completions",
            "status": self.status,
            "body": self.body,
        }


def _network_message(err: Exception) -> str:
    """Текст сетевой ошибки.

    Имя класса подставляется всегда: у части исключений httpx строковое
    представление пустое, и сообщение вырождалось в «Ошибка сети: » без
    единого слова — как раз в самом частом случае, при таймауте.
    """
    detail = str(err).strip()
    return f"Ошибка сети: {type(err).__name__}" + (f" — {detail}" if detail else "")


def _network_diagnostics(err: Exception, timeout: int) -> dict:
    """Что известно о сбое, когда ответа от сервера не было."""
    return {
        "тип": type(err).__name__,
        "url": f"{BASE_URL}/chat/completions",
        "сообщение": str(err) or "исключение без текста",
        "таймаут_секунд": timeout,
        "примечание": "ответа от сервера не поступило, тела ответа не существует",
    }


def _parse_body(text: str):
    """Разбирает тело ошибки как JSON, а если не вышло — отдаёт текстом."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text[:2000]


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
    response: dict = field(default_factory=dict)

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


def describe_response(body: dict, *, streamed: bool = False) -> dict:
    """Ответ API без самого текста ответа.

    Текст уже отрисован пользователю выше, повторять его в JSON незачем —
    он только мешает разглядеть служебные поля: usage, finish_reason,
    идентификатор запроса. Вместо текста остаётся его длина.
    """
    trimmed = copy.deepcopy(body)
    for choice in trimmed.get("choices") or []:
        for part_name in ("message", "delta"):
            part = choice.get(part_name)
            if not isinstance(part, dict):
                continue
            for field in ("content", "reasoning_content"):
                value = part.get(field)
                # Пустую строку оставляем как есть: в последнем куске потока
                # текста и правда нет, подпись «0 символов» только путала бы.
                if isinstance(value, str) and value:
                    part[field] = f"<{len(value)} символов, показано выше>"
    if streamed:
        trimmed["примечание"] = (
            "последний кусок потока: ответ пришёл частями, "
            "usage и finish_reason приходят в самом конце"
        )
    return trimmed


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
        raise LLMError(
            f"Ошибка API {err.response.status_code}: {err.response.text[:300]}",
            status=err.response.status_code,
            body=_parse_body(err.response.text),
        ) from err
    except requests.RequestException as err:
        raise LLMError(
            _network_message(err), diagnostics=_network_diagnostics(err, timeout)
        ) from err
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
        response=describe_response(body),
    )


async def stream(
    messages: list[dict], *, timeout: int = 300, **options
) -> AsyncIterator[dict]:
    """Потоковый вызов: отдаёт куски ответа по мере генерации.

    Генерирует события:
      {"type": "request", "request": {...}} — что именно уходит в API;
      {"type": "reasoning", "text": ...} — кусок рассуждения (если thinking включён);
      {"type": "content",   "text": ...} — кусок ответа;
      {"type": "response", "response": {...}} — ответ API без текста;
      {"type": "done", "finish_reason": ..., "usage": {...}, "elapsed": ...}.
    """
    payload = build_payload(messages, **options)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    yield {"type": "request", "request": describe_request(payload)}

    started = time.monotonic()
    finish_reason = "unknown"
    usage: dict = {}
    last_chunk: dict = {}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{BASE_URL}/chat/completions", headers=_headers(), json=payload
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise LLMError(
                        f"Ошибка API {response.status_code}: {detail[:300]}",
                        status=response.status_code,
                        body=_parse_body(detail),
                    )

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

                    last_chunk = chunk
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
        raise LLMError(
            _network_message(err), diagnostics=_network_diagnostics(err, timeout)
        ) from err

    if last_chunk:
        yield {"type": "response", "response": describe_response(last_chunk, streamed=True)}

    yield {
        "type": "done",
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed": round(time.monotonic() - started, 2),
    }
