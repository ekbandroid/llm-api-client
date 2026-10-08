"""Клиент к LLM по OpenAI-совместимому API: синхронный вызов и потоковый."""

import copy
import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from urllib.parse import urlparse

import httpx
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-flash")

# Адреса, по которым модель считается запущенной на этой же машине.
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0", "[::1]")

def is_local(url: str) -> bool:
    """Модель крутится на этой же машине?

    Отличать нужно ради двух вещей: ключа (Ollama и LM Studio его не
    спрашивают вовсе, а у облака без ключа запрос не имеет смысла) и
    терпения — пределы ожидания ниже. Проверяем по адресу, а не по настройке:
    настройку можно забыть переключить, адрес — нет.
    """
    host = (urlparse(url or "").hostname or "").lower()
    return host in LOCAL_HOSTS or host.endswith(".local")


# Диалект API. Формат запроса у всех общий — openai, — но у DeepSeek поверх
# него есть свои поля: thinking и reasoning_effort. Чужому серверу они не
# нужны, а некоторые на незнакомое поле отвечают ошибкой 400, и запрос
# ломается целиком. Определяем по адресу, переопределяется переменной.
DIALECT = (os.getenv("LLM_DIALECT")
           or ("deepseek" if "deepseek" in (urlparse(BASE_URL).hostname or "")
               else "openai")).lower()
THINKING = os.getenv("LLM_THINKING", "true").lower() == "true"
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "high")

# Системный промпт по умолчанию — разный у облака и у своей машины.
#
# «You are a helpful assistant» написано ни о чём, и большой модели этого
# хватает: задачу она понимает из самого запроса. Модель на три миллиарда
# параметров так не умеет — ей нужно сказать, что делать, коротко и прямо.
# Здесь сказано ровно то, чем эта модель занята в приложении: отвечает по
# выдержкам из документов пользователя. Оговорка «если в запросе есть
# выдержки» оставляет обычную беседу обычной.
LOCAL_SYSTEM_PROMPT = (
    "Отвечай по-русски, коротко и по делу. Если в запросе есть выдержки из "
    "документов — отвечай только по ним: приведи дословную цитату и назови "
    "файл. Чего в выдержках нет, того нет и в документах: не добавляй по "
    "памяти и не угадывай, а скажи, что не нашлось."
)

SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT") or (
    LOCAL_SYSTEM_PROMPT if is_local(BASE_URL) else "You are a helpful assistant.")


def _temperature() -> float | None:
    """Температура по умолчанию: своя для местной модели, никакой для облака.

    Ollama без этого поля берёт 0,8 — в самый раз для беседы и слишком много
    для задачи «ответь по выдержке и приведи из неё цитату»: модель
    пересказывает выдержку своими словами, и дословная цитата не сходится.
    Облаку значение не навязываем: его ответы устраивают, а менять поведение
    всех диалогов заодно с локальной настройкой значило бы протащить одно под
    видом другого.
    """
    задано = os.getenv("LLM_TEMPERATURE", "").strip().replace(",", ".")
    if not задано:
        return 0.2 if is_local(BASE_URL) else None
    try:
        # Границы те же, что у лаборатории температуры в интерфейсе: за ними
        # ответы перестают быть ответами, и молча пропускать такое незачем.
        return min(2.0, max(0.0, float(задано)))
    except ValueError:
        print(f"LLM_TEMPERATURE={задано!r} — не число, беру значение по умолчанию")
        return 0.2 if is_local(BASE_URL) else None


TEMPERATURE = _temperature()


# Сколько ждём установки соединения. Отдельно от ожидания ответа: сервер,
# который не отвечает вовсе, должен отваливаться быстро.
CONNECT_TIMEOUT = 10

# Во сколько раз терпеливее мы к модели на своей машине.
#
# Она думает дольше облачной и не параллелится: Ollama обслуживает запросы к
# одной модели по очереди, так что пока сервер поиска считает своё, ответ
# стоит и кусков не шлёт. Замер на qwen2.5:3b, время до первого куска: 6,4 с
# на 616 токенов, 13,8 с на 1823, 23,6 с на 3625 — около 6,5 с на каждую
# тысячу. Настоящий запрос приложения со схемами инструментов и выдержками
# вдвое больше, да ещё может ждать очереди; шестидесяти секунд ему мало, и
# пользователь получал ReadTimeout вместо ответа.
LOCAL_PATIENCE = 5 if is_local(BASE_URL) else 1

# Допустимая пауза между кусками потокового ответа. Это не лимит на всю
# генерацию: отсчёт начинается заново с каждым полученным куском. Нужен,
# чтобы зависшая модель не держала пользователя перед пустым экраном
# до самого общего таймаута.
STALL_TIMEOUT = 60 * LOCAL_PATIENCE

# Сколько ждём ПЕРВОГО куска текста. Отдельный предел нужен потому, что
# сервер может держать соединение живым служебными пакетами «: keep-alive»
# и при этом не начать отвечать вовсе: данные формально идут, таймаут чтения
# не срабатывает, и пользователь сидит перед пустым экраном до общего лимита.
FIRST_TOKEN_TIMEOUT = 60 * LOCAL_PATIENCE

# Пределы на весь вызов — тем же множителем. Служебные вызовы (переписывание
# запроса, судья, карточка фактов) идут через complete, и на местной модели
# каждый занимает десятки секунд; поток отвечает дольше, и ему дано больше.
COMPLETE_TIMEOUT = 120 * LOCAL_PATIENCE
STREAM_TIMEOUT = 300 * LOCAL_PATIENCE


# API отклоняет запрос целиком, если выбран формат json_object, а слова «json»
# нет ни в одном сообщении: «Prompt must contain the word 'json' in some form».
# Регистр не важен. Строку дописываем сами, чтобы настройка формата не могла
# уронить запрос.
JSON_REQUIRED_NOTE = "Ответ верни одним JSON-объектом."


def _mentions_json(messages: list[dict]) -> bool:
    return any("json" in (m.get("content") or "").lower() for m in messages)


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
        "таймаут_соединения": CONNECT_TIMEOUT,
        "пауза_между_кусками": STALL_TIMEOUT,
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
    seed: int | None = None,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
) -> dict:
    """Собирает тело запроса к /chat/completions."""
    use_thinking = THINKING if thinking is None else thinking
    # Служебные вызовы задают температуру сами (судья — 0, переписывание
    # вопроса — 0,3), и подставлять им общее значение нельзя. Подставляем
    # только там, где его не задали вовсе, — то есть в обычном разговоре.
    if temperature is None:
        temperature = TEMPERATURE
    payload: dict = {"model": model or MODEL, "messages": messages}
    # Поля рассуждений — диалект DeepSeek, и уходят только к нему. Локальной
    # модели они не нужны, а строгий сервер на незнакомое поле отвечает 400 и
    # роняет весь запрос.
    if DIALECT == "deepseek":
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
    # Зерно генератора. Приложению не нужно, нужно замеру: без него два
    # прогона одной и той же настройки расходятся сами по себе, и разницу
    # в десять процентов не отличить от шума. Поддерживают и Ollama, и
    # OpenAI-совместимые серверы; кто не поддерживает — поле проигнорирует.
    if seed is not None:
        payload["seed"] = seed
    # Схемы инструментов MCP. Решает по ним модель: приложение лишь выполняет
    # то, что она запросила, и возвращает результат следующим запросом.
    if tools:
        payload["tools"] = tools
    if response_format:
        payload["response_format"] = response_format
        if response_format.get("type") == "json_object" and not _mentions_json(messages):
            payload["messages"] = messages + [
                {"role": "system", "content": JSON_REQUIRED_NOTE}
            ]
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


def describe_response(
    body: dict, *, streamed: bool = False, chunks: int = 0, keep_text: bool = False
) -> dict:
    """Ответ API без самого текста ответа.

    Текст уже отрисован пользователю выше, повторять его в JSON незачем —
    он только мешает разглядеть служебные поля: usage, finish_reason,
    идентификатор запроса. Вместо текста остаётся его длина.

    keep_text=True — для служебных вызовов: там текст ответа это и есть
    карточка фактов или решение переключателя этапов, и больше его нигде
    не видно. Вырезать его значило бы показать пустой блок.
    """
    trimmed = copy.deepcopy(body)
    if not keep_text:
        for choice in trimmed.get("choices") or []:
            for part_name in ("message", "delta"):
                part = choice.get(part_name)
                if not isinstance(part, dict):
                    continue
                for field in ("content", "reasoning_content"):
                    value = part.get(field)
                    # Пустую строку оставляем как есть: в куске потока текста
                    # и правда нет, подпись «0 символов» только путала бы.
                    if isinstance(value, str) and value:
                        part[field] = f"<{len(value)} символов, показано выше>"
    if streamed:
        trimmed["примечание"] = (
            f"собрано приложением из {chunks} кусков потока: по отдельности "
            "каждый кусок несёт несколько символов текста, а usage и "
            "finish_reason приходят только в последнем"
        )
    return trimmed


def assemble_stream(
    header: dict, *, content: str, reasoning: str, finish_reason: str, usage: dict,
    tool_calls: list[dict] | None = None,
) -> dict:
    """Склеивает куски потока в ответ того же вида, что приходит без потока.

    Показывать последний кусок бессмысленно: в нём пустой delta.content и
    служебные поля, по которым не видно ни самого ответа, ни его длины.
    Показывать все куски тоже нельзя — их сотни. Поэтому собираем один объект:
    поля берём из первого куска, текст и причину остановки — накопленные.
    """
    message: dict = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": header.get("id"),
        "object": "chat.completion",
        "created": header.get("created"),
        "model": header.get("model"),
        "system_fingerprint": header.get("system_fingerprint"),
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


def headers() -> dict:
    """Заголовок авторизации — или ничего, если сервер его не ждёт."""
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    if is_local(BASE_URL):
        # Локальный сервер ключа не ждёт. Присылать «Bearer None» ему можно,
        # но это ложь в заголовке: лучше не присылать ничего.
        return {}
    raise LLMError("Не задан LLM_API_KEY. Скопируйте .env.example в .env и впишите ключ.")


def complete(
    messages: list[dict], *, timeout: int | None = None, keep_text: bool = False, **options
) -> Completion:
    """Синхронный вызов: ждёт ответ целиком и возвращает его с телеметрией.

    keep_text=True оставляет текст ответа в телеметрии — так вызывают
    служебные обращения, у которых этот текст нигде больше не показан.
    """
    payload = build_payload(messages, **options)
    timeout = COMPLETE_TIMEOUT if timeout is None else timeout

    started = time.monotonic()
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions", headers=headers(), json=payload,
            timeout=(CONNECT_TIMEOUT, timeout),
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
        response=describe_response(body, keep_text=keep_text),
    )


async def stream(
    messages: list[dict], *, timeout: int | None = None, **options
) -> AsyncIterator[dict]:
    """Потоковый вызов: отдаёт куски ответа по мере генерации.

    Генерирует события:
      {"type": "request", "request": {...}} — что именно уходит в API;
      {"type": "waiting", "elapsed": ...} — соединение живо, но текста ещё нет;
      {"type": "reasoning", "text": ...} — кусок рассуждения (если thinking включён);
      {"type": "content",   "text": ...} — кусок ответа;
      {"type": "response", "response": {...}} — ответ API без текста;
      {"type": "done", "finish_reason": ..., "usage": {...}, "elapsed": ...,
       "tool_calls": [...]} — вызовы инструментов, если модель их запросила.
    """
    payload = build_payload(messages, **options)
    timeout = STREAM_TIMEOUT if timeout is None else timeout
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    yield {"type": "request", "request": describe_request(payload)}

    started = time.monotonic()
    finish_reason = "unknown"
    usage: dict = {}
    # Куски потока копим не целиком: для показа нужны служебные поля первого
    # куска, накопленный текст и счёт кусков.
    header: dict = {}
    chunks = 0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    # Вызовы инструментов приходят кусками, как и текст: в первом куске имя и
    # id, в следующих — аргументы по частям. Собираем по индексу вызова.
    calls: dict[int, dict] = {}

    try:
        limits = httpx.Timeout(timeout, connect=CONNECT_TIMEOUT, read=STALL_TIMEOUT)
        async with httpx.AsyncClient(timeout=limits) as client:
            async with client.stream(
                "POST", f"{BASE_URL}/chat/completions", headers=headers(), json=payload
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise LLMError(
                        f"Ошибка API {response.status_code}: {detail[:300]}",
                        status=response.status_code,
                        body=_parse_body(detail),
                    )

                produced = False
                notified = 0.0
                async for line in response.aiter_lines():
                    waited = time.monotonic() - started
                    if not produced:
                        if waited > FIRST_TOKEN_TIMEOUT:
                            raise LLMError(
                                f"Модель не начала отвечать за {FIRST_TOKEN_TIMEOUT} с. "
                                "Сервер принял запрос и держит соединение, но текста не шлёт — "
                                "похоже на сбой модели на стороне провайдера.",
                                status=response.status_code,
                                body={"note": "соединение живо, получены только служебные пакеты",
                                      "waited_seconds": round(waited, 1)},
                            )
                        # Раз в пять секунд сообщаем, что ждём, — иначе окно
                        # выглядит зависшим, хотя запрос в работе.
                        if waited - notified >= 5:
                            notified = waited
                            yield {"type": "waiting", "elapsed": round(waited, 1)}
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    chunks += 1
                    if not header:
                        header = chunk
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        if reasoning := delta.get("reasoning_content"):
                            produced = True
                            reasoning_parts.append(reasoning)
                            yield {"type": "reasoning", "text": reasoning}
                        if content := delta.get("content"):
                            produced = True
                            content_parts.append(content)
                            yield {"type": "content", "text": content}
                        for call in delta.get("tool_calls") or []:
                            produced = True
                            slot = calls.setdefault(
                                call.get("index", 0),
                                {"id": "", "type": "function",
                                 "function": {"name": "", "arguments": ""}},
                            )
                            if call.get("id"):
                                slot["id"] = call["id"]
                            piece = call.get("function") or {}
                            if piece.get("name"):
                                slot["function"]["name"] = piece["name"]
                            if piece.get("arguments"):
                                slot["function"]["arguments"] += piece["arguments"]
    except httpx.HTTPError as err:
        raise LLMError(
            _network_message(err), diagnostics=_network_diagnostics(err, timeout)
        ) from err

    ordered = [calls[i] for i in sorted(calls)]
    if header:
        assembled = assemble_stream(
            header,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=ordered,
        )
        yield {
            "type": "response",
            "response": describe_response(assembled, streamed=True, chunks=chunks),
        }

    yield {
        "type": "done",
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed": round(time.monotonic() - started, 2),
        "tool_calls": ordered,
    }
