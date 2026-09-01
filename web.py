"""Веб-интерфейс к LLM: потоковый ответ и настройка ограничений из браузера."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import llm

STATIC = Path(__file__).parent / "static"

# Пресеты формата ответа. Пустая строка — ограничение не накладывается.
FORMAT_PRESETS: dict[str, dict[str, str]] = {
    "free": {"label": "Без ограничений", "spec": ""},
    "structured": {
        "label": "ТЕЗИС / ПРИЧИНЫ / ВЫВОД",
        "spec": (
            "Отвечай строго в следующем формате и ни в каком другом:\n\n"
            "ТЕЗИС: <одно предложение>\n"
            "ПРИЧИНЫ:\n- <пункт>\n- <пункт>\n- <пункт>\n"
            "ВЫВОД: <одно предложение>\n\n"
            "Ровно три пункта в разделе ПРИЧИНЫ. Никакого текста до строки «ТЕЗИС:». "
            "Без markdown-разметки и вводных фраз."
        ),
    },
    "bullets": {
        "label": "Только маркированный список",
        "spec": (
            "Отвечай только маркированным списком из 3–5 пунктов. "
            "Каждый пункт — одна строка, начинается с «- ». "
            "Никакого текста до и после списка, без заголовков и вводных фраз."
        ),
    },
    "json": {
        "label": "Строгий JSON",
        "spec": (
            "Отвечай одним валидным JSON-объектом и ничем больше:\n"
            '{"answer": "<краткий ответ одной строкой>", '
            '"key_points": ["<пункт>", "<пункт>", "<пункт>"]}\n'
            "Без markdown, без обрамляющих ```-блоков, без пояснений вне JSON."
        ),
    },
    "custom": {"label": "Свой формат", "spec": ""},
}

FALLBACK_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]


class Constraints(BaseModel):
    """Ограничения, выставленные в интерфейсе."""

    model: str | None = None
    format_preset: str = "free"
    custom_format: str = ""
    word_limit: int | None = Field(default=None, ge=1, le=5000)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    stop: str = ""
    thinking: bool = False


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    constraints: Constraints = Constraints()


app = FastAPI(title="LLM chat")


def build_system_prompt(c: Constraints) -> str:
    """Собирает системный промпт из выставленных ограничений."""
    parts = [llm.SYSTEM_PROMPT]

    spec = c.custom_format.strip() if c.format_preset == "custom" else FORMAT_PRESETS.get(
        c.format_preset, FORMAT_PRESETS["free"]
    )["spec"]
    if spec:
        parts.append(spec)

    limits = []
    if c.word_limit:
        limits.append(f"- уложись не более чем в {c.word_limit} слов во всём ответе")
    if c.stop.strip():
        limits.append(
            f"- закончив ответ, выведи отдельной строкой {c.stop.strip()} "
            "и немедленно прекрати генерацию"
        )
    if limits:
        parts.append("Ограничения:\n" + "\n".join(limits))

    return "\n\n".join(parts)


def sse(event: dict) -> str:
    """Оформляет событие в формат Server-Sent Events."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/api/config")
async def config() -> dict:
    """Отдаёт интерфейсу список моделей и пресетов формата."""
    models = FALLBACK_MODELS
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{llm.BASE_URL}/models", headers={"Authorization": f"Bearer {llm.API_KEY}"}
            )
        if response.status_code == 200:
            models = [m["id"] for m in response.json().get("data", [])] or FALLBACK_MODELS
    except httpx.HTTPError:
        pass  # список моделей не критичен — отдаём запасной

    return {
        "models": models,
        "default_model": llm.MODEL,
        "formats": [{"id": k, "label": v["label"]} for k, v in FORMAT_PRESETS.items()],
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Стримит ответ модели в виде SSE."""
    c = request.constraints
    system_prompt = build_system_prompt(c)
    messages = [{"role": "system", "content": system_prompt}]
    messages += [m.model_dump() for m in request.messages]

    stop = [s for s in (c.stop.strip(),) if s]

    async def events() -> AsyncIterator[str]:
        yield sse({"type": "meta", "system": system_prompt, "model": c.model or llm.MODEL})
        try:
            async for event in llm.stream(
                messages,
                model=c.model,
                max_tokens=c.max_tokens,
                stop=stop,
                thinking=c.thinking,
            ):
                yield sse(event)
        except llm.LLMError as err:
            yield sse({"type": "error", "message": str(err)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
