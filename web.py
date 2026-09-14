"""Веб-интерфейс к LLM: потоковый ответ и настройка ограничений из браузера."""

import asyncio
import json
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

import auth
import db
import benchmark
import history
import llm
import memory
import reasoning
import temperature as temperature_mod
import tokens as tokens_mod

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

FALLBACK_MODELS = ["deepseek-flash", "deepseek-v4-pro"]


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


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    yield


app = FastAPI(title="LLM chat", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=auth.SESSION_SECRET or secrets.token_urlsafe(48),
    session_cookie="llm_session",
    same_site="lax",
    # За TLS кука должна ходить только по HTTPS: COOKIE_SECURE=true в .env.
    https_only=os.getenv("COOKIE_SECURE", "false").lower() == "true",
    max_age=14 * 24 * 3600,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def revalidate_assets(request: Request, call_next):
    """Заставляет браузер проверять, не изменились ли страницы и статика.

    Без Cache-Control браузер сам решает, сколько считать файл свежим, и после
    выкатки показывает старый CSS, не спрашивая сервер. «no-cache» означает не
    «не кэшировать», а «кэшируй, но каждый раз переспрашивай»: вместе с ETag
    проверка стоит один ответ 304 без тела.
    """
    response = await call_next(request)
    is_page = response.headers.get("content-type", "").startswith("text/html")
    if is_page or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---------- доступ ----------

def current_user(request: Request) -> dict | None:
    """Пользователь текущей сессии или None."""
    user_id = request.session.get("user_id")
    return db.get(user_id) if user_id else None


def require_approved(request: Request) -> dict:
    """Пускает только подтверждённых. Для API-маршрутов."""
    user = current_user(request)
    if user is None:
        raise HTTPException(401, "Требуется вход")
    if user["status"] != db.APPROVED:
        raise HTTPException(403, "Доступ ещё не подтверждён администратором")
    return user


def require_admin(request: Request) -> dict:
    """Пускает только администраторов."""
    user = require_approved(request)
    if not user["is_admin"]:
        raise HTTPException(403, "Нужны права администратора")
    return user


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
async def config(_: dict = Depends(require_approved)) -> dict:
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
        "prices": benchmark.DEFAULT_PRICES,
        "price_note": "USD за 1 млн токенов, пиковые ставки без попадания в кэш",
    }


@app.post("/api/chat")
async def chat(
    request: ChatRequest, _: dict = Depends(require_approved)
) -> StreamingResponse:
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
            yield sse({"type": "error", "message": str(err), "response": err.response, "diagnostics": err.diagnostics})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- страницы ----------

@app.get("/")
async def index(request: Request):
    """Чат — только для подтверждённых пользователей."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["status"] != db.APPROVED:
        return RedirectResponse("/pending", status_code=302)
    return FileResponse(STATIC / "index.html")


@app.get("/legacy")
async def legacy_page(request: Request):
    """Прежний чат с ограничениями — без сохранения истории."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["status"] != db.APPROVED:
        return RedirectResponse("/pending", status_code=302)
    return FileResponse(STATIC / "legacy.html")


@app.get("/login")
async def login_page(request: Request):
    if current_user(request):
        return RedirectResponse("/", status_code=302)
    return FileResponse(STATIC / "login.html")


@app.get("/register")
async def register_page(request: Request):
    if current_user(request):
        return RedirectResponse("/", status_code=302)
    return FileResponse(STATIC / "register.html")


@app.get("/pending")
async def pending_page(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["status"] == db.APPROVED:
        return RedirectResponse("/", status_code=302)
    return FileResponse(STATIC / "pending.html")


@app.get("/reasoning")
async def reasoning_page(request: Request):
    """Режим сравнения способов рассуждения — для подтверждённых пользователей."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["status"] != db.APPROVED:
        return RedirectResponse("/pending", status_code=302)
    return FileResponse(STATIC / "reasoning.html")


@app.get("/temperature")
async def temperature_page(request: Request):
    """Режим сравнения температур — для подтверждённых пользователей."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["status"] != db.APPROVED:
        return RedirectResponse("/pending", status_code=302)
    return FileResponse(STATIC / "temperature.html")


@app.get("/models")
async def models_page(request: Request):
    """Режим сравнения моделей — для подтверждённых пользователей."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["status"] != db.APPROVED:
        return RedirectResponse("/pending", status_code=302)
    return FileResponse(STATIC / "models.html")


@app.get("/strategies")
async def strategies_page(request: Request):
    """Сравнение стратегий управления контекстом."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["status"] != db.APPROVED:
        return RedirectResponse("/pending", status_code=302)
    return FileResponse(STATIC / "strategies.html")


@app.get("/tokens")
async def tokens_page(request: Request):
    """Режим разбора расхода токенов."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if user["status"] != db.APPROVED:
        return RedirectResponse("/pending", status_code=302)
    return FileResponse(STATIC / "tokens.html")


@app.get("/admin")
async def admin_page(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    if not (user["status"] == db.APPROVED and user["is_admin"]):
        raise HTTPException(403, "Нужны права администратора")
    return FileResponse(STATIC / "admin.html")


# ---------- вход по логину и паролю ----------

class Credentials(BaseModel):
    login: str
    password: str
    email: str = ""
    name: str = ""


def _finish_login(request: Request, user: dict) -> dict:
    """Заводит сессию и говорит странице, куда идти дальше."""
    request.session["user_id"] = user["id"]
    return {
        "ok": True,
        "redirect": "/" if user["status"] == db.APPROVED else "/pending",
        "status": user["status"],
    }


@app.post("/api/auth/register")
def api_register(request: Request, creds: Credentials) -> dict:
    """Регистрирует нового пользователя со статусом «ожидает подтверждения».

    Синхронный def — FastAPI уводит его в пул потоков, и scrypt не блокирует
    событийный цикл на время хеширования.
    """
    login = creds.login.strip()
    if error := auth.validate_credentials(login, creds.password):
        raise HTTPException(400, error)

    is_admin = login.lower() in auth.admin_logins()
    user = db.create_user(
        login,
        auth.hash_password(creds.password),
        email=creds.email.strip(),
        name=creds.name.strip(),
        is_admin=is_admin,
    )
    if user is None:
        raise HTTPException(409, "Такой логин уже занят")
    return _finish_login(request, user)


@app.post("/api/auth/login")
def api_login(request: Request, creds: Credentials) -> dict:
    """Проверяет логин и пароль."""
    login = creds.login.strip()

    if wait := auth.throttle_check(login):
        raise HTTPException(429, f"Слишком много попыток. Повторите через {wait} с.")

    user = db.get_by_login(login)
    if user is None or not auth.verify_password(creds.password, user["password_hash"]):
        auth.throttle_fail(login)
        # Одна формулировка на оба случая: иначе форма подскажет, какие логины заняты.
        raise HTTPException(401, "Неверный логин или пароль")

    auth.throttle_reset(login)
    db.touch_login(user["id"])
    return _finish_login(request, user)


# ---------- память: профиль и проекты ----------
#
# Три слоя памяти лежат порознь: профиль — у пользователя, бриф и накопленные
# факты — у проекта, переписка — у диалога. Здесь только первые два: диалоговым
# слоем занимаются маршруты диалогов ниже.

class ProfileMemory(BaseModel):
    text: str = ""


class ProjectCreate(BaseModel):
    title: str = ""
    brief: str = ""


class ProjectPatch(BaseModel):
    title: str | None = None
    brief: str | None = None
    collect_facts: bool | None = None


def _owned_project(project_id: int, user: dict) -> dict:
    """Проект пользователя или 404 — как и у диалогов, чужой неотличим от несуществующего."""
    project = db.get_project(project_id, user["id"])
    if project is None:
        raise HTTPException(404, "Проект не найден")
    return project


@app.get("/api/me/memory")
async def profile_memory_get(user: dict = Depends(require_approved)) -> dict:
    """Долговременная память — то, что уходит во все диалоги с включённым слоем."""
    text = user["profile_memory"] or ""
    return {"text": text, "limit": db.PROFILE_LIMIT, "tokens": tokens_mod.estimate_tokens(text)}


@app.put("/api/me/memory")
async def profile_memory_put(
    payload: ProfileMemory, user: dict = Depends(require_approved)
) -> dict:
    db.set_profile_memory(user["id"], payload.text)
    return await profile_memory_get(db.get(user["id"]))


@app.get("/api/projects")
async def projects_list(user: dict = Depends(require_approved)) -> dict:
    return {"projects": db.list_projects(user["id"])}


@app.post("/api/projects")
async def project_create(
    payload: ProjectCreate, user: dict = Depends(require_approved)
) -> dict:
    return db.create_project(user["id"], title=payload.title, brief=payload.brief)


@app.get("/api/projects/{project_id}")
async def project_get(project_id: int, user: dict = Depends(require_approved)) -> dict:
    project = _owned_project(project_id, user)
    facts = memory.load_facts(project["facts"])
    return {
        "project": project,
        "facts": facts,
        "brief_limit": db.BRIEF_LIMIT,
        "facts_limit": memory.PROJECT_FACTS_LIMIT,
        # Считаем и для пустого проекта: название всё равно уходит в запрос.
        "tokens": tokens_mod.estimate_tokens(
            memory.project_text(memory.Layers(
                project_id=project["id"], project_title=project["title"],
                project_brief=project["brief"] or "", project_facts=facts,
            ))
        ),
    }


@app.patch("/api/projects/{project_id}")
async def project_patch(
    project_id: int, payload: ProjectPatch, user: dict = Depends(require_approved)
) -> dict:
    _owned_project(project_id, user)
    updated = db.update_project(
        project_id, user["id"], title=payload.title, brief=payload.brief,
        collect_facts=payload.collect_facts,
    )
    if updated is None:
        raise HTTPException(404, "Проект не найден")
    return updated


@app.delete("/api/projects/{project_id}")
async def project_delete(project_id: int, user: dict = Depends(require_approved)) -> dict:
    """Удаляет проект вместе с диалогами. Предупреждение показывает интерфейс."""
    _owned_project(project_id, user)
    killed = db.delete_project(project_id, user["id"])
    if killed is None:
        raise HTTPException(404, "Проект не найден")
    return {"ok": True, "conversations_deleted": killed}


@app.delete("/api/projects/{project_id}/facts/{key}")
async def project_fact_delete(
    project_id: int, key: str, user: dict = Depends(require_approved)
) -> dict:
    """Убирает один накопленный факт: память проекта должна быть управляемой."""
    project = _owned_project(project_id, user)
    facts = memory.load_facts(project["facts"])
    if facts.pop(key, None) is None:
        raise HTTPException(404, "Такого факта нет")
    db.set_project_facts(project_id, json.dumps(facts, ensure_ascii=False))
    return {"ok": True, "facts": facts}


# ---------- диалоги ----------

class ConversationCreate(BaseModel):
    title: str = ""
    model: str | None = None
    thinking: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=384_000)
    strategy: str = db.DEFAULT_STRATEGY
    context_n: int = Field(default=db.DEFAULT_CONTEXT_N, ge=1, le=200)
    # Диалог заводится сразу внутри проекта; переносить его потом нельзя.
    project_id: int | None = None


class ConversationPatch(BaseModel):
    title: str | None = None
    model: str | None = None
    thinking: bool | None = None
    # Ноль означает «снять лимит». Пропущенное поле означает «не трогать» —
    # по одному None эти два намерения не различить.
    max_tokens: int | None = Field(default=None, ge=0, le=384_000)
    strategy: str | None = None
    context_n: int | None = Field(default=None, ge=1, le=200)
    # Какие слои памяти подключать к запросам этого диалога.
    use_profile: bool | None = None
    use_project: bool | None = None


class NewMessage(BaseModel):
    content: str


def _owned(conversation_id: int, user: dict) -> dict:
    """Достаёт диалог, убеждаясь, что он принадлежит этому пользователю."""
    conversation = db.get_conversation(conversation_id, user["id"])
    if conversation is None:
        # Не различаем «нет такого» и «чужой»: иначе по коду ответа можно
        # перебором узнать, какие идентификаторы существуют.
        raise HTTPException(404, "Диалог не найден")
    return conversation


def _public_message(row: dict) -> dict:
    """Сообщение в том виде, в каком его ждёт страница."""
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "reasoning": row["reasoning"],
        "tokens_completion": row["tokens_completion"],
        "tokens_total": row["tokens_total"],
        "meta": json.loads(row["meta"]) if row["meta"] else None,
        "created_at": row["created_at"],
    }


@app.get("/api/conversations")
async def conversations_list(user: dict = Depends(require_approved)) -> dict:
    return {"conversations": db.list_conversations(user["id"])}


@app.post("/api/conversations")
async def conversation_create(
    payload: ConversationCreate, user: dict = Depends(require_approved)
) -> dict:
    return db.create_conversation(
        user["id"],
        title=payload.title or db.NEW_TITLE,
        model=payload.model,
        thinking=payload.thinking,
        max_tokens=payload.max_tokens,
        strategy=payload.strategy,
        context_n=payload.context_n,
        project_id=_owned_project(payload.project_id, user)["id"] if payload.project_id else None,
    )


@app.get("/api/conversations/{conversation_id}")
async def conversation_get(
    conversation_id: int, user: dict = Depends(require_approved)
) -> dict:
    conversation = _owned(conversation_id, user)
    return {
        "conversation": conversation,
        "messages": [_public_message(m) for m in db.list_messages(conversation_id)],
    }


@app.patch("/api/conversations/{conversation_id}")
async def conversation_patch(
    conversation_id: int, payload: ConversationPatch, user: dict = Depends(require_approved)
) -> dict:
    _owned(conversation_id, user)
    updated = db.update_conversation(
        conversation_id, user["id"],
        title=payload.title, model=payload.model, thinking=payload.thinking,
        max_tokens=payload.max_tokens or None,
        clear_max_tokens=payload.max_tokens == 0,
        strategy=payload.strategy,
        context_n=payload.context_n,
        use_profile=payload.use_profile,
        use_project=payload.use_project,
    )
    if updated is None:
        raise HTTPException(404, "Диалог не найден")
    return updated


class BranchRequest(BaseModel):
    from_message_id: int


@app.post("/api/conversations/{conversation_id}/branch")
async def conversation_branch(
    conversation_id: int, payload: BranchRequest, user: dict = Depends(require_approved)
) -> dict:
    """Создаёт ветку: копию диалога по указанное сообщение включительно."""
    _owned(conversation_id, user)
    branch = db.create_branch(conversation_id, user["id"], payload.from_message_id)
    if branch is None:
        raise HTTPException(400, "Нечего ветвить: сообщение не найдено")
    return branch


@app.delete("/api/conversations/{conversation_id}")
async def conversation_delete(
    conversation_id: int, user: dict = Depends(require_approved)
) -> dict:
    _owned(conversation_id, user)
    return {"ok": db.delete_conversation(conversation_id, user["id"])}


@app.post("/api/conversations/{conversation_id}/messages")
async def conversation_send(
    conversation_id: int, payload: NewMessage, user: dict = Depends(require_approved)
) -> StreamingResponse:
    """Принимает новое сообщение, историю поднимает сам и стримит ответ.

    Клиент присылает только текст: историю он не передаёт и подменить её
    не может. Оба сообщения сохраняются, поэтому после перезапуска диалог
    продолжается с того же места.
    """
    conversation = _owned(conversation_id, user)
    content = payload.content.strip()
    if not content:
        raise HTTPException(400, "Пустое сообщение")

    system_prompt = llm.SYSTEM_PROMPT
    model = conversation["model"] or llm.MODEL
    thinking = bool(conversation["thinking"])
    max_tokens = conversation["max_tokens"] or None

    asked = db.add_message(conversation_id, "user", content)
    # Первое сообщение даёт диалогу имя — иначе список будет из «Новых диалогов».
    if conversation["title"] == db.NEW_TITLE:
        db.update_conversation(conversation_id, user["id"], title=content)

    # Заново читаем диалог: только что добавленный вопрос должен войти в план.
    layers = memory.collect(conversation, user)
    plan = history.plan_request(
        _owned(conversation_id, user), system_prompt,
        memory_blocks=memory.blocks(layers), memory_info=memory.describe(layers),
    )
    messages = plan.messages

    async def events() -> AsyncIterator[str]:
        answer, reasoning = "", ""
        meta: dict = {"system": system_prompt, "model": model}
        completion_tokens = total_tokens = 0
        finish_reason = "unknown"
        elapsed = 0.0

        # Состав слоёв показываем до ответа: иначе понять, ушла ли память
        # в запрос, можно было бы только перезагрузив страницу.
        if plan.memory:
            yield sse({"type": "memory", "memory": plan.memory})

        try:
            async for event in llm.stream(
                messages, model=model, thinking=thinking, max_tokens=max_tokens
            ):
                if event["type"] == "content":
                    answer += event["text"]
                elif event["type"] == "reasoning":
                    reasoning += event["text"]
                elif event["type"] == "request":
                    meta["request"] = event["request"]
                elif event["type"] == "response":
                    meta["response"] = event["response"]
                elif event["type"] == "done":
                    usage = event.get("usage") or {}
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    finish_reason = event["finish_reason"]
                    elapsed = event["elapsed"]
                yield sse(event)
        except llm.LLMError as err:
            # Обмен не состоялся — убираем вопрос из истории. Иначе каждая
            # неудачная попытка оставалась бы в диалоге навсегда и оплачивалась
            # заново в каждом следующем запросе.
            db.delete_message(asked["id"])
            yield sse({"type": "error", "message": str(err), "response": err.response, "diagnostics": err.diagnostics})
            return

        meta.update(
            finish_reason=finish_reason, elapsed=elapsed,
            memory=plan.memory,
            context={
                "strategy": plan.strategy, "verbatim": plan.verbatim,
                "folded": plan.folded, "dropped": plan.dropped,
                "facts": len(plan.facts),
            },
        )
        saved = db.add_message(
            conversation_id, "assistant", answer,
            reasoning=reasoning or None,
            tokens_completion=completion_tokens, tokens_total=total_tokens,
            meta=json.dumps(meta, ensure_ascii=False),
        )
        yield sse({"type": "saved", "message_id": saved["id"], "title": _owned(conversation_id, user)["title"]})

        # Пересборку делаем после ответа: пользователь его уже видит, лишняя
        # задержка на сворачивание истории до него не доходит.
        # Обслуживание контекста идёт после ответа: пользователь его уже видит,
        # и задержка на конспект или карточку фактов до него не доходит.
        fresh = _owned(conversation_id, user)
        try:
            if history.needs_refresh(fresh):
                info = await asyncio.to_thread(history.refresh, fresh, model=model)
                if info:
                    yield sse({"type": "compressed", **info})
            elif (fresh["strategy"] or db.FULL) == db.FACTS:
                exchange = [{"role": "user", "content": content},
                            {"role": "assistant", "content": answer}]
                info = await asyncio.to_thread(
                    history.refresh_facts, fresh, exchange, model=model
                )
                if info:
                    yield sse({"type": "facts", **info})
                    # Факты диалога — то же знание, что нужно проекту. Переливаем
                    # без обращения к модели: они уже извлечены строкой выше.
                    if info.get("ok") and fresh["project_id"]:
                        moved = memory.absorb_facts(
                            db.get_project(fresh["project_id"], user["id"]), info["facts"]
                        )
                        if moved:
                            yield sse({"type": "project_facts", **moved})
        except llm.LLMError as err:
            yield sse({"type": "context_error", "message": str(err)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- сравнение стратегий ----------

class StrategyTest(BaseModel):
    question: str
    strategies: list[str] = Field(default_factory=lambda: list(db.STRATEGIES))


@app.post("/api/conversations/{conversation_id}/strategy-test")
async def strategy_test(
    conversation_id: int, payload: StrategyTest, user: dict = Depends(require_approved)
) -> dict:
    """Задаёт один вопрос по каждой стратегии сборки контекста.

    Диалог не меняется: ни вопрос, ни ответы в него не пишутся. Конспект и
    карточка фактов при необходимости строятся и сохраняются — они полезны
    сами по себе и не зависят от того, какая стратегия выбрана в диалоге.
    """
    conversation = _owned(conversation_id, user)
    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "Вопрос не может быть пустым")

    unknown = [s for s in payload.strategies if s not in db.STRATEGIES]
    if unknown:
        raise HTTPException(400, f"Неизвестные стратегии: {', '.join(unknown)}")

    model = conversation["model"] or llm.MODEL
    system_prompt = llm.SYSTEM_PROMPT
    rows = db.list_messages(conversation_id)
    if len(rows) < 4:
        raise HTTPException(400, f"В диалоге {len(rows)} сообщений — сравнивать нечего.")

    upkeep = 0
    if db.SUMMARY in payload.strategies:
        forced = dict(conversation, strategy=db.SUMMARY)
        if history.split(rows, conversation["summary_upto"])[1]:
            info = await asyncio.to_thread(history.refresh, forced, model=model)
            upkeep += info["cost_tokens"] if info else 0
    if db.FACTS in payload.strategies and not conversation["facts"]:
        exchange = [{"role": r["role"], "content": r["content"]} for r in rows]
        info = await asyncio.to_thread(
            history.refresh_facts, dict(conversation, strategy=db.FACTS), exchange, model=model
        )
        upkeep += info["cost_tokens"] if info else 0

    conversation = _owned(conversation_id, user)
    ask = {"role": "user", "content": question}
    results = []

    for name in payload.strategies:
        plan = history.plan_request(dict(conversation, strategy=name), system_prompt)
        try:
            result = await asyncio.to_thread(
                llm.complete, plan.messages + [ask], model=model, thinking=False
            )
        except llm.LLMError as err:
            raise HTTPException(502, f"{name}: {err}") from err
        results.append({
            "strategy": name,
            "answer": result.content.strip(),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "cost": tokens_mod.cost(result.prompt_tokens, result.completion_tokens, model),
            "elapsed": round(result.elapsed, 2),
            "verbatim": plan.verbatim,
            "dropped": plan.dropped,
            "folded": plan.folded,
            "facts": len(plan.facts),
            "request": result.request,
        })

    return {
        "model": model,
        "question": question,
        "messages_total": len(rows),
        "context_n": conversation["context_n"],
        "summary": conversation["summary"] or "",
        "facts": history.load_facts(conversation),
        "upkeep_tokens": upkeep,
        "results": results,
    }


# ---------- разбор расхода токенов ----------

@app.get("/api/conversations/{conversation_id}/tokens")
async def conversation_tokens(
    conversation_id: int, draft: str = "", user: dict = Depends(require_approved)
) -> dict:
    """Расход по обменам, накопительные суммы и оценка следующего запроса."""
    conversation = _owned(conversation_id, user)
    model = conversation["model"] or llm.MODEL
    turns = tokens_mod.dialog_growth(conversation_id, model)
    # Слои памяти уходят в тот же запрос, значит и в оценку: без них она
    # занижала бы размер ровно на ту часть, которую пользователь не видит
    # в переписке.
    described = memory.describe(memory.collect(conversation, user))
    memory_tokens = sum(layer["tokens"] for layer in described.values())

    return {
        "model": model,
        "priced": model in benchmark.DEFAULT_PRICES,
        "turns": [
            {
                "index": t.index,
                "question": t.question[:120],
                "prompt_tokens": t.prompt_tokens,
                "completion_tokens": t.completion_tokens,
                "total_tokens": t.total_tokens,
                "turn_cost": t.turn_cost,
                "cumulative_tokens": t.cumulative_tokens,
                "cumulative_cost": t.cumulative_cost,
            }
            for t in turns
        ],
        "next_request": tokens_mod.next_request_estimate(
            conversation_id, draft, memory_tokens=memory_tokens
        ),
        "memory": described,
        "ratios": tokens_mod.RATIOS,
    }


# ---------- сравнение способов рассуждения ----------

class ReasoningRequest(BaseModel):
    task: str
    reference: str = ""
    model: str | None = None


@app.post("/api/reasoning")
async def api_reasoning(
    request: ReasoningRequest, _: dict = Depends(require_approved)
) -> StreamingResponse:
    """Решает задачу четырьмя способами, отдавая результат каждого по мере готовности."""
    task = request.task.strip()
    if not task:
        raise HTTPException(400, "Задача не может быть пустой")

    async def events() -> AsyncIterator[str]:
        yield sse({"type": "start", "total": len(reasoning.METHODS)})
        for method in reasoning.METHODS:
            try:
                # Способы синхронные и идут по несколько секунд: в отдельном
                # потоке, иначе они заблокируют событийный цикл целиком.
                res = await asyncio.to_thread(method, task, request.model)
            except llm.LLMError as err:
                yield sse({"type": "error", "message": str(err), "response": err.response, "diagnostics": err.diagnostics})
                return
            yield sse({
                "type": "result",
                "key": res.key,
                "title": res.title,
                "note": res.note,
                "answer": res.answer,
                "extracted": reasoning.extract_answer(res.answer),
                "correct": reasoning.is_correct(res.answer, request.reference),
                "stages": [{"title": s.title, "text": s.text} for s in res.stages],
                "requests": res.requests,
                "responses": res.responses,
                "calls": res.calls,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "total_tokens": res.total_tokens,
                "elapsed": res.elapsed,
            })
        yield sse({"type": "done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- сравнение температур ----------

class TemperatureRequest(BaseModel):
    prompt: str
    reference: str = ""
    samples: int = Field(default=3, ge=1, le=8)
    temperatures: list[float] = Field(default=[0.0, 0.7, 1.2], max_length=6)
    model: str | None = None


@app.post("/api/temperature")
async def api_temperature(
    request: TemperatureRequest, _: dict = Depends(require_approved)
) -> StreamingResponse:
    """Гоняет один запрос при разных температурах, отдавая результат по мере готовности."""
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Запрос не может быть пустым")
    if any(not 0.0 <= t <= 2.0 for t in request.temperatures):
        raise HTTPException(400, "Температура должна быть от 0 до 2")

    async def events() -> AsyncIterator[str]:
        yield sse({"type": "start", "total": len(request.temperatures)})
        for value in request.temperatures:
            try:
                res = await asyncio.to_thread(
                    temperature_mod.run_temperature, prompt, value,
                    samples=request.samples, reference=request.reference,
                    model=request.model,
                )
            except llm.LLMError as err:
                yield sse({"type": "error", "message": str(err), "response": err.response, "diagnostics": err.diagnostics})
                return
            yield sse({
                "type": "result",
                "temperature": res.temperature,
                "samples": [
                    {"text": s.text, "correct": s.correct, "tokens": s.tokens}
                    for s in res.samples
                ],
                "unique": res.unique,
                "diversity": res.diversity,
                "lexical_richness": res.lexical_richness,
                "avg_words": res.avg_words,
                "accuracy": res.accuracy,
                "total_tokens": res.total_tokens,
                "elapsed": res.elapsed,
                "request": res.request,
                "responses": res.responses,
            })
        yield sse({"type": "done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- сравнение моделей ----------

class BenchConfig(BaseModel):
    model: str
    thinking: bool = False
    price_in: float = Field(default=0.0, ge=0)
    price_out: float = Field(default=0.0, ge=0)


class BenchRequest(BaseModel):
    prompt: str
    reference: str = ""
    configs: list[BenchConfig] = Field(min_length=1, max_length=6)


@app.post("/api/benchmark")
async def api_benchmark(
    request: BenchRequest, _: dict = Depends(require_approved)
) -> StreamingResponse:
    """Гоняет один запрос по нескольким моделям, отдавая замеры по мере готовности."""
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Запрос не может быть пустым")

    async def events() -> AsyncIterator[str]:
        yield sse({"type": "start", "total": len(request.configs)})
        for item in request.configs:
            config = benchmark.ModelConfig(
                model=item.model, thinking=item.thinking,
                price_in=item.price_in, price_out=item.price_out,
            )
            try:
                res = await asyncio.to_thread(
                    benchmark.run_config, prompt, config, reference=request.reference
                )
            except llm.LLMError as err:
                yield sse({"type": "error", "message": f"{config.title}: {err}", "response": err.response, "diagnostics": err.diagnostics})
                return
            yield sse({
                "type": "result",
                "title": config.title,
                "model": config.model,
                "thinking": config.thinking,
                "text": res.text,
                "reasoning": res.reasoning,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "reasoning_tokens": res.reasoning_tokens,
                "total_tokens": res.total_tokens,
                "elapsed": res.elapsed,
                "tokens_per_second": res.tokens_per_second,
                "correct": res.correct,
                "cost": res.cost,
                "request": res.request,
                "response": res.response,
            })
        yield sse({"type": "done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- OAuth ----------

@app.get("/auth/yandex")
async def auth_yandex(request: Request):
    """Отправляет пользователя на страницу входа Яндекса."""
    if not auth.is_configured():
        raise HTTPException(
            503,
            "Яндекс OAuth не настроен: заполните YANDEX_CLIENT_ID и "
            "YANDEX_CLIENT_SECRET в .env",
        )
    state = auth.new_state()
    request.session["oauth_state"] = state
    return RedirectResponse(auth.authorize_url(state), status_code=302)


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Принимает возврат от Яндекса и заводит сессию."""
    if error:
        raise HTTPException(400, f"Яндекс вернул ошибку: {error}")

    expected = request.session.pop("oauth_state", None)
    if not expected or not secrets.compare_digest(state, expected):
        raise HTTPException(400, "Неверный state — попробуйте войти заново")
    if not code:
        raise HTTPException(400, "Яндекс не передал код авторизации")

    try:
        token = await auth.exchange_code(code)
        profile = await auth.fetch_profile(token)
    except auth.AuthError as err:
        raise HTTPException(502, str(err)) from err

    user = db.upsert_from_yandex(profile, admin_logins=auth.admin_logins())
    request.session["user_id"] = user["id"]
    return RedirectResponse("/" if user["status"] == db.APPROVED else "/pending", status_code=302)


@app.get("/auth/logout")
async def auth_logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# ---------- API текущего пользователя ----------

@app.get("/api/me")
async def me(request: Request) -> dict:
    """Кто вошёл; используется страницами для отрисовки шапки."""
    user = current_user(request)
    if user is None:
        return {"authenticated": False, "oauth_configured": auth.is_configured()}
    return {
        "authenticated": True,
        "oauth_configured": auth.is_configured(),
        "login": user["login"],
        "name": user["name"],
        "email": user["email"],
        "status": user["status"],
        "is_admin": bool(user["is_admin"]),
    }


# ---------- админка ----------

class StatusUpdate(BaseModel):
    status: str


def _public_user(user: dict) -> dict:
    """Убирает из записи хеш пароля — наружу он не должен попадать никогда."""
    safe = {k: v for k, v in user.items() if k != "password_hash"}
    safe["has_password"] = bool(user.get("password_hash"))
    return safe


@app.get("/api/admin/users")
async def admin_users(_: dict = Depends(require_admin)) -> dict:
    return {"users": [_public_user(u) for u in db.list_all()]}


@app.post("/api/admin/users/{user_id}/status")
async def admin_set_status(
    user_id: int, update: StatusUpdate, admin: dict = Depends(require_admin)
) -> dict:
    """Подтверждает, блокирует или возвращает пользователя в ожидание."""
    if update.status not in db.STATUSES:
        raise HTTPException(400, f"Допустимые статусы: {', '.join(db.STATUSES)}")
    if user_id == admin["id"] and update.status != db.APPROVED:
        raise HTTPException(400, "Нельзя снять доступ у самого себя")
    if not db.set_status(user_id, update.status):
        raise HTTPException(404, "Пользователь не найден")
    return {"ok": True, "user": _public_user(db.get(user_id))}
