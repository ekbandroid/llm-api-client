"""Веб-интерфейс к LLM: потоковый ответ и настройка ограничений из браузера."""

import asyncio
import json
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

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
import invariants as invariants_mod
import llm
import mcp_tools
import memory
import reasoning
import scheduler
import task
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
    scheduler.init()
    runner = asyncio.create_task(_job_runner())
    try:
        yield
    finally:
        runner.cancel()


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

class ProfileCreate(BaseModel):
    title: str = ""
    about: str = ""
    style: str = ""
    constraints: str = ""
    response_format: str = db.TEXT_FORMAT


class ProfilePatch(BaseModel):
    title: str | None = None
    about: str | None = None
    style: str | None = None
    constraints: str | None = None
    response_format: str | None = None


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


def _owned_profile(profile_id: int, user: dict) -> dict:
    """Профиль пользователя или 404."""
    profile = db.get_profile(profile_id, user["id"])
    if profile is None:
        raise HTTPException(404, "Профиль не найден")
    return profile


def _profile_view(profile: dict) -> dict:
    """Профиль плюс его цена: блок уходит в каждый запрос диалога."""
    layers = memory.Layers(
        profile_id=profile["id"], profile_title=profile["title"],
        profile_about=profile["about"] or "", profile_style=profile["style"] or "",
        profile_constraints=profile["constraints"] or "",
        profile_format=profile["response_format"] or db.TEXT_FORMAT,
    )
    return dict(profile, tokens=tokens_mod.estimate_tokens(memory.profile_text(layers)))


@app.get("/api/profiles")
async def profiles_list(user: dict = Depends(require_approved)) -> dict:
    """Профили пользователя — долговременная память во множественном числе."""
    return {
        "profiles": [_profile_view(p) for p in db.list_profiles(user["id"])],
        "formats": [
            {"id": db.TEXT_FORMAT, "label": "Обычный текст"},
            {"id": db.JSON_FORMAT, "label": "JSON-объект"},
        ],
        "limit": db.PROFILE_LIMIT,
    }


@app.post("/api/profiles")
async def profile_create(
    payload: ProfileCreate, user: dict = Depends(require_approved)
) -> dict:
    try:
        profile = db.create_profile(
            user["id"], title=payload.title, about=payload.about, style=payload.style,
            constraints=payload.constraints, response_format=payload.response_format,
        )
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    return _profile_view(profile)


@app.get("/api/profiles/{profile_id}")
async def profile_get(profile_id: int, user: dict = Depends(require_approved)) -> dict:
    return _profile_view(_owned_profile(profile_id, user))


@app.patch("/api/profiles/{profile_id}")
async def profile_patch(
    profile_id: int, payload: ProfilePatch, user: dict = Depends(require_approved)
) -> dict:
    _owned_profile(profile_id, user)
    try:
        updated = db.update_profile(
            profile_id, user["id"], title=payload.title, about=payload.about,
            style=payload.style, constraints=payload.constraints,
            response_format=payload.response_format,
        )
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    if updated is None:
        raise HTTPException(404, "Профиль не найден")
    return _profile_view(updated)


@app.delete("/api/profiles/{profile_id}")
async def profile_delete(profile_id: int, user: dict = Depends(require_approved)) -> dict:
    """Удаляет профиль. Диалоги остаются, просто теряют долговременный слой."""
    _owned_profile(profile_id, user)
    orphaned = db.delete_profile(profile_id, user["id"])
    if orphaned is None:
        raise HTTPException(404, "Профиль не найден")
    return {"ok": True, "conversations_without_profile": orphaned}


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


class InvariantCreate(BaseModel):
    category: str = db.ARCHITECTURE
    rule: str
    rationale: str = ""


class InvariantPatch(BaseModel):
    category: str | None = None
    rule: str | None = None
    rationale: str | None = None
    active: bool | None = None


def _invariants_view(project: dict, user: dict) -> dict:
    """Инварианты проекта и цена их блока в каждом запросе."""
    items = db.list_invariants(project["id"], user["id"])
    active = [i for i in items if i["active"]]
    return {
        "project": {"id": project["id"], "title": project["title"]},
        "invariants": [dict(i, code=invariants_mod.code(i)) for i in items],
        "categories": [{"id": c, "label": invariants_mod.LABELS[c]} for c in db.INVARIANT_CATEGORIES],
        "tokens": (invariants_mod.describe(project, active) or {}).get("tokens", 0),
        "limit": db.INVARIANT_LIMIT,
    }


@app.get("/api/projects/{project_id}/invariants")
async def invariants_list(project_id: int, user: dict = Depends(require_approved)) -> dict:
    return _invariants_view(_owned_project(project_id, user), user)


@app.post("/api/projects/{project_id}/invariants")
async def invariant_create(
    project_id: int, payload: InvariantCreate, user: dict = Depends(require_approved)
) -> dict:
    project = _owned_project(project_id, user)
    try:
        db.create_invariant(project_id, user["id"], category=payload.category,
                            rule=payload.rule, rationale=payload.rationale)
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    return _invariants_view(project, user)


@app.patch("/api/invariants/{invariant_id}")
async def invariant_patch(
    invariant_id: int, payload: InvariantPatch, user: dict = Depends(require_approved)
) -> dict:
    current = db.get_invariant(invariant_id, user["id"])
    if current is None:
        raise HTTPException(404, "Инвариант не найден")
    try:
        db.update_invariant(invariant_id, user["id"], category=payload.category,
                            rule=payload.rule, rationale=payload.rationale, active=payload.active)
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    return _invariants_view(_owned_project(current["project_id"], user), user)


@app.delete("/api/invariants/{invariant_id}")
async def invariant_delete(invariant_id: int, user: dict = Depends(require_approved)) -> dict:
    """Убирает правило. Номер остаётся занятым: ссылки в истории не поплывут."""
    current = db.get_invariant(invariant_id, user["id"])
    if current is None:
        raise HTTPException(404, "Инвариант не найден")
    db.delete_invariant(invariant_id, user["id"])
    return _invariants_view(_owned_project(current["project_id"], user), user)


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


# ---------- серверы MCP ----------
#
# Сервер объявляет инструменты, включённые уходят схемами в запрос к модели.
# Схемы кэшируются в базе: ходить за ними на чужой сервер перед каждым
# сообщением значило бы добавлять его задержку к каждому ответу.

class MCPCreate(BaseModel):
    title: str = ""
    url: str


class MCPToggle(BaseModel):
    enabled: bool


# Адреса, по которым приложению ходить нечего: запрос уходит с сервера, и на
# проде «localhost» означал бы стук в собственную сеть, а не в машину того,
# кто вписал адрес.
PRIVATE_HOSTS = ("localhost", "127.", "0.", "10.", "192.168.", "169.254.",
                 "::1", "[::1]", "metadata.google.internal")


def _check_mcp_url(url: str) -> str:
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(400, "Адрес должен начинаться с http:// или https://")
    # Свой сервер погоды — исключение из запрета на приватные адреса: он и
    # должен слушать только петлю. Без этого удалённую заготовку нельзя было
    # бы вернуть руками.
    if url in db.OWN_MCP_URLS:
        return url
    host = parsed.hostname.lower()
    private = (
        host.startswith(PRIVATE_HOSTS)
        or host.endswith((".local", ".internal"))
        or any(host.startswith(f"172.{n}.") for n in range(16, 32))
    )
    if private:
        raise HTTPException(
            400, "Локальные и внутренние адреса недоступны: запрос уходит "
                 "с сервера приложения, а не из вашего браузера")
    return url


def _mcp_view(server: dict) -> dict:
    tools = mcp_tools.cached_tools(server)
    return {
        "id": server["id"],
        "title": server["title"],
        "url": server["url"],
        "enabled": bool(server["enabled"]),
        "slug": mcp_tools.server_slug(server),
        "tools": [{"name": t["name"], "description": mcp_tools.describe_tool(t, server),
                   "required": list((t.get("schema") or {}).get("required") or [])}
                  for t in tools],
        "count": len(tools),
        # Во сколько обойдутся схемы этого сервера в каждом запросе — цифра
        # для вкладки «Токены». Считаем по тому же виду, в каком они уходят
        # в запрос, иначе оценка разошлась бы со строкой в чате.
        "tokens": tokens_mod.estimate_tokens(json.dumps(
            mcp_tools.request_tools([server])[0], ensure_ascii=False)),
        "checked_at": server["checked_at"],
        "error": server["error"],
    }


async def _refresh_mcp(server: dict, user: dict) -> dict:
    """Сходить на сервер и обновить кэш схем. Ошибку не прячем, а показываем."""
    try:
        found = await mcp_tools.list_tools(server["url"])
    except mcp_tools.MCPError as err:
        db.set_mcp_tools(server["id"], user["id"], tools_json=None, error=str(err))
        return _mcp_view(db.get_mcp_server(server["id"], user["id"]))
    db.set_mcp_tools(server["id"], user["id"],
                     tools_json=mcp_tools.tools_json(found), error=None)
    view = _mcp_view(db.get_mcp_server(server["id"], user["id"]))
    view["server_name"] = found.name
    view["protocol"] = found.protocol
    return view


def _owned_mcp(server_id: int, user: dict) -> dict:
    server = db.get_mcp_server(server_id, user["id"])
    if server is None:
        raise HTTPException(404, "Сервер не найден")
    return server


@app.get("/api/mcp")
async def mcp_list(user: dict = Depends(require_approved)) -> dict:
    return {"servers": [_mcp_view(s) for s in db.list_mcp_servers(user["id"])]}


@app.post("/api/mcp")
async def mcp_create(payload: MCPCreate, user: dict = Depends(require_approved)) -> dict:
    """Добавляет сервер и сразу идёт к нему за списком инструментов."""
    url = _check_mcp_url(payload.url)
    server = db.create_mcp_server(user["id"], title=payload.title, url=url)
    return await _refresh_mcp(server, user)


@app.post("/api/mcp/{server_id}/check")
async def mcp_check(server_id: int, user: dict = Depends(require_approved)) -> dict:
    return await _refresh_mcp(_owned_mcp(server_id, user), user)


@app.post("/api/mcp/{server_id}/enabled")
async def mcp_enable(
    server_id: int, payload: MCPToggle, user: dict = Depends(require_approved)
) -> dict:
    """Тумблер. При включении схемы подтягиваются, если их ещё нет."""
    server = _owned_mcp(server_id, user)
    db.set_mcp_enabled(server_id, user["id"], payload.enabled)
    server = db.get_mcp_server(server_id, user["id"])
    if payload.enabled and not mcp_tools.cached_tools(server):
        return await _refresh_mcp(server, user)
    return _mcp_view(server)


@app.delete("/api/mcp/{server_id}")
async def mcp_delete(server_id: int, user: dict = Depends(require_approved)) -> dict:
    _owned_mcp(server_id, user)
    db.delete_mcp_server(server_id, user["id"])
    return {"ok": True}


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
    # Профиль, наоборот, переключается когда угодно.
    profile_id: int | None = None


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
    # Ноль означает «снять профиль», пропущенное поле — «не трогать».
    profile_id: int | None = Field(default=None, ge=0)


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
        profile_id=_owned_profile(payload.profile_id, user)["id"] if payload.profile_id else None,
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


@app.get("/api/conversations/{conversation_id}/messages")
async def conversation_messages(
    conversation_id: int, after: int = 0, user: dict = Depends(require_approved)
) -> dict:
    """Сообщения новее указанного. Нужен открытой странице: задания пишут в
    переписку в фоне, и без дозагрузки их ответы видны только после
    переоткрытия диалога."""
    _owned(conversation_id, user)
    fresh = db.list_messages_after(conversation_id, after)
    return {"messages": [_public_message(m) for m in fresh]}


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
        profile_id=(_owned_profile(payload.profile_id, user)["id"] if payload.profile_id else None),
        clear_profile=payload.profile_id == 0,
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
    # Задания живут в отдельной базе и внешним ключом к диалогу не связаны:
    # убираем их сами, иначе исполнитель будет ходить в удалённую переписку.
    dropped = scheduler.delete_chat_jobs(conversation_id)
    return {"ok": db.delete_conversation(conversation_id, user["id"]),
            "jobs_deleted": dropped}


ASSISTANT_TURN_NOTE = (
    "Ассистент продолжает работу: пользователь ничего не писал, ход у ассистента. "
    "Продолжай по текущему шагу задачи."
)


def _last_answer(conversation_id: int) -> str:
    """Последний настоящий ответ ассистента. Заготовку паузы пропускаем."""
    for m in reversed(db.list_messages(conversation_id)):
        if m["role"] != "assistant":
            continue
        meta = json.loads(m["meta"]) if m["meta"] else {}
        if not meta.get("paused"):
            return m["content"]
    return ""


# Сколько кругов «модель просит инструмент — приложение выполняет» допускается
# в одном обмене. Предохранитель: без него ошибка инструмента, на которую
# модель отвечает новым вызовом, крутилась бы без конца.
MCP_ROUNDS_LIMIT = 3

# Сколько текста инструмента уходит модели. Ответы бывают на десятки тысяч
# символов, и целиком они вытеснили бы из запроса саму переписку.
MCP_RESULT_LIMIT = 6000


async def _run_tool(call: dict, routes: dict, *, chat_id: int) -> dict:
    """Выполняет один запрошенный моделью вызов и описывает его для чата.

    Ошибка инструмента не прерывает обмен: её текст возвращается модели как
    результат. Так она объяснит пользователю, что случилось, — вместо того
    чтобы выдумать число, которого не получила.
    """
    name = (call.get("function") or {}).get("name") or ""
    raw = (call.get("function") or {}).get("arguments") or "{}"
    try:
        arguments = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        arguments = {}
    note = {"kind": "mcp", "name": name, "arguments": arguments, "ok": False}

    route = routes.get(name)
    if route is None:
        note["result"] = f"ОШИБКА: инструмент {name} не подключён"
        note["server"] = ""
        return note

    server, tool = route
    # Идентификатор диалога подставляет приложение: модель его не знает, и в
    # схеме этого параметра нет вовсе. Иначе планировщику некуда было бы
    # класть результат поручения.
    for field in mcp_tools.app_args(server, tool):
        arguments[field] = chat_id
    note.update(server=server["title"], tool=tool, url=server["url"])
    note["request"] = {"method": "tools/call", "url": server["url"],
                       "body": {"name": tool, "arguments": arguments}}
    started = time.monotonic()
    try:
        result = await mcp_tools.call_tool(server["url"], tool, arguments)
    except mcp_tools.MCPError as err:
        note["result"] = f"ОШИБКА: {err}"
    else:
        note["ok"] = not result.startswith("ОШИБКА")
        note["result"] = result[:MCP_RESULT_LIMIT]
    note["elapsed"] = round(time.monotonic() - started, 2)
    note["response"] = {"chars": len(note["result"]), "text": note["result"][:1000]}
    return note


async def _exchange(
    conversation_id: int, user: dict, content: str | None, *,
    outcome: dict, simulated: dict | None = None,
) -> AsyncIterator[str]:
    """Один обмен: сохранить реплику, собрать запрос, стримить ответ, обслужить.

    Через него идут и настоящие реплики, и реплики автопилота — одинаково:
    иначе автопилот проверял бы не то приложение, которое видит человек.
    simulated — телеметрия вызова, написавшего реплику за пользователя; она
    сохраняется в meta этой реплики.

    content=None — ход ассистента без реплики пользователя: в переписку идёт
    отметка о том, что ход перешёл к нему, и она же служит инструкцией.

    Итог пишется в outcome: ok, answer, tokens (основной ответ плюс служебные
    вызовы), paused. Вернуть значение из асинхронного генератора нельзя.
    """
    outcome["ok"] = False
    conversation = _owned(conversation_id, user)
    system_prompt = llm.SYSTEM_PROMPT
    model = conversation["model"] or llm.MODEL
    thinking = bool(conversation["thinking"])
    max_tokens = conversation["max_tokens"] or None
    # Телеметрия служебных вызовов копится с самого начала: переключатель
    # этапов может отработать ещё до ответа.
    service: list[dict] = []

    if content is None:
        asked = db.add_message(
            conversation_id, "system", ASSISTANT_TURN_NOTE,
            meta=json.dumps({"assistant_turn": True}, ensure_ascii=False),
        )
    else:
        asked = db.add_message(
            conversation_id, "user", content,
            meta=json.dumps({"simulated": True, **simulated}, ensure_ascii=False) if simulated else None,
        )
        # Первое сообщение даёт диалогу имя — иначе список будет из «Новых диалогов».
        if conversation["title"] == db.NEW_TITLE:
            db.update_conversation(conversation_id, user["id"], title=content)

    # Переход вызывает та сторона, чья реплика служит признаком. На
    # планировании это пользователь, поэтому переключатель работает ДО ответа:
    # иначе ответ на «план принят» готовился бы ещё на планировании.
    if content is not None and task.switches_before_answer(conversation):
        if previous := _last_answer(conversation_id):
            try:
                moved = await asyncio.to_thread(
                    task.refresh, conversation,
                    [{"role": "assistant", "content": previous},
                     {"role": "user", "content": content}],
                    model=model,
                )
            except llm.LLMError as err:
                moved = None
                yield sse({"type": "context_error", "message": str(err)})
            if moved:
                service.append(moved)
                yield sse({"type": "task_state", **moved})
                conversation = _owned(conversation_id, user)

    # Заново читаем диалог: только что добавленный вопрос должен войти в план.
    layers = memory.collect(conversation, user)
    described = memory.describe(layers)
    # Инварианты — правила проекта, отдельный блок сразу за рабочей памятью.
    project, rules = invariants_mod.load(conversation, user)
    if ruled := invariants_mod.describe(project, rules):
        described["invariants"] = ruled
    # Состояние задачи — отдельный блок и отдельный ключ телеметрии: это не
    # память, а то, где мы сейчас в работе.
    if state := task.describe(conversation):
        described["task"] = state
    # Инструменты подключённых серверов MCP. Схемы берутся из кэша; если
    # сервер включили, а за схемами ещё не ходили, сходим один раз здесь.
    servers = db.list_mcp_servers(user["id"], only_enabled=True)
    for server in servers:
        if not mcp_tools.cached_tools(server):
            await _refresh_mcp(server, user)
    servers = db.list_mcp_servers(user["id"], only_enabled=True)
    tool_schemas, tool_routes = mcp_tools.request_tools(servers)
    # Выполнение задания не заводит новых заданий: инструменты планирования
    # на это время снимаются. Одной просьбы в тексте мало — проверено, первое
    # же напоминание, выполняясь, завело себе копию и продолжило бы вечно.
    if simulated and simulated.get("scheduled"):
        tool_schemas = mcp_tools.without_scheduling(tool_schemas, tool_routes)
    if tool_schemas:
        described["mcp"] = {
            "servers": len(servers),
            "tools": len(tool_schemas),
            "tokens": tokens_mod.estimate_tokens(
                json.dumps(tool_schemas, ensure_ascii=False)),
        }
    plan = history.plan_request(
        _owned(conversation_id, user), system_prompt,
        memory_blocks=(memory.blocks(layers) + invariants_mod.blocks(project, rules)
                       + task.blocks(conversation) + mcp_tools.blocks(servers)),
        memory_info=described,
    )
    messages = plan.messages

    answer, reasoning = "", ""
    meta: dict = {"system": system_prompt, "model": model}
    completion_tokens = total_tokens = 0
    finish_reason = "unknown"
    elapsed = 0.0
    answered = False

    try:
        # Состав слоёв показываем до ответа: иначе понять, ушла ли память
        # в запрос, можно было бы только перезагрузив страницу.
        if plan.memory:
            yield sse({"type": "memory", "memory": plan.memory})

        # Пауза — правило приложения, а не просьба к модели: на паузе к API
        # не идём вовсе. Иначе прямое «продолжай» в последнем сообщении
        # перевешивает инструкцию в системном блоке, и работа продолжается.
        if task.is_paused(conversation):
            reply = task.paused_reply(conversation)
            yield sse({"type": "content", "text": reply})
            yield sse({"type": "done", "finish_reason": "paused", "usage": {}, "elapsed": 0})
            saved = db.add_message(
                conversation_id, "assistant", reply,
                meta=json.dumps({"system": system_prompt, "model": model,
                                 "memory": plan.memory, "paused": True,
                                 "finish_reason": "paused", "elapsed": 0},
                                ensure_ascii=False),
            )
            answered = True
            outcome.update(ok=True, answer=reply, tokens=0, paused=True)
            yield sse({"type": "saved", "message_id": saved["id"],
                       "title": _owned(conversation_id, user)["title"]})
            return

        # Кругов может быть несколько: модель просит инструмент, приложение
        # его выполняет и спрашивает снова. Текст всех кругов — один ответ:
        # пользователь видит его сплошным потоком, как обычно.
        rounds = 0
        while True:
            calls: list[dict] = []
            said = ""
            try:
                async for event in llm.stream(
                    messages, model=model, thinking=thinking, max_tokens=max_tokens,
                    # Формат ответа — настройка профиля: служебные вызовы
                    # (карточка фактов, конспект) его не наследуют.
                    response_format=memory.response_format(layers),
                    tools=tool_schemas or None,
                ):
                    if event["type"] == "content":
                        answer += event["text"]
                        said += event["text"]
                    elif event["type"] == "reasoning":
                        reasoning += event["text"]
                    elif event["type"] == "request":
                        meta["request"] = event["request"]
                    elif event["type"] == "response":
                        meta["response"] = event["response"]
                    elif event["type"] == "done":
                        usage = event.get("usage") or {}
                        completion_tokens += usage.get("completion_tokens", 0)
                        total_tokens += usage.get("total_tokens", 0)
                        finish_reason = event["finish_reason"]
                        elapsed += event["elapsed"]
                        calls = event.get("tool_calls") or []
                        # Ход не закончен: «done» в интерфейсе закрыл бы ответ,
                        # а после инструмента будет продолжение.
                        if calls and rounds < MCP_ROUNDS_LIMIT:
                            continue
                        if rounds:
                            # В последнем «done» показываем расход за все круги:
                            # иначе в чате была бы цена одного, а в сохранённом
                            # сообщении — сумма, и числа разошлись бы.
                            event = {**event, "usage": {
                                **usage, "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens, "rounds": rounds + 1}}
                    yield sse(event)
            except llm.LLMError as err:
                # Вопрос без ответа удалит finally ниже — так же, как при обрыве.
                yield sse({"type": "error", "message": str(err), "response": err.response, "diagnostics": err.diagnostics})
                return

            if not calls:
                break
            if rounds >= MCP_ROUNDS_LIMIT:
                # Предел исчерпан, а модель просит ещё. Молчать нельзя: ответ
                # оборван на полуслове, и пользователь должен знать почему.
                yield sse({"type": "context_error", "message":
                           f"Предел кругов вызова инструментов ({MCP_ROUNDS_LIMIT}) "
                           "исчерпан — ответ оборван. Спросите точнее или "
                           "повторите вопрос."})
                break
            rounds += 1
            messages = messages + [
                {"role": "assistant", "content": said, "tool_calls": calls}]
            for call in calls:
                note = await _run_tool(call, tool_routes, chat_id=conversation_id)
                service.append(note)
                yield sse({"type": "mcp", **note})
                messages.append({
                    "role": "tool", "tool_call_id": call.get("id") or "",
                    "content": note["result"],
                })

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
        answered = True
        yield sse({"type": "saved", "message_id": saved["id"], "title": _owned(conversation_id, user)["title"]})

        # Обслуживание контекста идёт после ответа: пользователь его уже видит,
        # и задержка на конспект или карточку фактов до него не доходит.
        fresh = _owned(conversation_id, user)
        # Служебные вызовы дописываются в meta ответа отдельной правкой —
        # иначе после перезагрузки страницы от них не осталось бы и следа.
        # Отказ по инварианту помечает реплику пользователя: иначе карточка
        # фактов и конспект принимают отклонённую просьбу за решение.
        rejected_by: list[str] = []
        try:
            # Судья идёт первым: вердикт должен стоять сразу под ответом, и
            # пометка об отказе нужна карточке фактов до её обновления.
            # Ревизор проверяет две пары правил сразу: инварианты проекта и
            # правило текущего этапа. Поэтому он нужен и там, где инвариантов
            # нет, но задача идёт по этапам.
            # Ревизору идёт не то правило, что модели: ему нужен закрытый
            # список «работа этапа / прыжок через этап», иначе он помечает
            # нарушением саму работу этапа.
            stage_rule = task.judge_rule(task.stage_of(fresh)) if task.is_task(fresh) else ""
            if rules or stage_rule:
                verdict = await asyncio.to_thread(
                    invariants_mod.check, rules,
                    content if content is not None else "(ход ассистента, реплики пользователя нет)",
                    answer,
                    stage_rule=stage_rule,
                    stage_label=task.LABELS[task.stage_of(fresh)] if stage_rule else "",
                    model=model,
                )
                service.append(verdict)
                rejected_by = verdict.get("defended") or []
                if rejected_by and content is not None:
                    marked = json.loads(asked["meta"]) if asked["meta"] else {}
                    marked["rejected_by"] = rejected_by
                    db.update_message_meta(asked["id"], json.dumps(marked, ensure_ascii=False))
                yield sse({"type": "invariants", **verdict, "message_id": asked["id"]})

            if history.needs_refresh(fresh):
                info = await asyncio.to_thread(history.refresh, fresh, model=model)
                if info:
                    service.append(info)
                    yield sse({"type": "compressed", **info})
            elif (fresh["strategy"] or db.FULL) == db.FACTS:
                exchange = ([{"role": "user", "content": content, "rejected_by": rejected_by}]
                            if content is not None else [])
                exchange += [{"role": "assistant", "content": answer}]
                info = await asyncio.to_thread(
                    history.refresh_facts, fresh, exchange, model=model
                )
                if info:
                    service.append(info)
                    yield sse({"type": "facts", **info})
                    # Факты диалога — то же знание, что нужно проекту. Переливаем
                    # без обращения к модели: они уже извлечены строкой выше.
                    if info.get("ok") and fresh["project_id"]:
                        moved = memory.absorb_facts(
                            db.get_project(fresh["project_id"], user["id"]), info["facts"]
                        )
                        if moved:
                            yield sse({"type": "project_facts", **moved})
            # Переключатель после ответа — только там, где переход вызывает
            # сам ассистент (этап выполнения: он предъявляет результат).
            # Остальные этапы проверены до ответа, второй раз платить незачем.
            after = _owned(conversation_id, user)
            if not task.switches_before_answer(conversation):
                pair = ([{"role": "user", "content": content, "rejected_by": rejected_by}]
                        if content is not None else [])
                moved = await asyncio.to_thread(
                    task.refresh, after, pair + [{"role": "assistant", "content": answer}],
                    model=model,
                )
                if moved:
                    service.append(moved)
                    yield sse({"type": "task_state", **moved})
        except llm.LLMError as err:
            yield sse({"type": "context_error", "message": str(err)})

        # Дописываем после except: даже если один из вызовов сорвался,
        # телеметрия предыдущих должна сохраниться.
        if service:
            meta["service"] = service
            db.update_message_meta(saved["id"], json.dumps(meta, ensure_ascii=False))

        outcome.update(
            ok=True, answer=answer, paused=False,
            tokens=total_tokens + sum(x.get("cost_tokens", 0) for x in service),
        )
    finally:
        # Ответ сохраняется только в самом конце. Если обмен сорвался — ошибка
        # API или закрытая вкладка, когда Starlette снимает поток на ближайшем
        # await, — в истории остался бы вопрос без ответа, который потом
        # оплачивался бы в каждом следующем запросе. Убираем его.
        if not answered:
            db.delete_message(asked["id"])


def _autopilot_context(conversation_id: int) -> tuple[str, str]:
    """Исходная задача и последний настоящий ответ ассистента.

    Заготовленный ответ «задача на паузе» пропускаем: реагировать на него
    модели-«пользователю» не на что, работа по задаче была до него.
    """
    rows = db.list_messages(conversation_id)
    first = next((m["content"] for m in rows if m["role"] == "user"), "")
    last = ""
    for m in reversed(rows):
        if m["role"] != "assistant":
            continue
        meta = json.loads(m["meta"]) if m["meta"] else {}
        if not meta.get("paused"):
            last = m["content"]
            break
    return first, last


async def _autopilot(conversation_id: int, user: dict, request: Request) -> AsyncIterator[str]:
    """Цикл ходов: модель пишет реплику за пользователя, дальше обычный обмен.

    Останавливается на этапе «готово», по лимиту ходов, на паузе (это и есть
    кнопка «Стоп»), если автопилот выключили посреди прогона, на ошибке и
    когда вкладку закрыли. Каждое условие проверяется перед ходом, поэтому
    начатый ход всегда доходит до конца.
    """
    conversation = _owned(conversation_id, user)
    if not task.is_autopilot(conversation):
        return

    start_stage = task.stage_of(conversation)
    limit = conversation["task_max_turns"] or 8
    model = conversation["model"] or llm.MODEL
    turns = spent = 0
    reason = "limit"

    while True:
        conversation = _owned(conversation_id, user)
        if task.stage_of(conversation) == db.DONE:
            reason = "done"
            break
        if conversation["task_paused"]:
            reason = "paused"
            break
        if not task.is_autopilot(conversation):
            reason = "off"
            break
        if turns >= limit:
            reason = "limit"
            break
        if await request.is_disconnected():
            reason = "stopped"
            break

        task_text, last_answer = _autopilot_context(conversation_id)
        if not last_answer:
            reason = "nothing"
            break
        try:
            reply = await asyncio.to_thread(
                task.simulate_user, conversation, task_text, last_answer, model=model
            )
        except llm.LLMError as err:
            yield sse({"type": "error", "message": str(err), "response": err.response,
                       "diagnostics": err.diagnostics})
            reason = "error"
            break
        if not reply["text"]:
            reason = "error"
            break

        turns += 1
        spent += reply["cost_tokens"]
        yield sse({"type": "autopilot_turn", "turn": turns, "max": limit, **reply})

        outcome: dict = {}
        async for chunk in _exchange(
            conversation_id, user, reply["text"], outcome=outcome, simulated=reply
        ):
            yield chunk
        spent += outcome.get("tokens", 0)
        if not outcome.get("ok"):
            reason = "error"
            break

    final = _owned(conversation_id, user)
    yield sse({
        "type": "autopilot_done", "reason": reason, "turns": turns, "max": limit,
        "from_stage": start_stage, "stage": task.stage_of(final), "tokens": spent,
    })


# Сколько ходов подряд ассистент делает сам, пока ход не вернётся к
# пользователю. Предохранитель: без него состояние «ход ассистента» могло бы
# крутить ходы до бесконечности.
ASSISTANT_TURNS_LIMIT = 3


async def _assistant_turns(
    conversation_id: int, user: dict, request: Request
) -> AsyncIterator[str]:
    """Ходы, которые ассистент делает сам, когда ход перешёл к нему.

    Это не автопилот: реплики за пользователя никто не выдумывает. Приложение
    лишь продолжает работу, которую состояние задачи уже числит за ассистентом,
    — иначе после перехода в выполнение обещанная реализация не появлялась бы,
    пока пользователь не напишет что-нибудь сам.
    """
    made = 0
    while made < ASSISTANT_TURNS_LIMIT:
        conversation = _owned(conversation_id, user)
        if not (task.is_task(conversation) and conversation["task_auto"]):
            break
        if conversation["task_autopilot"] or conversation["task_paused"]:
            break
        if task.stage_of(conversation) == db.DONE:
            break
        # Сам ассистент продолжает только там, где работа этапа за ним. На
        # проверке ход по смыслу пользователя: без этого условия ассистент
        # делал три хода подряд со словами «жду вашего ответа».
        if task.STAGE_TRIGGER[task.stage_of(conversation)] != db.ASSISTANT_ACTOR:
            break
        if (conversation["task_actor"] or db.USER_ACTOR) != db.ASSISTANT_ACTOR:
            break
        if await request.is_disconnected():
            break

        made += 1
        yield sse({"type": "next_turn", "kind": "assistant",
                   "turn": made, "max": ASSISTANT_TURNS_LIMIT})
        outcome: dict = {}
        async for chunk in _exchange(conversation_id, user, None, outcome=outcome):
            yield chunk
        if not outcome.get("ok"):
            break

    if made:
        yield sse({"type": "assistant_turns_done", "turns": made})


# ---------- исполнитель заданий ----------
#
# Планировщик (scheduler_mcp.py) только ведёт очередь. Выполняет поручения
# приложение — и тем же конвейером, что обычную реплику: заданию доступны
# память, инварианты, состояние задачи и все инструменты MCP, а ответ ложится
# сообщением в тот диалог, где поручение дали.

# Как часто заглядываем в очередь. Минута точности достаточна: расписание
# задаётся в минутах, а более частый опрос — это лишние чтения базы впустую.
JOB_TICK_SECONDS = int(os.getenv("JOB_TICK_SECONDS", "30"))

# Сколько результатов заданий-источников кладём в сводку.
SUMMARY_LIMIT = 40

# Замки на диалог: человек и исполнитель не должны писать в одну переписку
# одновременно — сообщения перемешались бы, а история стала бы нечитаемой.
_chat_locks: dict[int, asyncio.Lock] = {}


def _chat_lock(conversation_id: int) -> asyncio.Lock:
    lock = _chat_locks.get(conversation_id)
    if lock is None:
        lock = _chat_locks[conversation_id] = asyncio.Lock()
    return lock


# Шапка запланированного хода. Без неё поручение «напомни про чайник» читается
# как просьба завести напоминание, а не выполнить его.
JOB_HEADER = (
    "Сработало запланированное поручение «{title}». Выполни его прямо сейчас и "
    "ответь в этот диалог — это и есть выполнение, заводить новое задание не "
    "нужно.\n\nПоручение: {prompt}"
)


def _job_prompt(job: dict) -> str:
    """Поручение в том виде, в каком его получит агент.

    Для сводки к тексту поручения подкладываются результаты заданий-источников
    за прошедший период: модель не помнит, что писала в прошлые разы, и без
    этого блока обобщать ей было бы нечего.
    """
    if job["kind"] != scheduler.SUMMARY:
        return JOB_HEADER.format(title=job["title"], prompt=job["prompt"])

    sources = json.loads(job["sources"] or "null") or [
        other["id"] for other in scheduler.list_jobs(job["chat_id"])
        if other["id"] != job["id"]
    ]
    hours = max(1, round((job["every_minutes"] or 60) / 60))
    runs = scheduler.recent_runs(sources, hours=hours, limit=SUMMARY_LIMIT)
    head = JOB_HEADER.format(title=job["title"], prompt=job["prompt"])
    if not runs:
        return (f"{head}\n\n(Результатов за последние {hours} ч не "
                "накопилось — так и скажи.)")
    lines = "\n".join(
        f"- {r['ran_at']} · {r['title']}: {(r['answer'] or '').strip()[:300]}"
        for r in runs
    )
    return (f"{head}\n\nРезультаты заданий за последние {hours} ч "
            f"({len(runs)} шт.):\n{lines}")


async def _execute_job(job: dict) -> None:
    """Прогоняет поручение через обычный конвейер обмена."""
    user = db.conversation_owner(job["chat_id"])
    if user is None:
        # Диалог удалили вместе с заданиями, но это могло произойти в другом
        # процессе — просто убираем осиротевшее.
        scheduler.delete_chat_jobs(job["chat_id"])
        return

    outcome: dict = {}
    # События потока некому показывать, но сообщение об ошибке из них достать
    # надо: без него в истории запусков осталось бы «ответ не получен» без
    # единого слова о причине.
    failure = ""
    try:
        async for chunk in _exchange(
            job["chat_id"], user, _job_prompt(job), outcome=outcome,
            simulated={"scheduled": True, "job_id": job["id"],
                       "title": job["title"], "kind": job["kind"],
                       # Сам текст поручения: в ленте показываем его, а не
                       # служебную шапку, которая ушла модели.
                       "prompt": job["prompt"]},
        ):
            if '"type": "error"' in chunk or '"type":"error"' in chunk:
                failure = chunk[len("data: "):].strip()[:400]
    except Exception as err:  # noqa: BLE001 — падение задания не должно ронять петлю
        scheduler.record(job["id"], ok=False, error=f"{type(err).__name__}: {err}")
        print(f"планировщик: задание #{job['id']} упало — {type(err).__name__}: {err}")
        return

    ok = bool(outcome.get("ok"))
    if not ok:
        print(f"планировщик: задание #{job['id']} без ответа — {failure or 'причина неизвестна'}")
    scheduler.record(
        job["id"], ok=ok,
        answer=outcome.get("answer", ""), tokens=outcome.get("tokens", 0),
        error="" if ok else (failure or "ответ не получен"),
    )


async def _job_runner() -> None:
    """Фоновая петля: раз в JOB_TICK_SECONDS забирает созревшие задания."""
    while True:
        try:
            for job in scheduler.take_due():
                lock = _chat_lock(job["chat_id"])
                if lock.locked():
                    # В диалоге сейчас пишет человек — вернём задание в очередь
                    # и попробуем на следующем тике.
                    scheduler.postpone(job["id"])
                    continue
                async with lock:
                    await _execute_job(job)
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 — петля переживает любой сбой
            print(f"планировщик: тик не удался — {type(err).__name__}: {err}")
        await asyncio.sleep(JOB_TICK_SECONDS)


def _sse_response(events: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/conversations/{conversation_id}/messages")
async def conversation_send(
    conversation_id: int, payload: NewMessage, request: Request,
    user: dict = Depends(require_approved),
) -> StreamingResponse:
    """Принимает новое сообщение, историю поднимает сам и стримит ответ.

    Клиент присылает только текст: историю он не передаёт и подменить её
    не может. Оба сообщения сохраняются, поэтому после перезапуска диалог
    продолжается с того же места. Если включён автопилот, после ответа в том
    же потоке идут ходы за пользователя.
    """
    _owned(conversation_id, user)
    content = payload.content.strip()
    if not content:
        raise HTTPException(400, "Пустое сообщение")

    async def events() -> AsyncIterator[str]:
        # Замок держится весь обмен: пока человек разговаривает, исполнитель
        # заданий в этот диалог не пишет.
        async with _chat_lock(conversation_id):
            outcome: dict = {}
            async for chunk in _exchange(conversation_id, user, content, outcome=outcome):
                yield chunk
            # На паузе ни автопилот, ни самостоятельные ходы не стартуют: пауза —
            # тормоз человека над автоматом.
            if outcome.get("ok") and not outcome.get("paused"):
                if task.is_autopilot(_owned(conversation_id, user)):
                    async for chunk in _autopilot(conversation_id, user, request):
                        yield chunk
                else:
                    async for chunk in _assistant_turns(conversation_id, user, request):
                        yield chunk

    return _sse_response(events())


@app.get("/api/conversations/{conversation_id}/jobs")
async def jobs_list(
    conversation_id: int, user: dict = Depends(require_approved)
) -> dict:
    """Задания этого диалога для вкладки «Расписание»."""
    _owned(conversation_id, user)
    return {"jobs": [scheduler.describe(j)
                     for j in scheduler.list_jobs(conversation_id)]}


class JobStatus(BaseModel):
    status: str


@app.post("/api/conversations/{conversation_id}/jobs/{job_id}/status")
async def job_status(
    conversation_id: int, job_id: int, payload: JobStatus,
    user: dict = Depends(require_approved),
) -> dict:
    """Пауза, возобновление или отмена задания."""
    _owned(conversation_id, user)
    try:
        job = scheduler.set_status(job_id, payload.status, conversation_id)
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    if job is None:
        raise HTTPException(404, "Задание не найдено")
    return scheduler.describe(job)


@app.delete("/api/conversations/{conversation_id}/jobs/{job_id}")
async def job_delete(
    conversation_id: int, job_id: int, user: dict = Depends(require_approved)
) -> dict:
    _owned(conversation_id, user)
    if not scheduler.delete(job_id, conversation_id):
        raise HTTPException(404, "Задание не найдено")
    return {"ok": True}


@app.post("/api/conversations/{conversation_id}/autopilot")
async def autopilot_run(
    conversation_id: int, request: Request, user: dict = Depends(require_approved)
) -> StreamingResponse:
    """Запускает автопилот без новой реплики — так возобновляется работа после паузы."""
    conversation = _owned(conversation_id, user)
    if not task.is_autopilot(conversation):
        raise HTTPException(400, "Автопилот на этом диалоге не включён")
    if conversation["task_paused"]:
        raise HTTPException(400, "Задача на паузе — сначала снимите паузу")
    return _sse_response(_autopilot(conversation_id, user, request))


# ---------- состояние задачи ----------

class TaskPatch(BaseModel):
    mode: str | None = None
    stage: str | None = None
    # Отметки условий перехода: {"plan_approved": true}. Отдельное действие —
    # отметить условие значит утвердить работу предыдущего этапа.
    guards: dict[str, bool] | None = None
    paused: bool | None = None
    auto: bool | None = None
    autopilot: bool | None = None
    max_turns: int | None = Field(default=None, ge=1, le=db.MAX_TURNS_LIMIT)
    step: str | None = None
    expected: str | None = None
    actor: str | None = None
    note: str | None = None


def _task_view(conversation: dict) -> dict:
    """Состояние задачи вместе с журналом — всё, что нужно панели."""
    stage = task.stage_of(conversation)
    return {
        "mode": conversation["task_mode"] or db.CHAT_MODE,
        "stage": stage,
        "label": task.LABELS[stage],
        "step": conversation["task_step"] or "",
        "expected": conversation["task_expected"] or "",
        "actor": conversation["task_actor"] or db.USER_ACTOR,
        "paused": bool(conversation["task_paused"]),
        "auto": bool(conversation["task_auto"]),
        "autopilot": bool(conversation["task_autopilot"]),
        "max_turns": conversation["task_max_turns"] or 8,
        "max_turns_limit": db.MAX_TURNS_LIMIT,
        "updated_at": conversation["task_updated_at"],
        "stages": [{"id": st, "label": task.LABELS[st]} for st in db.STAGES],
        "allowed": list(task.allowed(stage)),
        "guards": task.guards(conversation),
        "blocked": {st: task.blocked(conversation, st) for st in task.allowed(stage)},
        "tokens": (task.describe(conversation) or {}).get("tokens", 0),
        "events": db.list_task_events(conversation["id"]),
    }


@app.get("/api/conversations/{conversation_id}/task")
async def task_get(conversation_id: int, user: dict = Depends(require_approved)) -> dict:
    return _task_view(_owned(conversation_id, user))


@app.post("/api/conversations/{conversation_id}/task")
async def task_set(
    conversation_id: int, payload: TaskPatch, user: dict = Depends(require_approved)
) -> dict:
    """Меняет состояние задачи. Переход между этапами проверяется автоматом."""
    conversation = _owned(conversation_id, user)

    try:
        if payload.mode is not None:
            conversation = db.update_task(conversation_id, user["id"], mode=payload.mode)
            # Первое включение режима начинает журнал: иначе у задачи не видно
            # момента, когда она вообще появилась.
            if payload.mode == db.TASK_MODE and not db.list_task_events(conversation_id):
                conversation = db.set_task_stage(
                    conversation_id, user["id"], task.stage_of(conversation),
                    note="задача заведена",
                )
        if any(v is not None for v in (payload.step, payload.expected, payload.actor,
                                       payload.auto, payload.autopilot, payload.max_turns)):
            conversation = db.update_task(
                conversation_id, user["id"], step=payload.step, expected=payload.expected,
                actor=payload.actor, auto=payload.auto, autopilot=payload.autopilot,
                max_turns=payload.max_turns,
            )
    except ValueError as err:
        raise HTTPException(400, str(err)) from err

    if payload.guards:
        for guard, value in payload.guards.items():
            try:
                conversation = db.set_task_guard(conversation_id, user["id"], guard, value)
            except ValueError as err:
                raise HTTPException(400, str(err)) from err

    if payload.paused is not None:
        conversation = db.set_task_pause(
            conversation_id, user["id"], payload.paused, note=payload.note
        )

    if payload.stage is not None:
        current = task.stage_of(conversation)
        if payload.stage != current:
            if payload.stage not in db.STAGES:
                raise HTTPException(400, f"Неизвестный этап: {payload.stage}")
            if not task.can_move(current, payload.stage):
                raise HTTPException(
                    400,
                    f"Переход «{task.LABELS[current]} → {task.LABELS[payload.stage]}» "
                    f"не разрешён. Отсюда можно: "
                    f"{', '.join(task.LABELS[st] for st in task.allowed(current)) or 'никуда'}.",
                )
            # Условие входа обязательно для всех: и для кнопки, и для API.
            if reason := task.blocked(conversation, payload.stage):
                raise HTTPException(400, reason[0].upper() + reason[1:] + ".")
            conversation = db.set_task_stage(
                conversation_id, user["id"], payload.stage, note=payload.note
            )

    if conversation is None:
        raise HTTPException(404, "Диалог не найден")
    return _task_view(conversation)


@app.post("/api/conversations/{conversation_id}/task/undo")
async def task_undo(conversation_id: int, user: dict = Depends(require_approved)) -> dict:
    """Отменяет последний переход — в том числе сделанный автоматически."""
    _owned(conversation_id, user)
    conversation = db.undo_task_stage(conversation_id, user["id"])
    if conversation is None:
        raise HTTPException(400, "Отменять нечего")
    return _task_view(conversation)


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
    if ruled := invariants_mod.describe(*invariants_mod.load(conversation, user)):
        described["invariants"] = ruled
    if state := task.describe(conversation):
        described["task"] = state
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
