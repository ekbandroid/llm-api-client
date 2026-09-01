"""Авторизация через Яндекс OAuth (Authorization Code Flow)."""

import os
import secrets
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("YANDEX_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("YANDEX_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("YANDEX_REDIRECT_URI", "http://localhost:8000/auth/callback")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")

AUTHORIZE_URL = "https://oauth.yandex.ru/authorize"
TOKEN_URL = "https://oauth.yandex.ru/token"
PROFILE_URL = "https://login.yandex.ru/info"


class AuthError(RuntimeError):
    """Ошибка на любом шаге OAuth-обмена."""


def admin_logins() -> set[str]:
    """Логины Яндекса, которым выдаётся админ-доступ."""
    raw = os.getenv("ADMIN_LOGINS", "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def is_configured() -> bool:
    """True, если приложение Яндекса заведено и ключи прописаны."""
    return bool(CLIENT_ID and CLIENT_SECRET)


def new_state() -> str:
    """Одноразовый state против CSRF на редиректе."""
    return secrets.token_urlsafe(24)


def authorize_url(state: str) -> str:
    """Ссылка, на которую отправляем пользователя для входа в Яндекс."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code(code: str) -> str:
    """Меняет одноразовый код на access_token."""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            },
        )
    if response.status_code != 200:
        raise AuthError(f"Яндекс отклонил код ({response.status_code}): {response.text}")

    token = response.json().get("access_token")
    if not token:
        raise AuthError("Яндекс не вернул access_token")
    return token


async def fetch_profile(token: str) -> dict:
    """Забирает профиль пользователя по access_token."""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            PROFILE_URL,
            params={"format": "json"},
            headers={"Authorization": f"OAuth {token}"},
        )
    if response.status_code != 200:
        raise AuthError(f"Не удалось получить профиль ({response.status_code}): {response.text}")

    profile = response.json()
    if not profile.get("id"):
        raise AuthError("В профиле Яндекса нет идентификатора")
    return profile
