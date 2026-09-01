"""Авторизация: вход по логину и паролю плюс Яндекс OAuth."""

import hashlib
import os
import re
import secrets
import time
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


# ============================================================================
# Вход по логину и паролю
# ============================================================================

LOGIN_RE = re.compile(r"^[a-zA-Z0-9._-]{3,32}$")
MIN_PASSWORD_LENGTH = 8

# Параметры scrypt: 128 * N * r = 16 МБ памяти на проверку — этого достаточно,
# чтобы перебор на видеокартах стал дорогим.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _DKLEN = 2**14, 8, 1, 32
_MAXMEM = 64 * 1024 * 1024


def hash_password(password: str) -> str:
    """Хеширует пароль scrypt'ом с индивидуальной солью."""
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        dklen=_DKLEN, maxmem=_MAXMEM,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${key.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    """Сверяет пароль с хешем. Любой разбор ошибки трактуется как несовпадение."""
    if not stored:
        return False
    try:
        algo, n, r, p, salt_hex, key_hex = stored.split("$")
        if algo != "scrypt":
            return False
        expected = bytes.fromhex(key_hex)
        key = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(expected), maxmem=_MAXMEM,
        )
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(key, expected)


def validate_credentials(login: str, password: str) -> str | None:
    """Возвращает текст ошибки или None, если логин и пароль допустимы."""
    if not LOGIN_RE.fullmatch(login):
        return ("Логин: от 3 до 32 символов, только латиница, цифры, точка, "
                "дефис и подчёркивание.")
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Пароль короче {MIN_PASSWORD_LENGTH} символов."
    return None


# ---------- защита от перебора ----------
# Счётчик живёт в памяти процесса: при нескольких воркерах uvicorn каждый считает
# своё, а рестарт обнуляет. Для настоящей защиты нужен общий стор (Redis) или
# ограничение на уровне nginx.

MAX_ATTEMPTS = 8
LOCKOUT_WINDOW = 300  # секунд

_attempts: dict[str, tuple[int, float]] = {}


def throttle_check(login: str) -> int:
    """Сколько секунд осталось ждать. 0 — можно пробовать."""
    count, first = _attempts.get(login.lower(), (0, 0.0))
    if count < MAX_ATTEMPTS:
        return 0
    left = LOCKOUT_WINDOW - (time.monotonic() - first)
    if left <= 0:
        _attempts.pop(login.lower(), None)
        return 0
    return int(left) + 1


def throttle_fail(login: str) -> None:
    """Отмечает неудачную попытку входа."""
    key = login.lower()
    count, first = _attempts.get(key, (0, time.monotonic()))
    if time.monotonic() - first > LOCKOUT_WINDOW:
        count, first = 0, time.monotonic()
    _attempts[key] = (count + 1, first)


def throttle_reset(login: str) -> None:
    """Сбрасывает счётчик после успешного входа."""
    _attempts.pop(login.lower(), None)
