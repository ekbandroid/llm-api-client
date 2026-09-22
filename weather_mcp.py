"""Свой MCP-сервер: погода, прогноз и качество воздуха вокруг Open-Meteo.

Вчера приложение научилось быть клиентом MCP — здесь вторая сторона. Сервер
объявляет инструменты со схемами аргументов, а выполняет их обычными запросами
к открытому API: Open-Meteo работает без ключа и регистрации.

Город принимается словом и переводится в координаты здесь же, геокодингом
Open-Meteo. Просить широту и долготу у модели значило бы просить её выдумать
числа: она их не знает, но охотно назовёт правдоподобные.

Ответ — короткий текст с единицами измерения, а не JSON: модель читает его как
факт, а голые числа достраивает сама. И в каждом ответе назван найденный город
со страной и регионом — иначе «Москва» в штате Айдахо прошла бы молча.

Ошибки возвращаются текстом, а не исключением: клиент передаст этот текст
модели, и она объяснит его пользователю вместо выдумки.

Запуск рядом с чатом, отдельным процессом:

    uvicorn weather_mcp:app --host 127.0.0.1 --port 8001
"""

from typing import Annotated

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import Field

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Открытый API без ключа отвечает быстро; ждать дольше незачем — вызов идёт
# внутри ответа модели, и пользователь всё это время смотрит на паузу.
TIMEOUT = 10

# Коды погоды WMO: сервер отдаёт число, человеку нужно слово.
WEATHER_CODES = {
    0: "ясно", 1: "преимущественно ясно", 2: "переменная облачность",
    3: "пасмурно", 45: "туман", 48: "изморозь",
    51: "морось слабая", 53: "морось", 55: "морось сильная",
    56: "ледяная морось", 57: "ледяная морось сильная",
    61: "дождь слабый", 63: "дождь", 65: "дождь сильный",
    66: "ледяной дождь", 67: "ледяной дождь сильный",
    71: "снег слабый", 73: "снег", 75: "снег сильный", 77: "снежная крупа",
    80: "ливень слабый", 81: "ливень", 82: "ливень сильный",
    85: "снегопад слабый", 86: "снегопад сильный",
    95: "гроза", 96: "гроза с градом", 99: "гроза с крупным градом",
}

# Европейский индекс качества воздуха: границы и слова из его же шкалы.
AQI_LEVELS = (
    (20, "хорошее"), (40, "приемлемое"), (60, "среднее"),
    (80, "плохое"), (100, "очень плохое"),
)


class WeatherError(RuntimeError):
    """Сервис не ответил или ответил не тем. Наружу уходит текстом."""


server = MCPServer(
    name="Погода (Open-Meteo)",
    version="1.0.0",
    instructions=(
        "Погода, прогноз и качество воздуха по названию города. Данные "
        "Open-Meteo, обновляются ежечасно. Координаты искать не нужно: "
        "передавайте название города как его написал пользователь."
    ),
)


async def _get(url: str, params: dict) -> dict:
    """Запрос к Open-Meteo. Любой сбой превращается в понятную ошибку."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as err:
        raise WeatherError(
            f"сервис погоды ответил {err.response.status_code}") from err
    except httpx.HTTPError as err:
        raise WeatherError(
            f"сервис погоды недоступен ({type(err).__name__})") from err


async def _find_city(city: str) -> dict:
    """Координаты города по названию. Берём первое совпадение."""
    name = (city or "").strip()
    if not name:
        raise WeatherError("не указан город")
    found = await _get(GEOCODING_URL, {
        "name": name, "count": 1, "language": "ru", "format": "json",
    })
    results = found.get("results") or []
    if not results:
        raise WeatherError(
            f"город «{name}» не найден. Проверьте написание или укажите "
            "город покрупнее рядом")
    return results[0]


def _place(found: dict) -> str:
    """Название найденного города так, как его стоит показать человеку."""
    parts = [found.get("name") or "", found.get("admin1") or "",
             found.get("country") or ""]
    return ", ".join(p for p in parts if p)


def _sky(code) -> str:
    return WEATHER_CODES.get(code, f"код погоды {code}")


def _aqi_level(value) -> str:
    if value is None:
        return "нет данных"
    for limit, label in AQI_LEVELS:
        if value <= limit:
            return label
    return "крайне плохое"


@server.tool()
async def weather_now(
    city: Annotated[str, Field(
        description="Город так, как его назвал пользователь: «Екатеринбург», "
                    "«Санкт-Петербург», «Berlin». Координаты не нужны.")],
) -> str:
    """Текущая погода в городе: температура, ощущения, влажность, ветер.

    Данные измерены, а не предсказаны, и обновляются каждый час.
    """
    try:
        found = await _find_city(city)
        data = await _get(FORECAST_URL, {
            "latitude": found["latitude"], "longitude": found["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "wind_speed_10m,precipitation,weather_code",
            "timezone": "auto",
        })
    except WeatherError as err:
        return f"Не удалось узнать погоду: {err}"

    now = data.get("current") or {}
    return (
        f"Погода сейчас — {_place(found)} "
        f"(местное время {now.get('time', '?').replace('T', ' ')}):\n"
        f"температура {now.get('temperature_2m')} °C, "
        f"ощущается как {now.get('apparent_temperature')} °C\n"
        f"{_sky(now.get('weather_code'))}, "
        f"влажность {now.get('relative_humidity_2m')} %, "
        f"ветер {now.get('wind_speed_10m')} км/ч, "
        f"осадки {now.get('precipitation')} мм"
    )


@server.tool()
async def forecast(
    city: Annotated[str, Field(
        description="Город так, как его назвал пользователь. Координаты не нужны.")],
    days: Annotated[int, Field(
        description="Сколько дней показать, считая сегодняшний: от 1 до 7",
        ge=1, le=7)] = 3,
) -> str:
    """Прогноз по дням: минимум и максимум температуры, осадки, облачность."""
    try:
        found = await _find_city(city)
        data = await _get(FORECAST_URL, {
            "latitude": found["latitude"], "longitude": found["longitude"],
            "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,"
                     "weather_code",
            "forecast_days": days, "timezone": "auto",
        })
    except WeatherError as err:
        return f"Не удалось получить прогноз: {err}"

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    lines = [f"Прогноз на {len(dates)} дн. — {_place(found)}:"]
    for i, date in enumerate(dates):
        lines.append(
            f"{date}: от {daily['temperature_2m_min'][i]} "
            f"до {daily['temperature_2m_max'][i]} °C, "
            f"{_sky(daily['weather_code'][i])}, "
            f"осадки {daily['precipitation_sum'][i]} мм"
        )
    return "\n".join(lines)


@server.tool()
async def air_quality(
    city: Annotated[str, Field(
        description="Город так, как его назвал пользователь. Координаты не нужны.")],
) -> str:
    """Качество воздуха: частицы PM2.5 и PM10 и европейский индекс.

    Индекс от 0 до 100+: чем больше, тем хуже. Полезно, когда спрашивают про
    смог, дым, проветривание или пробежку на улице.
    """
    try:
        found = await _find_city(city)
        data = await _get(AIR_URL, {
            "latitude": found["latitude"], "longitude": found["longitude"],
            "current": "pm10,pm2_5,european_aqi", "timezone": "auto",
        })
    except WeatherError as err:
        return f"Не удалось узнать качество воздуха: {err}"

    now = data.get("current") or {}
    aqi = now.get("european_aqi")
    return (
        f"Качество воздуха — {_place(found)} "
        f"(местное время {now.get('time', '?').replace('T', ' ')}):\n"
        f"европейский индекс {aqi} — {_aqi_level(aqi)}\n"
        f"PM2.5 {now.get('pm2_5')} мкг/м³, PM10 {now.get('pm10')} мкг/м³"
    )


# Приложение для uvicorn. Путь /mcp — тот же, по которому клиент ходит к
# чужим серверам: адрес получается однородным.
app = server.streamable_http_app(streamable_http_path="/mcp")
