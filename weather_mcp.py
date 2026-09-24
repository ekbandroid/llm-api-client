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

import os
from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import Field

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Запасной источник прогноза — Норвежский метеоинститут. Нужен не для красоты:
# хосты Open-Meteo стоят у разных провайдеров, и с нашего VPS перестала
# открываться вся сеть Hetzner, где живёт api.open-meteo.com. Геокодинг и
# качество воздуха у Open-Meteo лежат в других сетях и работали всё это время.
MET_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

# met.no требует представляться: запросы без внятного User-Agent он отклоняет.
# Личной почты здесь нет намеренно — репозиторий публичный; при желании адрес
# подставляется переменной окружения.
MET_AGENT = os.getenv(
    "MET_USER_AGENT",
    "llmchat.me.uk weather MCP (https://github.com/ekbandroid/llm-api-client)")

# Открытый API без ключа отвечает быстро; ждать дольше незачем — вызов идёт
# внутри ответа модели, и пользователь всё это время смотрит на паузу.
TIMEOUT = 10

# Соединение отдельно и коротко: недоступный источник должен отваливаться за
# три секунды, чтобы запасной успел ответить в те же десять.
CONNECT_TIMEOUT = 3

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

# Коды погоды met.no: слово вместо числа, суффиксы _day/_night отбрасываем.
MET_CODES = {
    "clearsky": "ясно", "fair": "малооблачно", "partlycloudy": "переменная облачность",
    "cloudy": "пасмурно", "fog": "туман",
    "lightrain": "дождь слабый", "rain": "дождь", "heavyrain": "дождь сильный",
    "lightrainshowers": "ливень слабый", "rainshowers": "ливень",
    "heavyrainshowers": "ливень сильный",
    "lightsleet": "мокрый снег слабый", "sleet": "мокрый снег",
    "heavysleet": "мокрый снег сильный",
    "lightsleetshowers": "мокрый снег слабый", "sleetshowers": "мокрый снег",
    "heavysleetshowers": "мокрый снег сильный",
    "lightsnow": "снег слабый", "snow": "снег", "heavysnow": "снег сильный",
    "lightsnowshowers": "снегопад слабый", "snowshowers": "снегопад",
    "heavysnowshowers": "снегопад сильный",
}

# Европейский индекс качества воздуха: границы и слова из его же шкалы.
AQI_LEVELS = (
    (20, "хорошее"), (40, "приемлемое"), (60, "среднее"),
    (80, "плохое"), (100, "очень плохое"),
)


class WeatherError(RuntimeError):
    """Сервис не ответил или ответил не тем. Наружу уходит текстом."""


server = MCPServer(
    name="Погода (Open-Meteo и met.no)",
    version="1.1.0",
    instructions=(
        "Погода, прогноз и качество воздуха по названию города. Координаты "
        "искать не нужно: передавайте название города как его написал "
        "пользователь. Источник данных — Open-Meteo, а если он недоступен, "
        "Норвежский метеоинститут; в ответе сказано, какой сработал."
    ),
)


async def _get(url: str, params: dict, *, headers: dict | None = None) -> dict:
    """Запрос к источнику. Любой сбой превращается в понятную ошибку.

    В тексте ошибки называется хост: когда с сервера перестала открываться
    целая сеть, разбираться пришлось по строке «ConnectTimeout» без единого
    намёка на то, куда именно не достучались.
    """
    host = httpx.URL(url).host
    limits = httpx.Timeout(TIMEOUT, connect=CONNECT_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=limits, headers=headers or {}) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as err:
        raise WeatherError(f"{host} ответил {err.response.status_code}") from err
    except httpx.HTTPError as err:
        raise WeatherError(f"{host} недоступен ({type(err).__name__})") from err


def _sky_met(code: str) -> str:
    """Слово по коду met.no: clearsky_day → ясно."""
    base = (code or "").split("_")[0]
    return MET_CODES.get(base, base or "нет данных")


async def _first_working(sources: tuple, *args) -> tuple[dict | list, str]:
    """Первый источник, который ответил. Возвращает данные и его имя.

    Порядок важен: Open-Meteo богаче (у него есть «ощущается как»), met.no —
    запасной. Когда маршрут до Open-Meteo восстановится, всё вернётся само.
    """
    troubles = []
    for name, fetch in sources:
        try:
            return await fetch(*args), name
        except WeatherError as err:
            troubles.append(f"{name} — {err}")
    raise WeatherError("; ".join(troubles))


# ---------- текущая погода ----------

async def _now_openmeteo(found: dict) -> dict:
    data = await _get(FORECAST_URL, {
        "latitude": found["latitude"], "longitude": found["longitude"],
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                   "wind_speed_10m,precipitation,weather_code",
        "timezone": "auto",
    })
    now = data.get("current") or {}
    return {
        "time": (now.get("time") or "?").replace("T", " "),
        "temperature": now.get("temperature_2m"),
        "feels": now.get("apparent_temperature"),
        "humidity": now.get("relative_humidity_2m"),
        "wind": now.get("wind_speed_10m"),
        "precipitation": now.get("precipitation"),
        "sky": _sky(now.get("weather_code")),
    }


async def _now_met(found: dict) -> dict:
    data = await _get(MET_URL, {
        "lat": round(float(found["latitude"]), 4),
        "lon": round(float(found["longitude"]), 4),
    }, headers={"User-Agent": MET_AGENT})
    points = (data.get("properties") or {}).get("timeseries") or []
    if not points:
        raise WeatherError("api.met.no вернул пустой прогноз")

    first = points[0]
    instant = ((first.get("data") or {}).get("instant") or {}).get("details") or {}
    hour = (first.get("data") or {}).get("next_1_hours") or {}
    zone = _zone(found)
    when = datetime.fromisoformat(first["time"].replace("Z", "+00:00"))
    return {
        "time": when.astimezone(zone).strftime("%Y-%m-%d %H:%M"),
        "temperature": instant.get("air_temperature"),
        # «Ощущается как» у met.no нет вовсе — показывать нечего, и выдумывать
        # это число мы не станем.
        "feels": None,
        # Влажность приходит дробной («84.0 %») — процент целым понятнее.
        "humidity": round(instant["relative_humidity"]) if instant.get("relative_humidity") is not None else None,
        # Ветер приходит в м/с, а у первого источника — в км/ч. Приводим, иначе
        # одно и то же место давало бы то 2.5, то 0.7 без объяснений.
        "wind": round((instant.get("wind_speed") or 0) * 3.6, 1),
        "precipitation": (hour.get("details") or {}).get("precipitation_amount"),
        "sky": _sky_met((hour.get("summary") or {}).get("symbol_code", "")),
    }


# ---------- прогноз по дням ----------

async def _forecast_openmeteo(found: dict, days: int) -> list[dict]:
    data = await _get(FORECAST_URL, {
        "latitude": found["latitude"], "longitude": found["longitude"],
        "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,weather_code",
        "forecast_days": days, "timezone": "auto",
    })
    daily = data.get("daily") or {}
    return [
        {"date": date,
         "low": daily["temperature_2m_min"][i], "high": daily["temperature_2m_max"][i],
         "precipitation": daily["precipitation_sum"][i],
         "sky": _sky(daily["weather_code"][i])}
        for i, date in enumerate(daily.get("time") or [])
    ]


async def _forecast_met(found: dict, days: int) -> list[dict]:
    """Собирает дни из почасового ряда met.no.

    Осадки берём из next_1_hours, а где его нет — из next_6_hours: в начале
    ряда шаг часовой, дальше шестичасовой, и такой выбор покрывает время без
    двойного счёта.
    """
    data = await _get(MET_URL, {
        "lat": round(float(found["latitude"]), 4),
        "lon": round(float(found["longitude"]), 4),
    }, headers={"User-Agent": MET_AGENT})
    zone = _zone(found)
    collected: dict[str, dict] = {}

    for point in (data.get("properties") or {}).get("timeseries") or []:
        when = datetime.fromisoformat(point["time"].replace("Z", "+00:00")).astimezone(zone)
        day = collected.setdefault(when.strftime("%Y-%m-%d"), {
            "date": when.strftime("%Y-%m-%d"), "low": None, "high": None,
            "precipitation": 0.0, "sky": "", "noon": 99,
        })
        block = point.get("data") or {}
        if (degrees := (block.get("instant", {}).get("details") or {}).get("air_temperature")) is not None:
            day["low"] = degrees if day["low"] is None else min(day["low"], degrees)
            day["high"] = degrees if day["high"] is None else max(day["high"], degrees)
        window = block.get("next_1_hours") or block.get("next_6_hours") or {}
        day["precipitation"] += (window.get("details") or {}).get("precipitation_amount") or 0.0
        # Небо дня — то, что ближе к полудню: ночной код описал бы не день.
        if abs(when.hour - 12) < day["noon"] and (window.get("summary") or {}).get("symbol_code"):
            day["noon"] = abs(when.hour - 12)
            day["sky"] = _sky_met(window["summary"]["symbol_code"])

    ordered = [d for _, d in sorted(collected.items())][:days]
    for day in ordered:
        day["precipitation"] = round(day["precipitation"], 1)
        day.pop("noon", None)
    return ordered


SOURCES_NOW = (("Open-Meteo", _now_openmeteo), ("met.no", _now_met))
SOURCES_FORECAST = (("Open-Meteo", _forecast_openmeteo), ("met.no", _forecast_met))


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


def _zone(found: dict) -> ZoneInfo:
    """Часовой пояс города: он приходит вместе с координатами от геокодинга."""
    try:
        return ZoneInfo(found.get("timezone") or "UTC")
    except Exception:  # noqa: BLE001 — незнакомая зона не повод падать
        return ZoneInfo("UTC")


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
        now, source = await _first_working(SOURCES_NOW, found)
    except WeatherError as err:
        return f"Не удалось узнать погоду: {err}"

    feels = f", ощущается как {now['feels']} °C" if now["feels"] is not None else ""
    return (
        f"Погода сейчас — {_place(found)} "
        f"(местное время {now['time']}, источник {source}):\n"
        f"температура {now['temperature']} °C{feels}\n"
        f"{now['sky']}, влажность {now['humidity']} %, "
        f"ветер {now['wind']} км/ч, осадки {now['precipitation']} мм"
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
        forecast, source = await _first_working(SOURCES_FORECAST, found, days)
    except WeatherError as err:
        return f"Не удалось получить прогноз: {err}"

    lines = [f"Прогноз на {len(forecast)} дн. — {_place(found)} "
             f"(источник {source}):"]
    for day in forecast:
        lines.append(
            f"{day['date']}: от {day['low']} до {day['high']} °C, "
            f"{day['sky']}, осадки {day['precipitation']} мм"
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
