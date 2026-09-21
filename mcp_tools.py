"""Связь с MCP-серверами: соединение, перечень инструментов, вызов.

MCP (Model Context Protocol) решает ту же задачу, что usb-разъём: вместо того
чтобы писать под каждый сервис свою обвязку, приложение говорит с любым
сервером одним протоколом. Сервер объявляет инструменты — имя, описание и
JSON-схему аргументов, — а клиент их перечисляет и вызывает. Ровно эти схемы
потом уходят в запрос к модели полем `tools`, и дальше модель сама решает,
какой инструмент ей нужен.

Модуль называется mcp_tools, а не mcp: файл `mcp.py` в корне проекта перекрыл
бы сам пакет `mcp`, и его импорт сломался бы. Это не догадка — тем же способом
проект уже ломался на стандартном пакете `compression`, см. первый абзац
докстринга history.py.

Запуск скриптом печатает сервер и его инструменты:

    python mcp_tools.py                                # курсы валют
    python mcp_tools.py https://mcp.deepwiki.com/mcp   # другой сервер
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field

from mcp import Client

# Сколько ждём сервер. Тот же смысл, что у таймаутов в llm.py: недоступный
# сервер должен отваливаться сам, а не держать ни терминал, ни веб-запрос.
TIMEOUT = 30

# Сервер по умолчанию: публичный, без ключа и регистрации, четыре инструмента.
DEFAULT_URL = "https://currency-mcp.wesbos.com/mcp"

# Готовые серверы, которые заводятся пользователю при первом входе. Все
# проверены живым запросом: отвечают без ключа и без регистрации.
KNOWN_SERVERS = [
    {"title": "Курсы валют", "url": DEFAULT_URL, "enabled": True},
    {"title": "DeepWiki — документация репозиториев",
     "url": "https://mcp.deepwiki.com/mcp", "enabled": False},
    {"title": "Context7 — документация библиотек",
     "url": "https://mcp.context7.com/mcp", "enabled": False},
    {"title": "GitMCP — документация по ссылке",
     "url": "https://gitmcp.io/docs", "enabled": False},
]


class MCPError(RuntimeError):
    """Сервер недоступен или ответил не по протоколу."""


def _flatten(err: BaseException) -> list[str]:
    """Разворачивает ExceptionGroup в плоский список сообщений.

    Клиент MCP работает на группе задач, поэтому наружу вылетает
    ExceptionGroup, у которой в тексте только «unhandled errors in a
    TaskGroup». Настоящая причина — внутри, и показывать нужно её.
    """
    if isinstance(err, BaseExceptionGroup):
        found: list[str] = []
        for sub in err.exceptions:
            found.extend(_flatten(sub))
        return found
    text = str(err).strip()
    return [f"{type(err).__name__}: {text}" if text else type(err).__name__]


def _explain(url: str, err: BaseException) -> MCPError:
    return MCPError(f"MCP {url} недоступен — {'; '.join(_flatten(err))}")


@dataclass
class Tool:
    """Инструмент так, как его объявил сервер."""

    name: str
    title: str
    description: str
    schema: dict

    @property
    def summary(self) -> str:
        """Описание для человека: у части серверов оно пустое."""
        return self.description or self.title or "(без описания)"

    @property
    def required(self) -> list[str]:
        return list(self.schema.get("required") or [])


@dataclass
class Server:
    """Что известно о сервере после рукопожатия."""

    url: str
    name: str
    version: str
    protocol: str
    instructions: str = ""
    tools: list[Tool] = field(default_factory=list)


def _tool(raw) -> Tool:
    return Tool(
        name=raw.name,
        title=raw.title or "",
        description=raw.description or "",
        schema=raw.input_schema or {},
    )


async def list_tools(url: str, *, timeout: float = TIMEOUT) -> Server:
    """Соединяется с сервером и возвращает его вместе со списком инструментов.

    Транспорт клиент выбирает сам по адресу: streamable HTTP для обычного URL
    и SSE для адресов, которые отвечают только им.
    """
    try:
        async with Client(url, read_timeout_seconds=timeout) as client:
            info = client.server_info
            result = await client.list_tools()
            return Server(
                url=url,
                name=getattr(info, "name", "") or "(без имени)",
                version=getattr(info, "version", "") or "",
                protocol=client.protocol_version or "",
                instructions=client.instructions or "",
                tools=[_tool(t) for t in result.tools],
            )
    except MCPError:
        raise
    except BaseException as err:  # noqa: BLE001 — наружу уходит своя ошибка
        raise _explain(url, err) from err


async def call_tool(
    url: str, name: str, arguments: dict, *, timeout: float = TIMEOUT
) -> str:
    """Вызывает инструмент и возвращает его ответ текстом.

    Ответ сервера — список блоков; текстовые склеиваются, остальные (картинки,
    ссылки на ресурсы) заменяются пометкой о типе: в запрос к модели уходит
    текст, а не байты.
    """
    try:
        async with Client(url, read_timeout_seconds=timeout) as client:
            result = await client.call_tool(name, arguments or {})
    except BaseException as err:  # noqa: BLE001
        raise _explain(url, err) from err

    parts: list[str] = []
    for block in result.content or []:
        if text := getattr(block, "text", None):
            parts.append(text)
        else:
            parts.append(f"[{getattr(block, 'type', 'неизвестный блок')}]")
    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))

    answer = "\n".join(parts).strip() or "(пустой ответ)"
    # Признак ошибки инструмента не исключение: модель должна увидеть текст
    # ошибки и объяснить его пользователю, а не молча остаться без данных.
    return f"ОШИБКА ИНСТРУМЕНТА: {answer}" if result.is_error else answer


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f"Соединяюсь: {url}")
    try:
        server = asyncio.run(list_tools(url))
    except MCPError as err:
        print(err)
        raise SystemExit(1)

    version = f" {server.version}" if server.version else ""
    print(f"Сервер: {server.name}{version} · протокол {server.protocol}")
    print(f"Инструментов: {len(server.tools)}\n")
    for tool in server.tools:
        required = ", ".join(tool.required) or "без обязательных"
        print(f"  {tool.name}  ({required})")
        print(f"      {tool.summary[:150]}")


if __name__ == "__main__":
    main()
