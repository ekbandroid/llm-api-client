"""MCP-сервер для цепочки: найти — прочитать — пересказать — сохранить.

Инструменты здесь нарочно мелкие и каждый делает одно. Смысл в том, что
собирает их в цепочку сам агент: `search` отдаёт заголовок, `article` по
заголовку — текст, `summarize` из текста — выжимку, `save_to_file` кладёт её
в файл. Приложение между звеньями ничего не склеивает: результат инструмента
возвращается модели строкой, и она сама решает, что передать дальше.

Отсюда требование к ответам: каждый должен содержать ровно то, что нужно
следующему звену. Поэтому `search` возвращает точные заголовки статей — по
приблизительному пересказу заголовка `article` ничего не найдёт.

Источник данных — Википедия: бесплатно, без ключа и регистрации, отдаёт текст
статьи простым текстом. Пересказ сервер делает сам, тем же ключом и той же
моделью, что и приложение: ключ уже лежит в .env рядом.

Запуск рядом с чатом:

    uvicorn research_mcp:app --host 127.0.0.1 --port 8003
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import Field

import files_store
import llm
import rag

WIKI_URL = "https://ru.wikipedia.org/w/api.php"

# Википедия просит представляться — иначе рано или поздно ответит 403.
WIKI_AGENT = os.getenv(
    "WIKI_USER_AGENT",
    "llmchat.me.uk research MCP (https://github.com/ekbandroid/llm-api-client)")

TIMEOUT = 15
CONNECT_TIMEOUT = 3

# Сколько текста статьи отдаём модели. Больше — дороже каждый следующий круг
# цепочки: текст проходит через контекст ещё раз как аргумент пересказа.
ARTICLE_LIMIT = 8000

# Предел на текст, который принимает пересказ.
SUMMARY_INPUT_LIMIT = 20000


class ResearchError(RuntimeError):
    """Источник не ответил или ответил не тем. Наружу уходит текстом."""


server = MCPServer(
    name="Поиск, пересказ и файлы",
    version="1.0.0",
    instructions=(
        "Инструменты для цепочки: search находит статьи Википедии, article "
        "отдаёт текст статьи по её точному заголовку, summarize делает "
        "выжимку из переданного текста, save_to_file сохраняет результат. "
        "Заголовок для article берите из ответа search дословно."
    ),
)


async def _wiki(params: dict) -> dict:
    """Запрос к API Википедии. Сбой превращается в понятную ошибку.

    Одна повторная попытка на сетевой сбой: замер показал, что примерно
    каждый третий запрос зависает на чтении, хотя соединение устанавливается
    за миллисекунды. Для цепочки это критично — сорвавшееся первое звено
    останавливает всю работу, и модель тратит круг на объяснения.
    """
    limits = httpx.Timeout(TIMEOUT, connect=CONNECT_TIMEOUT)
    trouble = None
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(
                timeout=limits, headers={"User-Agent": WIKI_AGENT}
            ) as client:
                response = await client.get(
                    WIKI_URL, params={**params, "format": "json"})
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as err:
            # Ответ получен, просто не тот — повторять незачем.
            raise ResearchError(
                f"Википедия ответила {err.response.status_code}") from err
        except httpx.HTTPError as err:
            trouble = err
            if attempt == 1:
                await asyncio.sleep(0.5)
    raise ResearchError(
        f"ru.wikipedia.org не ответила с двух попыток "
        f"({type(trouble).__name__})") from trouble


def _plain(html: str) -> str:
    """Фрагмент поиска приходит с разметкой подсветки — она тут не нужна."""
    return re.sub(r"<[^>]+>", "", html or "").replace("&quot;", '"').strip()


@server.tool()
async def search(
    query: Annotated[str, Field(
        description="Что искать: тема, название, имя. Обычными словами")],
    limit: Annotated[int, Field(
        description="Сколько статей вернуть, 1–5", ge=1, le=5)] = 3,
) -> str:
    """Ищет статьи в русской Википедии по запросу.

    Возвращает точные заголовки — их же нужно передавать в article, чтобы
    получить текст. Первый шаг цепочки «найти — прочитать — пересказать».
    """
    try:
        found = await _wiki({
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": limit,
        })
    except ResearchError as err:
        return f"Поиск не удался: {err}"

    hits = (found.get("query") or {}).get("search") or []
    if not hits:
        return (f"По запросу «{query}» в Википедии ничего не нашлось. "
                "Попробуйте другие слова или более общее название.")

    total = ((found.get("query") or {}).get("searchinfo") or {}).get("totalhits", 0)
    lines = [f"Найдено статей: {total}, показываю {len(hits)}."]
    for i, hit in enumerate(hits, 1):
        title = hit["title"]
        link = "https://ru.wikipedia.org/wiki/" + title.replace(" ", "_")
        lines.append(
            f"{i}. {title} — {hit.get('wordcount', '?')} слов\n"
            f"   {_plain(hit.get('snippet'))[:200]}\n"
            f"   {link}"
        )
    lines.append("Заголовок для article берите из списка дословно.")
    return "\n".join(lines)


@server.tool()
async def article(
    title: Annotated[str, Field(
        description="Точный заголовок статьи, как его вернул search")],
) -> str:
    """Отдаёт текст статьи Википедии простым текстом.

    Второй шаг цепочки: то, что дальше уходит в summarize.
    """
    try:
        data = await _wiki({
            "action": "query", "prop": "extracts", "explaintext": 1,
            "titles": title, "redirects": 1,
        })
    except ResearchError as err:
        return f"Статью получить не удалось: {err}"

    pages = (data.get("query") or {}).get("pages") or {}
    page = next(iter(pages.values()), {})
    if "missing" in page or not page.get("extract"):
        return (f"Статьи «{title}» в Википедии нет. Проверьте заголовок — "
                "он должен быть таким же, как в ответе search.")

    text = page["extract"].strip()
    cut = ""
    if len(text) > ARTICLE_LIMIT:
        text, cut = text[:ARTICLE_LIMIT], (
            f"\n\n[показаны первые {ARTICLE_LIMIT} из {len(page['extract'])} символов]")
    return f"{page.get('title', title)} — статья Википедии:\n\n{text}{cut}"


SUMMARY_SYSTEM = (
    "Ты делаешь выжимку из переданного текста. Пиши по существу, без вводных "
    "фраз вроде «в этой статье рассказывается». Сохраняй числа, имена и даты "
    "как есть. Не добавляй того, чего в тексте нет: выжимка — это сокращение, "
    "а не дополнение."
)


@server.tool()
async def summarize(
    text: Annotated[str, Field(
        description="Текст, который нужно сократить. Обычно это ответ article")],
    sentences: Annotated[int, Field(
        description="Сколько предложений оставить, 3–10", ge=3, le=10)] = 5,
    focus: Annotated[str, Field(
        description="На чём сосредоточиться: «на устройстве», «на истории». "
                    "Можно не указывать")] = "",
) -> str:
    """Делает выжимку из текста заданной длины.

    Третий шаг цепочки. Сервер обращается к модели сам, поэтому вызов стоит
    токенов — их количество называется в ответе.
    """
    source = (text or "").strip()
    if len(source) < 200:
        return ("Текста слишком мало для выжимки — передайте статью целиком, "
                "например ответ инструмента article.")

    task = (f"Сократи до {sentences} предложений"
            + (f", сосредоточься {focus}" if focus else "")
            + f".\n\nТекст:\n{source[:SUMMARY_INPUT_LIMIT]}")
    try:
        result = await asyncio.to_thread(
            llm.complete,
            [{"role": "system", "content": SUMMARY_SYSTEM},
             {"role": "user", "content": task}],
            thinking=False, max_tokens=1000, temperature=0.3, keep_text=True,
        )
    except llm.LLMError as err:
        return f"Выжимку сделать не удалось: {err}"

    return (f"{result.content.strip()}\n\n"
            f"[выжимка из {len(source)} символов, стоила "
            f"{result.total_tokens} токенов]")


@server.tool()
async def save_to_file(
    name: Annotated[str, Field(
        description="Имя файла с расширением; расширение задаёт вид: "
                    "«выжимка.md» для заметки, «отчёт.html» для страницы, "
                    "«данные.csv» для таблицы, «скрипт.py» для кода")],
    content: Annotated[str, Field(
        description="Что записать. Обычно это результат summarize — "
                    "передавайте его текстом, не пересказывая заново")],
    chat_id: int = 0,
) -> str:
    """Сохраняет текст в файл этого диалога.

    Последний шаг цепочки. Файл виден во вкладке «Файлы», откуда его можно
    скачать.
    """
    if not chat_id:
        return ("Не указан диалог. Этот сервер рассчитан на вызов из "
                "приложения: оно подставляет идентификатор само.")
    body = (content or "").strip()
    if not body:
        return "Нечего сохранять: передан пустой текст."

    try:
        path = files_store.save(chat_id, name, body)
    except OSError as err:
        return f"Файл не записался: {err}"

    note = ""
    if path.name != Path((name or "").strip()).name:
        note = f" (имя приведено к безопасному виду из «{name}»)"
    return (f"Сохранено: {path.name}, {path.stat().st_size} байт{note}. "
            f"Файл доступен во вкладке «Файлы» этого диалога.")


@server.tool()
async def list_files(chat_id: int = 0) -> str:
    """Перечисляет файлы, сохранённые в этом диалоге."""
    if not chat_id:
        return "Не указан диалог."
    items = files_store.listing(chat_id)
    if not items:
        return "Файлов в этом диалоге пока нет."
    lines = ["Файлы этого диалога:"]
    for item in items:
        when = datetime.fromisoformat(item["saved_at"])
        lines.append(f"  {item['name']} — {item['size']} байт, "
                     f"{when:%d.%m %H:%M} UTC")
    return "\n".join(lines)

# ---------- поиск по загруженным документам ----------
#
# Индексация и поиск живут здесь, а не в приложении, по одной причине: модели
# эмбеддингов держатся в памяти резидентно, и два процесса загрузили бы по
# своей копии. Приложение только принимает файлы и просит этот сервер их
# проиндексировать.


@server.tool()
async def search_docs(
    query: Annotated[str, Field(
        description="Что искать. Своими словами, как спросил пользователь")],
    limit: Annotated[int, Field(
        description="Сколько кусков вернуть, 1–8", ge=1, le=8)] = 4,
    chat_id: int = 0,
) -> str:
    """Ищет ответ в документах, подключённых к этому диалогу.

    Документы загружает человек во вкладке RAG и сам решает, какие из них
    подключить. Если подключённых наборов нет, искать негде — так и будет
    сказано.
    """
    if not chat_id:
        return "Не указан диалог."
    try:
        found = await asyncio.to_thread(rag.search_collections, query, chat_id, limit)
    except Exception as err:  # noqa: BLE001 — сбой поиска не должен ронять обмен
        return f"Поиск по документам не удался: {type(err).__name__}: {err}"

    if not found:
        attached = await asyncio.to_thread(rag.attached_collections, chat_id)
        if not attached:
            return ("К этому диалогу документы не подключены. Их загружают и "
                    "подключают во вкладке RAG.")
        return "В подключённых документах ничего похожего не нашлось."

    lines = [f"Найдено в документах ({len(found)}):"]
    for number, hit in enumerate(found, 1):
        lines.append(
            f"\n{number}. {hit['collection']} · {hit['source']} · "
            f"{hit['section']} (близость {hit['score']:.2f})\n{hit['text']}")
    lines.append("\nЭто выдержки из документов пользователя. Отвечай по ним и "
                 "указывай, из какого файла взято.")
    return "\n".join(lines)


@server.tool()
async def _index_collection(collection_id: int = 0) -> str:
    """Служебный: проиндексировать набор. Вызывает приложение после загрузки."""
    if not collection_id:
        return "Не указан набор."
    try:
        report = await asyncio.to_thread(rag.index_collection, int(collection_id))
    except Exception as err:  # noqa: BLE001
        rag.set_status(int(collection_id), "ошибка", f"{type(err).__name__}: {err}")
        return f"Индексация не удалась: {type(err).__name__}: {err}"
    skipped = f", пропущено: {'; '.join(report['skipped'])}" if report.get("skipped") else ""
    return (f"Проиндексировано: файлов {report.get('files', 0)}, "
            f"кусков {report['chunks']}, за {report.get('seconds', 0)} с{skipped}")


rag.init()

app = server.streamable_http_app(streamable_http_path="/mcp")
