"""Файлы, которые агент сохраняет в диалоге.

Общий модуль на два процесса: MCP-сервер пишет файлы инструментом
`save_to_file`, приложение показывает их во вкладке и отдаёт на скачивание.
Соглашение о каталогах должно быть одно на обоих, иначе сохранённое просто
не нашлось бы.

Раскладка простая: `FILES_DIR/chat-<id>/<имя>`. Каталог на диалог — чтобы
файлы одного разговора не смешивались с другим и удалялись вместе с ним.
"""

import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import db

# По умолчанию файлы лежат рядом с базой переписки, а не в текущем каталоге.
# Так и должно быть: путь к базе уже отличает рабочую машину от сервера, и
# второй переменной для того же самого не нужно. Прежний вариант стоил бага —
# сервер писал в /opt/llmchat/data/files, а приложение искало в каталоге
# кода, и вкладка «Файлы» показывала пустоту при существующих файлах.
FILES_DIR = Path(os.getenv("FILES_DIR") or db.DB_PATH.resolve().parent / "files")

# Имя файла: только то, что нельзя перепутать с путём. Каталоги, точки и
# слэши вырезаются целиком — содержимое приходит от модели, и складывать по
# её выбору куда угодно на диске мы не станем.
NAME_ALLOWED = re.compile(r"[^A-Za-zА-Яа-яЁё0-9 _-]+")

# Расширение — хвост из латинских букв и цифр. Список допустимых не ведём:
# расширение само по себе ничего не решает, опасен был бы неверный тип при
# отдаче, а его выбирает content_type ниже.
EXTENSION = re.compile(r"^[A-Za-z0-9]{1,8}$")
DEFAULT_EXTENSION = "md"

# Предел на размер файла: место на диске общее с базой переписки.
FILE_LIMIT = 100_000

# Типы, которые браузер показывает сам. Текстовое отдаём как text/plain, а не
# «настоящим» типом: csv и json браузер иначе скачает, а показать их полезнее.
TEXT_EXTENSIONS = (
    "md", "txt", "csv", "tsv", "json", "yaml", "yml", "xml", "log", "ini",
    "py", "js", "ts", "sql", "sh", "css", "c", "h", "java", "kt", "go", "rs",
)

# Эти показываем в песочнице: разметку и рисунок браузер выполняет, а написала
# их модель. Заголовок Content-Security-Policy при отдаче делает страницу
# чужим источником — к нашим кукам и API у неё доступа нет.
SANDBOX_TYPES = {"html": "text/html; charset=utf-8",
                 "htm": "text/html; charset=utf-8",
                 "svg": "image/svg+xml"}

# Что браузер умеет показывать без песочницы. Картинок и звука модель пока
# создать не может, но таблица нужна целиком: файлы появятся позже.
MEDIA_PREFIXES = ("image/", "audio/", "video/")
MEDIA_TYPES = {"pdf": "application/pdf"}


def content_type(name: str) -> tuple[str, bool, bool]:
    """Чем отдавать файл: тип, можно ли показать, нужна ли песочница.

    Единственное место, где это решается: интерфейс спрашивает у сервера, а не
    повторяет ту же таблицу у себя.
    """
    ext = Path(name).suffix.lstrip(".").lower()
    if ext in SANDBOX_TYPES:
        return SANDBOX_TYPES[ext], True, True
    if ext in TEXT_EXTENSIONS:
        return "text/plain; charset=utf-8", True, False
    if ext in MEDIA_TYPES:
        return MEDIA_TYPES[ext], True, False
    guess, _ = mimetypes.guess_type(name)
    if guess and guess.startswith(MEDIA_PREFIXES):
        return guess, True, False
    if guess and guess.startswith("text/"):
        return "text/plain; charset=utf-8", True, False
    # Неизвестное не показываем вовсе: пусть браузер честно скачает файл,
    # вместо того чтобы гадать о содержимом.
    return "application/octet-stream", False, False


def safe_name(name: str) -> str:
    """Имя файла без каталогов и сюрпризов.

    «../../etc/passwd.html» превращается в «passwd.html»: сначала отбрасывается
    путь, потом из имени вычищается всё, кроме букв, цифр, дефиса и
    подчёркивания. Расширение сохраняется любое — оно задаёт вид файла, а
    безопасность обеспечивает не оно, а тип при отдаче.
    """
    base = Path((name or "").strip()).name
    stem, dot, ext = base.rpartition(".")
    if not dot or not EXTENSION.match(ext):
        stem, ext = base, DEFAULT_EXTENSION
    clean = NAME_ALLOWED.sub("", stem).strip().replace(" ", "_")[:60]
    return f"{clean or 'заметка'}.{ext.lower()}"


def folder(chat_id: int, *, create: bool = False) -> Path:
    path = FILES_DIR / f"chat-{int(chat_id)}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def save(chat_id: int, name: str, content: str) -> Path:
    """Пишет файл и возвращает путь. Имя приводится к безопасному виду."""
    path = folder(chat_id, create=True) / safe_name(name)
    path.write_text((content or "")[:FILE_LIMIT], encoding="utf-8")
    return path


def listing(chat_id: int) -> list[dict]:
    """Файлы диалога, новые сверху. Недоступный каталог — пустой список."""
    path = folder(chat_id)
    try:
        items = [p for p in path.iterdir() if p.is_file()]
    except OSError:
        return []
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    listed = []
    for p in items:
        kind, viewable, _ = content_type(p.name)
        listed.append({
            "name": p.name,
            "size": p.stat().st_size,
            "saved_at": datetime.fromtimestamp(
                p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "type": kind,
            "viewable": viewable,
        })
    return listed


def find(chat_id: int, name: str) -> Path | None:
    """Путь к файлу диалога или None.

    Имя прогоняется через safe_name ещё раз: даже если в запросе пришло
    «../app.db», искать будут файл «app.db» внутри каталога диалога.
    """
    path = folder(chat_id) / safe_name(name)
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def delete(chat_id: int, name: str) -> bool:
    path = find(chat_id, name)
    if path is None:
        return False
    path.unlink()
    return True


def delete_chat_files(chat_id: int) -> int:
    """Убирает файлы вместе с диалогом: без него им негде показываться."""
    path = folder(chat_id)
    if not path.exists():
        return 0
    removed = 0
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
            removed += 1
    path.rmdir()
    return removed
