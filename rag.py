"""Индекс документов: разбиение на куски, эмбеддинги, метаданные.

Фундамент для поиска по своим документам. Здесь только индексация: корпус
читается с диска, режется на куски двумя способами, каждый кусок получает
вектор и метаданные и ложится в SQLite. Поиск тоже здесь, но пока как
инструмент проверки, а не как часть чата.

Почему именно так устроено:

**Эмбеддинги статические.** У DeepSeek эмбеддингов нет вовсе, а обычный путь
через onnxruntime закрыт: под Python 3.14 его не собирают. `model2vec` — это
таблица «строка на токен» (500 353 × 256 чисел), и эмбеддинг текста считается
усреднением строк его токенов. Ни матричных умножений, ни GPU, зависимости —
numpy и токенизатор. Платим качеством: статические векторы слабее
трансформерных, зато работают везде и мгновенно.

**Без FAISS.** На корпусе в полторы тысячи кусков полный перебор в numpy
занимает доли миллисекунды — меньше, чем накладные расходы индексных структур.
Тащить зависимость ради слова «векторная база» тут не за чем.

**Две стратегии разбиения** сравниваются на замерах, а не на ощущениях:
по фиксированному размеру и по структуре документа.

    python rag.py build                # собрать индекс обеими стратегиями
    python rag.py search "вопрос"      # что находится
    python rag.py compare              # сравнение стратегий
    python rag.py stats                # что лежит в индексе
"""

import argparse
import ast
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import db
import tokens as tokens_mod

# База индекса лежит рядом с базой переписки — тем же правилом, что файлы
# агента и очередь заданий: один путь уже отличает машину от сервера.
DB_PATH = Path(os.getenv("RAG_DB_PATH") or db.DB_PATH.resolve().parent / "rag.db")

# Многоязычная: корпус русский, английская модель на нём бесполезна.
MODEL_NAME = os.getenv("RAG_MODEL", "minishlab/potion-multilingual-128M")

FIXED, STRUCTURE, TITLED = "fixed", "structure", "titled"
STRATEGIES = (FIXED, STRUCTURE, TITLED)

# Куски: 1200 символов это примерно 350–500 токенов русского текста — столько
# не жалко положить в запрос по три штуки. Перекрытие нужно ровно затем, чтобы
# ответ, попавший на стык, нашёлся хотя бы в одном куске.
CHUNK_CHARS = 1200
OVERLAP_CHARS = 200

# Границы структурных кусков: раздел длиннее дорезается, короче — склеивается
# со следующим, иначе индекс засоряется огрызками по две строки.
MAX_CHUNK_CHARS = 2400
MIN_CHUNK_CHARS = 200

# Что индексируем. Каталоги с чужим кодом и данными не трогаем вовсе.
INCLUDE = ("README.md", "*.py", "static/*.js", "deploy/*.service")
EXCLUDE_NAMES = {".env", ".env.example", "DEPLOY.local.md"}
EXCLUDE_DIRS = {".venv", ".git", "files", "__pycache__", ".claude"}


@dataclass
class Chunk:
    """Кусок документа вместе с тем, откуда он взялся."""

    strategy: str
    source: str        # путь к файлу относительно корня корпуса
    title: str         # имя файла или заголовок статьи
    section: str       # путь заголовков или имя функции
    ordinal: int
    text: str
    line_from: int
    line_to: int

    @property
    def chunk_id(self) -> str:
        return f"{self.strategy}:{self.source}#{self.ordinal}"

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def tokens(self) -> int:
        return tokens_mod.estimate_tokens(self.text)


@dataclass
class Document:
    path: Path
    source: str
    text: str

    @property
    def kind(self) -> str:
        if self.source.endswith(".md"):
            return "markdown"
        if self.source.endswith(".py"):
            return "python"
        return "plain"


# ---------- корпус ----------

def read_corpus(root: Path) -> list[Document]:
    """Собирает файлы корпуса. Исключения — не украшение, а защита.

    `.env` и локальные заметки о развёртывании в индекс попадать не должны:
    иначе поиск станет способом прочитать то, что нарочно не в репозитории.
    """
    found: dict[str, Path] = {}
    for pattern in INCLUDE:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            if path.name in EXCLUDE_NAMES:
                continue
            if set(path.relative_to(root).parts) & EXCLUDE_DIRS:
                continue
            found[str(path.relative_to(root))] = path

    documents = []
    for source, path in sorted(found.items()):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if text.strip():
            documents.append(Document(path=path, source=source, text=text))
    return documents


def _line_of(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


# ---------- стратегия 1: по фиксированному размеру ----------

def _cut_point(text: str, start: int, target: int) -> int:
    """Ближайшая приличная граница около цели: абзац, строка, пробел.

    Резать ровно по счётчику символов значит рвать слова и строки кода
    посередине. Отступаем назад в пределах четверти куска.
    """
    if target >= len(text):
        return len(text)
    window = max(start + 1, target - CHUNK_CHARS // 4)
    for separator in ("\n\n", "\n", ". ", " "):
        found = text.rfind(separator, window, target)
        if found > start:
            return found + len(separator)
    return target


def chunk_fixed(doc: Document) -> list[Chunk]:
    """Режет текст окнами по CHUNK_CHARS с перекрытием."""
    chunks: list[Chunk] = []
    position, ordinal = 0, 0
    while position < len(doc.text):
        end = _cut_point(doc.text, position, position + CHUNK_CHARS)
        piece = doc.text[position:end].strip()
        if piece:
            chunks.append(Chunk(
                strategy=FIXED, source=doc.source, title=doc.path.name,
                section=f"символы {position}–{end}", ordinal=ordinal,
                text=piece,
                line_from=_line_of(doc.text, position),
                line_to=_line_of(doc.text, max(position, end - 1)),
            ))
            ordinal += 1
        if end >= len(doc.text):
            break
        position = max(position + 1, end - OVERLAP_CHARS)
    return chunks


# ---------- стратегия 2: по структуре ----------

def _split_long(text: str) -> list[str]:
    """Дорезает слишком длинный структурный кусок по размеру."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts, position = [], 0
    while position < len(text):
        end = _cut_point(text, position, position + MAX_CHUNK_CHARS)
        parts.append(text[position:end].strip())
        position = end
    return [p for p in parts if p]


def _emit(collected: list[Chunk], doc: Document, section: str,
          text: str, line_from: int, line_to: int) -> None:
    """Кладёт кусок, разрезав слишком длинный и пометив части.

    У частей считаются свои строки: одинаковый диапазон на всех частях врал бы
    ровно там, где метаданные и нужны — при проверке, откуда взялся ответ.
    """
    body = text.strip()
    if not body:
        return
    parts = _split_long(body)
    offset = 0
    for number, part in enumerate(parts, 1):
        mark = f" (часть {number} из {len(parts)})" if len(parts) > 1 else ""
        first = line_from + offset
        offset += part.count("\n") + 1
        collected.append(Chunk(
            strategy=STRUCTURE, source=doc.source, title=doc.path.name,
            section=section + mark, ordinal=len(collected),
            text=part, line_from=first,
            line_to=min(line_to, first + part.count("\n")),
        ))


def chunk_markdown(doc: Document) -> list[Chunk]:
    """Режет статью по заголовкам, храня их путь в section."""
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal buffer
        body = "\n".join(buffer)
        if len(body.strip()) >= MIN_CHUNK_CHARS:
            path = " / ".join(name for _, name in stack) or doc.path.name
            _emit(chunks, doc, path, body, start_line, end_line)
            buffer = []
        # Короткий раздел не выбрасываем и не плодим: он дописывается к
        # следующему — огрызок в две строки всё равно ничего не находит.

    for number, line in enumerate(doc.text.splitlines(), 1):
        if line.startswith("#") and " " in line[:7]:
            level = len(line) - len(line.lstrip("#"))
            flush(number - 1)
            if not buffer:
                start_line = number
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, line.lstrip("# ").strip()))
        buffer.append(line)

    flush(len(doc.text.splitlines()))
    if buffer and "\n".join(buffer).strip():
        path = " / ".join(name for _, name in stack) or doc.path.name
        _emit(chunks, doc, path, "\n".join(buffer), start_line,
              len(doc.text.splitlines()))
    return chunks


def chunk_python(doc: Document) -> list[Chunk]:
    """Режет модуль по определениям: докстринг, функции, классы.

    Код между определениями не пропускается: в этом проекте самое ценное —
    длинные константы с промптами (STAGE_RULES, JUDGE_SYSTEM), и потерять их
    значило бы выбросить половину смысла.
    """
    try:
        tree = ast.parse(doc.text)
    except SyntaxError:
        return chunk_fixed(doc)

    lines = doc.text.splitlines()
    chunks: list[Chunk] = []
    covered: list[tuple[int, int, str]] = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        kind = "класс" if isinstance(node, ast.ClassDef) else "функция"
        covered.append((start, node.end_lineno, f"{kind} {node.name}"))

    covered.sort()
    segments: list[tuple[str, int, int]] = []
    position = 1
    for start, end, name in covered:
        if start > position:
            segments.append(("уровень модуля", position, start - 1))
        segments.append((name, start, end))
        position = end + 1
    if position <= len(lines):
        segments.append(("уровень модуля", position, len(lines)))

    # Мелочь склеивается с соседом: кусок в две строки ничего не находит, но
    # место в индексе занимает и разбавляет выдачу.
    for name, start, end in segments:
        text = "\n".join(lines[start - 1:end])
        if not text.strip():
            continue
        if chunks and len(text) < MIN_CHUNK_CHARS and chunks[-1].source == doc.source:
            previous = chunks[-1]
            merged = previous.text + "\n" + text
            if len(merged) <= MAX_CHUNK_CHARS:
                chunks[-1] = Chunk(
                    strategy=STRUCTURE, source=doc.source, title=doc.path.name,
                    section=previous.section, ordinal=previous.ordinal,
                    text=merged, line_from=previous.line_from, line_to=end,
                )
                continue
        _emit(chunks, doc, name, text, start, end)
    return chunks


def chunk(doc: Document, strategy: str) -> list[Chunk]:
    """Куски по выбранной стратегии.

    `titled` — та же структурная разбивка, но в начало куска дописывается
    строка «файл · раздел». Затея не косметическая: вопрос «что делает
    safe_name» содержит имя функции, а в тексте самой функции оно встречается
    один раз среди сотни слов — усреднённый вектор его почти не замечает.
    Заголовок возвращает имени вес и заодно виден модели, когда кусок уедет
    в запрос.
    """
    if strategy == FIXED:
        return chunk_fixed(doc)

    if doc.kind == "markdown":
        pieces = chunk_markdown(doc)
    elif doc.kind == "python":
        pieces = chunk_python(doc)
    else:
        pieces = [Chunk(**{**vars(c), "strategy": STRUCTURE}) for c in chunk_fixed(doc)]

    if strategy == STRUCTURE:
        return pieces
    return [Chunk(**{**vars(c), "strategy": TITLED,
                     "text": f"{c.source} · {c.section}\n\n{c.text}"})
            for c in pieces]


# ---------- эмбеддинги ----------

_model = None


def model():
    """Модель грузится один раз и лениво: 489 МБ читать зря незачем."""
    global _model
    if _model is None:
        from model2vec import StaticModel
        _model = StaticModel.from_pretrained(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Векторы, нормированные на единицу: тогда косинус — скалярное произведение."""
    vectors = np.asarray(model().encode(texts), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


# ---------- хранилище ----------

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  TEXT PRIMARY KEY,
    strategy  TEXT    NOT NULL,
    source    TEXT    NOT NULL,
    title     TEXT    NOT NULL,
    section   TEXT    NOT NULL,
    ordinal   INTEGER NOT NULL,
    line_from INTEGER NOT NULL,
    line_to   INTEGER NOT NULL,
    chars     INTEGER NOT NULL,
    tokens    INTEGER NOT NULL,
    text      TEXT    NOT NULL,
    -- Вектор лежит рядом с куском: отдельная таблица ничего бы не дала,
    -- читаем их всегда вместе.
    vector    BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy, source);

-- Поиск словами по тем же кускам. Нужен для сравнения: векторы хорошо
-- находят прозу и плохо — код, где вопрос содержит точное имя функции.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, chunk_id UNINDEXED, tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def save(chunks: list[Chunk], vectors: np.ndarray) -> None:
    with connect() as conn:
        if chunks and chunks[0].strategy == STRUCTURE:
            conn.executemany(
                "INSERT INTO chunks_fts (text, chunk_id) VALUES (?, ?)",
                [(f"{c.source} {c.section}\n{c.text}", c.chunk_id) for c in chunks])
        conn.executemany(
            "INSERT OR REPLACE INTO chunks (chunk_id, strategy, source, title,"
            " section, ordinal, line_from, line_to, chars, tokens, text, vector)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(c.chunk_id, c.strategy, c.source, c.title, c.section, c.ordinal,
              c.line_from, c.line_to, c.chars, c.tokens, c.text,
              vectors[i].astype(np.float32).tobytes())
             for i, c in enumerate(chunks)],
        )


def load_index(strategy: str) -> tuple[list[dict], np.ndarray]:
    """Все куски стратегии и матрица их векторов."""
    with connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM chunks WHERE strategy = ? ORDER BY source, ordinal",
            (strategy,))]
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.vstack([np.frombuffer(r.pop("vector"), dtype=np.float32) for r in rows])
    return rows, matrix


# ---------- сборка и поиск ----------

def build(root: Path, strategies=STRATEGIES) -> dict:
    """Полная пересборка индекса. Дешевле, чем следить за изменениями."""
    init()
    documents = read_corpus(root)
    report = {"documents": len(documents),
              "chars": sum(len(d.text) for d in documents), "strategies": {}}

    with connect() as conn:
        conn.execute("DELETE FROM chunks WHERE strategy IN (%s)"
                     % ",".join("?" * len(strategies)), tuple(strategies))
        conn.execute("DELETE FROM chunks_fts")

    for strategy in strategies:
        started = time.monotonic()
        pieces: list[Chunk] = []
        for doc in documents:
            pieces.extend(chunk(doc, strategy))
        cut = time.monotonic() - started

        started = time.monotonic()
        vectors = embed([c.text for c in pieces])
        embedded = time.monotonic() - started
        save(pieces, vectors)

        report["strategies"][strategy] = {
            "chunks": len(pieces),
            "chars_avg": round(sum(c.chars for c in pieces) / max(len(pieces), 1)),
            "tokens_avg": round(sum(c.tokens for c in pieces) / max(len(pieces), 1)),
            "cut_seconds": round(cut, 2),
            "embed_seconds": round(embedded, 2),
        }

    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                     ("model", MODEL_NAME))
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                     ("built_at", time.strftime("%Y-%m-%d %H:%M:%S")))
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                     ("root", str(root)))
    with connect() as conn:
        # Без контрольной точки данные лежат в файле журнала, и размер базы
        # показал бы четыре килобайта вместо настоящего.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    report["db_bytes"] = DB_PATH.stat().st_size
    return report


def search(query: str, strategy: str = STRUCTURE, limit: int = 5) -> list[dict]:
    """Ближайшие куски. Перебором: на нашем объёме это доли миллисекунды."""
    rows, matrix = load_index(strategy)
    if not rows:
        return []
    vector = embed([query])[0]
    scores = matrix @ vector
    best = np.argsort(-scores)[:limit]
    return [{**rows[i], "score": float(scores[i])} for i in best]


def search_lexical(query: str, limit: int = 5) -> list[dict]:
    """Поиск словами по структурным кускам, ранжирование bm25.

    Стеммера для русского в FTS5 нет, так что совпадают только точные формы
    слов. Именно поэтому он силён там, где вектор слаб: в вопросе про
    `safe_name` имя написано ровно так же, как в коде.
    """
    words = [w for w in re.findall(r"[\w_]+", query.lower()) if len(w) > 2]
    if not words:
        return []
    match = " OR ".join(f'"{w}"' for w in words)
    with connect() as conn:
        try:
            rows = conn.execute(
                "SELECT c.*, bm25(chunks_fts) AS rank FROM chunks_fts"
                " JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id"
                " WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
    return [{**dict(r), "score": -float(r["rank"])} for r in rows]


# ---------- сравнение стратегий ----------
#
# Вопросы к собственному проекту, у каждого известен файл, где ответ есть на
# самом деле. Эталон проверялся глазами: это не «похоже на правду», а «здесь
# про это и написано».

# Вопросы двух родов, и это важно для честности замера.
#
# «Документация» — то, что рассказано в README своими словами. «Реализация» —
# то, чего в README нет вовсе: имена констант, таблиц и функций, проверено
# поиском по тексту. Без такого разделения сравнение было бы бессмысленным:
# README на 115 КБ прозы говорит о тех же вещах теми же словами, что и
# вопросы, и перетягивает почти любую выдачу на себя.
DOCS, CODE = "документация", "реализация"

QUESTIONS = [
    ("чем пауза отличается от ожидания ответа задачи", {"README.md", "task.py"}, DOCS),
    ("сколько символов приходится на токен в русском тексте", {"README.md", "tokens.py"}, DOCS),
    ("во сколько токенов обошлась цепочка инструментов", {"README.md"}, DOCS),
    ("зачем html открывается в песочнице", {"README.md"}, DOCS),
    ("почему нельзя проверять нарушение инварианта по ключевым словам", {"README.md", "invariants.py"}, DOCS),
    ("как автопилот отвечает за пользователя", {"README.md", "task.py"}, DOCS),

    ("сколько ждём первый кусок потокового ответа", {"llm.py"}, CODE),
    ("в какой таблице лежат отметки о смене этапа задачи", {"db.py"}, CODE),
    ("как код погоды превращается в слово", {"weather_mcp.py"}, CODE),
    ("что делает safe_name с именем файла", {"files_store.py"}, CODE),
    ("как отклонённая просьба выглядит в строке для служебного вызова", {"history.py"}, CODE),
    ("сколько кругов вызова инструментов разрешено в одном обмене", {"web.py"}, CODE),
]


LEXICAL = "structure+слова"


def compare(limit: int = 3, detail: bool = False) -> dict:
    """Одинаковые вопросы обеим стратегиям, метрики рядом."""
    # Модель грузится заранее: иначе её загрузка попала бы в замер времени
    # первой стратегии и та выглядела бы в тысячу раз медленнее второй.
    model()
    outcome = {}
    for strategy in STRATEGIES:
        rows, matrix = load_index(strategy)
        if not rows:
            continue
        hits1 = hits3 = 0
        by_group: dict[str, list[int]] = {DOCS: [], CODE: []}
        ranks, sizes, token_costs = [], [], []
        started = time.monotonic()
        for question, expected, group in QUESTIONS:
            vector = embed([question])[0]
            scores = matrix @ vector
            best = np.argsort(-scores)[:limit]
            sources = [rows[i]["source"] for i in best]
            found_at = next((n for n, s in enumerate(sources, 1) if s in expected), 0)
            hits1 += sources[0] in expected
            hits3 += bool(found_at)
            ranks.append(found_at)
            by_group[group].append(found_at)
            sizes.extend(rows[i]["chars"] for i in best)
            token_costs.append(sum(rows[i]["tokens"] for i in best))
            if detail:
                mark = f"место {found_at}" if found_at else "НЕ НАЙДЕНО"
                print(f"  [{strategy:<9}] {mark:<11} {question[:46]:<46} "
                      f"→ {', '.join(sources)}")
        found = [r for r in ranks if r]
        outcome[strategy] = {
            "chunks": len(rows),
            "hit1": hits1, "hit3": hits3, "total": len(QUESTIONS),
            "rank_avg": round(sum(found) / len(found), 2) if found else None,
            "docs_hit3": sum(1 for r in by_group[DOCS] if r),
            "docs_total": len(by_group[DOCS]),
            "code_hit3": sum(1 for r in by_group[CODE] if r),
            "code_total": len(by_group[CODE]),
            "chunk_chars_avg": round(sum(sizes) / len(sizes)),
            "top_tokens_avg": round(sum(token_costs) / len(token_costs)),
            "search_ms": round((time.monotonic() - started) * 1000 / len(QUESTIONS), 2),
        }

    # Та же выдача, но найденная словами — чтобы было видно, чего стоят
    # векторы там, где вопрос содержит точное имя из кода.
    hits1 = hits3 = 0
    by_group = {DOCS: [], CODE: []}
    ranks, sizes, token_costs = [], [], []
    started = time.monotonic()
    for question, expected, group in QUESTIONS:
        best = search_lexical(question, limit)
        sources = [r["source"] for r in best]
        found_at = next((n for n, s in enumerate(sources, 1) if s in expected), 0)
        hits1 += bool(sources) and sources[0] in expected
        hits3 += bool(found_at)
        ranks.append(found_at)
        by_group[group].append(found_at)
        sizes.extend(r["chars"] for r in best)
        token_costs.append(sum(r["tokens"] for r in best))
        if detail:
            mark = f"место {found_at}" if found_at else "НЕ НАЙДЕНО"
            print(f"  [{LEXICAL:<9}] {mark:<11} {question[:46]:<46} "
                  f"→ {', '.join(sources) or 'пусто'}")
    found = [r for r in ranks if r]
    if sizes:
        outcome[LEXICAL] = {
            "chunks": outcome.get(STRUCTURE, {}).get("chunks", 0),
            "hit1": hits1, "hit3": hits3, "total": len(QUESTIONS),
            "rank_avg": round(sum(found) / len(found), 2) if found else None,
            "docs_hit3": sum(1 for r in by_group[DOCS] if r),
            "docs_total": len(by_group[DOCS]),
            "code_hit3": sum(1 for r in by_group[CODE] if r),
            "code_total": len(by_group[CODE]),
            "chunk_chars_avg": round(sum(sizes) / len(sizes)),
            "top_tokens_avg": round(sum(token_costs) / len(token_costs)),
            "search_ms": round((time.monotonic() - started) * 1000 / len(QUESTIONS), 2),
        }
    return outcome


# ---------- командная строка ----------

def _print_build(report: dict) -> None:
    print(f"Документов: {report['documents']}, "
          f"символов: {report['chars']:,}".replace(",", " "))
    for strategy, numbers in report["strategies"].items():
        print(f"\n  {strategy}: кусков {numbers['chunks']}, "
              f"в среднем {numbers['chars_avg']} символов "
              f"(~{numbers['tokens_avg']} токенов)")
        print(f"    разбиение {numbers['cut_seconds']} с, "
              f"эмбеддинги {numbers['embed_seconds']} с")
    print(f"\nБаза: {DB_PATH} — {report['db_bytes'] / 1024:.0f} КБ")


def _print_search(results: list[dict]) -> None:
    if not results:
        print("Ничего не найдено — возможно, индекс ещё не собран.")
        return
    for number, row in enumerate(results, 1):
        print(f"\n{number}. {row['score']:.3f}  {row['source']} "
              f"строки {row['line_from']}–{row['line_to']}")
        print(f"   раздел: {row['section']}")
        snippet = " ".join(row["text"].split())[:200]
        print(f"   {snippet}…")


def _print_compare(outcome: dict) -> None:
    if not outcome:
        print("Индекс пуст — сначала build.")
        return
    print(f"{'':<12}{'кусков':>8}{'топ-1':>8}{'топ-3':>8}{'доки':>8}{'код':>8}"
          f"{'ранг':>7}{'символов':>10}{'токенов':>9}{'мс':>7}")
    for strategy, n in outcome.items():
        print(f"{strategy:<12}{n['chunks']:>8}{n['hit1']:>5}/{n['total']:<2}"
              f"{n['hit3']:>5}/{n['total']:<2}"
              f"{n['docs_hit3']:>5}/{n['docs_total']:<2}"
              f"{n['code_hit3']:>5}/{n['code_total']:<2}"
              f"{str(n['rank_avg']):>7}{n['chunk_chars_avg']:>10}"
              f"{n['top_tokens_avg']:>9}{n['search_ms']:>7}")
    print("\nтоп-1 и топ-3 — в скольких вопросах нужный источник оказался первым "
          "и в тройке;\nдоки и код — те же попадания в тройку отдельно по "
          "вопросам о документации\nи о реализации; ранг — средняя позиция "
          "верного куска; токенов — цена выдачи\nтройки в запросе к модели.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Индекс документов для поиска")
    commands = parser.add_subparsers(dest="command", required=True)

    builder = commands.add_parser("build", help="собрать индекс")
    builder.add_argument("--path", default=".", help="корень корпуса")

    finder = commands.add_parser("search", help="найти куски по вопросу")
    finder.add_argument("query")
    finder.add_argument("--strategy", choices=STRATEGIES, default=STRUCTURE)
    finder.add_argument("--limit", type=int, default=5)
    finder.add_argument("--words", action="store_true",
                        help="искать словами (bm25) вместо векторов")

    comparer = commands.add_parser("compare", help="сравнить стратегии разбиения")
    comparer.add_argument("--detail", action="store_true",
                          help="показать каждый вопрос отдельно")
    commands.add_parser("stats", help="что лежит в индексе")

    args = parser.parse_args()
    if args.command == "build":
        _print_build(build(Path(args.path).resolve()))
    elif args.command == "search":
        _print_search(search_lexical(args.query, args.limit) if args.words
                      else search(args.query, args.strategy, args.limit))
    elif args.command == "compare":
        _print_compare(compare(detail=args.detail))
    else:
        with connect() as conn:
            meta = {r["key"]: r["value"] for r in conn.execute("SELECT * FROM meta")}
            print("Модель:", meta.get("model", "—"))
            print("Собран:", meta.get("built_at", "—"), "из", meta.get("root", "—"))
            print()
            for row in conn.execute(
                "SELECT strategy, count(*) n, sum(chars) c, count(DISTINCT source) f"
                " FROM chunks GROUP BY strategy"
            ):
                print(f"  {row['strategy']:<12} кусков {row['n']:>5}, "
                      f"файлов {row['f']:>3}, символов {row['c']:,}".replace(",", " "))


if __name__ == "__main__":
    main()
