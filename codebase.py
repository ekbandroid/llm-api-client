"""Репозиторий, над которым работает ассистент: файлы из git и состояние git.

Отдельный модуль, а не часть rag.py, потому что здесь другая забота. rag.py
знает, как резать текст и считать векторы; здесь — что считать текстом проекта
и откуда брать его состояние. Имя не `project.py` намеренно: в приложении уже
есть «проекты» (таблица projects — контейнер памяти с брифом и фактами), и два
разных смысла под одним словом в коде путали бы.

Файлы берутся из git, а не обходом файловой системы. Разница не в удобстве:
`INCLUDE` в rag.py включает `*.py` в корне, и любой неотслеживаемый файл,
оказавшийся рядом, попал бы в индекс — а индекс это способ прочитать
содержимое. `git ls-files` отдаёт только то, что в репозитории, и соблюдает
.gitignore сам собой. Сверка на этом проекте: обход даёт 32 файла, из них вне
git — ни одного, то есть сегодня разницы нет, и она появится ровно в тот день,
когда кто-то положит рядом файл с выгрузкой.

    python -m codebase          # что попадёт в индекс и что скажет git
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Корень репозитория. Из переменной, а не от того, кто зовёт: инструмент,
# которому корень приносит модель, читал бы любой репозиторий на машине.
ROOT = Path(os.getenv("PROJECT_ROOT") or Path(__file__).resolve().parent).resolve()

# Названия наборов. По ним же команда находит свои наборы при пересборке,
# поэтому менять их нельзя без переиндексации.
DOCS_TITLE = "Документация проекта"
CODE_TITLE = "Код проекта"
# Вопросы и ответы поддержки — отдельным набором, а не внутри документации.
# Причина из дня 31: на ранжирование разделение не влияет, но в заголовке
# выдержки видно, откуда пришёл ответ — из коротких вопросов-ответов или из
# README на двести килобайт. Поддержке это важнее, чем разработчику.
SUPPORT_TITLE = "Поддержка: вопросы и ответы"
SUPPORT_GLOB = "support/*.md"

# Что считаем документацией. Остальное из отобранного — код.
DOCS_SUFFIXES = {".md", ".rst", ".txt"}

# Подкоманды git, которые нам разрешены. Список, а не «любой git, который
# попросят»: инструментом распоряжается модель, и `git push --force` в нём
# быть не должно даже теоретически.
ALLOWED = ("rev-parse", "status", "diff", "log", "ls-files", "rev-list", "show")

# Сколько ждём git. Секунды хватает всему, кроме diff на огромном дереве.
TIMEOUT = 20


class GitError(RuntimeError):
    """git не смог. Текст пригоден для показа человеку и модели."""


def git(*args: str, timeout: int = TIMEOUT) -> str:
    """Запускает git в корне проекта и отдаёт stdout.

    Без shell=True и без строки-команды: аргументы списком, поэтому ни пробел
    в пути, ни точка с запятой в имени файла ничего не значат.
    """
    if not args or args[0] not in ALLOWED:
        raise GitError(f"подкоманда {args[0] if args else '—'} не разрешена")
    try:
        готово = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as err:
        raise GitError("git не найден на этой машине") from err
    except subprocess.TimeoutExpired as err:
        raise GitError(f"git {args[0]} не ответил за {timeout} с") from err
    if готово.returncode != 0:
        raise GitError((готово.stderr or готово.stdout).strip()[:300]
                       or f"git {args[0]} вернул {готово.returncode}")
    return готово.stdout


def is_repo() -> bool:
    """Это git-репозиторий?

    Проверять приходится: на сервере кода нет истории вовсе — он кладётся
    `git archive`'ом, без .git. Инструменты должны сказать об этом словами, а
    не упасть с исключением на ровном месте.
    """
    try:
        return git("rev-parse", "--is-inside-work-tree", timeout=5).strip() == "true"
    except GitError:
        return False


def inside(path: str) -> Path:
    """Путь от модели — внутрь корня, и никуда больше.

    Проверяем после resolve(), а не поиском «..» в строке: ссылка на каталог
    выше не содержит «..» вовсе, а уводит так же.
    """
    цель = (ROOT / path).resolve()
    if цель != ROOT and ROOT not in цель.parents:
        raise GitError(f"путь {path} вне проекта")
    if not цель.exists():
        raise GitError(f"пути {path} в проекте нет")
    return цель


def tracked_files() -> list[str]:
    """Файлы репозитория, отобранные теми же правилами, что корпус rag.

    Правила берём из rag, а не повторяем: там они написаны и там же объяснено,
    почему `.env` исключён. Импорт внутри функции — чтобы сервер MCP, которому
    нужен только git, не тащил за собой numpy и машинерию индексации.
    """
    import rag

    отслеженные = [s for s in git("ls-files", "-z").split("\0") if s]
    отобранные = []
    for путь in отслеженные:
        дорожка = Path(путь)
        if дорожка.name in rag.EXCLUDE_NAMES:
            continue
        if set(дорожка.parts) & rag.EXCLUDE_DIRS:
            continue
        # Path.match сверяет с конца пути: «*.py» подходит файлу на любой
        # глубине, «static/*.js» — только внутри static. Для ассистента это
        # то, что нужно, и это не совпадает с Path.glob в rag.read_corpus —
        # оговорка стоит рядом с самим INCLUDE.
        if any(дорожка.match(образец) for образец in rag.INCLUDE):
            отобранные.append(путь)
    return sorted(отобранные)


def split_files(paths: list[str]) -> dict[str, list[str]]:
    """Раскладывает отобранное по наборам: поддержка, документация, код."""
    поддержка = [p for p in paths if Path(p).match(SUPPORT_GLOB)]
    осталось = [p for p in paths if p not in set(поддержка)]
    docs = [p for p in осталось if Path(p).suffix.lower() in DOCS_SUFFIXES]
    code = [p for p in осталось if p not in set(docs)]
    return {DOCS_TITLE: docs, CODE_TITLE: code, SUPPORT_TITLE: поддержка}


def flat_name(path: str) -> str:
    """Путь в имя файла: static/common.js → static__common.js.

    Папка набора плоская, и одноимённые файлы из разных папок затирали бы друг
    друга. Имя файла — это то, что модель называет как источник в ответе,
    поэтому оно должно говорить, откуда кусок, а не только как он назван.
    """
    return path.replace("/", "__")


def build_collections(user_id: int, *, model: str | None = None,
                      docs_strategy: str | None = None,
                      code_strategy: str | None = None) -> dict:
    """Собирает два набора по файлам репозитория. Идемпотентно.

    Свои наборы находит по названию и пересобирает их на месте, а не плодит
    копии: иначе каждая пересборка после правок добавляла бы ещё одну пару, и
    через неделю в списке было бы десять «Кодов проекта».
    """
    import rag

    if not is_repo():
        raise GitError(f"{ROOT} — не git-репозиторий, брать список файлов негде")

    по_наборам = split_files(tracked_files())
    if not any(по_наборам.values()):
        raise GitError("под правила rag.INCLUDE не подошёл ни один файл репозитория")

    отчёт = {"root": str(ROOT), "collections": []}
    стратегии = {DOCS_TITLE: docs_strategy or rag.TITLED,
                 CODE_TITLE: code_strategy or rag.STRUCTURE,
                 # Вопросы и ответы размечены заголовками, и заголовок внутри
                 # куска здесь важен: без него «Что проверить» теряет вопрос.
                 SUPPORT_TITLE: rag.TITLED}
    for title, пути in по_наборам.items():
        стратегия = стратегии[title]
        if not пути:
            continue
        набор = _collection_by_title(user_id, title)
        if набор is None:
            набор = rag.create_collection(
                user_id, title=title, model=model or rag.DEFAULT_MODEL,
                strategy=стратегия)
        elif набор["strategy"] != стратегия:
            rag.set_shape(набор["id"], strategy=стратегия)

        папка = rag.collection_dir(набор["id"], create=True)
        # Папку чистим: файл, удалённый из репозитория, иначе остался бы в
        # наборе навсегда — ровно та беда, от которой в deploy.sh сверка
        # состава с git ls-files.
        for лишний in папка.iterdir():
            if лишний.is_file():
                лишний.unlink()
        for путь in пути:
            (папка / flat_name(путь)).write_bytes((ROOT / путь).read_bytes())

        начало = time.monotonic()
        итог = rag.index_collection(набор["id"])
        отчёт["collections"].append({
            "id": набор["id"], "title": title, "strategy": стратегия,
            "files": len(пути), **итог,
            "seconds": round(time.monotonic() - начало, 2),
        })
    return отчёт


def _collection_by_title(user_id: int, title: str) -> dict | None:
    """Набор этого пользователя с таким названием, если он уже есть."""
    import rag

    for набор in rag.list_collections(user_id):
        if набор["title"] == title:
            return набор
    return None


def state() -> dict:
    """Состояние репозитория одним словарём — для инструментов MCP."""
    if not is_repo():
        return {"repo": False, "root": str(ROOT)}
    ветка = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    отметки = [s for s in git("status", "--porcelain").splitlines() if s.strip()]
    return {"repo": True, "root": str(ROOT), "branch": ветка,
            "changed": len(отметки), "lines": отметки}


# Диалог, к которому замер подключает наборы. Не настоящий: номер взят вне
# диапазона реальных, как 900 у rag_compare.
MEASURE_CHAT = 901


def measure(path: str, *, limit: int = 3, pool: int = 30, only: str = "") -> None:
    """hit@1 и hit@3 по файлу вопросов, через настоящий путь поиска.

    Формат строки: «вопрос | файл, ещё файл | группа». Группа произвольна —
    по ней считается отдельный итог, и смысл в том, чтобы отделить вопросы по
    документации от вопросов по реализации: README на 202 КБ говорит о тех же
    вещах теми же словами, что и вопросы, и без разделения средний результат
    скрывал бы, что поиск по коду работает хуже.

    Файл вопросов **намеренно вне репозитория**, и это не вкусовщина. Прежний
    набор `rag.QUESTIONS` лежит внутри rag.py, то есть внутри индекса, и все
    12 его вопросов нашлись в кусках дословно: кусок с ними набрал близость
    17,66 против 0,57 у README. Индексируя свой репозиторий, индексируешь и
    свой тестовый набор — измерять им нечего.
    """
    import rag

    вопросы = []
    for строка in open(path, encoding="utf-8"):
        строка = строка.strip()
        if not строка or строка.startswith("#"):
            continue
        части = [ч.strip() for ч in строка.split("|")]
        if len(части) < 2 or not части[0]:
            continue
        ожидаемые = {ф.strip() for ф in части[1].split(",") if ф.strip()}
        группа = части[2] if len(части) > 2 and части[2] else "без группы"
        вопросы.append((части[0], ожидаемые, группа))
    if not вопросы:
        raise SystemExit(f"в {path} нет вопросов")

    # only="code" — замер без README. Это и есть обещание двух наборов:
    # когда спрашиваешь про реализацию, документацию можно отцепить.
    нужны = {"docs": (DOCS_TITLE,), "code": (CODE_TITLE,)}.get(only,
                                                               (DOCS_TITLE, CODE_TITLE))
    for название in (DOCS_TITLE, CODE_TITLE):
        набор = _collection_by_title_any(название)
        if набор:
            rag.attach(MEASURE_CHAT, набор["id"], название in нужны)
    подключены = rag.attached_collections(MEASURE_CHAT)
    if not подключены:
        raise SystemExit("наборы проекта не собраны: manage.py index-project <логин>")
    print(f"наборы: {', '.join(c['title'] for c in подключены)}")

    rag.model()  # загрузка заранее: иначе попала бы в замер первого вопроса
    по_группам: dict[str, list[int]] = {}
    загрязнены = 0
    print(f"\n{'группа':<14} {'место':<6} вопрос")
    for вопрос, ожидаемые, группа in вопросы:
        # Оба этапа, как в search_docs: сперва широко по пулу, потом отсев
        # кусков без слов вопроса и склейка соседей. Мерить один первый этап
        # значило бы мерить то, чего приложение не делает.
        найдено = rag.search_collections([вопрос], MEASURE_CHAT, pool)
        найдено = rag.keep_wordy(найдено, [вопрос])
        найдено = rag.merge_neighbours(найдено)[:limit]
        источники = [h["source"] for h in найдено]
        место = next((n for n, s in enumerate(источники, 1) if s in ожидаемые), 0)
        по_группам.setdefault(группа, []).append(место)
        # Вопрос, лежащий в индексе дословно, измеряет не поиск, а сам себя.
        if any(вопрос in h["text"] for h in найдено):
            загрязнены += 1
            источники[0] += " (ВОПРОС В ИНДЕКСЕ)"
        print(f"{группа:<14} {str(место) if место else 'НЕТ':<6} "
              f"{вопрос[:50]:<50} → {', '.join(источники[:limit])}")

    print()
    всё: list[int] = []
    for группа, места in по_группам.items():
        всё += места
        n = len(места)
        print(f"  {группа:<14} hit@1 {sum(1 for m in места if m == 1)}/{n}  "
              f"hit@{limit} {sum(1 for m in места if m)}/{n}")
    print(f"  {'всего':<14} hit@1 {sum(1 for m in всё if m == 1)}/{len(всё)}  "
          f"hit@{limit} {sum(1 for m in всё if m)}/{len(всё)}")
    if загрязнены:
        print(f"\n  ВНИМАНИЕ: {загрязнены} вопросов нашлись в индексе дословно — "
              f"эти строки измеряют не поиск")


def _collection_by_title_any(title: str) -> dict | None:
    """Набор с таким названием у любого пользователя — замеру хозяин неважен."""
    import rag

    rag.init()
    with rag.connect() as conn:
        row = conn.execute(
            "SELECT * FROM collections WHERE title = ? ORDER BY id DESC LIMIT 1",
            (title,)).fetchone()
    return dict(row) if row else None


def main() -> None:
    if len(sys.argv) > 1:
        return measure(sys.argv[1], only=sys.argv[2] if len(sys.argv) > 2 else "")
    print(f"корень: {ROOT}")
    if not is_repo():
        print("не git-репозиторий")
        raise SystemExit(1)
    пути = tracked_files()
    по_наборам = split_files(пути)
    print(f"отобрано файлов: {len(пути)} — "
          + ", ".join(f"{имя.lower()} {len(с)}" for имя, с in по_наборам.items()))
    for подпись, набор in по_наборам.items():
        всего = sum((ROOT / p).stat().st_size for p in набор)
        print(f"  {подпись}: {len(набор)} файлов, {всего // 1024} КБ")
        for путь in набор[:4]:
            print(f"    {путь} → {flat_name(путь)}")
        if len(набор) > 4:
            print(f"    … ещё {len(набор) - 4}")
    состояние = state()
    print(f"ветка: {состояние['branch']}, изменено файлов: {состояние['changed']}")


if __name__ == "__main__":
    sys.exit(main())
