"""Проверка разбора javascript перед выкаткой.

Зачем. Питон при выкатке хотя бы импортируется, а javascript до браузера никто
не читает. Одна опечатка обошлась в полдня: в `common.js` две строковые
константы стояли рядом без `+` — по-питоньи склейка, по-javascript
синтаксическая ошибка. Разбор файла упал целиком, все его функции пропали, и
отправка сообщений умерла во всех диалогах, ничего не сказав. Проверка ниже
ловит ровно это и делает так, чтобы такое не уезжало на сервер.

Что проверяется: каждый `static/*.js` и каждый встроенный `<script>` внутри
`static/*.html`. Встроенные куски записываются во временный файл с отступом из
пустых строк, поэтому номер строки в ошибке — настоящий номер в html.

Чем проверяется: `node --check`, если node есть; иначе `jsc` из macOS, где
`checkSyntax(путь)` делает то же самое. Если не нашлось ничего, проверка не
молчит, а падает: «проверил и всё хорошо» и «проверить было нечем» — разные
вещи, и путать их опаснее, чем не проверять вовсе.

    python deploy/check-js.py            # весь static
    python deploy/check-js.py static/common.js
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

# macOS держит движок JavaScriptCore внутри фреймворка, в PATH его нет.
JSC = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A"
           "/Helpers/jsc")

# Встроенный скрипт: берём только классические. У модуля другие правила разбора
# (import и await на верхнем уровне), и проверять его как обычный скрипт значило
# бы получать выдуманные ошибки.
SCRIPT = re.compile(r"<script([^>]*)>(.*?)</script\s*>", re.S | re.I)
CLASSIC = re.compile(r'type\s*=\s*["\']?(text|application)/javascript', re.I)


def inline_scripts(page: Path) -> list[tuple[int, str]]:
    """Встроенные скрипты страницы: (номер первой строки, текст)."""
    text = page.read_text()
    found = []
    for match in SCRIPT.finditer(text):
        attributes, body = match.group(1), match.group(2)
        if "src=" in attributes.lower():
            continue  # это ссылка на отдельный файл, он проверяется сам
        if "type=" in attributes.lower() and not CLASSIC.search(attributes):
            continue
        if not body.strip():
            continue
        line = text.count("\n", 0, match.start(2)) + 1
        found.append((line, body))
    return found


def targets(paths: list[Path]) -> list[tuple[Path, Path, str]]:
    """Что проверяем: (файл для движка, что показать человеку, пояснение).

    Для встроенных скриптов первый и второй пути разные: движок читает
    временный файл, а человеку называется страница.
    """
    plan: list[tuple[Path, Path, str]] = []
    holder = Path(tempfile.mkdtemp(prefix="check-js-"))
    for path in paths:
        if path.suffix == ".js":
            plan.append((path, path, ""))
            continue
        for number, (line, body) in enumerate(inline_scripts(path), 1):
            # Пустые строки вместо начала страницы: движок считает строки от
            # начала файла, и без отступа он показывал бы номер внутри куска.
            copy = holder / f"{path.stem}-{number}.js"
            copy.write_text("\n" * (line - 1) + body)
            plan.append((copy, path, f"встроенный скрипт {number}"))
    return plan


def shortly(path: Path) -> str:
    """Путь от корня проекта: полный только мешает читать."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def by_node(plan: list[tuple[Path, Path, str]], node: str) -> list[str]:
    problems = []
    for file, shown, note in plan:
        done = subprocess.run([node, "--check", str(file)],
                              capture_output=True, text=True)
        if done.returncode:
            first = (done.stderr.strip().splitlines() or [""])[0]
            problems.append(
                f"{shortly(shown)}{' — ' + note if note else ''}: {first}")
    return problems


def by_jsc(plan: list[tuple[Path, Path, str]]) -> list[str]:
    # Один запуск на все файлы: движок стартует не мгновенно, а файлов десяток.
    harness = ("for (const path of arguments) {\n"
               "  try { checkSyntax(path); }\n"
               "  catch (e) { print(path + '|' + e.line + '|' + e.message); }\n"
               "}\n")
    holder = Path(tempfile.mkdtemp(prefix="check-js-"))
    script = holder / "harness.js"
    script.write_text(harness)
    done = subprocess.run([str(JSC), str(script), "--", *[str(f) for f, _, _ in plan]],
                          capture_output=True, text=True)
    named = {str(file): (shown, note) for file, shown, note in plan}
    problems = []
    for line in done.stdout.splitlines():
        path, _, rest = line.partition("|")
        number, _, message = rest.partition("|")
        shown, note = named.get(path, (path, ""))
        where = f"{shortly(shown)}:{number}" + (f" ({note})" if note else "")
        problems.append(f"{where}: {message}")
    if done.returncode and not problems:
        problems.append(f"jsc не смог проверить: {done.stderr.strip()[:200]}")
    return problems


def main() -> None:
    given = [Path(a) for a in sys.argv[1:]]
    paths = given or sorted([*STATIC.glob("*.js"), *STATIC.glob("*.html")])
    if not paths:
        sys.exit("Нечего проверять: в static нет ни .js, ни .html")

    plan = targets(paths)
    if node := shutil.which("node"):
        problems = by_node(plan, node)
        engine = "node --check"
    elif JSC.exists():
        problems = by_jsc(plan)
        engine = "jsc checkSyntax"
    else:
        sys.exit("Проверить нечем: нет ни node, ни jsc. Поставьте node "
                 "(brew install node) — молча пропускать проверку нельзя.")

    if problems:
        print(f"Разбор javascript не прошёл ({engine}):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        sys.exit(1)
    print(f"javascript разбирается: проверено {len(plan)} "
          f"шт. ({engine})")


if __name__ == "__main__":
    main()
