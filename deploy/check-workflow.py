"""Разбор файлов .github/workflows: находит то, что GitHub найдёт сам, но позже.

Появилась после круга через GitHub впустую. Шаг назывался `id: ключ`, а выход
`есть` — кириллицей. GitHub разбирает их как имена, а не как текст, поэтому
файл целиком оказался негодным: в списке заданий вместо имени появился путь к
нему, и прогон упал, ничего не сделав. Ошибка видна только после пуша, а пуш —
это круг в несколько минут.

Та же ловушка, на которой этот проект трижды попался с именами переменных в sh.
Правило одно и то же: **имена — латиницей, русский остаётся в name, комментариях
и сообщениях.**

Проверяется три вещи:
  1. yaml вообще разбирается;
  2. в идентификаторах (id шага, steps.*, outputs.*) нет кириллицы;
  3. каждый блок `run:` разбирается как shell — `sh -n`.

Третья нужна потому, что ошибку в shell внутри workflow иначе видно только в
середине прогона, когда уже поставлены зависимости и скачана модель.

    python deploy/check-workflow.py
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

КАТАЛОГ = Path(__file__).resolve().parent.parent / ".github" / "workflows"

# Выражения ${{ ... }} подставляет GitHub; для sh это мусор, заменяем заглушкой.
ВЫРАЖЕНИЕ = re.compile(r"\$\{\{[^}]*\}\}")

# Места, где строка — имя, а не текст: GitHub ждёт латиницу.
ИМЕНА = re.compile(r"(?:^\s*id:\s*|steps\.|outputs\.|needs\.|jobs\.)"
                   r"([^\s'\"=.}]*[А-Яа-яЁё][^\s'\"=.}]*)", re.M)


def блоки_run(текст: str) -> list[tuple[int, str]]:
    """Содержимое всех `run:` — с номером строки, чтобы было куда смотреть.

    Разбираем отступами, а не yaml: pyyaml в зависимостях проекта не значится
    (он приезжает попутно и может не приехать), а проверка не должна зависеть
    от удачи. Нам нужны только блочные скаляры — ими и пишут shell.
    """
    найдено = []
    строки = текст.splitlines()
    i = 0
    while i < len(строки):
        совпало = re.match(r"^(\s*)-?\s*run:\s*([|>][-+]?)?\s*$", строки[i])
        однострочный = re.match(r"^(\s*)-?\s*run:\s+(\S.*)$", строки[i])
        if совпало:
            отступ = len(совпало.group(1))
            тело, i = [], i + 1
            while i < len(строки):
                if строки[i].strip() and (len(строки[i]) - len(строки[i].lstrip())) <= отступ:
                    break
                тело.append(строки[i])
                i += 1
            найдено.append((i - len(тело) + 1, "\n".join(тело)))
            continue
        if однострочный:
            найдено.append((i + 1, однострочный.group(2)))
        i += 1
    return найдено


def проверить(путь: Path) -> list[str]:
    текст = путь.read_text(encoding="utf-8")
    беды = []

    if кириллица := ИМЕНА.findall(текст):
        беды.append(f"кириллица в именах, которые GitHub разбирает как имена: "
                    f"{', '.join(sorted(set(кириллица)))}. "
                    f"Файл станет негодным целиком — имена только латиницей.")

    try:
        import yaml
    except ImportError:
        print(f"  {путь.name}: yaml не проверен — модуль pyyaml не установлен")
    else:
        try:
            yaml.safe_load(текст)
        except yaml.YAMLError as err:
            беды.append(f"yaml не разбирается: {str(err).splitlines()[0]}")

    for номер, код in блоки_run(текст):
        пробный = ВЫРАЖЕНИЕ.sub("X", код)
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                         encoding="utf-8") as f:
            f.write(пробный)
            временный = f.name
        готово = subprocess.run(["sh", "-n", временный], capture_output=True, text=True)
        os.unlink(временный)
        if готово.returncode:
            беды.append(f"строка {номер}: shell не разбирается — "
                        f"{готово.stderr.strip().splitlines()[-1][:160]}")
    return беды


def main() -> int:
    if not КАТАЛОГ.exists():
        print("заданий нет: каталога .github/workflows не существует")
        return 0
    файлы = sorted(КАТАЛОГ.glob("*.yml")) + sorted(КАТАЛОГ.glob("*.yaml"))
    if not файлы:
        print("заданий нет")
        return 0

    всего = 0
    for путь in файлы:
        беды = проверить(путь)
        всего += len(беды)
        if беды:
            for беда in беды:
                print(f"  {путь.name}: {беда}", file=sys.stderr)
        else:
            блоков = len(блоки_run(путь.read_text(encoding="utf-8")))
            print(f"  {путь.name}: разбирается, блоков shell {блоков}")
    return 1 if всего else 0


if __name__ == "__main__":
    sys.exit(main())
