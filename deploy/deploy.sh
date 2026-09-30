#!/bin/sh
# Выкатка на llmchat.me.uk. Запускать из корня проекта:
#
#     deploy/deploy.sh
#     deploy/deploy.sh --no-restart      # только код, без перезапуска служб
#
# Уезжает то, что закоммичено в master: git archive берёт дерево из коммита, а
# не с диска. .env и базы не в git, tar их не трогает.
#
# Перед отправкой — проверки. Питон при запуске хотя бы импортируется, и
# ошибка в нём видна сразу по упавшей службе; javascript до браузера не читает
# никто, и однажды это стоило полдня: опечатка в common.js уронила разбор
# файла целиком и молча убила отправку сообщений во всех диалогах. Поэтому
# разбор javascript проверяется здесь, и без него выкатки не будет.

set -eu

HOST=llmvps
APP=/opt/llmchat/app
UNITS="llmchat llmchat-weather llmchat-scheduler llmchat-research"
PYTHON=${PYTHON:-.venv/bin/python}

[ -d .git ] || { echo "Запускать из корня проекта" >&2; exit 1; }
[ -x "$PYTHON" ] || PYTHON=python3

echo "· проверяю разбор javascript"
"$PYTHON" deploy/check-js.py

echo "· проверяю разбор python"
"$PYTHON" -m compileall -q *.py >/dev/null

# Незакоммиченное не уедет — лучше сказать об этом до, а не после. Базы и .env
# в .gitignore, в этот список они не попадают.
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
    echo "· ВНИМАНИЕ: это не уедет, оно не в коммите:"
    echo "$DIRTY" | sed 's/^/    /'
fi

echo "· отправляю master на $HOST:$APP"
git archive master | ssh "$HOST" "sudo -u llmchat tar -x -C $APP"

# tar только добавляет и перезаписывает: файл, удалённый из репозитория,
# остаётся на сервере навсегда. Однажды так пережила выкатку страница из
# старой версии — она лежала там полмесяца и открывалась из интернета.
# Сами ничего не удаляем: решать, что на проде лишнее, — не дело скрипта.
echo "· сверяю состав каталога с git"
WANT=$(mktemp); HAVE=$(mktemp)
git ls-files | sort > "$WANT"
# find запускаем из самого каталога: у llmchat нет доступа к домашнему
# каталогу того, кто вошёл по ssh, и вернуться туда в конце обхода он не смог бы.
LOOK="cd $APP && find . -type f ! -name .env ! -name '*.pyc' \
      ! -path '*/__pycache__/*' -printf '%P\n'"
ssh "$HOST" "sudo -u llmchat sh -c \"$LOOK\"" | sort > "$HAVE"
EXTRA=$(comm -13 "$WANT" "$HAVE")
rm -f "$WANT" "$HAVE"
if [ -n "$EXTRA" ]; then
    echo "  на сервере есть файлы, которых нет в git:"
    echo "$EXTRA" | sed 's/^/    /'
    echo "  убрать при необходимости:"
    echo "$EXTRA" | sed "s|^|    ssh $HOST 'sudo -u llmchat rm $APP/|;s|\$|'|"
fi

if [ "${1:-}" = "--no-restart" ]; then
    echo "· службы не трогаю (--no-restart)"
    exit 0
fi

echo "· перезапускаю службы"
ssh "$HOST" "sudo systemctl restart $UNITS"

# Перезапуск сам по себе ничего не доказывает: служба может подняться и тут же
# упасть на импорте. Поэтому спрашиваем состояние после паузы.
sleep 4
echo "· состояние:"
ssh "$HOST" "systemctl is-active $UNITS" | paste -d' ' - - - - | sed 's/^/    /'
ssh "$HOST" "curl -s -o /dev/null -w '    чат отвечает: %{http_code} за %{time_total} с\n' https://llmchat.me.uk/"
