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
