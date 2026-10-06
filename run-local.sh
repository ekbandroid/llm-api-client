#!/bin/sh
# Проверка, что приложение поднимается на этой машине и отвечает.
#
#     ./run-local.sh            # поднять, проверить, погасить
#     ./run-local.sh --keep     # оставить работать для ручных опытов
#     PORT=9000 ./run-local.sh  # другой порт
#
# Модель не нужна: ключ проверяется только в момент запроса, а до запроса мы
# не доходим. Проверяется ровно то, что ломается при переносе на новую
# машину, — зависимости, разбор файлов, создание баз, ответы страниц.
#
# База берётся временная. Иначе проверка писала бы в ту же app.db, где лежит
# настоящая переписка, и «тест» однажды оказался бы единственным, что
# происходило с рабочими данными.

set -eu

PORT=${PORT:-8765}
VENV=.venv
KEEP=
[ "${1:-}" = "--keep" ] && KEEP=1

[ -f web.py ] || { echo "Запускать из корня проекта" >&2; exit 1; }

DATA=$(mktemp -d "${TMPDIR:-/tmp}/llmchat-test.XXXXXX")
SERVER=

cleanup() {
    [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null || true
    [ -n "$KEEP" ] || rm -rf "$DATA"
}
trap cleanup EXIT INT TERM

step() { printf '· %s\n' "$1"; }
fail() { printf '  ПЛОХО: %s\n' "$1" >&2; exit 1; }

# ---------- окружение ----------

step "python"
command -v python3 >/dev/null || fail "python3 не найден"
printf '  %s\n' "$(python3 -V)"

step "виртуальное окружение"
if [ ! -x "$VENV/bin/python" ]; then
    printf '  создаю %s\n' "$VENV"
    python3 -m venv "$VENV"
fi
# Ставим зависимости, только если их нет: pip даже вхолостую думает секунду,
# а скрипт хочется запускать часто.
if ! "$VENV/bin/python" -c "import fastapi, uvicorn, mcp" 2>/dev/null; then
    printf '  ставлю зависимости\n'
    "$VENV/bin/pip" install -q -r requirements.txt
fi
printf '  %s\n' "$("$VENV/bin/python" -c 'import fastapi; print("fastapi", fastapi.__version__)')"

# ---------- разбор файлов ----------

step "разбор python"
"$VENV/bin/python" -m compileall -q ./*.py >/dev/null || fail "python не разбирается"

step "разбор javascript"
# Без конвейера. В sh нет pipefail, и `check-js.py | sed` вернул бы статус
# sed — то есть успех всегда. Проверено: подложенная ошибка в common.js так
# и прошла мимо. Проверка, которая не умеет упасть, хуже отсутствия проверки.
if JS=$("$VENV/bin/python" deploy/check-js.py 2>&1); then
    printf '%s\n' "$JS" | sed 's/^/  /'
else
    printf '%s\n' "$JS" | sed 's/^/  /'
    fail "javascript не разбирается"
fi

# ---------- запуск ----------

step "свободен ли порт $PORT"
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/"; then
    fail "порт $PORT уже занят — укажите другой через PORT=..."
fi

step "поднимаю приложение на временной базе"
printf '  данные: %s\n' "$DATA"
DB_PATH="$DATA/app.db" \
SESSION_SECRET="$("$VENV/bin/python" -c 'import secrets; print(secrets.token_urlsafe(48))')" \
COOKIE_SECURE=false \
    "$VENV/bin/uvicorn" web:app --host 127.0.0.1 --port "$PORT" \
    --log-level warning > "$DATA/server.log" 2>&1 &
SERVER=$!

# Ждём, пока отзовётся. Первый запуск дольше: создаются таблицы.
WAITED=0
until curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/login"; do
    kill -0 "$SERVER" 2>/dev/null || { sed 's/^/  /' "$DATA/server.log"; fail "приложение не поднялось"; }
    WAITED=$((WAITED + 1))
    [ "$WAITED" -gt 30 ] && fail "не дождались ответа за 30 с"
    sleep 1
done
printf '  поднялось за %s с\n' "$WAITED"

# ---------- проверки ----------

expect() {   # путь ожидаемый-код пояснение
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT$1")
    if [ "$CODE" = "$2" ]; then
        printf '  %-28s %s — %s\n' "$1" "$CODE" "$3"
    else
        fail "$1 ответил $CODE вместо $2 ($3)"
    fi
}

step "страницы и api"
expect /                    302 "без входа уводит на /login"
expect /login               200 "страница входа"
expect /static/common.js    200 "статика раздаётся"
expect /api/config          401 "api закрыто без входа"
expect /api/conversations   401 "переписка закрыта без входа"

step "страница входа собрана"
curl -s "http://127.0.0.1:$PORT/login" | grep -q "LLM чат" \
    || fail "в странице входа нет заголовка"
printf '  заголовок на месте\n'

step "служебные команды"
DB_PATH="$DATA/app.db" "$VENV/bin/python" manage.py list >/dev/null \
    || fail "manage.py не работает"
printf '  manage.py list отвечает\n'

step "созданные файлы"
for f in app.db scheduler.db; do
    [ -f "$DATA/$f" ] && printf '  %-14s %s байт\n' "$f" "$(wc -c < "$DATA/$f" | tr -d ' ')" \
                      || printf '  %-14s не создан (нормально, если не понадобился)\n' "$f"
done

step "ошибки в журнале"
if grep -iE "traceback|error" "$DATA/server.log" >/dev/null 2>&1; then
    grep -iE "traceback|error" "$DATA/server.log" | head -5 | sed 's/^/  /'
    fail "в журнале есть ошибки"
fi
printf '  чисто\n'

# ---------- итог ----------

echo
if [ -n "$KEEP" ]; then
    echo "Приложение работает: http://127.0.0.1:$PORT"
    echo "  данные:    $DATA"
    echo "  учётка:    DB_PATH=$DATA/app.db $VENV/bin/python manage.py create-admin ivan"
    echo "  погасить:  kill $SERVER && rm -rf $DATA"
    trap - EXIT INT TERM
else
    echo "Локальный запуск работает. Чтобы поиграть руками: ./run-local.sh --keep"
fi
