#!/bin/sh
# Проверка, что приложение поднимается на этой машине и отвечает.
#
#     ./run-local.sh                   # поднять, проверить, погасить
#     ./run-local.sh --keep            # оставить работать для ручных опытов
#     ./run-local.sh --with-rag        # и сервер поиска по документам
#     ./run-local.sh --keep --lan      # и открыть его устройствам в этом Wi-Fi
#     LAN_PORT=9100 ./run-local.sh --lan   # другой порт для сети
#     PORT=9000 ./run-local.sh         # другой порт
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
RAG_PORT=${RAG_PORT:-8763}
VENV=.venv
KEEP=
WITH_RAG=

# Приложение всегда слушает только петлю. Устройствам в сети его отдаёт
# отдельный процесс на системном питоне — почему именно так, подробно в
# lan_proxy.py: коротко, брандмауэр macOS не умеет пропускать неподписанную
# программу, а питон из Homebrew не подписан.
LAN=
LAN_PORT=
for arg in "$@"; do
    case "$arg" in
        --keep) KEEP=1 ;;
        --with-rag) WITH_RAG=1 ;;
        --lan) LAN=1 ;;
        *) echo "Неизвестный ключ: $arg" >&2; exit 1 ;;
    esac
done
# Без if нельзя: при `set -e` неудачная проверка [ -n ] погасила бы скрипт.
if [ -n "$LAN" ]; then
    LAN_PORT=${LAN_PORT:-$((PORT + 1))}
fi

# Адрес этой машины в сети — его и набирают на телефоне. Интерфейс заранее не
# известен: Wi-Fi это обычно en0, но после док-станции или второй карты бывает
# и en1. Перебираем, берём первый с адресом.
lan_address() {
    for IFACE in en0 en1 en2 en3 en7; do
        ADDR=$(ipconfig getifaddr "$IFACE" 2>/dev/null) || continue
        if [ -n "$ADDR" ]; then printf '%s' "$ADDR"; return 0; fi
    done
    return 1
}

[ -f web.py ] || { echo "Запускать из корня проекта" >&2; exit 1; }

# Где лежат данные. Для проверки — временный каталог: она не должна писать в
# ту же app.db, где настоящая переписка. Для --keep наоборот нужен постоянный:
# иначе каждый перезапуск давал бы пустое приложение — ни учётки, ни диалогов,
# ни настроек, — а rag.db при этом переживает перезапуск, и подключённый набор
# оставался за прежним номером пользователя и пропадал из списка. Так и вышло.
if [ -n "$KEEP" ]; then
    DATA=$PWD/local-data
    mkdir -p "$DATA"
else
    DATA=$(mktemp -d "${TMPDIR:-/tmp}/llmchat-test.XXXXXX")
fi
SERVER=
RAGSERVER=
LANPROXY=

# Переписка — во временную базу, а индекс документов — настоящий. Разводим
# намеренно: проверять поиск по пустому индексу бессмысленно, а писать
# проверочные диалоги в рабочую переписку незачем. Поиск индекс только читает.
export RAG_DB_PATH="$PWD/rag.db"
export RAG_SOURCES_DIR="$PWD/rag-sources"
# Адрес сервера поиска нужен обоим: чат по нему ходит, и он же попадает в
# список MCP-серверов при заведении учётки.
export RESEARCH_MCP_URL="http://127.0.0.1:$RAG_PORT/mcp"

cleanup() {
    [ -n "$LANPROXY" ] && kill "$LANPROXY" 2>/dev/null || true
    [ -n "$RAGSERVER" ] && kill "$RAGSERVER" 2>/dev/null || true
    [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null || true
    # Временный каталог убираем, постоянный — никогда: в нём переписка.
    [ -n "$KEEP" ] || rm -rf "$DATA"
}
trap cleanup EXIT INT TERM

step() { printf '· %s\n' "$1"; }

# Кто держит порт и чем его погасить. Советовать «возьмите другой порт»
# почти всегда мимо: в девяти случаях из десяти порт занят нашим же
# процессом от прошлого запуска с --keep.
busy() {   # порт, имя переменной для подмены
    # Только слушающие сокеты. Без -sTCP:LISTEN lsof отдаёт и тех, кто просто
    # соединён с этим портом, — то есть открытый браузер. Проверено: при
    # открытой странице скрипт насчитал владельцами порта два процесса и
    # посоветовал «kill 2747 93296», где второй был Google Chrome. Совет
    # гасить браузер, чтобы освободить порт, — это хуже, чем молчание.
    OWNERS=$(lsof -ti:"$1" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' | sed 's/ *$//')
    if [ -z "$OWNERS" ]; then
        fail "порт $1 занят, но кем — выяснить не удалось (нет lsof?). Укажите другой через $2=..."
    fi
    printf '  порт %s занят:\n' "$1" >&2
    for PID in $OWNERS; do
        # Длинный путь к интерпретатору съедает всю строку, а узнать процесс
        # можно только по тому, что идёт после него. Путь сокращаем.
        # От пути к исполняемому файлу оставляем только имя: длинный путь к
        # интерпретатору съедает строку, а узнаётся процесс по тому, что идёт
        # после него — «-m http.server 8763» или «uvicorn research_mcp:app».
        ps -o pid=,etime=,command= -p "$PID" 2>/dev/null \
            | sed -E 's#^( *[0-9]+ +[0-9:.-]+ +)[^ ]*/#\1#' \
            | cut -c1-120 | sed 's/^/    /' >&2
    done
    printf '  погасить:  kill %s\n' "$OWNERS" >&2
    printf '  или взять другой порт:  %s=<номер> %s\n' "$2" "$0" >&2
    exit 1
}
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

# Спрашиваем у самого приложения, а не разбираем .env: видно именно то, что
# оно прочитало, со всеми правилами про переменные окружения. Правка, ушедшая
# не в тот файл, становится заметна сразу — на этом уже потерян вечер.
step "провайдер"
"$VENV/bin/python" - <<'PYEOF' | sed 's/^/  /'
import json, urllib.error, urllib.request
import llm

где = "локальный" if llm.is_local(llm.BASE_URL) else "внешний"
строка = f"{llm.BASE_URL} · {где} · диалект {llm.DIALECT}"
try:
    запрос = urllib.request.Request(f"{llm.BASE_URL}/models", headers=llm.headers())
    with urllib.request.urlopen(запрос, timeout=5) as ответ:
        сколько = len(json.load(ответ).get("data", []))
    print(f"{строка} · отвечает, моделей {сколько}")
except llm.LLMError as err:
    print(f"{строка}\n  НЕ ПРОВЕРЕН: {err}")
except (urllib.error.URLError, OSError, ValueError) as err:
    # Не повод валить проверку: приложение поднимается и без модели, а
    # скрипт проверяет именно подъём.
    print(f"{строка}\n  НЕ ОТВЕЧАЕТ: {type(err).__name__}: {err}")

# Окно контекста. Спрашиваем только у местной модели и только потому, что
# задать его в запросе нельзя: OpenAI-совместимый /v1 такого поля не знает, и
# окно берётся либо из параметров модели, либо из OLLAMA_CONTEXT_LENGTH у
# сервера. Маленькое окно не ошибка, а молчаливая обрезка: запрос с выдержками
# не доезжает, и со стороны это выглядит как глупость модели. Один раз на это
# уже ушёл день.
if llm.is_local(llm.BASE_URL):
    корень = llm.BASE_URL.rsplit("/v1", 1)[0]
    try:
        запрос = urllib.request.Request(
            f"{корень}/api/show", data=json.dumps({"model": llm.MODEL}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(запрос, timeout=5) as ответ:
            параметры = json.load(ответ).get("parameters") or ""
        окно = next((s.split()[-1] for s in параметры.splitlines()
                     if s.startswith("num_ctx")), "")
        if окно:
            print(f"окно контекста {окно} — своё, из Modelfile")
        else:
            print("окно контекста НЕ ЗАДАНО у модели: возьмётся из "
                  "OLLAMA_CONTEXT_LENGTH или по умолчанию, а маленькое молча "
                  "обрежет запрос с выдержками")
            print("  соберите свою: ollama create llmchat-rag:3b -f local/Modelfile")
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        # Не Ollama, модели нет или сервер молчит — проверке это не мешает.
        pass
PYEOF

if [ -n "$WITH_RAG" ]; then
    # Сервер поиска — отдельный процесс не для красоты: модель эмбеддингов
    # держится в памяти резидентно, около гигабайта, и второй её экземпляр
    # рядом с приложением удвоил бы расход. Поэтому и флагом, а не всегда.
    step "сервер поиска по документам"
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$RAG_PORT/"; then
        busy "$RAG_PORT" RAG_PORT
    fi
    DB_PATH="$DATA/app.db" \
        "$VENV/bin/uvicorn" research_mcp:app --host 127.0.0.1 --port "$RAG_PORT" \
        --log-level warning > "$DATA/research.log" 2>&1 &
    RAGSERVER=$!
    WAITED=0
    until curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$RAG_PORT/mcp"; do
        kill -0 "$RAGSERVER" 2>/dev/null || { sed 's/^/  /' "$DATA/research.log"; fail "сервер поиска не поднялся"; }
        WAITED=$((WAITED + 1))
        [ "$WAITED" -gt 60 ] && fail "сервер поиска не ответил за 60 с"
        sleep 1
    done
    printf '  поднялся за %s с на порту %s\n' "$WAITED" "$RAG_PORT"
    "$VENV/bin/python" - <<'PYEOF' | sed 's/^/  /'
import asyncio, os
import mcp_tools, rag

сервер = asyncio.run(mcp_tools.list_tools(os.environ["RESEARCH_MCP_URL"]))
имена = [t.name for t in сервер.tools]
print("инструменты:", ", ".join(имена))
print("search_docs на месте:", "search_docs" in имена)
with rag.connect() as c:
    наборы = c.execute("SELECT count(*), coalesce(sum(chunks), 0) FROM collections"
                       " WHERE status = 'готов'").fetchone()
print(f"индекс: наборов {наборы[0]}, кусков {наборы[1]} ({rag.DB_PATH})")
PYEOF
fi

step "свободен ли порт $PORT"
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/"; then
    busy "$PORT" PORT
fi

step "поднимаю приложение на временной базе"
printf '  данные: %s\n' "$DATA"
printf '  слушает: 127.0.0.1:%s\n' "$PORT"
# По сети отдаётся ТОЛЬКО приложение — у него есть вход по логину. Сервер
# поиска и Ollama остаются на петле намеренно: аутентификации нет ни у того,
# ни у другого, и в сети они означали бы открытый всем доступ к загруженным
# документам и к самой модели.
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

if [ -n "$LAN" ]; then
    step "доступ из своей сети"
    if [ ! -x /usr/bin/python3 ]; then
        printf '  нет /usr/bin/python3 — поставьте инструменты разработчика:\n'
        printf '    xcode-select --install\n'
    elif curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$LAN_PORT/"; then
        busy "$LAN_PORT" LAN_PORT
    else
        # Сеть слушает СИСТЕМНЫЙ питон, а не наш: подробности в lan_proxy.py.
        # Коротко — брандмауэр macOS опознаёт программу по подписи, питон из
        # Homebrew не подписан вовсе, и разрешить его через socketfilterfw
        # нельзя: правило записывается, `--getappblocked` говорит «permitted»,
        # а соединение всё равно рвётся. Системный питон подписан Apple и
        # проходит, поэтому он и стоит на входе.
        /usr/bin/python3 lan_proxy.py "$LAN_PORT" "$PORT" \
            > "$DATA/lan.log" 2>&1 &
        LANPROXY=$!
        WAITED=0
        until curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$LAN_PORT/login"; do
            kill -0 "$LANPROXY" 2>/dev/null \
                || { sed 's/^/  /' "$DATA/lan.log"; fail "перелив не поднялся"; }
            WAITED=$((WAITED + 1))
            [ "$WAITED" -gt 15 ] && fail "перелив не ответил за 15 с"
            sleep 1
        done
        printf '  перелив поднялся: сеть %s → петля %s\n' "$LAN_PORT" "$PORT"
        # Проверяем по настоящему сетевому адресу, а не по петле: петля
        # работает всегда, и именно поэтому прежние проверки ничего не ловили.
        if ADDR=$(lan_address); then
            CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 \
                   "http://$ADDR:$LAN_PORT/login" || true)
            if [ "$CODE" = "200" ]; then
                printf '  http://%s:%s отвечает по сети\n' "$ADDR" "$LAN_PORT"
            else
                printf '  http://%s:%s НЕ отвечает (код %s)\n' "$ADDR" "$LAN_PORT" "$CODE"
                printf '  Проверьте брандмауэр: он мог запретить и системный питон.\n'
                printf '    /usr/libexec/ApplicationFirewall/socketfilterfw --getappblocked /usr/bin/python3\n'
            fi
        else
            printf '  адрес в сети не найден — машина не в сети?\n'
        fi
    fi
fi

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
    if [ -n "$LANPROXY" ]; then
        if ADDR=$(lan_address); then
            echo "  с телефона: http://$ADDR:$LAN_PORT (тот же Wi-Fi)"
        else
            echo "  с телефона: адрес в сети не найден — машина не в сети?"
        fi
        echo "             вход по логину остаётся; сайт виден всем в этой сети"
    fi
    echo "  данные:    $DATA (постоянные, переживают перезапуск)"
    echo "  учётка:    DB_PATH=$DATA/app.db $VENV/bin/python manage.py create-admin ivan"
    if [ -n "$RAGSERVER" ]; then
        echo "  поиск:     http://127.0.0.1:$RAG_PORT/mcp (процесс $RAGSERVER)"
    fi
    # Пробелы схлопываем: без --with-rag и без --lan переменные пусты, и в
    # строке, которую человек копирует, оставались бы дыры.
    STOP=$(printf '%s %s %s' "$SERVER" "$RAGSERVER" "$LANPROXY" | tr -s ' ' | sed 's/ *$//')
    echo "  погасить:  kill $STOP"
    trap - EXIT INT TERM
else
    echo "Локальный запуск работает. Чтобы поиграть руками: ./run-local.sh --keep"
fi
