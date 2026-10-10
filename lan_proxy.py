"""Отдаёт приложение устройствам в своей сети. Слушает сеть, переливает на петлю.

Зачем вообще отдельный процесс, когда uvicorn умеет `--host 0.0.0.0`.

Привязки к 0.0.0.0 на macOS мало: входящие соединения фильтрует брандмауэр, и
решает он по программе, а не по порту. Опознаёт он её по подписи — а питон из
Homebrew **не подписан вовсе**:

    codesign -dv /usr/bin/python3                       → CodeDirectory, Signature
    codesign -dv .../python@3.14/.../MacOS/Python       → ни строки

Поэтому `socketfilterfw --add` для него бесполезен: правило записывается,
`--getappblocked` отвечает «is permitted», а соединение всё равно обрывается с
«connection reset by peer». Проверено на этой машине: простейший http.server на
системном питоне по сетевому адресу отвечает 200, тот же http.server на питоне
venv — ничего.

Отсюда и способ. Сеть слушает **системный** `/usr/bin/python3` — он подписан
Apple, и брандмауэр его пропускает, — а всё, что пришло, переливает на
приложение по петле. Пароль не нужен, настройки безопасности не трогаются, и
приложение остаётся на 127.0.0.1, то есть снаружи доступно ровно столько, сколько
мы решили отдать.

Только стандартная библиотека: запускается не из venv.

    /usr/bin/python3 lan_proxy.py 8766 8765
"""

import asyncio
import sys

КУСОК = 65536


async def перелить(читать: asyncio.StreamReader, писать: asyncio.StreamWriter) -> None:
    """Качает байты в одну сторону, пока не кончатся."""
    try:
        while кусок := await читать.read(КУСОК):
            писать.write(кусок)
            await писать.drain()
    except (ConnectionError, asyncio.IncompleteReadError, OSError):
        # Любой обрыв — обычное дело: страницу закрыли, телефон уснул,
        # поток SSE прервали. Поводов шуметь в журнале тут нет.
        pass
    finally:
        писать.close()


async def соединение(порт: int, читать: asyncio.StreamReader,
                     писать: asyncio.StreamWriter) -> None:
    """Одно соединение: открыть встречное к приложению и связать их."""
    try:
        к_приложению_ч, к_приложению_п = await asyncio.open_connection("127.0.0.1", порт)
    except OSError as err:
        print(f"приложение на 127.0.0.1:{порт} не отвечает: {err}", file=sys.stderr)
        писать.close()
        return
    # Обе стороны качаем одновременно: SSE идёт от приложения к браузеру
    # потоком, и ждать, пока закроется запрос, нельзя — ответ так и не придёт.
    await asyncio.gather(
        перелить(читать, к_приложению_п),
        перелить(к_приложению_ч, писать),
    )


async def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Нужно два числа: порт в сети и порт приложения на петле")
    снаружи, внутри = int(sys.argv[1]), int(sys.argv[2])
    сервер = await asyncio.start_server(
        lambda ч, п: соединение(внутри, ч, п), "0.0.0.0", снаружи)
    print(f"сеть {снаружи} → петля {внутри}", flush=True)
    async with сервер:
        await сервер.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
