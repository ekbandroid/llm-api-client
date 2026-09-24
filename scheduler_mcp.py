"""MCP-сервер планировщика: модель заводит поручения, которые выполнит агент.

Сервер ничего не выполняет сам — он только ведёт очередь. Исполнитель живёт в
приложении: выполнять поручение должен агент, а модель, ключ, память и
переписка есть только там. MCP-серверу пришлось бы завести второй экземпляр
всего этого.

Очередь — общий файл SQLite, см. scheduler.py. Приложение берёт задания из
него напрямую, поэтому служебных инструментов «отдай созревшее» здесь нет:
два процесса на одной машине, и лишний сетевой круг ничего бы не добавил.

У каждого задания есть chat_id — диалог, в котором оно появится. Модель этого
идентификатора не знает: приложение подставляет его само перед вызовом, а из
схемы, которую видит модель, параметр вырезан.

Запуск рядом с чатом:

    uvicorn scheduler_mcp:app --host 127.0.0.1 --port 8002
"""

import os
from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer
from pydantic import Field

import scheduler

# Сервер живёт в UTC, человек — нет. Время в ответах показываем в этой зоне.
TZ = ZoneInfo(os.getenv("SCHEDULER_TZ", "Europe/Moscow"))

KIND_LABELS = {
    scheduler.ONCE: "разовое",
    scheduler.REPEAT: "повторяющееся",
    scheduler.SUMMARY: "сводка",
}
STATUS_LABELS = {
    scheduler.ACTIVE: "активно",
    scheduler.PAUSED: "на паузе",
    scheduler.DONE: "выполнено",
}

server = MCPServer(
    name="Планировщик поручений",
    version="1.0.0",
    instructions=(
        "Отложенные и повторяющиеся поручения. Выполнит их этот же ассистент "
        "в этом же диалоге, когда придёт время, и ответ появится в переписке. "
        "Поручение пиши как обычную просьбу — «посмотри погоду в Екатеринбурге "
        "и скажи, нужен ли зонт», — а не как имя инструмента: во время "
        "выполнения будут доступны все те же инструменты, что сейчас."
    ),
)


def _local(stamp: str | None) -> str:
    """Время запуска словами и по местным часам."""
    moment = scheduler._parse(stamp)
    if moment is None:
        return "не запланировано"
    shown = moment.astimezone(TZ).strftime("%d.%m %H:%M")
    return f"{shown} ({scheduler.human_delay(stamp)})"


def _line(job: dict) -> str:
    view = scheduler.describe(job)
    parts = [f"#{view['id']} {view['title']}",
             f"{KIND_LABELS[view['kind']]}, {STATUS_LABELS[view['status']]}"]
    if view["every_minutes"]:
        parts.append(f"каждые {view['every_minutes']} мин")
    if view["status"] == scheduler.ACTIVE:
        parts.append(f"следующий запуск {_local(view['next_run_at'])}")
    if view["runs"]:
        parts.append(f"выполнено раз: {view['runs']}")
    line = " · ".join(parts)
    if answer := view.get("last_answer"):
        line += f"\n    последний результат: {answer[:200]}"
    return line


def _chat(chat_id: int | None) -> int:
    """Проверяет, что приложение подставило диалог."""
    if not chat_id:
        raise ValueError(
            "не указан диалог. Этот сервер рассчитан на вызов из приложения: "
            "оно подставляет идентификатор диалога само")
    return int(chat_id)


@server.tool()
async def schedule_once(
    prompt: Annotated[str, Field(
        description="Что сделать, когда придёт время. Обычная просьба своими "
                    "словами: «напомни про чайник», «посмотри погоду в "
                    "Екатеринбурге и скажи, нужен ли зонт»")],
    in_minutes: Annotated[int, Field(
        description="Через сколько минут выполнить: 15 — через четверть часа, "
                    "120 — через два часа", ge=1, le=10080)],
    title: Annotated[str, Field(
        description="Короткое название для списка. Можно не указывать")] = "",
    chat_id: int = 0,
) -> str:
    """Отложенное поручение: выполнить один раз через заданное время."""
    try:
        job = scheduler.create(_chat(chat_id), kind=scheduler.ONCE,
                               prompt=prompt, in_minutes=in_minutes, title=title)
    except ValueError as err:
        return f"Не удалось завести задание: {err}"
    return (f"Задание #{job['id']} заведено: выполню {_local(job['next_run_at'])} "
            f"и напишу результат сюда же.")


@server.tool()
async def schedule_repeat(
    prompt: Annotated[str, Field(
        description="Что делать каждый раз. Обычная просьба своими словами")],
    every_minutes: Annotated[int, Field(
        description="Период в минутах: 60 — раз в час, 1440 — раз в сутки. "
                    "Каждое выполнение стоит токенов, слишком частым не делай",
        ge=scheduler.MIN_PERIOD, le=scheduler.MAX_PERIOD)],
    title: Annotated[str, Field(
        description="Короткое название для списка. Можно не указывать")] = "",
    chat_id: int = 0,
) -> str:
    """Повторяющееся поручение: выполнять раз в N минут, результат — в диалог."""
    try:
        job = scheduler.create(_chat(chat_id), kind=scheduler.REPEAT,
                               prompt=prompt, every_minutes=every_minutes,
                               title=title)
    except ValueError as err:
        return f"Не удалось завести задание: {err}"
    return (f"Задание #{job['id']} заведено: каждые {every_minutes} мин, "
            f"первый запуск {_local(job['next_run_at'])}.")


@server.tool()
async def schedule_summary(
    prompt: Annotated[str, Field(
        description="Как обобщать. Например: «коротко: что менялось, есть ли "
                    "тренд, на что обратить внимание»")],
    every_minutes: Annotated[int, Field(
        description="Как часто делать сводку, в минутах",
        ge=scheduler.MIN_PERIOD, le=scheduler.MAX_PERIOD)],
    sources: Annotated[list[int] | None, Field(
        description="Номера заданий, чьи результаты обобщать. Пусто — все "
                    "задания этого диалога")] = None,
    title: Annotated[str, Field(
        description="Короткое название для списка. Можно не указывать")] = "",
    chat_id: int = 0,
) -> str:
    """Периодическая сводка: обобщение результатов нескольких заданий.

    Перед выполнением к поручению подкладываются результаты заданий-источников
    за прошедший период — ассистент увидит их и напишет обобщение.
    """
    try:
        job = scheduler.create(_chat(chat_id), kind=scheduler.SUMMARY,
                               prompt=prompt, every_minutes=every_minutes,
                               sources=sources, title=title or "Сводка")
    except ValueError as err:
        return f"Не удалось завести сводку: {err}"
    what = f"по заданиям {', '.join('#' + str(s) for s in sources)}" if sources \
        else "по всем заданиям этого диалога"
    return (f"Сводка #{job['id']} заведена {what}: каждые {every_minutes} мин, "
            f"первая {_local(job['next_run_at'])}.")


@server.tool()
async def list_jobs(chat_id: int = 0) -> str:
    """Все поручения этого диалога: вид, состояние, следующий запуск."""
    try:
        jobs = scheduler.list_jobs(_chat(chat_id))
    except ValueError as err:
        return str(err)
    if not jobs:
        return "Заданий в этом диалоге нет."
    return "Задания этого диалога:\n" + "\n".join(_line(j) for j in jobs)


@server.tool()
async def cancel_job(
    job_id: Annotated[int, Field(description="Номер задания из списка")],
    chat_id: int = 0,
) -> str:
    """Отменить поручение. Оно останется в списке с пометкой «выполнено»."""
    try:
        job = scheduler.set_status(job_id, scheduler.DONE, _chat(chat_id))
    except ValueError as err:
        return str(err)
    if job is None:
        return f"Задания #{job_id} в этом диалоге нет."
    return f"Задание #{job_id} «{job['title']}» отменено."


scheduler.init()

app = server.streamable_http_app(streamable_http_path="/mcp")
