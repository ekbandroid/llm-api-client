"""Состояние задачи как конечный автомат: этап, шаг, ожидаемое действие, пауза.

Память отвечает на вопрос «что мы знаем», состояние — «где мы сейчас в работе».
Это разные вещи, поэтому и модуль отдельный от memory.py.

Автомат здесь настоящий, а не подпись к диалогу: переходы разрешены только по
цепочке, и запрос на прыжок «планирование → готово» отклоняется — кем бы он ни
был предложен, человеком через API или моделью в автоматическом режиме.

Пауза — ортогональный флаг, а не пятый этап. Разница между паузой и ожиданием
ответа видна в одном месте: что агент делает, когда придёт следующее сообщение.
Ожидание — норма хода, работа идёт; на паузе работа приостановлена, и агент
обязан не продолжать.
"""

import json

import db
import history
import llm
import tokens as tokens_mod

# Подписи живут в db.py: ими же подписывается отметка о смене этапа, которую
# хранилище добавляет прямо в переписку.
LABELS = db.STAGE_LABELS
CHAIN = " → ".join(LABELS.values())

# Чей ход двигает этап дальше. От этого зависит, когда работает переключатель:
# переход вызывает та сторона, чья реплика служит признаком. Раньше он всегда
# работал после ответа — и реплику «план принят, приступай» ассистент отвечал,
# находясь ещё на планировании: либо отказывался писать код, либо писал и
# получал от ревизора «забежал вперёд». Виновато было приложение, не модель.
STAGE_TRIGGER = {
    db.PLANNING: db.USER_ACTOR,        # пользователь утверждает план
    db.EXECUTION: db.ASSISTANT_ACTOR,  # ассистент предъявляет результат
    db.VALIDATION: db.USER_ACTOR,      # пользователь принимает или находит расхождения
    db.DONE: db.USER_ACTOR,            # пользователь возвращает задачу в работу
}


def switches_before_answer(conversation: dict) -> bool:
    """Нужно ли проверить переход до ответа — по этапу на начало обмена.

    Позиция выбирается один раз, поэтому вызов переключателя остаётся один.
    """
    if not is_task(conversation) or not conversation.get("task_auto"):
        return False
    if conversation.get("task_paused"):
        return False
    return STAGE_TRIGGER[stage_of(conversation)] == db.USER_ACTOR


# Разрешённые переходы. Назад — не «отмена», а нормальная часть работы:
# план оказался негодным, проверка не прошла, готовую задачу вернули в работу.
TRANSITIONS = {
    db.PLANNING: (db.EXECUTION,),
    db.EXECUTION: (db.VALIDATION, db.PLANNING),
    db.VALIDATION: (db.DONE, db.EXECUTION),
    db.DONE: (db.EXECUTION,),
}

# Ради этих инструкций состояние и существует: без них этап остаётся подписью,
# которая ни на что не влияет. Форма — как у инвариантов: неотменяемое правило,
# причина и что делать вместо. Замер показал, почему это важно: с мягкой
# процессной формулировкой модели перепрыгивали этап в 5 случаях из 6 — выдавали
# код на планировании и закрывали задачу мимо проверки, стоило попросить прямо.
# С формулировкой ниже — 0 прыжков из 6 на обеих моделях.
STAGE_RULES = {
    db.PLANNING: (
        "Этап планирования: уточняй непонятное и предлагай план по шагам.\n"
        "Правило жизненного цикла, и просьба пользователя его не отменяет: "
        "на этом этапе не выдавай реализацию — ни кода, ни конфигов, ни готовых "
        "решений, — даже если просят прямо и торопят. Почему: работа без "
        "утверждённого плана переделывается целиком.\n"
        "Если просят реализацию — откажись от этой части, объясни причину, "
        "предложи план по шагам и попроси утвердить его: после утверждения "
        "этап переключится, и код можно будет писать."
    ),
    db.EXECUTION: (
        "Этап выполнения: работай по утверждённому плану, не возвращайся к его "
        "обсуждению без запроса, отмечай, какой шаг закрыт.\n"
        "Правило жизненного цикла, и просьба пользователя его не отменяет: "
        "задачу нельзя объявить законченной, минуя этап проверки. Почему: "
        "непроверенный результат выдаётся за готовый.\n"
        "Если просят подвести финальный итог или закрыть задачу — откажись, "
        "объясни причину и предложи перейти к проверке."
    ),
    db.VALIDATION: (
        "Этап проверки: сверяй результат с планом, ищи расхождения и предлагай "
        "правки.\n"
        "Правило жизненного цикла, и просьба пользователя его не отменяет: "
        "новая работа на этом этапе не начинается. Почему: новая работа посреди "
        "проверки оставляет непроверенным и её, и прежнее.\n"
        "Если просят новую функциональность — откажись, объясни причину и "
        "предложи вынести её в отдельную задачу или вернуть текущую в выполнение."
    ),
    db.DONE: (
        "Задача завершена: отвечай кратко и справочно.\n"
        "Правило жизненного цикла, и просьба пользователя его не отменяет: "
        "новую работу по завершённой задаче не начинай. Почему: сделанное после "
        "закрытия остаётся непроверенным.\n"
        "Если просят продолжить — скажи, что задачу нужно вернуть в выполнение."
    ),
}

# Что ревизор проверяет на каждом этапе. Правило выше написано как инструкция
# модели — «работай по утверждённому плану, не возвращайся к его обсуждению»;
# ревизору такая форма не годится. Проверено на живом диалоге: из неё он
# вычитал запрет предъявлять код на этапе выполнения и пометил нарушением
# ровно ту работу, ради которой этап существует. Поэтому ревизору идёт
# закрытый список: что на этапе положено и что единственное считается прыжком
# через этап. Открытая формулировка оставляет место догадке, закрытая — нет.
STAGE_CHECK = {
    db.PLANNING: {
        "expected": "вопросы по задаче, обсуждение, план по шагам, просьба "
                    "утвердить план, отказ писать код до утверждения",
        "forbidden": "реализация: код, конфиги, команды, тексты файлов, "
                     "готовое решение",
    },
    db.EXECUTION: {
        "expected": "работа по утверждённому плану — код, конфиги, команды, "
                    "полная реализация шагов; отметки о закрытых шагах; "
                    "предъявление готового результата и предложение перейти "
                    "к проверке",
        "forbidden": "объявление задачи законченной, готовой или проверенной "
                     "в обход этапа проверки",
    },
    db.VALIDATION: {
        "expected": "сверка результата с планом, поиск расхождений, список "
                    "правок, вывод о готовности задачи",
        "forbidden": "новая работа: функциональность, которой не было в плане",
    },
    db.DONE: {
        "expected": "короткие справочные ответы по завершённой задаче",
        "forbidden": "новая работа по завершённой задаче",
    },
}


def judge_rule(stage: str) -> str:
    """Правило этапа в том виде, в каком его проверяет ревизор."""
    check = STAGE_CHECK[stage]
    return (
        f"На этапе «{LABELS[stage]}» ассистент обязан делать именно это: "
        f"{check['expected']}. Это работа самого этапа — сколько бы её ни было "
        "и чем бы она ни заканчивалась, нарушением она не является.\n"
        f"Нарушение правила этапа — ТОЛЬКО это: {check['forbidden']}.\n"
        "Ничто другое нарушением этапа не считается."
    )


PAUSED_RULE = (
    "Задача на паузе. Остановись: коротко подтверди, на чём остановились — "
    "этап и текущий шаг, — и жди возобновления. Работу не продолжай и по "
    "существу задачи не отвечай."
)

ACTOR_LABELS = {db.USER_ACTOR: "пользователя", db.ASSISTANT_ACTOR: "ассистента"}


def paused_reply(conversation: dict) -> str:
    """Ответ на сообщение в паузу — его формирует приложение, а не модель.

    Проверено на живой модели: одной инструкции в системном блоке мало. Стоит
    пользователю написать «продолжай», и модель продолжает работу, потому что
    прямая просьба в последнем сообщении перевешивает указание в системном
    блоке. Пауза — правило, а правила соблюдает код: к API мы просто не идём.
    Заодно это ничего не стоит и отвечает мгновенно.
    """
    stage = stage_of(conversation)
    lines = [f"Задача на паузе. Остановились на этапе «{LABELS[stage]}»."]
    if step := (conversation.get("task_step") or "").strip():
        lines.append(f"Текущий шаг: {step}")
    if expected := (conversation.get("task_expected") or "").strip():
        actor = ACTOR_LABELS.get(conversation.get("task_actor") or db.USER_ACTOR, "пользователя")
        lines.append(f"Ожидается: {expected} — ход {actor}.")
    lines.append(
        "Сообщение сохранено в диалоге. Работа продолжится после нажатия "
        "«Продолжить» во вкладке «Задача»."
    )
    return "\n\n".join(lines)


def is_paused(conversation: dict) -> bool:
    """Приостановлена ли задача. Для обычного чата — всегда нет."""
    return is_task(conversation) and bool(conversation.get("task_paused"))


def is_task(conversation: dict) -> bool:
    """Включён ли у диалога режим задачи."""
    return (conversation.get("task_mode") or db.CHAT_MODE) == db.TASK_MODE


def stage_of(conversation: dict) -> str:
    return conversation.get("task_stage") or db.PLANNING


def allowed(stage: str) -> tuple:
    """Куда можно уйти с этого этапа."""
    return TRANSITIONS.get(stage, ())


def can_move(from_stage: str, to_stage: str) -> bool:
    """Разрешён ли переход по таблице. Условие входа проверяется отдельно."""
    return to_stage in allowed(from_stage)


# Этап → условие, без которого в него не пускают. Назад условий нет: иначе
# ошибку нельзя было бы исправить.
GUARD_FOR_STAGE = {stage: guard for guard, stage in db.GUARD_STAGE.items()}


def guard_for(stage: str) -> str | None:
    """Условие входа в этап или None, если его нет."""
    return GUARD_FOR_STAGE.get(stage)


def guard_met(conversation: dict, guard: str) -> bool:
    return bool(conversation.get(db.GUARD_COLUMN[guard]))


def guards(conversation: dict) -> list[dict]:
    """Все условия с отметками — для панели и телеметрии."""
    return [
        {
            "id": guard,
            "label": db.GUARD_LABELS[guard],
            "opens": stage,
            "opens_label": LABELS[stage],
            "met": guard_met(conversation, guard),
        }
        for guard, stage in db.GUARD_STAGE.items()
    ]


def blocked(conversation: dict, target: str) -> str | None:
    """Почему нельзя войти в этап. None — можно.

    Переход вперёд без отметки невозможен никому: ни человеку, ни
    переключателю. Отметка отделена от перехода намеренно — она и есть акт
    утверждения, и сделать её мимоходом нельзя.
    """
    guard = guard_for(target)
    if guard and not guard_met(conversation, guard):
        return (f"нельзя в «{LABELS[target]}»: сначала отметьте "
                f"«{db.GUARD_LABELS[guard]}»")
    return None


def state_text(conversation: dict) -> str:
    """Текст блока состояния для запроса."""
    stage = stage_of(conversation)
    parts = [
        f"Состояние задачи. Этап: {LABELS[stage]} ({CHAIN}).",
        # В истории остаются реплики, сказанные на прежнем этапе. Одной
        # оговорки мало — проверено: модель повторяла «сейчас этап
        # планирования», когда задача уже была в выполнении. Поэтому смена
        # этапа ещё и отмечается сообщением прямо в переписке.
        "Этап мог смениться по ходу разговора: верен тот, что назван здесь, "
        "а не тот, что упоминался в прежних репликах.",
    ]

    if step := (conversation.get("task_step") or "").strip():
        parts.append(f"Текущий шаг: {step}")

    expected = (conversation.get("task_expected") or "").strip()
    actor = ACTOR_LABELS.get(conversation.get("task_actor") or db.USER_ACTOR, "пользователя")
    if expected:
        parts.append(f"Сейчас ход {actor}, ожидается: {expected}")
    else:
        parts.append(f"Сейчас ход {actor}.")

    # Правило паузы важнее правила этапа: оно отменяет работу целиком.
    parts.append(PAUSED_RULE if conversation.get("task_paused") else STAGE_RULES[stage])

    # Что нужно, чтобы двинуться дальше: модель должна понимать, чего просить.
    for nxt in allowed(stage):
        if db.STAGES.index(nxt) > db.STAGES.index(stage):
            guard = guard_for(nxt)
            if guard and not guard_met(conversation, guard):
                parts.append(
                    f"Переход к этапу «{LABELS[nxt]}» откроется, когда будет "
                    f"отмечено условие «{db.GUARD_LABELS[guard]}»."
                )
    return "\n".join(parts)


def blocks(conversation: dict) -> list[dict]:
    """System-сообщение с состоянием — только для диалогов в режиме задачи."""
    if not is_task(conversation):
        return []
    return [{"role": "system", "content": state_text(conversation)}]


def describe(conversation: dict) -> dict:
    """Что состояние добавило к запросу — для интерфейса и meta сообщения."""
    if not is_task(conversation):
        return {}
    text = state_text(conversation)
    return {
        "stage": stage_of(conversation),
        "label": LABELS[stage_of(conversation)],
        "paused": bool(conversation.get("task_paused")),
        "auto": bool(conversation.get("task_auto")),
        "autopilot": is_autopilot(conversation),
        "tokens": tokens_mod.estimate_tokens(text),
    }


def is_autopilot(conversation: dict) -> bool:
    """Включён ли автопилот: модель отвечает и за пользователя."""
    return is_task(conversation) and bool(conversation.get("task_autopilot"))


# ---------- автоматическое переключение ----------
#
# Служебный запрос после ответа. В него уходит не вся переписка (на длинном
# диалоге это тысячи токенов), а состояние, карточка фактов и последний обмен:
# карточка уже хранит цель, договорённости и принятые решения, то есть
# согласованный план в сжатом виде.

SWITCH_SYSTEM = (
    "Ты следишь за состоянием задачи и не участвуешь в разговоре. Тебе дают "
    "текущее состояние, известные факты и последний обмен репликами. Верни "
    "ТОЛЬКО json-объект вида "
    '{"stage": "<этап или null>", "step": "<текущий шаг>", '
    '"expected": "<ожидаемое действие>", "actor": "user|assistant", '
    '"guard_met": true|false, "reason": "<коротко, почему переход>"}.\n'
    "Этапы: planning → execution → validation → done.\n"
    "Переход засчитывается только по признаку:\n"
    "- planning → execution: план сформулирован по шагам И пользователь его "
    "принял (явное согласие либо он сам перешёл к деталям реализации);\n"
    "- execution → validation: все шаги плана закрыты, результат предъявлен;\n"
    "- execution → planning: план оказался негодным — новое требование или "
    "противоречие;\n"
    "- validation → done: проверка проведена, расхождений нет либо они "
    "исправлены и приняты;\n"
    "- validation → execution: найдены расхождения, которые надо править;\n"
    "- done → execution: пользователь просит доработать.\n"
    "Если признака нет или есть сомнение — верни \"stage\": null. Остаться "
    "дешевле, чем откатывать ошибочный переход.\n"
    "Переход назад засчитывается только по сигналу от пользователя, а не по "
    "рассуждению ассистента.\n"
    "Поля step, expected и actor заполняй всегда по последнему обмену, даже "
    "когда этап не меняется. Без непустого reason переход не будет принят.\n"
    "У перехода вперёд есть условие входа — оно названо во входных данных. "
    "Ставь \"guard_met\": true, только если условие действительно выполнено по "
    "переписке, и назови признак в reason. Если признака нет — верни "
    '"stage": null: отметить условие значит утвердить работу предыдущего этапа, '
    "и мимоходом это не делается."
)


def _parse(text: str) -> dict | None:
    """Разбирает ответ переключателя. Не JSON — значит, состояние не трогаем."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def refresh(conversation: dict, exchange: list[dict], *, model: str | None = None) -> dict | None:
    """Обновляет состояние по последнему обмену. Возвращает сведения или None.

    Возвращает None, когда переключатель не должен работать вовсе: обычный чат,
    ручной режим или пауза. Пауза — именно остановка: пока она стоит, этапы не
    двигаются, и снимать её может только человек.
    """
    if not is_task(conversation) or not conversation.get("task_auto"):
        return None
    if conversation.get("task_paused"):
        return None
    if not exchange:
        return None

    stage = stage_of(conversation)
    # Отклонённая по инварианту просьба идёт без текста: иначе «переводим
    # на MongoDB, решение принято» выглядело бы как сигнал к переходу.
    transcript = "\n".join(history.transcript_line(m) for m in exchange)
    facts = conversation.get("facts") or ""
    user_part = (
        f"Этап: {stage}\n"
        f"Текущий шаг: {conversation.get('task_step') or '(не задан)'}\n"
        f"Ожидается: {conversation.get('task_expected') or '(не задано)'}\n"
        f"Чей ход: {conversation.get('task_actor') or db.USER_ACTOR}\n"
        f"Допустимые переходы отсюда: {', '.join(allowed(stage)) or 'нет'}\n"
        + "".join(
            f"Условие входа в «{LABELS[nxt]}»: {db.GUARD_LABELS[guard_for(nxt)]} — "
            f"{'отмечено' if guard_met(conversation, guard_for(nxt)) else 'НЕ отмечено'}\n"
            for nxt in allowed(stage) if guard_for(nxt)
        )
        + "\n"
        + (f"Известные факты: {facts}\n\n" if facts else "")
        + f"Последний обмен:\n{transcript}"
    )

    result = llm.complete(
        [{"role": "system", "content": SWITCH_SYSTEM},
         {"role": "user", "content": user_part}],
        model=model, thinking=False, max_tokens=400,
        response_format={"type": db.JSON_FORMAT},
        # Решение о переходе — тоже вердикт: на одном и том же обмене оно
        # должно быть одним и тем же.
        temperature=0,
        # Ответ переключателя — его решение; в интерфейсе он виден только здесь.
        keep_text=True,
    )
    telemetry = {"kind": "task", "request": result.request, "response": result.response}
    parsed = _parse(result.content)
    if parsed is None:
        return {"ok": False, "reason": "ответ переключателя не разобран как JSON",
                "cost_tokens": result.total_tokens, **telemetry}

    info = {
        "ok": True,
        "cost_tokens": result.total_tokens,
        **telemetry,
        "from_stage": stage,
        "stage": stage,
        "moved": False,
        "reason": (parsed.get("reason") or "").strip(),
    }

    # Шаг, ожидаемое действие и чей ход обновляем всегда: именно свежесть
    # этих полей избавляет от повторных объяснений после паузы.
    step = (parsed.get("step") or "").strip()
    expected = (parsed.get("expected") or "").strip()
    actor = parsed.get("actor") if parsed.get("actor") in db.ACTORS else None
    if step or expected or actor:
        db.update_task(
            conversation["id"], conversation["user_id"],
            step=step or None, expected=expected or None, actor=actor,
        )
        info.update(step=step, expected=expected, actor=actor)

    proposed = parsed.get("stage")
    if not proposed or proposed == stage:
        return info

    # Выбор перехода модели доверяем, соблюдение правил — нет.
    if not can_move(stage, proposed):
        info["rejected"] = f"переход {stage} → {proposed} недопустим"
        return info
    if not info["reason"]:
        info["rejected"] = "переход без причины"
        return info

    # Условие входа обязательно и для модели. Отметить его она может, но
    # только явно — заявив guard_met и назвав признак в reason.
    guard = guard_for(proposed)
    if guard and not guard_met(conversation, guard):
        if not parsed.get("guard_met"):
            info["rejected"] = f"условие «{db.GUARD_LABELS[guard]}» не отмечено"
            return info
        db.set_task_guard(conversation["id"], conversation["user_id"], guard, True,
                          note=info["reason"], author="model")
        info["guard_set"] = guard
        info["guard_label"] = db.GUARD_LABELS[guard]

    db.set_task_stage(conversation["id"], conversation["user_id"], proposed,
                      note=info["reason"], author="model")
    info.update(stage=proposed, moved=True)
    return info


# ---------- автопилот: модель за пользователя ----------
#
# Отдельный служебный вызов пишет следующую реплику пользователя. Реплика потом
# идёт через тот же конвейер, что и настоящая: слои памяти, блок состояния,
# карточка фактов, переключатель. Иначе автопилот проверял бы не то приложение,
# которое видит человек.
#
# Модель-«пользователь» неизбежно принимает решения, которых человек не
# принимал: ОС, пути, названия. Иначе ей нечем отвечать на уточняющие вопросы.
# Поэтому её реплики помечаются в базе и в интерфейсе.

SIMULATED_USER_SYSTEM = (
    "Ты играешь пользователя, который поставил задачу ассистенту и хочет "
    "довести её до конца. Пиши от первого лица, коротко, как человек в чате, "
    "без приветствий и подписей.\n"
    "Спрашивают — отвечай на вопросы, принимая разумные решения сам.\n"
    "Предлагают план — прими его или попроси одну конкретную правку.\n"
    "Показывают результат — проверь по существу: подтверди, что всё сходится, "
    "или назови конкретное расхождение.\n"
    "Задача сделана и проверена — так и скажи, коротко.\n"
    "Не пиши код и не делай работу за ассистента. Не благодари без дела. "
    "Верни только текст реплики."
)

STAGE_HINTS = {
    db.PLANNING: "ассистент уточняет задачу и предлагает план",
    db.EXECUTION: "ассистент выполняет утверждённый план",
    db.VALIDATION: "ассистент проверяет результат",
    db.DONE: "задача завершена",
}


def simulate_user(
    conversation: dict, task_text: str, last_answer: str, *, model: str | None = None,
) -> dict:
    """Пишет следующую реплику за пользователя.

    На вход — не вся переписка, а то, без чего нельзя ответить: исходная
    задача, карточка фактов, этап и последний ответ ассистента. Так вызов
    стоит одинаково на любой длине диалога.
    """
    stage = stage_of(conversation)
    facts = conversation.get("facts") or ""
    user_part = (
        f"Исходная задача:\n{task_text}\n\n"
        + (f"Что уже решено: {facts}\n\n" if facts else "")
        + f"Этап: {LABELS[stage]} — {STAGE_HINTS[stage]}.\n\n"
        f"Последний ответ ассистента:\n{last_answer}\n\n"
        "Напиши свою следующую реплику."
    )
    result = llm.complete(
        [{"role": "system", "content": SIMULATED_USER_SYSTEM},
         {"role": "user", "content": user_part}],
        model=model, thinking=False, max_tokens=300,
        # Реплика больше нигде не видна целиком вместе с её запросом.
        keep_text=True,
    )
    return {
        "kind": "simulated_user",
        "text": result.content.strip(),
        "cost_tokens": result.total_tokens,
        "request": result.request,
        "response": result.response,
    }

