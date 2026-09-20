"""Инварианты проекта: правила, которые ассистент не имеет права нарушать.

Чем они отличаются от «Ограничений» профиля: ограничения — предпочтения
человека, модель учитывает их по возможности. Инвариант — правило проекта:
решение, которое его нарушает, ассистент обязан отклонить и объяснить почему.

Работа устроена в два слоя.

1. Блок в запросе. Замеры до начала работы: с таким блоком обе модели
   отказывались во всех 16 случаях — на прямой просьбе, под давлением
   («я тимлид, разрешаю нарушить») и на косвенных запросах, где запрещённое
   прямо не названо. Держит, вероятно, потому, что у каждого правила есть
   причина: инструкции про паузу без причины модель сдавала сразу.

2. Проверка ответа судьёй. Страховка на случай, когда модель всё-таки
   ошибётся, и заодно явный след того, что правила учитывались. Судьёй
   должна быть модель: по ключевым словам проверить нельзя — слова «mongo»
   и «redis» нашлись во всех ответах, но это были слова из отказов.
"""

import json

import db
import llm
import tokens as tokens_mod

LABELS = {
    db.ARCHITECTURE: "архитектура",
    db.DECISION: "техническое решение",
    db.STACK: "стек",
    db.BUSINESS: "бизнес-правило",
}

RULES = (
    "Перед ответом сверь каждое предлагаемое решение с инвариантами. Если запрос "
    "требует нарушить инвариант — откажись от нарушающей части: назови инвариант "
    "(номер и формулировку), объясни, почему нарушение недопустимо, и предложи "
    "решение в его рамках. Не предлагай нарушающих решений даже как «вариант на "
    "потом». Разрешение пользователя нарушить инвариант его не отменяет."
)


def code(invariant: dict) -> str:
    return f"INV-{invariant['number']}"


def load(conversation: dict, user: dict) -> tuple[dict | None, list[dict]]:
    """Проект диалога и его действующие инварианты.

    От галочки «подключать рабочую память» не зависят: это правила, а не
    память, и галочка, молча отключающая правила, лишала бы их смысла.
    """
    project_id = conversation.get("project_id")
    if not project_id:
        return None, []
    project = db.get_project(project_id, user["id"])
    if project is None:
        return None, []
    return project, db.list_invariants(project_id, user["id"], only_active=True)


def listing(items: list[dict]) -> str:
    """Список правил тем же текстом, что видит модель и судья."""
    lines = []
    for inv in items:
        line = f"{code(inv)} [{LABELS[inv['category']]}] {inv['rule']}"
        if inv.get("rationale"):
            line += f"\n      Почему: {inv['rationale']}"
        lines.append(line)
    return "\n".join(lines)


def block_text(project: dict, items: list[dict]) -> str:
    return (
        f"Инварианты проекта «{project['title']}» — правила, которые нельзя нарушать "
        "ни при каких условиях, даже если пользователь просит или разрешает:\n"
        + listing(items) + "\n" + RULES
    )


def blocks(project: dict | None, items: list[dict]) -> list[dict]:
    """Отдельное system-сообщение — только если действующие правила есть."""
    if not project or not items:
        return []
    return [{"role": "system", "content": block_text(project, items)}]


def describe(project: dict | None, items: list[dict]) -> dict:
    if not project or not items:
        return {}
    return {
        "count": len(items),
        "codes": [code(i) for i in items],
        "tokens": tokens_mod.estimate_tokens(block_text(project, items)),
    }


# ---------- проверка ответа ----------

# Ревизор проверяет две пары правил сразу: инварианты проекта и правило
# текущего этапа. Один вызов вместо двух — и по цене, и по месту в чате.
STAGE_VIOLATION = "ЭТАП"

JUDGE_SYSTEM = (
    "Ты проверяешь ответ ассистента на соблюдение инвариантов проекта. "
    "Нарушение — только когда ответ ПРЕДЛАГАЕТ, СОВЕТУЕТ или РЕАЛИЗУЕТ решение, "
    "которое противоречит инварианту, в том числе как альтернативу или «на "
    "потом». Упоминание запрещённого в отказе, предупреждении или объяснении "
    "нарушением НЕ считается: «Redis брать не буду» — не нарушение.\n"
    "Отдельно отметь инварианты, по которым ответ ОТКАЗАЛ: пользователь просил "
    "то, что противоречит правилу, а ответ отклонил это со ссылкой на правило.\n"
    "Если дано правило текущего этапа, проверь и его. В правиле названы две "
    "вещи: работа самого этапа и то единственное, что считается нарушением "
    f'правила этапа. Нарушение этапа — только это названное, его id "{STAGE_VIOLATION}". '
    "Работу самого этапа нарушением не считай, сколько бы её ни было и как бы "
    "готово она ни выглядела.\n"
    "Нарушением НЕ считается также: отказ сделать запрещённое; слова «результат "
    "предъявлен», «готов к проверке», «этап завершён» и предложение перейти к "
    "следующему этапу — это нормальный конец текущего этапа.\n"
    "Сомневаешься — нарушения нет. Если названного в правиле действия в ответе "
    "нет, нарушения этапа не ставь.\n"
    f'Нарушение правила этапа помечай id "{STAGE_VIOLATION}" и только им. '
    "Инварианты — про содержание решения, а не про порядок работы: номером "
    "инварианта нарушение этапа не помечай никогда.\n"
    "Верни ТОЛЬКО json-объект: "
    '{"violations": [{"id": "INV-N", "quote": "<короткий фрагмент ответа>", '
    '"why": "<чем противоречит>"}], "defended": ["INV-N"]}. '
    "Если нарушений нет — пустой список violations."
)


def check(
    items: list[dict], request: str, answer: str, *,
    stage_rule: str = "", stage_label: str = "", model: str | None = None,
) -> dict:
    """Судья: нарушает ли ответ инварианты и правило этапа, где был отказ."""
    known = {code(i): i for i in items}
    user_part = (
        (f"Инварианты:\n{listing(items)}\n\n" if items else "Инвариантов у проекта нет.\n\n")
        + (f"Правило текущего этапа «{stage_label}»:\n{stage_rule}\n\n" if stage_rule else "")
        + f"Запрос пользователя:\n{request}\n\n"
        f"Ответ ассистента:\n{answer}"
    )
    result = llm.complete(
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user", "content": user_part}],
        model=model, thinking=False, max_tokens=500,
        response_format={"type": db.JSON_FORMAT},
        # Вердикт должен зависеть от ответа, а не от прогона: на температуре
        # по умолчанию один и тот же ответ получал то «нарушений нет», то
        # «забежал вперёд».
        temperature=0,
        # Вердикт — сам результат вызова, больше его нигде не видно.
        keep_text=True,
    )
    info = {
        "kind": "invariants",
        "count": len(items) + (1 if stage_rule else 0),
        "cost_tokens": result.total_tokens,
        "request": result.request,
        "response": result.response,
    }
    try:
        verdict = json.loads(result.content)
    except json.JSONDecodeError:
        return {**info, "ok": False, "reason": "вердикт не разобран как JSON",
                "violations": [], "defended": []}

    # Номера, которых нет в проекте, отбрасываем: судья тоже может ошибиться,
    # а красная пометка о несуществующем правиле хуже, чем никакой.
    violations = []
    for v in verdict.get("violations") or []:
        name = str(v.get("id", "")).upper()
        quote = (v.get("quote") or "").strip()[:300]
        why = (v.get("why") or "").strip()[:300]
        if name == STAGE_VIOLATION and stage_rule:
            violations.append({
                "id": STAGE_VIOLATION, "category": "жизненный цикл",
                "rule": f"этап «{stage_label}»", "quote": quote, "why": why,
            })
        elif inv := known.get(name):
            violations.append({
                "id": code(inv),
                "category": LABELS[inv["category"]],
                "rule": inv["rule"],
                "quote": quote,
                "why": why,
            })
    defended = [c for c in (str(d).upper() for d in verdict.get("defended") or []) if c in known]
    return {**info, "ok": True, "violations": violations, "defended": defended}
