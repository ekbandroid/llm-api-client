"""Слои памяти агента: что помнится всегда, что — про задачу, что — про диалог.

Слои различаются не форматом, а сроком жизни и владельцем:

    долговременная — профиль пользователя, `users.profile_memory`;
    рабочая        — проект: бриф, написанный руками, и факты, накопленные
                     из диалогов проекта;
    краткосрочная  — сам диалог: сообщения, конспект, карточка фактов.

Здесь собираются первые два: краткосрочным занимается history.py, он же
вставляет блоки этого модуля в запрос.

Каждый слой уходит в запрос ОТДЕЛЬНЫМ system-сообщением. Так он виден в
аккордеоне «Запрос к API» сам по себе, и любой слой можно отключить, не
пересобирая остальные. Порядок — от постоянного к сиюминутному: сначала
человек, затем задача, затем разговор.
"""

import json
from dataclasses import dataclass, field

import db
import tokens as tokens_mod

PROFILE_PREFIX = "Долговременная память — что известно о собеседнике:\n"

PROJECT_PREFIX = "Рабочая память. Проект «{title}».\n"
PROJECT_BRIEF_PREFIX = "Описание задачи:\n"
PROJECT_FACTS_PREFIX = "Накоплено в диалогах проекта:\n"
PROJECT_EMPTY = "Пока о проекте ничего не записано — не домысливай за него."

# Фактов у проекта больше, чем у диалога (FACTS_LIMIT = 25): сюда стекается
# несколько диалогов сразу. Но потолок всё равно нужен — блок уходит в каждый
# запрос проекта и оплачивается заново.
PROJECT_FACTS_LIMIT = 40


@dataclass
class Layers:
    """Содержимое слоёв, подключённых к конкретному диалогу."""

    profile: str = ""
    project_id: int | None = None
    project_title: str = ""
    project_brief: str = ""
    project_facts: dict = field(default_factory=dict)

    @property
    def has_project(self) -> bool:
        """Проект подключён — блок уходит, даже пока он пуст.

        Пустой проект раньше не давал в запрос ничего, и первый обмен в новом
        проекте шёл вообще без рабочего слоя: бриф ещё не написан, факты
        появляются только после ответа. Со стороны это выглядело поломкой —
        галочка включена, а слоя в запросе нет. Одно название уже задаёт рамку
        разговора, а строка о пустоте удерживает модель от выдумывания.
        """
        return self.project_id is not None


def load_facts(raw: str | None) -> dict:
    """Разбирает карточку фактов. Испорченный JSON трактуем как пустую."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def collect(conversation: dict, user: dict) -> Layers:
    """Собирает слои, подключённые к диалогу.

    Отключённый флажком слой сюда не попадает вовсе — не «попадает и потом
    отфильтровывается». Иначе его содержимое рано или поздно просочилось бы
    в запрос через describe() или отладочный вывод.
    """
    layers = Layers()

    if conversation.get("use_profile", 1):
        layers.profile = (user.get("profile_memory") or "").strip()

    project_id = conversation.get("project_id")
    if project_id and conversation.get("use_project", 1):
        project = db.get_project(project_id, user["id"])
        if project:
            layers.project_id = project["id"]
            layers.project_title = project["title"]
            layers.project_brief = (project["brief"] or "").strip()
            layers.project_facts = load_facts(project["facts"])

    return layers


def project_text(layers: Layers) -> str:
    """Текст блока рабочей памяти."""
    parts = [PROJECT_PREFIX.format(title=layers.project_title)]
    if layers.project_brief:
        parts.append(PROJECT_BRIEF_PREFIX + layers.project_brief)
    if layers.project_facts:
        lines = [f"- {k}: {v}" for k, v in list(layers.project_facts.items())[:PROJECT_FACTS_LIMIT]]
        parts.append(PROJECT_FACTS_PREFIX + "\n".join(lines))
    if not layers.project_brief and not layers.project_facts:
        parts.append(PROJECT_EMPTY)
    return "\n".join(parts)


def blocks(layers: Layers) -> list[dict]:
    """Готовые system-сообщения для запроса — в порядке убывания срока жизни."""
    out = []
    if layers.profile:
        out.append({"role": "system", "content": PROFILE_PREFIX + layers.profile})
    if layers.has_project:
        out.append({"role": "system", "content": project_text(layers)})
    return out


def describe(layers: Layers) -> dict:
    """Что именно дал каждый слой — для интерфейса и для meta сообщения.

    Храним размеры, а не сам текст: он уже виден в аккордеоне запроса, а в
    meta повторялся бы при каждом сообщении и раздувал базу.
    """
    info: dict = {}
    if layers.profile:
        info["profile"] = {
            "chars": len(layers.profile),
            "tokens": tokens_mod.estimate_tokens(PROFILE_PREFIX + layers.profile),
        }
    if layers.has_project:
        text = project_text(layers)
        info["project"] = {
            "title": layers.project_title,
            "chars": len(text),
            "tokens": tokens_mod.estimate_tokens(text),
            "facts": len(layers.project_facts),
            "brief_chars": len(layers.project_brief),
        }
    return info


def absorb_facts(project: dict, facts: dict) -> dict | None:
    """Переливает карточку фактов диалога в рабочую память проекта.

    К модели не обращаемся: факты уже извлечены при обновлении карточки
    диалога, второй раз за ту же работу платить незачем.

    Новое значение вытесняет старое по тому же ключу — считаем, что диалог
    говорит о задаче свежее, чем накопленное раньше. При переполнении
    отбрасываются самые давние ключи: у проекта, идущего месяцами, иначе
    не останется места для того, что выяснилось сегодня.
    """
    if not project or not facts or not project.get("collect_facts", 1):
        return None

    current = load_facts(project.get("facts"))
    merged = {**current, **{str(k): v for k, v in facts.items()}}
    added = [k for k in merged if k not in current]
    changed = [k for k in facts if k in current and current[k] != facts[k]]
    if not added and not changed:
        return None

    if len(merged) > PROJECT_FACTS_LIMIT:
        merged = dict(list(merged.items())[-PROJECT_FACTS_LIMIT:])

    db.set_project_facts(project["id"], json.dumps(merged, ensure_ascii=False))
    return {
        "project_id": project["id"],
        "title": project["title"],
        "added": added,
        "changed": changed,
        "count": len(merged),
    }
