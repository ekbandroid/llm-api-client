"""Решение одной задачи четырьмя способами рассуждения и сравнение результатов.

Способы:
  1. direct   — прямой вопрос без каких-либо дополнительных инструкций;
  2. cot      — инструкция «решай пошагово»;
  3. meta     — модель сначала пишет промпт для решения, затем решает по нему;
  4. experts  — аналитик, инженер и критик решают независимо, затем синтез.

thinking выключен во всех способах: иначе модель рассуждает внутри себя
независимо от промпта и разница между способами смазывается.
"""

import re
import sys
from dataclasses import dataclass, field

import llm

# Задача подобрана так, чтобы прямой ответ на ней ошибался: модель сравнивает
# дробные части 11 и 9 как целые числа и объявляет 9.11 большим.
DEFAULT_TASK = "Что больше: 9.11 или 9.9? В ответе укажите большее число."
DEFAULT_REFERENCE = "9.9"

ANSWER_TAG = "ОТВЕТ:"
_TAIL = f'\n\nЗаверши ответ отдельной строкой вида «{ANSWER_TAG} <краткий ответ>».'


@dataclass
class Stage:
    """Один вызов API внутри способа."""

    title: str
    text: str


@dataclass
class MethodResult:
    """Итог одного способа рассуждения."""

    key: str
    title: str
    note: str
    answer: str
    stages: list[Stage] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class _Runner:
    """Делает вызовы и копит по ним статистику для одного способа."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.elapsed = 0.0

    def ask(self, system: str, user: str) -> str:
        result = llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=self.model,
            thinking=False,
        )
        self.calls += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.elapsed += result.elapsed
        return result.content.strip()

    def finish(self, key: str, title: str, note: str, answer: str, stages: list[Stage]) -> MethodResult:
        return MethodResult(
            key=key, title=title, note=note, answer=answer, stages=stages,
            calls=self.calls, prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens, elapsed=round(self.elapsed, 2),
        )


# ---------- способ 1: прямой ответ ----------

def method_direct(task: str, model: str | None = None) -> MethodResult:
    """Без единой дополнительной инструкции — как есть."""
    r = _Runner(model)
    answer = r.ask("You are a helpful assistant.", task)
    return r.finish(
        "direct", "1. Прямой ответ",
        "Системный промпт по умолчанию, никаких указаний о способе решения.",
        answer, [],
    )


# ---------- способ 2: пошагово ----------

def method_cot(task: str, model: str | None = None) -> MethodResult:
    """Явная инструкция рассуждать по шагам."""
    r = _Runner(model)
    system = (
        "Решай задачу пошагово. Разбей рассуждение на пронумерованные шаги, "
        "в каждом шаге делай ровно одно действие и показывай промежуточный результат. "
        "Не переходи к ответу, пока не выполнил все шаги." + _TAIL
    )
    answer = r.ask(system, task)
    return r.finish(
        "cot", "2. Пошаговое рассуждение",
        "Инструкция разбить решение на нумерованные шаги.",
        answer, [],
    )


# ---------- способ 3: модель сама пишет промпт ----------

def method_meta(task: str, model: str | None = None) -> MethodResult:
    """Сначала просим составить промпт, затем решаем по нему."""
    r = _Runner(model)
    crafted = r.ask(
        "Ты специалист по промпт-инжинирингу. По присланной задаче составь "
        "наилучший промпт для её решения другой языковой моделью: укажи роль, "
        "метод рассуждения и требования к формату ответа. "
        "Верни только текст промпта, без пояснений и без решения самой задачи.",
        f"Задача:\n{task}",
    )
    answer = r.ask(crafted + _TAIL, task)
    return r.finish(
        "meta", "3. Промпт, составленный моделью",
        "Первый вызов создаёт промпт, второй решает задачу по нему.",
        answer, [Stage("Промпт, который придумала модель", crafted)],
    )


# ---------- способ 4: группа экспертов ----------

EXPERTS = [
    ("Аналитик", "Ты аналитик. Твоя сила — вычленить условия задачи, отделить данные "
                 "от предположений и проверить, нет ли скрытой ловушки в формулировке."),
    ("Инженер", "Ты инженер. Твоя сила — формализовать задачу: ввести переменные, "
                "составить уравнения и получить результат вычислением, а не интуицией."),
    ("Критик", "Ты критик. Твоя сила — искать ошибку: перепроверять счёт, подставлять "
               "результат обратно в условие и явно называть типичную ловушку этой задачи."),
]


def method_experts(task: str, model: str | None = None) -> MethodResult:
    """Три эксперта решают независимо, затем сведение в общий ответ.

    Эксперты вызываются отдельными запросами намеренно: в одном общем промпте
    критик видит ответ аналитика и склонен с ним соглашаться.
    """
    r = _Runner(model)
    stages: list[Stage] = []
    for name, persona in EXPERTS:
        opinion = r.ask(persona + _TAIL, task)
        stages.append(Stage(name, opinion))

    digest = "\n\n".join(f"### {s.title}\n{s.text}" for s in stages)
    answer = r.ask(
        "Ты ведущий совещания. Тебе дали независимые решения трёх экспертов. "
        "Сравни их, укажи, где они расходятся, и на чьей стороне правота. "
        "Если все сошлись — проверь их общий ответ самостоятельно." + _TAIL,
        f"Задача:\n{task}\n\nРешения экспертов:\n{digest}",
    )
    return r.finish(
        "experts", "4. Группа экспертов",
        "Аналитик, инженер и критик отвечают независимо, затем сведение.",
        answer, stages,
    )


METHODS = [method_direct, method_cot, method_meta, method_experts]


# ---------- разбор и сверка ответов ----------

_ANSWER_HINT = re.compile(r"^\W*(?:итоговый\s+)?ответ\W*[:：-]?\s*(.+)$", re.IGNORECASE)


def extract_answer(text: str) -> str:
    """Достаёт краткий ответ из свободного текста.

    У первого способа инструкции о формате нет по условию задания, поэтому
    метки ОТВЕТ: в его ответе не будет — приходится угадывать по признакам:
    сначала явная метка, затем строка со словом «ответ», затем последняя
    строка с числом, и лишь в крайнем случае просто последняя строка.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""

    for line in reversed(lines):
        if ANSWER_TAG in line:
            return line.split(ANSWER_TAG, 1)[1].strip(" *_.") or line.strip()

    for line in reversed(lines):
        if match := _ANSWER_HINT.match(line.replace("*", "")):
            return match.group(1).strip(" *_.")

    for line in reversed(lines):
        if re.search(r"\d", line):
            return line.strip(" *_")

    return lines[-1]


def is_correct(text: str, reference: str) -> bool | None:
    """Сверяет ответ с эталоном. None — эталон не задан, сверять не с чем."""
    if not reference:
        return None
    short = extract_answer(text)
    if re.fullmatch(r"-?\d+(?:[.,]\d+)?", reference.strip()):
        # Число ищем по границам слова, иначе «50» найдётся внутри «1050».
        numbers = re.findall(r"-?\d+(?:[.,]\d+)?", short.replace(" ", ""))
        return reference.strip().replace(",", ".") in [n.replace(",", ".") for n in numbers]
    return reference.strip().lower() in short.lower()


def solve_all(task: str, *, model: str | None = None) -> list[MethodResult]:
    """Прогоняет задачу всеми четырьмя способами."""
    return [method(task, model) for method in METHODS]


# ---------- вывод в консоль ----------

def report(results: list[MethodResult], task: str, reference: str) -> None:
    print(f"Модель: {llm.MODEL}\nЗадача: {task}")
    if reference:
        print(f"Эталонный ответ: {reference}")

    for res in results:
        print(f"\n{'=' * 72}\n{res.title}\n{res.note}\n{'=' * 72}")
        for stage in res.stages:
            print(f"\n--- {stage.title} ---\n{stage.text}")
        if res.stages:
            print("\n--- Итоговый ответ ---")
        print(res.answer)

    print(f"\n{'=' * 72}\nСРАВНЕНИЕ\n{'=' * 72}")
    print(f"{'способ':<32}{'вызовов':>8}{'токенов':>9}{'время':>8}  {'вердикт':<10}ответ")
    print("-" * 90)
    for res in results:
        ok = is_correct(res.answer, reference)
        mark = {True: "верно", False: "НЕВЕРНО", None: "—"}[ok]
        short = extract_answer(res.answer)[:28]
        print(f"{res.title:<32}{res.calls:>8}{res.total_tokens:>9}{res.elapsed:>7.1f}с  {mark:<10}{short}")


def stability(task: str, reference: str, runs: int) -> dict[str, list[bool | None]]:
    """Прогоняет каждый способ несколько раз и собирает, где ответ верен.

    Один прогон ничего не говорит о точности: модель недетерминирована и на
    одной и той же задаче отвечает по-разному. Сравнивать имеет смысл долю
    верных ответов, а не единичный результат.
    """
    tally: dict[str, list[bool | None]] = {}
    for _ in range(runs):
        for res in solve_all(task):
            tally.setdefault(res.title, []).append(is_correct(res.answer, reference))
    return tally


def report_stability(tally: dict[str, list[bool | None]], runs: int) -> None:
    print(f"\n{'=' * 72}\nУСТОЙЧИВОСТЬ: {runs} прогонов каждого способа\n{'=' * 72}")
    print(f"{'способ':<32}{'верных':>9}  результат по прогонам")
    print("-" * 72)
    for title, marks in tally.items():
        good = sum(1 for m in marks if m)
        trail = " ".join("+" if m else "-" for m in marks)
        print(f"{title:<32}{good:>4}/{len(marks):<4}  {trail}")


def main() -> None:
    args = sys.argv[1:]
    task = args[0] if args else DEFAULT_TASK
    reference = args[1] if len(args) > 1 else (DEFAULT_REFERENCE if not args else "")
    runs = int(args[2]) if len(args) > 2 else 1

    try:
        report(solve_all(task), task, reference)
        if runs > 1:
            report_stability(stability(task, reference, runs), runs)
    except llm.LLMError as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
