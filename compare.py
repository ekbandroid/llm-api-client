"""Сравнение ответов LLM на один и тот же вопрос: без ограничений и с ограничениями.

Ограниченный вариант задаёт три вещи одновременно:
  1. формат ответа  — жёсткая структура в системном промпте;
  2. длину ответа   — словесный лимит в промпте + max_tokens на стороне API;
  3. завершение     — стоп-последовательность в API + инструкция её выводить.
"""

import sys

import llm

DEFAULT_PROMPT = "Почему в Python стоит использовать виртуальное окружение?"

BASELINE_SYSTEM = "You are a helpful assistant."

WORD_LIMIT = 80
MAX_TOKENS = 250
STOP_MARKER = "###END###"

CONSTRAINED_SYSTEM = f"""Ты отвечаешь строго в следующем формате и ни в каком другом:

ТЕЗИС: <одно предложение>
ПРИЧИНЫ:
- <пункт>
- <пункт>
- <пункт>
ВЫВОД: <одно предложение>

Требования:
- не более {WORD_LIMIT} слов во всём ответе;
- ровно три пункта в разделе ПРИЧИНЫ, каждый не длиннее одной строки;
- никакого текста до строки «ТЕЗИС:»;
- после строки «ВЫВОД:» выведи отдельной строкой {STOP_MARKER} и немедленно прекрати генерацию;
- без markdown-разметки, заголовков и вводных фраз."""


def words(text: str) -> int:
    """Число слов в тексте."""
    return len(text.split())


def check_format(text: str) -> list[tuple[str, bool]]:
    """Проверяет ответ на соответствие затребованному формату."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    bullets = [line for line in lines if line.startswith("-")]
    return [
        ("начинается с «ТЕЗИС:»", bool(lines) and lines[0].startswith("ТЕЗИС:")),
        ("есть раздел «ПРИЧИНЫ:»", any(line.startswith("ПРИЧИНЫ:") for line in lines)),
        ("ровно 3 пункта списка", len(bullets) == 3),
        ("есть раздел «ВЫВОД:»", any(line.startswith("ВЫВОД:") for line in lines)),
        (f"уложился в {WORD_LIMIT} слов", words(text) <= WORD_LIMIT),
        ("нет markdown-мусора", "**" not in text and "##" not in text),
        (f"маркер {STOP_MARKER} обрезан API", STOP_MARKER not in text),
    ]


def run(prompt: str, *, thinking: bool) -> tuple[llm.Completion, llm.Completion]:
    """Делает два вызова с одним и тем же вопросом и возвращает оба ответа."""
    baseline = llm.complete(
        [
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        thinking=thinking,
    )
    constrained = llm.complete(
        [
            {"role": "system", "content": CONSTRAINED_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        stop=[STOP_MARKER],
        thinking=thinking,
    )
    return baseline, constrained


def show(title: str, note: str, result: llm.Completion) -> None:
    """Печатает один ответ с заголовком."""
    print(f"\n{'=' * 70}\n{title}\n{note}\n{'=' * 70}")
    print(result.content.strip())
    if result.truncated:
        print(f"\n[!] Обрыв по max_tokens={MAX_TOKENS} — ответ не дописан до конца.")


def table(baseline: llm.Completion, constrained: llm.Completion) -> None:
    """Печатает сравнительную таблицу метрик."""
    rows = [
        ("символов", len(baseline.content), len(constrained.content)),
        ("слов", words(baseline.content), words(constrained.content)),
        ("строк", baseline.content.count("\n") + 1, constrained.content.count("\n") + 1),
        ("токенов ответа", baseline.completion_tokens, constrained.completion_tokens),
        ("токенов всего", baseline.total_tokens, constrained.total_tokens),
        ("finish_reason", baseline.finish_reason, constrained.finish_reason),
        ("время, с", f"{baseline.elapsed:.1f}", f"{constrained.elapsed:.1f}"),
    ]
    print(f"\n{'=' * 70}\nСРАВНЕНИЕ\n{'=' * 70}")
    print(f"{'метрика':<18}{'без ограничений':>20}{'с ограничениями':>20}")
    print("-" * 58)
    for name, left, right in rows:
        print(f"{name:<18}{str(left):>20}{str(right):>20}")

    if baseline.completion_tokens and constrained.completion_tokens:
        ratio = baseline.completion_tokens / constrained.completion_tokens
        print(f"\nОтвет без ограничений длиннее в {ratio:.1f} раза по токенам.")

    print("\nСоблюдение формата (проверка ограниченного ответа):")
    for label, ok in check_format(constrained.content):
        print(f"  [{'v' if ok else 'x'}] {label}")


def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_PROMPT
    # thinking по умолчанию выключен: рассуждения тратят токены ответа,
    # и max_tokens начинает резать их, а не сам ответ — сравнение перестаёт быть честным.
    thinking = False

    print(f"Модель: {llm.MODEL}\nВопрос: {prompt}")
    try:
        baseline, constrained = run(prompt, thinking=thinking)
    except llm.LLMError as err:
        sys.exit(str(err))

    show("БЕЗ ОГРАНИЧЕНИЙ", "system = «You are a helpful assistant.», без max_tokens и stop", baseline)
    show(
        "С ОГРАНИЧЕНИЯМИ",
        f"формат задан явно, лимит {WORD_LIMIT} слов, max_tokens={MAX_TOKENS}, stop=[{STOP_MARKER}]",
        constrained,
    )
    table(baseline, constrained)


if __name__ == "__main__":
    main()
