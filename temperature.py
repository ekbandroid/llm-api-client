"""Сравнение ответов модели при разной температуре.

Температура задаёт разброс при выборе следующего токена: 0 — всегда самый
вероятный вариант, выше — тем охотнее модель берёт менее вероятные.

Каждая температура прогоняется несколько раз. По одному ответу разнообразие
измерить нельзя: сравнивать не с чем, а именно разнообразие — главное, на что
температура влияет.
"""

import re
import sys
from dataclasses import dataclass, field
from itertools import combinations

import llm
from reasoning import extract_answer, is_correct

DEFAULT_PROMPT = (
    "В каком году вышел первый iPhone? Ответь одним предложением "
    "с коротким пояснением."
)
DEFAULT_REFERENCE = "2007"
DEFAULT_TEMPERATURES = (0.0, 0.7, 1.2)
DEFAULT_SAMPLES = 3

SYSTEM = "You are a helpful assistant."

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def words(text: str) -> list[str]:
    """Слова текста в нижнем регистре, без пунктуации и цифр."""
    return _WORD.findall(text.lower())


def jaccard(a: set[str], b: set[str]) -> float:
    """Доля общих слов: 1 — тексты из одних и тех же слов, 0 — ни одного общего."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


@dataclass
class Sample:
    """Один ответ модели."""

    text: str
    correct: bool | None
    tokens: int
    elapsed: float


@dataclass
class TempResult:
    """Итог по одной температуре."""

    temperature: float
    samples: list[Sample] = field(default_factory=list)
    # Запрос одинаков для всех выборок одной температуры — хранится один раз.
    request: dict = field(default_factory=dict)

    @property
    def texts(self) -> list[str]:
        return [s.text for s in self.samples]

    @property
    def unique(self) -> int:
        """Сколько среди ответов действительно разных (по краткому ответу)."""
        return len({extract_answer(t).strip().lower() for t in self.texts})

    @property
    def diversity(self) -> float:
        """0 — ответы дословно совпадают, 1 — не имеют ни одного общего слова.

        Считается как единица минус средняя попарная доля общих слов.
        """
        sets = [set(words(t)) for t in self.texts]
        pairs = list(combinations(sets, 2))
        if not pairs:
            return 0.0
        return round(1 - sum(jaccard(a, b) for a, b in pairs) / len(pairs), 3)

    @property
    def lexical_richness(self) -> float:
        """Доля неповторяющихся слов в ответе, усреднённая по выборкам.

        Косвенный признак «непредсказуемости» формулировок, не креативности
        как таковой — измерить её числом нельзя, тексты нужно читать глазами.
        """
        ratios = [len(set(w)) / len(w) for w in map(words, self.texts) if w]
        return round(sum(ratios) / len(ratios), 3) if ratios else 0.0

    @property
    def avg_words(self) -> float:
        counts = [len(words(t)) for t in self.texts]
        return round(sum(counts) / len(counts), 1) if counts else 0.0

    @property
    def accuracy(self) -> float | None:
        """Доля верных ответов. None — эталон не задан."""
        graded = [s.correct for s in self.samples if s.correct is not None]
        return round(sum(graded) / len(graded), 3) if graded else None

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.samples)

    @property
    def elapsed(self) -> float:
        return round(sum(s.elapsed for s in self.samples), 2)


def run_temperature(
    prompt: str, temperature: float, *, samples: int, reference: str, model: str | None = None
) -> TempResult:
    """Прогоняет один и тот же запрос несколько раз при заданной температуре."""
    result = TempResult(temperature=temperature)
    for _ in range(samples):
        completion = llm.complete(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=temperature,
            thinking=False,  # рассуждение сгладило бы влияние температуры на ответ
            model=model,
        )
        text = completion.content.strip()
        result.request = completion.request
        result.samples.append(Sample(
            text=text,
            correct=is_correct(text, reference),
            tokens=completion.total_tokens,
            elapsed=completion.elapsed,
        ))
    return result


def compare(
    prompt: str, *, temperatures=DEFAULT_TEMPERATURES, samples: int = DEFAULT_SAMPLES,
    reference: str = "", model: str | None = None,
) -> list[TempResult]:
    return [
        run_temperature(prompt, t, samples=samples, reference=reference, model=model)
        for t in temperatures
    ]


def report(results: list[TempResult], prompt: str, reference: str, samples: int) -> None:
    print(f"Модель: {llm.MODEL}\nЗапрос: {prompt}")
    if reference:
        print(f"Эталон: {reference}")
    print(f"Выборок на каждую температуру: {samples}")

    for res in results:
        print(f"\n{'=' * 72}\nTEMPERATURE = {res.temperature}\n{'=' * 72}")
        for i, s in enumerate(res.samples, 1):
            mark = {True: " [верно]", False: " [неверно]", None: ""}[s.correct]
            print(f"\n--- выборка {i}{mark} ---\n{s.text}")

    print(f"\n{'=' * 72}\nСРАВНЕНИЕ\n{'=' * 72}")
    header = f"{'temperature':>12}{'точность':>11}{'разных':>8}{'разнообразие':>14}{'уник. слов':>12}{'слов':>7}{'токенов':>9}"
    print(header)
    print("-" * len(header))
    for res in results:
        acc = "—" if res.accuracy is None else f"{res.accuracy:.0%}"
        print(f"{res.temperature:>12}{acc:>11}{res.unique:>8}{res.diversity:>14.3f}"
              f"{res.lexical_richness:>12.3f}{res.avg_words:>7}{res.total_tokens:>9}")


def main() -> None:
    args = sys.argv[1:]
    prompt = args[0] if args else DEFAULT_PROMPT
    reference = args[1] if len(args) > 1 else (DEFAULT_REFERENCE if not args else "")
    samples = int(args[2]) if len(args) > 2 else DEFAULT_SAMPLES
    try:
        results = compare(prompt, samples=samples, reference=reference)
    except llm.LLMError as err:
        sys.exit(str(err))
    report(results, prompt, reference, samples)


if __name__ == "__main__":
    main()
