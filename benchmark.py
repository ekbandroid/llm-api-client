"""Сравнение одного запроса на разных моделях: время, токены, стоимость.

Сравниваются конфигурации, а не только имена моделей: одна и та же модель
с включённым thinking тратит на ответ заметно больше вычислений, и это такая
же ступень мощности, как переход на более сильную модель.

Цены API не отдаёт, поэтому задаются вручную — за миллион токенов, отдельно
за входные и выходные. Не заданы — стоимость просто не считается.
"""

import sys
from dataclasses import dataclass, field

import llm
from reasoning import extract_answer, is_correct

# Тарифы DeepSeek, долларов за миллион токенов:
# https://api-docs.deepseek.com/quick_start/pricing
#
# Взяты ПИКОВЫЕ ставки без попадания в кэш — верхняя граница, чтобы оценка
# не оказалась заниженной. В непиковые часы вдвое дешевле, а при попадании
# в кэш входные токены дешевле примерно в тридцать раз.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {"in": 0.44, "out": 1.32},
    "deepseek-v4-pro": {"in": 1.32, "out": 3.96},
    "deepseek-v4-flash-vision-exp": {"in": 0.44, "out": 1.32},
}

DEFAULT_PROMPT = (
    "Объясни, чем отличается процесс от потока в операционной системе, "
    "и приведи по одному примеру, когда нужен каждый."
)


@dataclass
class ModelConfig:
    """Что именно сравниваем: модель плюс режим рассуждения."""

    model: str
    thinking: bool = False
    price_in: float = 0.0    # за 1 млн входных токенов
    price_out: float = 0.0   # за 1 млн выходных токенов

    @property
    def title(self) -> str:
        return f"{self.model}{' + thinking' if self.thinking else ''}"

    @property
    def priced(self) -> bool:
        return self.price_in > 0 or self.price_out > 0


@dataclass
class BenchResult:
    """Итог одного запуска."""

    config: ModelConfig
    text: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int
    elapsed: float
    correct: bool | None
    request: dict = field(default_factory=dict)

    @property
    def cost(self) -> float | None:
        """Стоимость запроса. None — цены не заданы."""
        if not self.config.priced:
            return None
        return (
            self.prompt_tokens / 1_000_000 * self.config.price_in
            + self.completion_tokens / 1_000_000 * self.config.price_out
        )

    @property
    def tokens_per_second(self) -> float:
        """Скорость генерации — сопоставима между моделями, в отличие от общего времени."""
        return round(self.completion_tokens / self.elapsed, 1) if self.elapsed else 0.0


def run_config(
    prompt: str, config: ModelConfig, *, reference: str = "", timeout: int = 180
) -> BenchResult:
    """Выполняет запрос в одной конфигурации и снимает замеры."""
    completion = llm.complete(
        [{"role": "user", "content": prompt}],
        model=config.model,
        thinking=config.thinking,
        timeout=timeout,
    )
    text = completion.content.strip()
    return BenchResult(
        config=config,
        text=text,
        reasoning=completion.reasoning,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        total_tokens=completion.total_tokens,
        reasoning_tokens=completion.reasoning_tokens,
        elapsed=round(completion.elapsed, 2),
        correct=is_correct(text, reference),
        request=completion.request,
    )


def compare(
    prompt: str, configs: list[ModelConfig], *, reference: str = ""
) -> list[BenchResult]:
    return [run_config(prompt, c, reference=reference) for c in configs]


def default_configs() -> list[ModelConfig]:
    """Ступени мощности на том, что доступно в аккаунте DeepSeek."""
    return [
        ModelConfig("deepseek-v4-flash"),
        ModelConfig("deepseek-v4-pro"),
        ModelConfig("deepseek-v4-pro", thinking=True),
    ]


def report(results: list[BenchResult], prompt: str, reference: str) -> None:
    print(f"Запрос: {prompt}")
    if reference:
        print(f"Эталон: {reference}")

    for res in results:
        print(f"\n{'=' * 72}\n{res.config.title}\n{'=' * 72}")
        if res.reasoning:
            print(f"[рассуждение, {len(res.reasoning)} символов — скрыто]")
        print(res.text)

    print(f"\n{'=' * 72}\nСРАВНЕНИЕ\n{'=' * 72}")
    header = (f"{'конфигурация':<30}{'время':>8}{'ток/с':>8}{'вход':>7}"
              f"{'выход':>7}{'всего':>7}{'верно':>7}{'цена':>10}")
    print(header)
    print("-" * len(header))
    for r in results:
        mark = {True: "да", False: "нет", None: "—"}[r.correct]
        cost = "—" if r.cost is None else f"{r.cost:.6f}"
        print(f"{r.config.title:<30}{r.elapsed:>7.1f}с{r.tokens_per_second:>8}"
              f"{r.prompt_tokens:>7}{r.completion_tokens:>7}{r.total_tokens:>7}"
              f"{mark:>7}{cost:>10}")

    cheapest = min(results, key=lambda r: r.total_tokens)
    slowest = max(results, key=lambda r: r.elapsed)
    fastest = min(results, key=lambda r: r.elapsed)
    print(f"\nСамая быстрая: {fastest.config.title} ({fastest.elapsed} с).")
    print(f"Самая медленная: {slowest.config.title} ({slowest.elapsed} с), "
          f"в {slowest.elapsed / fastest.elapsed:.1f} раза дольше.")
    print(f"Меньше всего токенов: {cheapest.config.title} ({cheapest.total_tokens}).")


def main() -> None:
    args = sys.argv[1:]
    prompt = args[0] if args else DEFAULT_PROMPT
    reference = args[1] if len(args) > 1 else ""
    try:
        results = compare(prompt, default_configs(), reference=reference)
    except llm.LLMError as err:
        sys.exit(str(err))
    report(results, prompt, reference)


if __name__ == "__main__":
    main()
