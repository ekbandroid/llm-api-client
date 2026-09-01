"""CLI-чат с LLM через OpenAI-совместимый API, с поддержкой thinking."""

import llm


def main() -> None:
    thinking_info = f", thinking={llm.REASONING_EFFORT}" if llm.THINKING else ""
    print(f"Модель: {llm.MODEL}{thinking_info}. Ctrl+C — выход.\n")
    history: list[dict] = [{"role": "system", "content": llm.SYSTEM_PROMPT}]

    while True:
        user_input = input("Вы: ").strip()
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        try:
            result = llm.complete(history)
        except llm.LLMError as err:
            print(f"{err}\n")
            history.pop()
            continue

        history.append({"role": "assistant", "content": result.content})

        if result.reasoning:
            print(f"\n[Рассуждение]\n{result.reasoning}\n")
        print(f"\nLLM: {result.content}\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nПока!")
