"""CLI-чат с DeepSeek через OpenAI-совместимый API, с поддержкой thinking."""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")
THINKING = os.getenv("LLM_THINKING", "true").lower() == "true"
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "high")
SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful assistant.")

if not API_KEY:
    sys.exit("Не задан LLM_API_KEY. Скопируйте .env.example в .env и впишите ключ.")


def ask(messages: list[dict]) -> tuple[str, str | None]:
    """Отправляет историю сообщений в API и возвращает (ответ, рассуждение)."""
    payload: dict = {"model": MODEL, "messages": messages}
    if THINKING:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = REASONING_EFFORT

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return message["content"], message.get("reasoning_content")


def main() -> None:
    thinking_info = f", thinking={REASONING_EFFORT}" if THINKING else ""
    print(f"Модель: {MODEL}{thinking_info}. Ctrl+C — выход.\n")
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("Вы: ").strip()
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        try:
            answer, reasoning = ask(history)
        except requests.HTTPError as err:
            print(f"Ошибка API {err.response.status_code}: {err.response.text}\n")
            history.pop()
            continue
        except requests.RequestException as err:
            print(f"Ошибка сети: {err}\n")
            history.pop()
            continue

        history.append({"role": "assistant", "content": answer})

        if reasoning:
            print(f"\n[Рассуждение]\n{reasoning}\n")
        print(f"\nLLM: {answer}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПока!")
