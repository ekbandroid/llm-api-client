# llm-cli

Минимальный консольный чат с LLM через OpenAI-совместимый API. Хранит историю диалога в рамках сессии.

## Установка

```bash
git clone https://github.com/ВАШ_ЛОГИН/llm-cli.git
cd llm-cli

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Настройка

```bash
cp .env.example .env
```

Откройте `.env` и впишите свой ключ. Файл `.env` в `.gitignore`, в репозиторий он не попадёт.

## Запуск

```bash
python app.py
```

## Другие провайдеры

Приложение работает с любым API, совместимым с форматом OpenAI. Меняются только две переменные в `.env`:

| Провайдер | `LLM_BASE_URL` | `LLM_MODEL` |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.5` |
| Ollama (локально) | `http://localhost:11434/v1` | `llama3.2` |

Названия моделей меняются — актуальные смотрите в документации провайдера.

## Структура

```
app.py            — всё приложение
requirements.txt  — зависимости
.env.example      — шаблон конфигурации
.gitignore        — исключает .env и мусор
```
