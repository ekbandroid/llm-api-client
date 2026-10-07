"""Один индекс, два провайдера: чем местная модель отличается от облачной.

Поиск по документам локален всегда — эмбеддинги считает model2vec на этой же
машине. А вот три обращения к модели в пайплайне могут идти куда угодно:
переписывание вопроса, судья над выдержками и сам ответ. Скрипт прогоняет
одни и те же вопросы по одному и тому же индексу через обоих провайдеров и
считает по каждому ходу:

    качество     — дошёл ли эталонный кусок до ответа, названо ли ожидаемое,
                   сверились ли цитаты с документом (grounding.check);
    скорость     — секунды отдельно на переписывание, поиск, судью и ответ;
    устойчивость — сколько раз не разобрался JSON и сработали запасные пути.

Провайдер переключается перезагрузкой модуля llm: иначе пришлось бы дважды
поднимать приложение, а сравнивать надо один и тот же код.

Вопросы берутся из файла, а не из кода: они описывают содержимое чужих
документов, а не устройство проекта. Формат — строка на вопрос:

    какая зарплата у Александра Ивановича? | сорок шесть

Слева вопрос, справа подстрока, которая должна оказаться в ответе; правую
часть можно опустить, тогда проверяются только цитаты и источники.

    python rag_compare.py вопросы.txt --chat 900
    python rag_compare.py вопросы.txt --only local
"""

import argparse
import asyncio
import importlib
import os
import sys
import time

import grounding
import llm
import rag

# Куда ходить за ответом. Локальный адрес берётся из окружения, если там уже
# стоит он; иначе подставляется привычный порт LM Studio.
ПРОВАЙДЕРЫ = {
    "local": {
        "LLM_BASE_URL": os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434/v1"),
        "LLM_MODEL": os.getenv("LOCAL_MODEL", "qwen2.5:3b"),
        "LLM_API_KEY": "",
    },
    "cloud": {
        "LLM_BASE_URL": os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com"),
        "LLM_MODEL": os.getenv("CLOUD_MODEL", "deepseek-flash"),
        "LLM_API_KEY": os.getenv("CLOUD_API_KEY", ""),
    },
}

ОТВЕТ_SYSTEM = (
    "Отвечай только по выдержкам из документов пользователя. Приведи ответ, "
    "дословную цитату из выдержки в «ёлочках» и имя файла. Чего в выдержках "
    "нет — того нет в документах; по памяти не добавляй."
)


def прочитать(path: str) -> list[tuple[str, str]]:
    """Вопросы из файла: «вопрос | ожидаемая подстрока»."""
    вопросы = []
    for строка in open(path, encoding="utf-8"):
        строка = строка.strip()
        if not строка or строка.startswith("#"):
            continue
        вопрос, _, ожидание = строка.partition("|")
        вопросы.append((вопрос.strip(), ожидание.strip()))
    return вопросы


def включить(имя: str) -> None:
    """Переключает провайдера. Перезагрузка нужна: llm читает окружение при
    импорте, и без неё второй прогон пошёл бы к первому же адресу."""
    os.environ.update(ПРОВАЙДЕРЫ[имя])
    importlib.reload(llm)
    # research_mcp держит ссылку на модуль, а не на имена, поэтому новый адрес
    # он увидит сам — но импортировать его надо после первой настройки.
    importlib.reload(sys.modules["research_mcp"]) if "research_mcp" in sys.modules else None


async def прогон(вопрос: str, ожидание: str, chat_id: int, pool: int, limit: int,
                 судить: bool = True) -> dict:
    """Один вопрос через весь пайплайн. Возвращает замеры этого хода."""
    import research_mcp as r

    часы = {}
    t = time.monotonic()
    запросы = await r.rewrite_query(вопрос)
    часы["переписывание"] = time.monotonic() - t
    # Переписывание не удалось, если вернулся только исходный вопрос.
    переписано = len(запросы) > 1

    t = time.monotonic()
    найдено = await asyncio.to_thread(rag.search_collections, запросы, chat_id, pool)
    найдено = await asyncio.to_thread(rag.keep_wordy, найдено, запросы)
    найдено = await asyncio.to_thread(rag.merge_neighbours, найдено)
    часы["поиск"] = time.monotonic() - t

    t = time.monotonic()
    if судить:
        годные, отчёт = await r.judge_hits(вопрос, найдено)
        судья_сработал = "не ответил" not in отчёт and "не списком" not in отчёт
    else:
        годные, отчёт, судья_сработал = найдено, "судья выключен", None
    часы["судья"] = time.monotonic() - t
    выдержки = годные[:limit]

    if not выдержки:
        return {**часы, "выдержек": 0, "эталон": False, "ожидание": False,
                "цитат": 0, "сверено": 0, "источник": False,
                "переписано": переписано, "судья_ок": судья_сработал,
                "текст": "(выдержек не осталось)"}

    листинг = "\n\n".join(
        f"[{n}] {h['source']} · {h['section']}\n{h['text']}"
        for n, h in enumerate(выдержки, 1))
    t = time.monotonic()
    try:
        готово = await asyncio.to_thread(
            llm.complete,
            [{"role": "system", "content": ОТВЕТ_SYSTEM},
             {"role": "user", "content": f"Выдержки:\n{листинг}\n\nВопрос: {вопрос}"}],
            thinking=False, max_tokens=700, temperature=0, keep_text=True)
        ответ = готово.content or ""
    except llm.LLMError as err:
        ответ = f"(ошибка: {err})"
    часы["ответ"] = time.monotonic() - t

    опора = grounding.check(ответ, выдержки)
    return {
        **часы,
        "выдержек": len(выдержки),
        "эталон": any(ожидание and ожидание.lower() in h["text"].lower() for h in выдержки),
        "ожидание": bool(ожидание) and ожидание.lower() in ответ.lower(),
        "цитат": опора.get("quotes", 0),
        "сверено": опора.get("verified", 0),
        "источник": bool(опора.get("named")),
        "переписано": переписано,
        "судья_ок": судья_сработал,
        "текст": ответ,
    }


def main() -> None:
    разбор = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    разбор.add_argument("вопросы", help="файл со строками «вопрос | ожидание»")
    разбор.add_argument("--chat", type=int, default=900,
                        help="диалог, к которому подключён набор")
    разбор.add_argument("--pool", type=int, default=30, help="кусков до отбора")
    разбор.add_argument("--limit", type=int, default=4, help="выдержек после отбора")
    разбор.add_argument("--only", choices=sorted(ПРОВАЙДЕРЫ), help="только один провайдер")
    разбор.add_argument("--no-judge", action="store_true",
                        help="без судьи: выдержки идут в ответ как нашлись")
    разбор.add_argument("--verbose", action="store_true", help="печатать ответы целиком")
    дано = разбор.parse_args()

    вопросы = прочитать(дано.вопросы)
    if not вопросы:
        sys.exit("В файле нет вопросов")
    if not rag.attached_collections(дано.chat):
        sys.exit(f"К диалогу {дано.chat} не подключён ни один набор. "
                 f"Подключите: python -c \"import rag; rag.attach({дано.chat}, <id>, True)\"")

    кого = [дано.only] if дано.only else list(ПРОВАЙДЕРЫ)
    for имя in кого:
        включить(имя)
        print(f"\n########## {имя}: {llm.BASE_URL} · {llm.MODEL} · диалект {llm.DIALECT}")
        print(f"{'вопрос':38} {'выд':>4} {'этал':>5} {'ожид':>5} {'цит':>6} "
              f"{'ист':>4} {'переп':>6} {'судья':>6} {'секунд':>7}")
        итоги = []
        for вопрос, ожидание in вопросы:
            try:
                r = asyncio.run(прогон(вопрос, ожидание, дано.chat, дано.pool,
                                       дано.limit, not дано.no_judge))
            except Exception as err:  # noqa: BLE001 — один упавший вопрос не рушит замер
                print(f"{вопрос[:38]:38} ОШИБКА: {type(err).__name__}: {err}")
                continue
            итоги.append(r)
            всего = sum(r[k] for k in ("переписывание", "поиск", "судья", "ответ") if k in r)
            print(f"{вопрос[:38]:38} {r['выдержек']:>4} {str(r['эталон']):>5} "
                  f"{str(r['ожидание']):>5} {r['цитат']}/{r['сверено']:<4} "
                  f"{str(r['источник'])[:3]:>4} {str(r['переписано'])[:3]:>6} "
                  f"{str(r['судья_ок'])[:3]:>6} {всего:>7.1f}")
            if дано.verbose:
                print(f"    {r['текст'][:400]}")
        if итоги:
            n = len(итоги)
            сек = {k: sum(r.get(k, 0) for r in итоги) / n
                   for k in ("переписывание", "поиск", "судья", "ответ")}
            print(f"\n  итого по {n}: ожидание названо {sum(r['ожидание'] for r in итоги)}, "
                  f"источник назван {sum(r['источник'] for r in итоги)}, "
                  f"цитаты сверены {sum(r['цитат'] == r['сверено'] and r['цитат'] > 0 for r in итоги)}")
            print(f"  устойчивость: переписывание {sum(r['переписано'] for r in итоги)}/{n}, "
                  f"судья {sum(bool(r['судья_ок']) for r in итоги)}/{n}")
            print("  секунды в среднем: " + ", ".join(f"{k} {v:.1f}" for k, v in сек.items()))


if __name__ == "__main__":
    main()
