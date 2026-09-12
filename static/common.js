// Общее для всех режимов: показ запроса, который приложение отправляет в API.
// Ключ на сервере уже заменён звёздочками, наружу он не уходит.

/** Сворачиваемый блок с JSON одного или нескольких вызовов. */
function jsonDetails(items, label) {
  const list = Array.isArray(items) ? items : [items];
  if (!list.length || !list[0]) return null;

  const d = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = label;
  d.appendChild(summary);

  list.forEach((item, i) => {
    if (list.length > 1) {
      const h = document.createElement("div");
      h.className = "hint";
      h.style.margin = "8px 0 0";
      h.textContent = `Вызов ${i + 1} из ${list.length}`;
      d.appendChild(h);
    }
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(item, null, 2);
    d.appendChild(pre);
  });
  return d;
}

/** Что приложение отправило в API. Ключ на сервере уже заменён звёздочками. */
function requestDetails(requests, label) {
  const n = Array.isArray(requests) ? requests.length : 1;
  return jsonDetails(requests, label || (n > 1 ? `Запросы к API (${n})` : "Запрос к API"));
}

/** Что API вернуло. Текст ответа вырезан на сервере — он показан выше. */
function responseDetails(responses, label) {
  const n = Array.isArray(responses) ? responses.length : 1;
  return jsonDetails(responses, label || (n > 1 ? `Ответы API (${n})` : "Ответ API"));
}

// ---------------------------------------------------------------------------
// Грубая оценка числа токенов — та же, что на сервере в tokens.py.
// Нужна, чтобы показывать размер запроса при наборе, не дёргая API на каждую
// букву. Точное число приходит только в usage после вызова.

// Коэффициенты и правила должны совпадать с RATIOS и ratio_for в tokens.py:
// иначе цифра в чате разойдётся с цифрой на странице разбора.
const TOKEN_RATIOS = { ru: 3.33, en: 5.85, code: 2.41 };
const MESSAGE_OVERHEAD = 4;
const CODEY = /[{}()\[\];=<>]|\bdef\b|\bfunction\b|\bimport\b/g;

function estimateTokens(text) {
  if (!text) return 0;
  if ((text.match(CODEY) || []).length > text.length / 120) {
    return Math.max(1, Math.round(text.length / TOKEN_RATIOS.code));
  }
  const letters = (text.match(/\p{L}/gu) || []).length || 1;
  const cyrillic = (text.match(/[а-яёА-ЯЁ]/g) || []).length;
  const share = cyrillic / letters;
  const ratio = TOKEN_RATIOS.ru * share + TOKEN_RATIOS.en * (1 - share);
  return Math.max(1, Math.round(text.length / ratio));
}

function estimateMessages(texts) {
  return texts.reduce((sum, t) => sum + estimateTokens(t) + MESSAGE_OVERHEAD, 0);
}
