// Общее для всех режимов: показ запроса, который приложение отправляет в API.
// Ключ на сервере уже заменён звёздочками, наружу он не уходит.

function requestDetails(requests, label) {
  const list = Array.isArray(requests) ? requests : [requests];
  if (!list.length || !list[0]) return null;

  const d = document.createElement("details");
  const s = document.createElement("summary");
  s.textContent = label || (list.length > 1 ? `Запросы к API (${list.length})` : "Запрос к API");
  d.appendChild(s);

  list.forEach((req, i) => {
    if (list.length > 1) {
      const h = document.createElement("div");
      h.className = "hint";
      h.style.margin = "8px 0 0";
      h.textContent = `Вызов ${i + 1} из ${list.length}`;
      d.appendChild(h);
    }
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(req, null, 2);
    d.appendChild(pre);
  });
  return d;
}
