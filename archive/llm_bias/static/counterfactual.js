const $ = (id) => document.getElementById(id);
let pairs = [];
let latestResult = null;

function esc(value) { return String(value).replace(/[&<>\"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[char])); }
function token(data, id) { const text = data.vocab[String(id)] ?? data.vocab[id] ?? `#${id}`; return text.replace(/\n/g, "⏎").replace(/\t/g, "⇥").replace(/^ /, "·"); }
function number(value) { return value == null || Number.isNaN(value) ? "—" : Number(value).toFixed(3); }

function renderGrid(name, data, tone, panel) {
  const rows = data.grid.map((row) => {
    const cells = row.top_ids.map((ids, position) => `<div class="cell ${tone}" data-panel="${panel}" data-row="${data.grid.indexOf(row)}" data-position="${position}" title="layer ${row.layer}, position ${position}"><span class="tok">${esc(token(data, ids[0]))}</span><span class="prob">${row.top_probs[position][0]}</span></div>`).join("");
    return `<div class="grid-row" style="--cols:${data.seq_len}"><div class="layer-label">L${row.layer}</div>${cells}</div>`;
  }).join("");
  const labels = data.context_ids.map((id, position) => `<div class="token-label">${position}: ${esc(token(data, id))}</div>`).join("");
  return `<div class="grid-card"><h2>${name} · top-1 readout <small>(hover for top-${data.grid[0]?.top_ids[0]?.length ?? 0})</small></h2><div class="grid"><div class="grid-row" style="--cols:${data.seq_len}"><div></div>${labels}</div>${rows}</div></div>`;
}

function render(result) {
  latestResult = result;
  const pair = result.pair;
  $("source-prompt").textContent = pair.source_prompt;
  $("target-prompt").textContent = pair.target_prompt;
  $("patched-prompt").textContent = `${pair.source_prompt}\n\npatch L${result.patch.layer}, position ${result.patch.position} ← target entity residual`;
  const metrics = result.metrics;
  $("metrics").innerHTML = [["source margin", metrics.source_margin], ["target margin", metrics.target_margin], ["patched margin", metrics.patched_margin], ["normalized transfer", metrics.transfer], ["source answer rank", metrics.source_answer_rank], ["patched target rank", metrics.patched_target_answer_rank]].map(([label, value]) => `<div class="metric"><strong>${number(value)}</strong><span>${label}</span></div>`).join("");
  $("grids").innerHTML = renderGrid("Source", result.source, "source", "source") + renderGrid("Target", result.target, "target", "target") + renderGrid("Patched", result.patched, "patched", "patched");
  const rows = result.comparison.filter((row) => row.position === result.patch.position || row.position === pair.source_entity_start);
  $("comparison").innerHTML = `<table><thead><tr><th>Layer</th><th>Position</th><th>Source</th><th>Target</th><th>Patched</th><th>Δ patched prob</th></tr></thead><tbody>${rows.map((row) => `<tr><td>L${row.layer}</td><td>${row.position}</td><td>${esc(token(result.source, row.source_top1))}</td><td>${esc(token(result.target, row.target_top1))}</td><td>${esc(token(result.patched, row.patched_top1))}</td><td>${number(row.patched_top1_prob - row.source_top1_prob)}</td></tr>`).join("")}</tbody></table>`;
  $("status").textContent = `Rendered ${pair.category}/${pair.function} · L${result.patch.layer} · ${result.source.seq_len} tokens`;
}

const tooltip = $("tooltip");
$("grids").addEventListener("mousemove", (event) => {
  const cell = event.target.closest(".cell");
  if (!cell || !$("grids").contains(cell)) { tooltip.hidden = true; return; }
  const data = latestResult?.[cell.dataset.panel];
  if (!data) { tooltip.hidden = true; return; }
  const row = data.grid[Number(cell.dataset.row)];
  const position = Number(cell.dataset.position);
  const rows = row.top_ids[position].map((id, index) => `<tr><td>${esc(token(data, id))}</td><td class="p">${(row.top_probs[position][index] * 100).toFixed(1)}%</td></tr>`).join("");
  tooltip.innerHTML = `<table>${rows}</table><div class="meta">${row.is_output ? "output" : `L${row.layer}`} · pos ${position} · H=${row.entropy[position]} · kurt=${row.kurtosis[position]}</div>`;
  tooltip.hidden = false;
  const width = tooltip.offsetWidth, height = tooltip.offsetHeight;
  tooltip.style.left = `${Math.min(event.clientX + 14, innerWidth - width - 8)}px`;
  tooltip.style.top = `${Math.min(event.clientY + 14, innerHeight - height - 8)}px`;
});
$("grids").addEventListener("mouseleave", () => { tooltip.hidden = true; });

async function read() {
  const pair = pairs[Number($("pair").value)];
  if (!pair) return;
  $("read").disabled = true; $("status").textContent = "Running source, target, and patched forward passes…";
  try {
    const response = await fetch("/api/counterfactual", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pair_id: pair.pair_id, patch_layer: Number($("layer").value), mode: $("mode").value, top_k: Number($("top-k").value) }) });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    render(await response.json());
  } catch (error) { $("status").textContent = error.message; } finally { $("read").disabled = false; }
}

async function init() {
  const info = await (await fetch("/api/info")).json();
  $("model-info").textContent = `${info.model_id} · ${info.n_layers}L · ${info.device} · lens ${info.fitted_layers.join(", ")}`;
  $("layer").max = String(Math.max(0, info.n_layers - 2));
  pairs = await (await fetch("/api/pairs")).json();
  $("pair").innerHTML = pairs.map((pair, index) => `<option value="${index}">${pair.category} / ${pair.function} · ${pair.source_entity} → ${pair.target_entity}</option>`).join("");
  $("layer").addEventListener("input", () => { $("layer-value").value = $("layer").value; $("layer-value").textContent = $("layer").value; });
  $("read").addEventListener("click", read); await read();
}
init().catch((error) => { $("status").textContent = error.message; });
