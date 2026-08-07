(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { result: null, pinnedCell: null, vocabularyFocus: null };
  const MAX_VOCABULARY_OPTIONS = 20;
  const compact = (value) => String(value).replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/^ /, "·") || "∅";
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;",
  })[char]);
  const token = (result, id) => compact(result.vocab[String(id)] ?? result.vocab[id] ?? `#${id}`);
  const percent = (value) => `${(Number(value) * 100).toFixed(Number(value) >= 0.1 ? 1 : 2)}%`;

  function compareTokenIds(left, right) {
    const leftNumber = Number(left);
    const rightNumber = Number(right);
    if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {
      return leftNumber - rightNumber;
    }
    return String(left).localeCompare(String(right));
  }

  function rankVocabulary(counter) {
    return [...counter.entries()]
      .sort(([leftId, leftCount], [rightId, rightCount]) =>
        rightCount - leftCount || compareTokenIds(leftId, rightId));
  }

  function collectVocabularyStats(result) {
    const global = new Map();
    const rows = result.grid.map((row, rowIndex) => {
      const counts = new Map();
      let cellCount = 0;
      (row.top_ids || []).forEach((ids) => {
        if (!Array.isArray(ids) || !ids.length) return;
        cellCount += 1;
        new Set(ids.map((id) => String(id))).forEach((id) => {
          counts.set(id, (counts.get(id) || 0) + 1);
          global.set(id, (global.get(id) || 0) + 1);
        });
      });
      return { row, rowIndex, counts, cellCount };
    });
    return {
      global,
      rows,
      totalCells: rows.reduce((total, row) => total + row.cellCount, 0),
    };
  }

  function applyVocabularyFocus() {
    const select = $("vocabulary-focus");
    const option = select?.selectedOptions[0];
    const scope = option?.dataset.scope;
    const tokenId = option?.dataset.tokenId;
    const rowIndex = option?.dataset.rowIndex;
    state.vocabularyFocus = scope && tokenId ? { scope, tokenId, rowIndex } : null;

    document.querySelectorAll(".readout-cell").forEach((cell) => {
      cell.classList.remove("vocab-match");
      delete cell.dataset.vocabRank;
      if (!state.vocabularyFocus) return;
      const cellRow = Number(cell.dataset.row);
      const cellPosition = Number(cell.dataset.position);
      const ids = state.result?.grid?.[cellRow]?.top_ids?.[cellPosition];
      const rank = Array.isArray(ids)
        ? ids.findIndex((id) => String(id) === tokenId)
        : -1;
      const rowMatches = scope === "global" || cellRow === Number(rowIndex);
      if (rank >= 0 && rowMatches) {
        cell.classList.add("vocab-match");
        cell.dataset.vocabRank = String(rank + 1);
      }
    });
  }

  function renderVocabularyFocus(result) {
    const select = $("vocabulary-focus");
    if (!select) return;
    const stats = collectVocabularyStats(result);
    select.innerHTML = "<option value=\"\">選擇常見 top-k token…</option>";

    function addGroup(label, scope, counter, cellCount, rowIndex = null) {
      const entries = rankVocabulary(counter).slice(0, MAX_VOCABULARY_OPTIONS);
      if (!entries.length) return;
      const group = document.createElement("optgroup");
      group.label = label;
      entries.forEach(([tokenId, count]) => {
        const option = document.createElement("option");
        option.value = `${scope}:${rowIndex ?? ""}:${tokenId}`;
        option.dataset.scope = scope;
        option.dataset.tokenId = tokenId;
        if (rowIndex !== null) option.dataset.rowIndex = String(rowIndex);
        option.textContent = `${token(result, tokenId)} · #${tokenId} · ${count}/${cellCount} cells`;
        group.append(option);
      });
      select.append(group);
    }

    addGroup("全域 / Global · all readout rows", "global", stats.global, stats.totalCells);
    stats.rows.forEach(({ row, rowIndex, counts, cellCount }) => {
      addGroup(
        `By layer / ${row.is_output ? "OUTPUT" : `L${row.layer}`}`,
        "row",
        counts,
        cellCount,
        rowIndex,
      );
    });
    select.disabled = !select.querySelector("option[data-token-id]");
    select.onchange = applyVocabularyFocus;
  }

  function setBusy(busy) {
    $("run").disabled = busy;
    $("run").querySelector(".button-label").textContent = busy ? "正在讀取 residual…" : "分析 prompt";
  }

  function renderInspector(rowIndex, position) {
    const result = state.result;
    const row = result.grid[rowIndex];
    const ids = row.top_ids[position];
    const probabilities = row.top_probs[position];
    const inputToken = token(result, result.context_ids[position]);
    const maximum = Math.max(...probabilities, 0.00001);
    $("inspector").innerHTML = `
      <p class="inspector-kicker">Cell inspector</p>
      <h3>${row.is_output ? "Final output" : `Layer ${row.layer}`} · position ${position}</h3>
      <div class="inspector-meta">
        <div><strong>${escapeHtml(inputToken)}</strong><span>input token</span></div>
        <div><strong>${escapeHtml(token(result, ids[0]))}</strong><span>top prediction</span></div>
        <div><strong>${Number(row.entropy[position]).toFixed(3)}</strong><span>entropy (nats)</span></div>
        <div><strong>${Number(row.kurtosis[position]).toFixed(2)}</strong><span>excess kurtosis</span></div>
      </div>
      <div class="distribution">
        ${ids.map((id, index) => `
          <div class="distribution-row">
            <span class="distribution-rank">${index + 1}</span>
            <span class="distribution-token" style="--width:${(probabilities[index] / maximum) * 100}%"><span>${escapeHtml(token(result, id))}</span></span>
            <span class="distribution-prob">${percent(probabilities[index])}</span>
          </div>
        `).join("")}
      </div>`;
  }

  function focusPosition(position) {
    document.querySelectorAll(".readout-cell").forEach((cell) => {
      cell.classList.toggle("focus-column", Number(cell.dataset.position) === position);
    });
    document.querySelectorAll(".grid-header:not(.grid-corner)").forEach((header, index) => {
      header.classList.toggle("focus-header", index === position);
    });
    const focusedCell = document.querySelector(`.readout-cell[data-position="${position}"]`);
    const shell = document.querySelector(".grid-shell");
    if (focusedCell && shell) {
      shell.scrollLeft = Math.max(
        0,
        focusedCell.offsetLeft - (shell.clientWidth - focusedCell.clientWidth) / 2,
      );
    }
  }

  function renderGrid(result) {
    const grid = $("grid");
    grid.style.setProperty("--columns", result.seq_len);
    const headers = result.context_ids.map((id, position) => `
      <div class="grid-header" role="columnheader">
        <strong>${escapeHtml(token(result, id))}</strong>
        <span>#${position}</span>
      </div>`).join("");
    const rows = result.grid.map((row, rowIndex) => {
      const cells = row.top_ids.map((ids, position) => {
        const probability = row.top_probs[position][0];
        return `
          <button
            class="readout-cell"
            type="button"
            role="cell"
            data-row="${rowIndex}"
            data-position="${position}"
            style="--confidence:${Math.min(1, Math.max(0, probability))}"
            aria-label="${row.is_output ? "final output" : `layer ${row.layer}`}, position ${position}"
          >
            <span class="token">${escapeHtml(token(result, ids[0]))}</span>
            <span class="probability">${percent(probability)}</span>
          </button>`;
      }).join("");
      return `
        <div class="layer-label ${row.is_output ? "output" : ""}" role="rowheader">
          <span>${row.is_output ? "OUTPUT" : `L${row.layer}`}</span>
          <small>${row.is_output ? "J = I" : result.mode}</small>
        </div>${cells}`;
    }).join("");
    grid.innerHTML = `<div class="grid-header grid-corner">layer ↓<br>token →</div>${headers}${rows}`;

    grid.querySelectorAll(".readout-cell").forEach((cell) => {
      cell.addEventListener("mouseenter", () => {
        if (!state.pinnedCell) renderInspector(Number(cell.dataset.row), Number(cell.dataset.position));
      });
      cell.addEventListener("click", () => {
        const wasPinned = cell === state.pinnedCell;
        grid.querySelectorAll(".pinned").forEach((node) => node.classList.remove("pinned"));
        state.pinnedCell = wasPinned ? null : cell;
        if (state.pinnedCell) state.pinnedCell.classList.add("pinned");
        renderInspector(Number(cell.dataset.row), Number(cell.dataset.position));
      });
    });
  }

  function metricCell(value, maximum, digits = 3) {
    const ratio = maximum > 0 ? Math.min(1, Math.max(0, value / maximum)) : 0;
    return `<span><i class="meter"><i style="--width:${ratio * 100}%"></i></i>${Number(value).toFixed(digits)}</span>`;
  }

  function renderMetrics(metrics) {
    const maxKurtosis = Math.max(...metrics.map((item) => Math.max(0, item.mean_kurtosis)), 1);
    $("layer-metrics").innerHTML = `
      <div class="metric-row">
        <span class="metric-layer">Layer</span><span>Next-token acc.</span><span>Mean kurtosis</span><span>Top-1 autocorr.</span>
      </div>
      ${metrics.map((item, index) => `
        <div class="metric-row">
          <span class="metric-layer">${index === metrics.length - 1 ? "OUTPUT" : `L${item.layer}`}</span>
          ${metricCell(item.next_token_acc, 1)}
          ${metricCell(item.mean_kurtosis, maxKurtosis, 2)}
          ${metricCell(item.top1_autocorr, 1)}
        </div>`).join("")}`;
  }

  function render(result) {
    state.result = result;
    state.pinnedCell = null;
    state.vocabularyFocus = null;
    $("results").hidden = false;
    $("summary").innerHTML = [
      [result.seq_len, "tokens"],
      [result.layers.length, "readout rows"],
      [result.response_token_count || 0, "output tokens"],
      [result.mode === "jlens" ? "J-lens" : "Logit", "mode"],
      [result.thinking_enabled ? "On" : "Off", "reasoning"],
    ].map(([value, label]) => `<div class="summary-item"><strong>${value}</strong><span>${label}</span></div>`).join("");
    $("continuation-card").hidden = !result.continuation;
    $("continuation-text").textContent = result.continuation || "";

    $("position").innerHTML = result.context_ids.map((id, position) =>
      `<option value="${position}" ${position === result.prompt_len - 1 ? "selected" : ""}>#${position} · ${escapeHtml(token(result, id))}</option>`
    ).join("");
    $("position").onchange = () => focusPosition(Number($("position").value));

    renderGrid(result);
    renderVocabularyFocus(result);
    renderMetrics(result.layer_metrics);
    const initialPosition = Math.max(0, result.prompt_len - 1);
    focusPosition(initialPosition);
    const outputRow = result.grid.length - 1;
    renderInspector(outputRow, initialPosition);
    if (result.truncated) {
      $("error").textContent = `Prompt 超過 ${result.requested_max_seq_len} tokens，readout 已截斷。`;
      $("error").hidden = false;
    }
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function run() {
    const prompt = $("prompt").value.trim();
    if (!prompt) {
      $("error").textContent = "請先輸入 prompt。";
      $("error").hidden = false;
      return;
    }
    setBusy(true);
    $("error").hidden = true;
    try {
      const response = await fetch("/api/readout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          mode: $("mode").value,
          top_k: Number($("top-k").value),
          max_seq_len: Number($("max-seq-len").value),
          chat: $("chat").checked,
          enable_thinking: $("reasoning").checked,
          generate_continuation: $("continuation").checked,
          max_new_tokens: Number($("max-new-tokens").value),
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || response.statusText);
      render(payload);
    } catch (error) {
      $("error").textContent = error.message;
      $("error").hidden = false;
    } finally {
      setBusy(false);
    }
  }

  async function init() {
    $("prompt").addEventListener("input", () => {
      $("character-count").textContent = `${$("prompt").value.length.toLocaleString()} characters`;
    });
    $("prompt").dispatchEvent(new Event("input"));
    $("prompt").addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") run();
    });
    $("run").addEventListener("click", run);
    $("chat").addEventListener("change", () => {
      $("reasoning").disabled =
        !$("chat").checked || !$("reasoning").dataset.supported;
      if ($("reasoning").disabled) $("reasoning").checked = false;
    });

    const response = await fetch("/api/info");
    const info = await response.json();
    if (!response.ok) throw new Error(info.detail || response.statusText);
    $("model-meta").textContent = `${info.model_id} · ${info.n_layers}L · ${info.device} · fitted ${info.fitted_layers.join(", ")}`;
    if (info.missing_fitted_layers.length) {
      $("lens-warning").textContent =
        `目前 Jacobian lens coverage 是 ${info.fitted_layer_count}/${info.expected_fitted_layer_count} source layers；` +
        `缺少 L${info.missing_fitted_layers.join(", L")}。要顯示每一層，請改用 layer-stride=1 fit 的 lens。` +
        "Logit lens 模式不依賴 fitted Jacobian，仍會顯示全部層。";
      $("lens-warning").hidden = false;
    }
    $("chat").checked = info.has_chat_template;
    $("chat").disabled = !info.has_chat_template;
    $("reasoning").dataset.supported = info.supports_enable_thinking ? "1" : "";
    $("reasoning").checked = false;
    $("chat").dispatchEvent(new Event("change"));
  }

  init().catch((error) => {
    $("model-meta").textContent = "model unavailable";
    $("error").textContent = error.message;
    $("error").hidden = false;
  });
})();
