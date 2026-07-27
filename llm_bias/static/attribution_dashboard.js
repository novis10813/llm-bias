(function () {
  "use strict";

  const data = window.ATTRIBUTION_DATA;
  const indexLabels = { sp500: "S&P 500", russell1000: "Russell 1000", russell2000: "Russell 2000" };
  const contextLabels = { without: "without context", with: "with context" };
  const select = document.getElementById("date-select");
  const panels = document.getElementById("panels");
  const tooltip = document.getElementById("tooltip");

  function compact(value) {
    return String(value).replace(/\s+/g, " ").trim() || "∅";
  }

  function color(value, maximum) {
    const ratio = maximum > 0 ? Math.max(0, Math.min(1, value / maximum)) : 0;
    return "hsl(" + (316 - ratio * 35) + ", 78%, " + (96 - ratio * 56) + "%)";
  }

  function showTooltip(event, html) {
    tooltip.innerHTML = html;
    tooltip.style.display = "block";
    tooltip.style.left = Math.min(event.clientX + 14, window.innerWidth - tooltip.offsetWidth - 12) + "px";
    tooltip.style.top = Math.min(event.clientY + 14, window.innerHeight - tooltip.offsetHeight - 12) + "px";
  }

  function hideTooltip() { tooltip.style.display = "none"; }

  function inputSummary(panel, rowIndex) {
    return panel._matrix[rowIndex]
      .map(function (value, colIndex) { return { value: value || 0, token: panel._condition.input_tokens[colIndex] }; })
      .filter(function (item) { return item.value > 0; })
      .sort(function (left, right) { return right.value - left.value; })
      .slice(0, 8)
      .map(function (item) { return compact(item.token.token) + " (" + item.value.toFixed(4) + ")"; })
      .join(", ");
  }

  function resetPanel(panel) {
    panel.querySelectorAll(".output-token, .input-token").forEach(function (node) {
      node.classList.remove("muted", "focused");
    });
    panel.querySelectorAll(".input-token").forEach(function (node) {
      const value = Number(node.dataset.baseValue || 0);
      node.style.backgroundColor = value > 0 ? color(value, Number(panel.dataset.maximum)) : "";
      node.classList.toggle("zero", value <= 0);
    });
  }

  function focusOutput(panel, rowIndex) {
    panel.querySelectorAll(".output-token").forEach(function (node) {
      const active = Number(node.dataset.row) === rowIndex;
      node.classList.toggle("muted", !active);
      node.classList.toggle("focused", active);
    });
    panel.querySelectorAll(".input-token").forEach(function (node) {
      const value = panel._matrix[rowIndex][Number(node.dataset.col)] || 0;
      node.style.backgroundColor = value > 0 ? color(value, Number(panel.dataset.maximum)) : "";
      node.classList.toggle("zero", value <= 0);
    });
  }

  function focusInput(panel, colIndex) {
    panel.querySelectorAll(".input-token").forEach(function (node) {
      node.classList.toggle("focused", Number(node.dataset.col) === colIndex);
    });
    panel.querySelectorAll(".output-token").forEach(function (node) {
      const value = panel._matrix[Number(node.dataset.row)][colIndex] || 0;
      node.classList.toggle("muted", value <= 0);
      node.classList.toggle("focused", value > 0);
    });
  }

  function renderPanel(condition, dateMaximum) {
    const panel = document.createElement("article");
    panel.className = "panel";
    panel.dataset.maximum = dateMaximum;
    panel._condition = condition;
    panel._matrix = condition.matrix;

    const title = document.createElement("h2");
    title.textContent = (indexLabels[condition.index] || condition.index) + " · " + (contextLabels[condition.context] || condition.context);
    panel.appendChild(title);
    const subtitle = document.createElement("div");
    subtitle.className = "panel-subtitle";
    subtitle.textContent = "Qwen output tokens above · complete input prompt below · wraps to fit panel";
    panel.appendChild(subtitle);
    if (condition.validation_summary) {
      const validation = document.createElement("div");
      validation.className = "validation-summary";
      validation.textContent = "Semantic Scope AOPC: " + condition.validation_summary.semantic_scope_aopc_mean.toFixed(6) +
        " · random: " + condition.validation_summary.random_aopc_mean.toFixed(6) +
        " · more negative = stronger ablation effect";
      panel.appendChild(validation);
    }
    const promptText = document.createElement("pre");
    promptText.className = "prompt-text";
    promptText.textContent = condition.prompt;
    panel.appendChild(promptText);

    const flow = document.createElement("div");
    flow.className = "flow";
    const outputLine = document.createElement("div");
    outputLine.className = "flow-line output-line";
    const outputLabel = document.createElement("div");
    outputLabel.className = "flow-label";
    outputLabel.textContent = "Output";
    outputLine.appendChild(outputLabel);
    const outputTrack = document.createElement("div");
    outputTrack.className = "token-track";
    condition.output_tokens.forEach(function (output, rowIndex) {
      if (/^<\|.*\|>$/.test(output.token)) return;
      const token = document.createElement("span");
      token.className = "token output-token";
      token.dataset.row = rowIndex;
      token.textContent = compact(output.token);
      token.title = "Output position " + output.position;
      token.addEventListener("mouseenter", function (event) {
        focusOutput(panel, rowIndex);
        const targetScore = output.target_logit === undefined
          ? "<b>Log probability</b>: " + output.log_probability.toFixed(5)
          : "<b>Target logit</b>: " + output.target_logit.toFixed(5) + "<br><b>Log probability</b>: " + output.log_probability.toFixed(5);
        const validationScore = output.semantic_scope_aopc === undefined
          ? ""
          : "<br><b>Semantic Scope AOPC</b>: " + output.semantic_scope_aopc.toFixed(6) +
            "<br><b>Random AOPC</b>: " + output.random_aopc.toFixed(6) +
            "<br><b>Δ log p @20%</b>: " + output.semantic_scope_log_probability_delta[3].toFixed(6);
        showTooltip(event, "<b>Output token</b>: " + compact(output.token) + " (position " + output.position + ")<br>" + targetScore + validationScore + "<br><b>Top input influence</b>: " + inputSummary(panel, rowIndex));
      });
      token.addEventListener("mousemove", function (event) { showTooltip(event, tooltip.innerHTML); });
      token.addEventListener("mouseleave", function () { resetPanel(panel); hideTooltip(); });
      outputTrack.appendChild(token);
    });
    outputLine.appendChild(outputTrack);
    flow.appendChild(outputLine);

    const inputLine = document.createElement("div");
    inputLine.className = "flow-line input-line";
    const inputLabel = document.createElement("div");
    inputLabel.className = "flow-label";
    inputLabel.textContent = "Input";
    inputLine.appendChild(inputLabel);
    const inputTrack = document.createElement("div");
    inputTrack.className = "token-track";
    condition.input_tokens.forEach(function (input, colIndex) {
      const token = document.createElement("span");
      token.className = "token input-token";
      token.dataset.col = colIndex;
      const baseValue = Math.max.apply(null, condition.matrix.map(function (row) { return row[colIndex] || 0; }));
      token.dataset.baseValue = baseValue;
      token.textContent = compact(input.token);
      token.title = "Input position " + input.position;
      token.style.backgroundColor = baseValue > 0 ? color(baseValue, dateMaximum) : "";
      token.classList.toggle("zero", baseValue <= 0);
      token.addEventListener("mouseenter", function (event) {
        focusInput(panel, colIndex);
        showTooltip(event, "<b>Input token</b>: " + compact(input.token) + " (position " + input.position + ")<br><b>Maximum attribution</b>: " + baseValue.toFixed(6));
      });
      token.addEventListener("mousemove", function (event) { showTooltip(event, tooltip.innerHTML); });
      token.addEventListener("mouseleave", function () { resetPanel(panel); hideTooltip(); });
      inputTrack.appendChild(token);
    });
    inputLine.appendChild(inputTrack);
    flow.appendChild(inputLine);
    panel.appendChild(flow);
    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = condition.input_attribution_complete
      ? "Complete input-token Semantic Scope scores are saved for every output token."
      : "Input tokens not present in the saved top-k attribution are shown neutrally.";
    panel.appendChild(hint);
    return panel;
  }

  function renderDate(dateValue) {
    const selected = data.dates.find(function (item) { return item.date === dateValue; });
    if (!selected) return;
    panels.replaceChildren();
    const dateMaximum = Math.max.apply(null, selected.conditions.map(function (condition) { return condition.max_attribution || 0; }));
    selected.conditions.forEach(function (condition) { panels.appendChild(renderPanel(condition, dateMaximum)); });
  }

  data.dates.forEach(function (item, index) {
    const option = document.createElement("option");
    option.value = item.date;
    option.textContent = item.date;
    if (index === 0) option.selected = true;
    select.appendChild(option);
  });
  select.addEventListener("change", function () { renderDate(select.value); });
  if (data.dates.length) renderDate(data.dates[0].date);
})();
