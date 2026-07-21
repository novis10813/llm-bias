# 環境需求

- 使用 `uv` 管理 python 虛擬環境
- `uv add` 添加套件

# Repository scope

這份文件適用於整個 repository；子目錄中的 `AGENTS.md` 會補充更具體的
模組規則。

## 專案目的

本 repo 實驗 entity-level bias 如何在 decoder LLM 的 residual stream 中
形成，以及把 source entity 的 activation 替換成 target entity 後，是否會
改變最終答案分布。

主要流程是：

1. 從 `third_party/jacobian-lens` 的 factual experiment spec 建立 token-aligned
   counterfactual pairs。
2. 使用 `jlens` fitting Jacobian lens。
3. 執行 residual activation patching，計算 answer-margin transfer。
4. 使用 `jspace-viz` 的 model/lens 格式與 grid 概念呈現 source、target、patched
   三種狀態。

## 依賴與外部 checkout

- Python 依賴與 lockfile 一律由 `uv` 管理；新增套件使用 `uv add`。
- `third_party/jacobian-lens` 與 `third_party/jspace-viz` 是 editable workspace
  members，但整個 `third_party/` 被 `.gitignore` 忽略。
- 新環境請依照 `README.md` clone 兩個外部 repo 後再執行 `uv sync`。
- 不要把模型權重、patch 結果、lens binary 或第三方 checkout 加入 root Git
  repository。這些內容位於 `.cache/`、`artifacts/`、`third_party/`。

## 常用驗證

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
```

視覺化 server：

```bash
uv run python -m llm_bias serve-viz \
  --host 0.0.0.0 \
  --port 8321 \
  --lens artifacts/entity_control/jacobian_lens_16.pt \
  --pairs artifacts/entity_control/pairs.jsonl
```

# 外部套件與工具

## jacobian-lens
Anthropic 於 2026 年推出的一項大語言模型（LLM）可解釋性工具。它能「解碼」模型在各個隱藏層（中間層）的內部神經元活動，直接翻譯出該狀態即將輸出的概念，進而揭示 AI 模型內部的「全局工作空間 (Global Workspace)」與隱性推理過程。
