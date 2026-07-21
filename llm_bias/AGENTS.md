# `llm_bias/` scope

這份文件說明主實驗 Python package 的責任邊界；上層規則見 repository root
的 `AGENTS.md`。

## 模組責任

- `data.py`：讀取 factual generalization spec，建立 token-aligned `Pair`，並
  儲存/載入 `pairs.jsonl`。若改變 token span 過濾條件，必須更新資料測試。
- `model.py`：提供原始 `jlens` 實驗使用的 Hugging Face model loader，以及預設
  local model path。
- `interventions.py`：記錄 residual stream、執行單點 activation patch，以及
  輸出 patched residuals；patch hook 必須保留 decoder block output 的 tuple、list
  或 model-output 結構。
- `analysis.py`：fitting lens、跑完整 patch experiment、計算 normalized transfer
  與產生靜態 summary/heatmap。不要把 static summary 當成 per-cell interactive
  readout。
- `visualization.py`：使用 `jspace_viz` 的 `WrappedModel` 與 `JacobianLens`，把
  source/target/patched activation 轉成相同的 layer × position grid schema，並
  提供 FastAPI endpoints。
- `cli.py`：只負責 command dispatch；實驗邏輯放在上述模組，不要在 CLI 中直接
  寫模型推理流程。

## Counterfactual API

`POST /api/counterfactual` 接收 pair id、patch layer、patch position、readout
mode 與 top-k，回傳：

- `source`：原始 source prompt 的 grid
- `target`：target prompt 的 grid
- `patched`：source prompt 注入 target entity residual 後的 grid
- `metrics`：source/target/patched answer margins、answer ranks、normalized transfer
- `comparison`：各 layer/position 的 top-1 對照

Grid 的 `top_ids` 與 `top_probs` 必須保留完整 top-k；前端可以只顯示 top-1，
但不能在 backend 提前截斷成單一 token。

## 修改與驗證原則

- 優先使用 `apply_patch` 修改檔案。
- 修改 activation patch 語意時，至少執行 `uv run pytest -q`，並用實際 local
  model 做一次 API smoke test。
- 不要儲存完整 raw activations；interactive endpoint 應即時重算 selected pair
  與 selected layer，或只儲存 compact top-k/rank 結果。
