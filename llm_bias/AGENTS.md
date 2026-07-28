# `llm_bias/` scope

這份文件說明主實驗 Python package 的責任邊界；上層規則見 repository root
的 `AGENTS.md`。

## Package 邊界

- `core/`：只放 model loading、prompt formatting 等模型無關的共用基礎設施。
- `lens_fitting/`：獨立 fitting Jacobian lens；不可 import 任一 experiment。
- `counterfactual_patching/`：擁有 `Pair`、residual patch、transfer analysis
  與 interactive counterfactual visualization。
- `prompt_analysis/`：擁有 CSV prompt readout、generated-token attribution、
  attribution validation 與結果視覺化。

兩個 experiment package 不可互相 import。共同能力必須先確認確實與研究語意
無關，才可放進 `core/`。三個 CLI 入口分別是 `fit-jacobian-lens`、
`counterfactual-patching` 與 `prompt-analysis`；experiment CLI 不可自行 fitting
lens。

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
