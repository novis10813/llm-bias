# `tests/` scope

這個目錄放置 entity-bias experiment 的 regression tests；上層規則見 root
`AGENTS.md`。

## 測試分層

- `test_data.py`：Pair serialization 與 transfer 公式。
- `test_workflow_boundaries.py`：三個 CLI、experiment import 邊界與 calibration
  prompt loading。
- `test_interventions.py`：activation patch hook 對 tensor、tuple、list 與 model
  output 結構的保留行為。
- `test_visualization.py`：counterfactual grid comparison payload 的 layer/position
  對齊。

## 新增測試原則

- 純函式與資料格式優先使用 deterministic unit tests，不要為了單元測試重新
  載入 2.5GB Llama checkpoint。
- 任何 patch hook 或輸出 schema 改動，都要補上 regression test。
- 模型載入與 GPU inference 屬於 smoke/integration test，應明確標示，避免讓
  `uv run pytest -q` 依賴下載或特定 GPU。
