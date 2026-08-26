# `tests/` scope

這個目錄放置 entity-bias experiment 的 regression tests；上層規則見 root
`AGENTS.md`。Counterfactual 線測試已隨程式移至 `archive/tests/`。

## 目前主要 regression tests

- `test_jspace_intervention.py` / `test_jspace_intervention_workflow.py`：
  coordinate swap/gain math、live dose diagnostics、dose-matched controls、
  hook cleanup 與 artifact schema 相容性。
- `test_baseline_trial_pipeline.py`：baseline trial workflow 行為。
- `test_prompt_input.py`、`test_core_inference.py`、`test_core_analysis.py`、
  `test_core_artifacts.py`：shared workflow 核心子套件。
- `test_continuation_scoring.py`：buy/sell margin 與 KL 計算。
- `test_lens_artifacts.py`、`test_lens_loader.py`、`test_lens_promotion.py`、
  `test_lens_registry.py`：canonical lens 存取與 promotion 規則。
- `test_workflow_boundaries.py`：CLI 與 experiment import 邊界。

## 新增測試原則

- 純函式與資料格式優先使用 deterministic unit tests、fake model、monkeypatch
  與 temporary directories；不要為了單元測試載入大 checkpoint
  （例如 Qwen3.5-4B 約 8.7GB BF16）。
- 任何 patch hook 或輸出 schema 改動，都要補上 regression test。
- 模型載入與 GPU inference 屬於 smoke/integration test，應明確標示，避免讓
  `uv run pytest -q` 依賴下載或特定 GPU。
