# `tests/` scope

這個目錄放置 entity-bias experiment 的 regression tests；上層規則見 root
`AGENTS.md`。Counterfactual 線測試已隨程式移至 `archive/tests/`。

## 目前主要 regression tests

- `test_jspace_intervention.py` / `test_jspace_intervention_workflow.py`：
  coordinate swap/gain math、live dose diagnostics、dose-matched controls、
  hook cleanup 與 artifact schema 相容性。
- `test_jspace_valence_readout.py`：valence pair 驗證/選擇/rendering、
  full-softmax average-before-top-k contract、contrast 分數 closed form、
  frozen candidate 候選規則與上限、fake model/lens 的 workflow 與 CLI 行為。
- `test_jspace_token_screen.py`：V1 token causal screen 的 config 驗證（對稱
  劑量、token_id 唯一、answer word、SHA）、fake model 的 dose math / scale
  floor / side invariance / matched-random 同範數 / hook cleanup、
  symmetric-slope 分析（equal-ticker 聚合、specificity、Holm、shortlist
  上限）、pipeline lifecycle（preflight 不建 run、SHA tamper 拒絕、
  determinism）與 CLI 行為。
- `test_jspace_outcome_flip.py`：V2 outcome-conditioned decision flip 的 config
  驗證、decision-position margin 計分位置語義、direction fitting（gradient
  確定性/ uphill / hook cleanup、label-permutation signs）、paired arms 的 dose
  matching 與 flip role、ticker-level estimands（exact paired test、strata、
  gates、calibration selection 與 fail-closed）、pipeline lifecycle（identity
  tamper 拒絕、missing input 拒絕、frozen combination 綁定）與 CLI 行為；
  全部用 fake model、無 GPU。
- `test_jspace_outcome_geometry.py`：V2 outcome direction geometric projection
  的 dot projection 公式（general coefficient、normalization invariance、
  orthogonal/antiparallel、degenerate 與 invalid input）、fake model sector state
  累積對照 closed form、analyze compact metrics 與 TF-IDF 段、pipeline lifecycle
  （direction identity tamper fail-closed、missing input 與 bad binding 拒絕、
  TF-IDF arm 與 fake lens、logodds prototype 拒絕）與 CLI 行為；全部用 fake
  model、無 GPU。
- `test_baseline_trial_pipeline.py`：baseline trial workflow 行為。
- `test_span_sensitivity.py`：header condition rendering、paired estimands、artifact lifecycle
  與 CLI。
- `test_prompt_input.py`、`test_core_inference.py`、`test_core_analysis.py`、
  `test_core_artifacts.py`：shared workflow 核心子套件。
- `test_continuation_scoring.py`：buy/sell margin 與 KL 計算。
- `test_lens_artifacts.py`、`test_lens_install.py`、`test_lens_loader.py`、
  `test_lens_promotion.py`、`test_lens_registry.py`：canonical lens 安裝、存取與
  promotion 規則。
- `test_prompt_readout.py`、`test_attribution_validation.py`、
  `test_generation_artifact.py`：prompt-analysis readout、generation 與 attribution
  artifact contracts。
- `test_workflow_boundaries.py`：root guidance contract、CLI 與 experiment import 邊界。

這份清單按 workflow 選列；新增或刪除 test file 時不要求在此枚舉整個目錄，但新的
獨立 workflow 必須加入一行入口。

## Instruction Index

目前 `tests/` 沒有含 `AGENTS.md` 的直接子目錄。Integration/GPU tests 若日後移入獨立
子目錄並形成專屬 fixture/runtime 規則，再新增局部 `AGENTS.md` 並更新本節。

## 新增測試原則

- 純函式與資料格式優先使用 deterministic unit tests、fake model、monkeypatch
  與 temporary directories；不要為了單元測試載入大 checkpoint
  （例如 Qwen3.5-4B 約 8.7GB BF16）。
- 任何 patch hook 或輸出 schema 改動，都要補上 regression test。
- 模型載入與 GPU inference 屬於 smoke/integration test，應明確標示，避免讓
  `uv run pytest -q` 依賴下載或特定 GPU。

## 局部驗證

```bash
uv run pytest -q tests/<target_test_file>.py
```

修改 shared workflow contract 或 import boundary 時，另執行
`uv run pytest -q tests/test_workflow_boundaries.py`。
