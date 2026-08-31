# Research scripts reference

`scripts/` 保存 active workflow 的 operator、資料準備器、diagnostic 與 report
renderer。從 repository root 使用 `uv run python scripts/<name>.py` 或
`bash scripts/<name>.sh`。Canonical command blocks 與參數仍以各 workflow 文件為準。

## Jacobian-lens calibration, evaluation, and promotion

| Script | Responsibility | Main outputs | Canonical document |
|---|---|---|---|
| `prepare_qwen_calibration.py` | 建立 English、Simplified Chinese、mixed calibration corpora，寫入 balance 與 SHA-256 manifest | `data/calibration/<model-slug>/*.jsonl`, `manifest.json` | [Qwen Jacobian-lens selection](jacobian-lens-selection/proposal.md) |
| `prepare_qwen_lens_eval.py` | 建立 bilingual intermediate-layer holdout | `data/evaluations/<model-slug>/*.jsonl`, manifest | 同上 |
| `run_qwen_lens_candidates.sh` | 依序 fit、evaluate、promote Qwen3.5-4B candidates；支援 tmux 與 digest checkpoint resume | candidate lens、`evaluation.json`、promotion log | 同上 |
| `evaluate_qwen_lens_candidates.py` | 比較多個 candidate lens 並選出 winner | `evaluation.json` | 同上 |
| `promote_qwen_lens_candidate.py` | 驗證 hash、shape 與 metadata，archive 舊 lens 後寫入 canonical lens | `jacobian_lens.pt`, metadata, `selection.json` | 同上 |
| `run_qwen27b_lens.sh` | 執行 Qwen3.6-27B probes、benchmarks、48-hour gate、full fit、evaluation 與 promotion | `artifacts/qwen3.6-27b/jacobian-lens/candidates/` 與 run gate files | 同上 |
| `probe_qwen27b.py` | 驗證 two-GPU placement、VRAM 與短 autograd path；不 fitting lens | probe JSON | 同上 |
| `check_qwen_benchmark.py` | 檢查 lens finiteness、完整 layer coverage、resume 與 VRAM stability | quality-gate JSON | 同上 |
| `evaluate_qwen_lens_single.py` | 評估單一 Qwen3.6-27B candidate | evaluation JSON | 同上 |

只有 `promote_qwen_lens_candidate.py` 可寫入 active canonical lens。Candidate fitting
與 checkpoints 遵循 selection 文件的 candidate-adjacent layout；一般 canonical
fitting checkpoints 則使用 archive checkpoint root。

## Baseline and prompt-analysis operations

| Script | Responsibility | Main outputs | Canonical document |
|---|---|---|---|
| `convert_baseline_trial_plan.py` | 將外部 baseline `trial_plan.jsonl` 轉為 prompt CSV 並保存 source provenance | `data/baseline/.../trial_plan_prompts.csv` 與 provenance JSON | [Baseline trial plan prompts](baseline-trial/proposal.md) |
| `run_prompt_analysis.sh` | 以 tmux/env contract 執行 readout、generation、generated-token attribution lifecycle | model/dataset-scoped run root | [Prompt-analysis reproducibility](baseline-trial/report-reproducibility.md) |
| `run_mag7_8k_return_prompt_analysis.sh` | 套用 MAG7 8-K return-pairs preset 後轉交 `run_prompt_analysis.sh` | 同上 | 同上 |
| `visualize_prompt_analysis.sh` | 從完成的 run 建立 uncertainty plots 與 attribution dashboard | run-scoped visualization files | [Prompt-analysis reproducibility](baseline-trial/report-reproducibility.md) |

Shell runner 的 env defaults 與 artifact lifecycle 由 canonical workflow 文件定義。
修改 env 名稱、預設值或 run-root 拒絕覆寫行為時，要同步更新對應文件與 regression
tests。

## J-space diagnostics and reports

| Script | Responsibility | Inputs and outputs | Related document |
|---|---|---|---|
| `jspace_layer_stats.py` | 計算 layer accuracy、kurtosis、lag agreement、effective dimension 與 CKA | lens/model → `layer_stats*.json`, `cka.npz` | [J-space sector intervention](jspace-sector-intervention/report.md) |
| `jspace_outcome_token_attribution.py` | 對 Buy/Sell decision token 做 per-(layer, position) 一階 attribution（embedding + 全層 residual 的 per-position gradient norm、position-class 聚合）；V2 position-rule 診斷 | outcome_flip_config + split prompts → `outcome_token_attribution.json` + metadata | [J-space outcome-conditioned decision-flip V2](jspace-token-experiments/proposal-v2.md) |
| `render_jspace_report.py` | 將 layer stats 與 CKA 畫成 diagnostic figures | stats/NPZ → PNG | 同上 |
| `jspace_tfidf_analysis.py` | 從 compact lens readout 計算 sector/company TF-IDF 與 Monroe log-odds；可做 concept normalization | readout + CSV → keyword/log-odds JSONL、summary、optional NPZ | 同上 |
| `compare_jspace_token_runs.py` | 比較 raw-token 與 normalized-concept ranking overlap | 兩個 TF-IDF run → comparison JSON | 同上 |
| `render_jspace_tfidf_report.py` | 將 TF-IDF run 轉成 Markdown report | run directory → Markdown | 同上 |
| `render_jspace_publication_figures.py` | 產生 sector cards、summary、echo scatter 與 company confusion figures | TF-IDF run → PNG | 同上 |
| `render_jspace_intervention_report_figures.py` | 重建 interim report 的 dose/control figures | 固定的 calibration run IDs → `docs/assets/jspace-sector-intervention/*.png` | 同上 |
| `render_causal_tracing_figures.py` | 由 Phase 3 discovery 與 held-out confirmation 的 compact artifacts 建立 position-transfer figures（discovery 曲線 + 3×3 matrix 主圖、direction-split 補充圖） | discovery/confirmation run artifacts → `docs/assets/jspace-causal-tracing/`（PDF+PNG+provenance JSON） | [Activation patching causal tracing](activation-patching-causal-tracing/proposal.md) |

`render_jspace_intervention_report_figures.py` 綁定報告中的既有 run IDs；若要支援新
run，先在報告中新增結果與 provenance，再改 renderer。J-space analysis scripts 只讀
compact artifacts，不可把 raw activations、residuals 或 Jacobians 寫入輸出。

## Test-coupled interfaces

- `tests/test_lens_promotion.py` 直接載入 `promote_qwen_lens_candidate.py` 的
  `promote` 與 `complete_lens_metadata`。
- `tests/test_jspace_token_normalization.py` 直接載入 `jspace_tfidf_analysis.py` 的
  `ConceptNormalizer`。
- `tests/test_lens_artifacts.py` 與 `tests/test_workflow_boundaries.py` 執行或檢查
  lens/prompt-analysis shell runners，包括 model slug derivation 與 canonical lens path。

重命名上述 scripts、symbols、env variables 或 path contract 時，必須同步更新這些
regression tests。
