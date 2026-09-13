# Research scripts reference

`scripts/` 保存 active workflow 的 operator、資料準備器、diagnostic 與 report
renderer。從 repository root 使用 `uv run python scripts/<name>.py` 或
`bash scripts/<name>.sh`。Canonical command blocks 與參數仍以各 workflow 文件為準。

## Investment-dial auxiliary diagnostic

`investment_dial_fine_a.py` reads the complete V1 calibration run and generates only
A balanced prompts at nine fixed deltas for L15/n8490. It writes registered per-delta
responses, summaries and provenance; no B/test generation or refitting. Commands and
transfer requirements: [A-only diagnostic](investment-dial/details/diagnostic-fine-a.md).
Regression coupling: `tests/test_investment_dial_fine_a.py` (CLI, subset guards,
mocked full lifecycle) and `tests/test_investment_dial.py` (owning workflow mechanics).

Report renderer: `plot_fine_a_curve.py` 重建 [fine-a 報告](investment-dial/details/report-fine-a.md)
的曲線圖；只讀取 `fine-a-gpu-bf16-01` 與 parent calibration run 的 compact
artifacts，寫入 `docs/assets/investment-dial/fine_a_curve.{pdf,png}`。

## Investment-dial calibration V2

`scripts/investment_dial_calibration_v2.py` implements the frozen V2 fine-grid A calibration
and B reevaluation operator. It verifies fixed V1/fine-a parent manifest digests, model/runtime/
tokenization identity, exact 340-row/85-company populations, and compact row identities before
loading the model. It writes only registered compact JSON/JSONL outputs and has a no-output
`--smoke` preflight. First formal GPU run `calib-v2-gpu-bf16-01` completed 2026-09-09 with
gate `pass` (RMSE 0.0579, max error 0.0824, `certified=true`);
report: [V2 report](investment-dial/details/report-v2.md).
Regression coupling: `tests/test_investment_dial_calibration_v2.py` (parent guards, exact
curve assembly, inversion, smoke lifecycle, gate outcomes, artifact lifecycle, and CLI help).
Canonical contract: [investment-dial calibration V2 proposal](investment-dial/details/proposal-v2.md).

Report renderer: `plot_calibration_v2.py` 重建 [V2 報告](investment-dial/details/report-v2.md)
的組裝曲線、inversion 與 B 比較圖；只讀取 `calib-v2-gpu-bf16-01` 與 V1 run 的
compact artifacts，寫入 `docs/assets/investment-dial/calibration_v2_curve.{pdf,png}`。

Explanation renderer: `plot_calibration_v2_explained.py` 產出解釋圖（V1 粗 grid
為何失敗、V2 細 grid 恢復什麼、per-target error 對比）；同源 compact artifacts，
寫入 `docs/assets/investment-dial/calibration_v2_explained.{pdf,png}`。

## Balanced Evidence Gap (Phase 1 behavioral confirmation)

`scripts/balanced_evidence_gap.py` implements the Phase 1 balanced-evidence
behavioral confirmation operator. It builds 256 prompts (128 named + 128
anonymous) from 16 companies in the investment-dial `test` split across 4
sectors, with fixed `positive_count=2` (2 positive + 2 negative evidence
items), runs forward inference, and computes per-company margin and
named-vs-anonymous gap statistics.
Protocol: [Phase 1 protocol](balanced-evidence-gap/details/proposal-phase1.md)

## Balanced Evidence Gap (Phase 2 cross-entity probe and patching)

`scripts/balanced_evidence_gap_phase2.py` implements experiment 2A, the
cross-entity probe. It builds 64 prompts (16 tickers × 2 reverse options ×
2 evidence orders) from the frozen shared-evidence template, runs clean
forwards with the L15/n8490 dial readout (H4), and evaluates gate 2A
(IQR, Spearman vs Phase 1, framing stability).
`scripts/balanced_evidence_gap_phase2_patch.py` implements experiments 2B
(`sweep`, entity-state residual layer sweep over 8 top/bottom directions ×
32 layers × 4 spans) and 2C (`attribute`, attention-edge zeroing on
full-attention layers plus MLP gradient×activation attribution inside the
2B handoff interval). The `sweep` subcommand checks gate 2A in the 2A run
by default, or gate 2A Rev 2 in a re-evaluation run via `--gate-run`.
`scripts/balanced_evidence_gap_phase2_rev2.py` is the CPU-only Rev 2 gate
re-evaluation (no model load): it re-analyzes an existing 2A forward run
against the Phase 1 named-vs-anonymous gap reference with the group
construct check, and writes a standalone analysis run with provenance to
the source records. Package: `llm_bias/balanced_evidence_gap/`.
Protocol: [Phase 2 Rev 1 protocol](balanced-evidence-gap/details/proposal-phase2.md) and [Phase 2 Rev 2 protocol](balanced-evidence-gap/details/proposal-phase2-rev2.md)

Figure renderers: `scripts/plot_balanced_evidence_gap.py`（Phase 1）→
`docs/assets/balanced-evidence-gap/balanced_evidence_gap.{pdf,png}`；
`scripts/plot_balanced_evidence_gap_phase2b.py`（2B layer sweep）→
`docs/assets/balanced-evidence-gap/phase2b_layer_sweep.{pdf,png}`；
`scripts/plot_balanced_evidence_gap_phase2c.py`（2C components）→
`docs/assets/balanced-evidence-gap/phase2c_components.{pdf,png}`。
`scripts/balanced_evidence_gap_phase2c_gate_reanalysis.py`：CPU-only 2C
gate re-evaluation（gate 實作修正後對已完成 2C run 的存檔記錄重算，
SHA-256 provenance，無 GPU）。

`scripts/balanced_evidence_gap_phase3.py`：Phase 3 neuron causal
validation（2C 三 candidate coordinate 的 additive `mlp_addition` grid
＋ 30 matched controls ＋ dial descriptive arm；stages
`prepare / pilot / intervene / analyze`；gate 3A）→
`artifacts/<model-slug>/balanced-evidence-gap-phase3/runs/<run-id>/`。
Package: `llm_bias/balanced_evidence_gap/neuron_causal.py`。
Protocol: [Phase 3 protocol](balanced-evidence-gap/details/proposal-phase3.md)（Rev 2 frozen）；
Report: [Phase 3 report](balanced-evidence-gap/details/report-phase3.md)（gate 3A fail，0/3 confirmed）。

Figure renderers 補充：`scripts/plot_balanced_evidence_gap_phase3.py`
（Phase 3 ΔM 曲線 vs controls ＋ dial 對照）→
`docs/assets/balanced-evidence-gap/phase3_neuron_causal.{pdf,png}`。

`scripts/balanced_evidence_gap_jlens_neurons.py`：收線後結構診斷——
Phase 3 三座標＋dial 的 down-projection 行經 canonical J-lens transport
到 final layer 再 unembed 的 vocabulary readout（無 forward；
descriptive、non-causal）→
`artifacts/<model-slug>/balanced-evidence-gap-phase3/runs/<run-id>/`。
Package: `llm_bias/balanced_evidence_gap/jlens_neurons.py`。
Protocol: [J-lens diagnostic](balanced-evidence-gap/details/diagnostic-jlens-neurons.md)（frozen 2026-09-10）。
Renderer: `scripts/plot_balanced_evidence_gap_jlens_neurons.py` →
`docs/assets/balanced-evidence-gap/jlens_neuron_structure.{pdf,png}`。

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
| `generate_entity_cell_generic_baseline.py` | 以本地 OpenAI-compatible Qwen3.5-9B endpoint 產生、驗證並固定選取 adapted generic cloze baseline | `data/entity-cell/generic-baseline-qwen3.5-9b-v1.jsonl` 與相鄰 provenance JSON | [Entity cell localization versions](entity-cell-localization/details/README.md)（V1/V2） / [V1 report](entity-cell-localization/details/report-v1.md) / [V2 report](entity-cell-localization/details/report-v2.md) |
| `entity_cell_amnesia_recheck.py` | 針對性重算 E1 V2 frozen candidates 的 amnesia endpoint gate（v2 儀器、CPU-only），綁定 `entity-cell-e1-discovery-v2` 的 candidate 表 | `--output` 指定的 compact JSON（endpoint rows 與 optional dose curve） | [Entity cell localization V2 report 附錄](entity-cell-localization/details/report-v2.md)（instrument revision re-verification） |
| `entity_cell_factual_recall_preflight.py` | Factual recall preflight（proposed，非 frozen）：重用 E1 V2 frozen factual-cloze frames，在 clean baseline 捕捉各 entity 的模型自身完成式（top-5 與 5-token greedy）與 Anonymous 對照（每 frame 一次），判斷是否存在 entity-specific 事實訊號；`--entities TICKER:Name...`（`|` 接多 surface variants）可覆蓋內建 entity 集 | `--output` 指定的 compact JSON（per-frame top-k、greedy 續寫、metadata） | [Entity cell localization README「Proposed: factual recall preflight」](entity-cell-localization/details/README.md) |
| `entity_cell_factual_amnesia_probe.py` | Factual amnesia probe（proposed，非 frozen）：對指定 entity cell 做 dose curve 壓制，量測 frozen gold 序列（clean greedy 前 3 token）的 joint log-probability 崩塌，含 name-only scope、wrong-entity / matched-random / 跨 entity 對照；`--entity` / `--cell` / `--wrong-cell` / `--cross` / `--source-run` 參數化；`run_probe` 可被批次 runner 直接呼叫 | `--output` 指定的 compact JSON（per-condition gold logp、top-1 診斷、frame summary） | [HFM discovery report §4](entity-cell-localization/details/report-hfm-discovery.md) / [HFM-2 report §3](entity-cell-localization/details/report-hfm2-discovery.md) / [E1 V3 proposal §9](entity-cell-localization/details/proposal-v3.md) |
| `entity_cell_fact_probe_batch.py` | 上述 probe 的批次 runner（proposed）：model 一次載入，依 targets JSON 串行跑多個 (entity, cell) targets；`--indices` 支援多 worker 分片（跨 GPU 並行） | 每 target 一份 compact JSON（targets 檔指定路徑） | [E1 V3 proposal §9](entity-cell-localization/details/proposal-v3.md)（calibration population 來源） |
| `entity_cell_decision_flip_probe.py` | Decision-level suppression/flip probe（proposed，非 frozen）：frozen amnesia-endpoint margin（logP(buy)−logP(sell)，FP32 tail）的方向掃描（entity vs anonymous header）、dose sweep 翻轉測試、可選 fact block、wrong-entity / matched-random / 跨 ticker 對照；`--flip-cell` / `--wrong-cell` / `--financial-prompts` / `--reuse-scan` / `--fact-frame` 參數化 | `--output` 指定的 compact JSON（direction scan、flip curves、cross-ticker control、可選 fact block） | [Entity cell decision-level probe report](entity-cell-localization/details/report-decision-probe.md) |
| `entity_cell_e3_frozen_records_proposed.py` | E3 frozen-format records with explicit selection（proposed selection，frozen readout）：直接呼叫 frozen `run_upstream_suppression_record` / `run_downstream_suppression_record`，對明確選取的 cell/heads 產出 byte-compatible E3-A/E3-B compact records（含 flip 欄位與 mediation）；用於 formal eligibility 為空、無法驅動 frozen E3 CLI 的 discovery run | `--output` 指定的 compact JSON（E3-A/E3-B records、selection provenance） | [JNJ/JPM battery report](entity-cell-localization/details/report-jnj-jpm-battery.md) |
| `entity_cell_readout_delta_probe.py` | E4 residual stream 壓抑 readout 比較 probe（proposed，非 frozen）：對 4 個 frozen V3 entity cells 在 fact frame readout 位置做 clean / target / matched-random / wrong-entity 四條件的逐層 residual stream transported readout（L0–L30 經 canonical lens，L31 identity），輸出每 (condition, layer) 的 gold log-prob 與 top-10 token，並計算 first_divergence_layer、控制組 max Δ、top-10 overlap；gold 綁定 source run 的人驗記錄（fail-closed）；`--target` / `--expected-lens-sha256` 參數化 | `--output` 指定的 compact JSON（per-(condition,layer) readout rows + per-frame summary + lens/model provenance） | [E4 proposal](entity-cell-localization/details/proposal-e4.md) |
| `run_prompt_analysis.sh` | 以 tmux/env contract 執行 readout、generation、generated-token attribution lifecycle | model/dataset-scoped run root | [Prompt-analysis reproducibility](baseline-trial/report.md) |
| `run_mag7_8k_return_prompt_analysis.sh` | 套用 MAG7 8-K return-pairs preset 後轉交 `run_prompt_analysis.sh` | 同上 | 同上 |
| `visualize_prompt_analysis.sh` | 從完成的 run 建立 uncertainty plots 與 attribution dashboard | run-scoped visualization files | [Prompt-analysis reproducibility](baseline-trial/report.md) |

Shell runner 的 env defaults 與 artifact lifecycle 由 canonical workflow 文件定義。
修改 env 名稱、預設值或 run-root 拒絕覆寫行為時，要同步更新對應文件與 regression
tests。

## J-space diagnostics and reports

| Script | Responsibility | Inputs and outputs | Related document |
|---|---|---|---|
| `jspace_layer_stats.py` | 計算 layer accuracy、kurtosis、lag agreement、effective dimension 與 CKA | lens/model → `layer_stats*.json`, `cka.npz` | [J-space sector intervention](jspace-sector-intervention/report.md) |
| `jspace_outcome_token_attribution.py` | 對 Buy/Sell decision token 做 per-(layer, position) 一階 attribution（embedding + 全層 residual 的 per-position gradient norm、position-class 聚合）；V2 position-rule 診斷 | outcome_flip_config + split prompts → `outcome_token_attribution.json` + metadata | [J-space outcome-conditioned decision-flip V2](jspace-token-experiments/details/proposal-v2.md) |
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
