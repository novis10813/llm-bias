# J-space token causal screen V1：representation-nominated token directions

## 文件狀態與科學邊界

**Version status：completed、exploratory、closed。** V1 的正式 discovery run 已完成，
shortlist 為空；本版本不再事後放寬 gate 或改寫 estimand。版本關係與目前研究入口見
[J-space token experiments](jspace-token-causal-screen.md)。

本文件描述目前已實作的 `jspace-intervention run-token-screen` workflow：對
[Valence readout](jspace-valence-vocabulary-readout.md) 產出的
**completed** `frozen_candidate_suggestions.json` 候選 token，做一次
**最小、可重複的第一輪因果篩選**（exploratory discovery screen）。它回答
「沿候選 token 的 transported 方向做對稱小劑量 additive steering，是否比
matched-norm random 方向更改變 buy/sell continuation margin」，並產出 shortlist
給後續 signed steering / gain / swap 的正式驗證。

**這是 discovery split 上的 exploratory screen，不是 confirmatory causal
evidence。** 候選提名、劑量、與 shortlist 規則在看到 screen 結果前由
SHA-bound config 凍結；shortlist 仍需在 calibration/test split 上做
preregistered 的 causal 驗證才能作為結論。Jacobian lens 是 transported
representation readout，不是 chain-of-thought、discrete reasoning path 或
standalone causal claim。

**`representation_side` 只是 provenance。** screen 永不以 candidate 的正負
side 乘以任何劑量符號：正 alpha 一律**加上**候選 token 方向，負 alpha 一律
**減去**。一個 side=negative 的候選在 +1 劑量下與 side=positive 的同一 token
得到完全相同的 intervention（有 regression test 保證）。

## 輸入

- **Baseline prompt CSV**（例：`data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv`），
  含 `ticker`、`name`、`sector`、`marketcap` 與 `prompt_with_context_*` 欄位。
  預設只使用 `prompt_with_context_attribute_0`（`--prompt-column` 可改）。
- **Split manifest**：既有 `jspace_intervention_splits` JSON。
- **Token screen config**：由 `prepare-token-screen-config` 從
  `frozen_candidate_suggestions.json` 凍結（見下）。
- **Model** 與 **validated canonical lens**（workflow 只消費既有 lens，
  不 fitting、不修改）。

## Config：`prepare-token-screen-config`

```bash
uv run jspace-intervention prepare-token-screen-config \
  --candidates artifacts/qwen3.5-4b/jspace-valence-readout/runs/<valence-run>/analyze/frozen_candidate_suggestions.json \
  --model .cache/models/qwen3.5-4b \
  --source-sector Technology \
  --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
  --layers 14,15,16,17,18,19,20,21,22,23,24,25,26 \
  [--alphas -1,0,1] [--top-positions 3] [--loading-threshold 0.0] \
  [--controls token,matched_random] \
  --output artifacts/qwen3.5-4b/jspace-token-screen/config.json
```

- `TokenScreenConfig` 把候選**逐字拷貝**自 candidate artifact
  （`token`、`token_id`、`representation_side`，以及 compact readout scores：
  `mean_positive`、`mean_negative`、`band_probability_diff`、
  `band_smoothed_log_ratio`、`band_js_contribution`），並綁定
  `candidate_artifact_path` + **SHA-256** 與 **split manifest SHA-256**。
  run 時若任一檔案與 config 的 SHA 不符，直接在建立 run 前失敗。
- 驗證規則：候選 `token_id` 唯一；`alphas` 必須**恰好是一對對稱非零劑量加 0**
  （預設 `-1,0,1`，排序後 `[-a, 0, a]`，a > 0、finite）；候選 token 不得是
  answer word（buy/sell）；`controls` 只能取自 `{token, matched_random}`；
  layers 非空且唯一；`top_positions` > 0；`loading_threshold` finite。
- `validate-config` 也能驗證 token screen config（依 `candidates` +
  `candidate_artifact_sha256` 欄位分派）。

## Forward：一次 clean pass，候選共享

對每個選中的 prompt（split 預設 `discovery`、sector 取 config 的
`source_sector`）：

1. **一次** clean forward 記錄所有候選共享層（`--layers`）的 residuals，並
   **一次** clean scoring 計算 `M = logP(buy) − logP(sell)`
   （固定答案 token continuation scoring，`decision_prefix` 後接 buy/sell；
   不比較 top-1 token，不保存完整 vocab 分布）。
2. 每個候選 token：layer 方向為 `W_U J_l[token_id]`（residual 座標，由
   canonical lens 的 Jacobian 與 model unembedding 預計算、不持久化）。
   在 evidence span 內以既有 median-cosine loading 選 **PRIMARY evidence
   positions**（`--top-positions`，`--loading-threshold`；與 swap/gain 共用
   `select_loaded_positions`）。
3. 每層 local scale = 選定 positions 上 `concept_coordinate` 的
   **median absolute value**；**只有 exact/numerical zero 才 floor 到 1e-6**
   （非零小值不放大）。
4. 交付的 coordinate delta = `alpha * local_scale / sqrt(n_layers)`，以既有
   `steer_positions` 在 PRIMARY positions 上**加減**方向（正 alpha 加、負
   alpha 減）。
5. Arms（`--controls`，預設兩者）：
   - `token`：候選方向本身；
   - `matched_random`：**同一批 PRIMARY positions、同一組 local scales**，
     換成 seeded **same-norm** random 方向（`matched_random_direction`）。
     因為方向同範數、positions 與 delta 相同，**不需要任何 norm rescaling**，
     也**沒有**其他 position control（沒有 shuffled/final-position 對照）。
6. alpha = 0 不施加 transform、直接重用 clean margin（no-op 行）。
7. Buy/Sell 都是單一 continuation token。為避免 BF16 lm-head logits 把小型
   intervention effect 量化成 0.125 的階梯，outcome scorer 只把 final residual
   的 final norm 與 unembedding 改用 FP32；decoder forward 與 intervention 仍是
   BF16。Artifact 明確標記 `single_token_fp32_final_norm_unembedding`。
8. 只 persist **compact 行**：candidate provenance（token/token_id/
   representation_side + readout scores）、arm、alpha、loaded positions、
   scale min/mean/max、clean/intervened/delta margin、delivered
   coordinate before/after、perturbation/state/relative norm 診斷。
   **不 persist** raw residual/activation/tensor、KL，也不做 full-vocab
   distribution（本 screen 不計算 next-token KL）。
9. Hook 由 `residual_interventions` 管理；forward 拋錯也保證移除
   （regression test 涵蓋）。

## Analyze：對稱斜率、specificity、shortlist

**Estimand（每 prompt、每候選、每 arm）：對稱斜率**

```
slope = (ΔM(+a) − ΔM(−a)) / (2a)      M = logP(buy) − logP(sell)
```

其中 `ΔM` 相對於 clean margin、`a` 是 config 的正劑量。對稱差把 dose-response
的奇數（線性）成分分離出來，並消去 clean 狀態的偶數偏移。

- **Specificity**（每 prompt）= `slope_token − slope_matched_random`：
  候選方向比 matched-norm random 方向多造成多少 buy/sell margin 位移。
- 聚合：先在 **ticker 內平均 prompts**，再對 tickers **等權平均**
  （ticker 是獨立單位）；CI 用既有 hierarchical **ticker bootstrap**
  （先抽 tickers、再抽 ticker 內 prompts，seed 固定）。
- 每候選報告 token arm、matched_random arm 與 specificity 的
  mean、CI95、**sign consistency**（與總體同号的 ticker 數）、
  **sign-flip p**（ticker means 上）；specificity 的 p 值對所有候選做
  **Holm–Bonferroni** 調整（`sign_flip_p_holm`）。
- **Shortlist**：要求 specificity CI **不含 0**、loading coverage ≥ 90%，
  且 specificity ticker sign consistency ≥ 70%；通過者依 |mean specificity|
  排序，**最多 2 個 buy-shifting**（CI 全正）與**最多 2 個 sell-shifting**
  （CI 全負）。Summary 同時報告 loading coverage 與最大/平均 relative
  perturbation norm，供判斷效果是否由 loading failure 或過大 dose 驅動。
- 整個 summary 標記 `"interpretation": "exploratory_discovery"`。

## Workflow 與 artifacts

`prepare → forward → analyze → finalize`（`ArtifactRun`），preflight
（chat-template 長度、sector/split 匹配、SHA 綁定）全部在建立 run 之前
完成；失敗不產生 run 目錄。

Run root：`artifacts/<model-slug>/jspace-token-screen/runs/<run-id>/`

```
manifest.json
prepare/
  prompt_records.jsonl             # 每個選中 prompt 一筆（record_id、ticker、prompt 等）
  metadata.json                    # input/split/candidate/config SHA、layers、alphas、controls、positive_dose
forward/
  token_screen_results.jsonl       # 每 (prompt, candidate, arm, alpha) 一筆 compact 行
  metadata.json                    # layers、alphas、controls、estimand、record_count
analyze/
  token_screen_summary.json        # 每候選 token/random/specificity 統計、Holm p、shortlist
  metadata.json                    # interpretation、shortlist counts
```

Manifest 的 `prepare`/`forward`/`analyze` 三 stage 全 complete 才能 finalize；
input refs 綁定 CSV、split manifest、config、candidate artifact 與 canonical lens。

## CLI

```bash
# 1) 凍結 config（綁定 candidate artifact 與 split manifest 的 SHA）
uv run jspace-intervention prepare-token-screen-config \
  --candidates .../frozen_candidate_suggestions.json \
  --model .cache/models/qwen3.5-4b --source-sector Technology \
  --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
  --output artifacts/qwen3.5-4b/jspace-token-screen/config.json

# 2) 跑 screen（預設 dataset jspace-token-screen、split discovery、
#    prompt column prompt_with_context_attribute_0）
uv run jspace-intervention run-token-screen \
  --input data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv \
  --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
  --config artifacts/qwen3.5-4b/jspace-token-screen/config.json \
  --model .cache/models/qwen3.5-4b \
  [--lens artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt] \
  --run-id token-screen-technology-1 \
  [--split discovery] [--max-records N] [--max-seq-len 1024] \
  [--prompt-column prompt_with_context_attribute_0]

# 3) 驗證任一 config（swap/gain/token screen 自動分派）
uv run jspace-intervention validate-config --config <config.json>
```

## Technology discovery 結果

正式 FP32-outcome run：

`artifacts/qwen3.5-4b/jspace-token-screen/runs/token-screen-technology-discovery-fp32-20260827T065908Z`

規模為 35 個 Technology discovery tickers、12 candidates、token/matched-random
兩 arms、`alpha = -0.5, 0, +0.5`，共 2,520 rows。所有 stages 完成；所有
candidate 的 loading coverage 都是 35/35，最大 relative perturbation norm 為
3.11%。

沒有 candidate 通過 frozen shortlist gate。兩個 CI 排除 0 的 exploratory signals：

| Token | Representation side | Token slope | Random slope | Specificity mean | Specificity CI95 | Ticker sign consistency | Holm p | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ` upgrade` | positive | +0.009739 | +0.000713 | +0.009026 | [+0.000893, +0.017322] | 24/35（68.6%） | 0.5717 | fail：低於 70% |
| ` downgrade` | negative | −0.015113 | −0.002756 | −0.012357 | [−0.022607, −0.002242] | 20/35（57.1%） | 0.4078 | fail：低於 70% |

其餘 10 candidates 的 specificity CI 都跨 0；所有 12 candidates 的
Holm-adjusted sign-flip p 都未達 0.05。因此 discovery shortlist 為空，不啟動
calibration、gain 或 positive↔negative swap。`upgrade` 與 `downgrade` 只能標記為
未通過 frozen gate 的 near-signals，不可事後放寬門檻。

先前一個 BF16 outcome diagnostic run 顯示 margin effect 被 lm-head logits 量化成
約 0.125 的階梯，已排除，不作結果解讀。正式 run 使用 config 綁定的
`single_token_fp32_final_norm_unembedding` scorer；decoder/intervention 本身仍為 BF16。

## 限制

- **Discovery-only、exploratory**：候選與參數都來自 discovery split；
  shortlist 不是結論，必須在 calibration/test 上做 follow-up。
- **第一輪 screen 刻意最小**：沒有 shuffled/final-position 對照、沒有
  next-token KL、沒有 dose 掃描（只有一對對稱劑量）。這些屬於後續
  formal experiment 的對照。
- Specificity 只控制**matched-norm random 方向**在相同 positions/scales 下
  的位移；它不控制方向與 prompt 內容的語意相關性，也不控制候選間的多重
  比較以外的 confound（Holm 只處理候選間 p 值）。
- Local scale 是 per-layer、per-prompt 的 median absolute coordinate；
  exact zero 才 floor 到 1e-6，不做其他正規化或跨候選 dose matching
  （token 與 random arm 同範數、同 positions、同 delta，norm 天然一致）。
- 候選 artifact 必須是 **completed** valence run 的
  `frozen_candidate_suggestions.json`（`prepare-token-screen-config` 檢查
  `artifact_type`）；SHA 綁定之後，候選內容任何改動都會讓 run 拒絕執行。
- `side` 不進入任何運算；若兩個候選 token_id 相同（正負側都有），
  config 驗證會拒絕（token_id 必須唯一）。

## Tests

`tests/test_jspace_token_screen.py`：config 驗證（對稱劑量、唯一性、
answer word、SHA）、fake model 的 dose math（closed-form margin、scale
floor、side invariance、matched-random 同範數、hook cleanup）、
analysis（對稱斜率分母、equal-ticker 聚合、specificity、Holm、shortlist
上限）、pipeline lifecycle（preflight 失敗不建 run、SHA tamper 拒絕、
determinism）與 CLI（config 凍結 SHA、validation 分派、run 預設值）。
