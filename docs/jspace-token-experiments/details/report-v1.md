# J-space token causal screen V1：實驗報告

## 文件狀態與科學邊界

**Version status：completed、exploratory、closed。** V1 的正式 discovery run 已完成，
shortlist 為空；本版本不再事後放寬 gate 或改寫 estimand。版本關係與目前研究入口見
[J-space token experiments](README.md)。

本文件描述目前已實作的 `jspace-intervention run-token-screen` workflow：對
[Valence readout](../../jspace-valence-readout/proposal.md) 產出的
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


方法與 frozen screen contract 見 [V1 提案](proposal-v1.md)。

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
