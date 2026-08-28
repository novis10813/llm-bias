# J-space token experiment versions

本頁是 J-space token intervention 的版本入口。討論、issue、config、run ID 與結果報告
必須標明版本，避免把 representation association、margin sensitivity 與 decision flip
混成同一個 estimand。

## Version matrix

| Version | Direction source | Primary outcome | Implementation | Evidence status |
|---|---|---|---|---|
| [V1](jspace-token-causal-screen-v1.md) | Valence readout 提名的 vocabulary token directions | 對稱 Buy/Sell margin slope 與 matched-random specificity | `prepare-token-screen-config`、`run-token-screen` 已實作 | Completed discovery screen；shortlist 為空 |
| [V2](jspace-outcome-direction-flip-v2.md) | Discovery prompts 上直接 fitting 的 outcome-gradient direction | Held-out Buy↔Sell decision flips；完整 generation 作必要 behavioral validation | `prepare-outcome-flip-config`、`run-outcome-flip` 已實作（Draft 1 凍結值）；尚無正式 run | Implemented，未產生 evidence |

## V1 結論

V1 測試 12 個 representation-nominated token directions。`upgrade` 與 `downgrade`
specificity CI 排除 0，但未通過 frozen ticker-consistency gate，所有 Holm-adjusted p
也未達 0.05。840 個非零 token-arm interventions 只有 1 個 fixed-choice margin sign
flip，且不是通過 gate 的 candidate。因此 V1 不支持 vocabulary direction 可穩健改變
Buy/Sell decision，也不進入 calibration、gain 或 swap。

Formal V1 run：

`artifacts/qwen3.5-4b/jspace-token-screen/runs/token-screen-technology-discovery-fp32-20260827T065908Z`

## V2 目的

V2 不從詞彙 association 選方向。它直接以 discovery prompts 的 Buy/Sell outcome
gradient fitting 一條 antisymmetric axis，並以 held-out decision flip 作主要結果：

- `+d` 應造成 Sell→Buy；
- `−d` 應造成 Buy→Sell；
- outcome direction 必須優於 same-norm random、label-permutation 與 position controls；
- fixed-choice flip 必須有 deterministic full-generation decision agreement，才可宣稱
  behavioral steering。

V2 的詳細 split、fitting、controls、estimands 與 freeze gates 見
[J-space outcome-conditioned decision-flip V2](jspace-outcome-direction-flip-v2.md)。
V2 有自己的 CLI（`run-outcome-flip`）、config schema 與 artifact identity；永遠不得用
V1 的 `run-token-screen` command 或 `jspace-token-screen` artifacts 冒充 V2。

## Versioning rules

1. V1 artifacts 與文件保持 immutable historical interpretation；修正 typo 或 path 時不
   改變 estimand、gate 或結果。
2. V2 implementation 必須使用新的 config/artifact type、dataset slug 與 run ID prefix，
   不覆寫 `jspace-token-screen` V1 artifacts。
3. 每個結果段落要寫明 version、split、direction source、primary outcome 與 controls。
4. 若 V2 protocol 在第一次 formal run 前調整，更新 draft revision 記錄；第一次正式
   run 後若改 estimand或 gate，建立 V3，不回填 V2。
