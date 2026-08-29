# J-space token experiment versions

本頁是 J-space token intervention 的版本入口。討論、issue、config、run ID 與結果報告
必須標明版本，避免把 representation association、margin sensitivity 與 decision flip
混成同一個 estimand。

## Version matrix

| Version | Direction source | Primary outcome | Implementation | Evidence status |
|---|---|---|---|---|
| [V1](jspace-token-causal-screen-v1.md) | Valence readout 提名的 vocabulary token directions | 對稱 Buy/Sell margin slope 與 matched-random specificity | `prepare-token-screen-config`、`run-token-screen` 已實作 | Completed discovery screen；shortlist 為空 |
| [V2](jspace-outcome-direction-flip-v2.md) | Discovery prompts 上直接 fitting 的 outcome-gradient direction | Held-out Buy↔Sell decision flips；完整 generation 作必要 behavioral validation | `prepare-outcome-flip-config`、`run-outcome-flip` 已實作（Draft 1 凍結值）；第一次正式 pipeline run 已完成 | Test run 完成：`success=false`（sell 方向 Holm gate 因 n=2 結構性不可顯著；buy 方向 9/9 vs 0/9、Holm p=0.003 通過） |

## V1 結論

V1 測試 12 個 representation-nominated token directions。`upgrade` 與 `downgrade`
specificity CI 排除 0，但未通過 frozen ticker-consistency gate，所有 Holm-adjusted p
也未達 0.05。840 個非零 token-arm interventions 只有 1 個 fixed-choice margin sign
flip，且不是通過 gate 的 candidate。因此 V1 不支持 vocabulary direction 可穩健改變
Buy/Sell decision，也不進入 calibration、gain 或 swap。

Formal V1 run：

`artifacts/qwen3.5-4b/jspace-token-screen/runs/token-screen-technology-discovery-fp32-20260827T065908Z`

## V2 第一次正式 run 結果（2026-08-28）

Full pipeline：discovery（deterministic mode）→ calibration（選定 L10–30、
`evidence_item_end`、r=0.4）→ test（11 個 held-out Technology tickers）。
Verdict：`success=false`；失敗 gate 僅 sell 方向的 Holm-adjusted specificity——
sell-eligible 只有 2 個 tickers，exact paired test 在 n=2 時最小 p_one_sided 為
0.25，結構性不可顯著。Buy 方向 9/9 vs matched-random 0/9（label-permutation
reverse 9/9），Holm-adjusted p=0.003，specificity CI [1,1]，generation parse
與 agreement 皆 100%。final-position control 在 calibration 與 test 皆雙方向
100% flip，position specificity 不成立；解釋上限為 outcome-axis steering of
the decision，非 evidence-position 因果。run-once 性質已消耗；sell 方向重驗
需新版本（更大/更平衡 test split）。

Formal V2 runs（config：`config-technology-draft1.json`）：

- discovery：`artifacts/qwen3.5-4b/jspace-outcome-direction-flip/runs/outcome-flip-tech-discovery-det-20260828T015605Z`
- calibration：`artifacts/qwen3.5-4b/jspace-outcome-direction-flip/runs/outcome-flip-tech-calibration-det-20260828T015659Z`
- test：`artifacts/qwen3.5-4b/jspace-outcome-direction-flip/runs/outcome-flip-tech-test-20260828T051624Z`

輔助診斷（非 V2 protocol 的一部分）：per-(layer, position) attribution screen
（`scripts/jspace_outcome_token_attribution.py`，
`artifacts/qwen3.5-4b/jspace-outcome-token-attribution/`）顯示 evidence item-end
的 per-token 一階敏感度集中在 L0–L14（L15–L18 斷崖），而 aggregate direction
的跨 ticker 對齊在 L15–L26 最高，兩者不矛盾。outcome direction geometric
projection（逐層 Technology−Financial Services sector state difference 在
frozen outcome directions 上的 dot projection 與 parallel/perpendicular 分解；
描述性幾何、非 causal）見
[outcome direction 幾何投影分解](jspace-outcome-direction-geometry.md)；第一次
正式 run（2026-08-28）顯示 sector state difference 在 L10–30 幾乎全部
orthogonal 於 outcome direction（angle 82°–92°、parallel energy fraction ≤
0.019），TF-IDF sector prototypes 亦然（|coefficient| ≤ 0.036）。Direction
decode（`jspace-intervention decode-outcome-direction`，以 canonical Jacobian
lens 解碼 frozen $d_l$ 的 transported direction logit 與完整詞彙 softmax，見 V2
文件「Direction decode」節）同為輔助診斷，不引入新的 estimand 或 gate。

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
