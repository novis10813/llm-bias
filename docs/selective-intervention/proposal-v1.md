# Selective-intervention V1：L15 k=8 subspace removal（protocol proposal）

**狀態**：frozen（2026-09-13，用戶批准完整臂 formal 預算）；formal run
`selective-intervention-v1-gpu-bf16-01` 完成（2026-09-14）：gate `fail`
（G1a/G1b/G2 pass、G3/G4 fail）→ 依 frozen 決策表 V1 於 full strength 為
負結果；完整結果見 [report-v1.md](report-v1.md)  
**對象模型**：Qwen3.5-4B（bf16，32 層，hidden 2560；`.cache/models/qwen3.5-4b`）  
**上游依賴**：
- `balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01`（64-prompt population、archived margins、spans）
- `entity-to-dial/runs/entity-to-dial-e-01`（`analyze/summary.json` 之 `pca_basis_vectors` / `pca_singular_values`，top-16 右奇異向量）  
**Roadmap 對應**：[M5 — Implement and evaluate selective intervention](../proposal/entity-bias-roadmap.md#m5--implement-and-evaluate-selective-intervention-2026-09-01--2026-09-07)；
[research proposal §4.4](../proposal/entity-bias-research-proposal.md) 候選形式 #3
（low-rank projection/subtraction from an entity-difference subspace）。

---

## 1. 動機與待測宣稱

Entity-to-dial 線已確認：8 個凍結方向的 L15 指令區間 state difference 被約束在
8 維線性子空間中（k=8 投影恢復 full-swap 效應的 98.3%）。本 V1 是該發現的第一個
推論期應用：對單一 forward 的 L15 post-block 殘差，在指令區間位置移除 entity
contrast 的 8 維子空間分量，測試三個預先註冊假設：

- **H1（steering efficacy）**：full-strength（α = 1.0）subspace removal 使 group
  gap 與 16 公司 margin IQR 至少縮減一半。
- **H2（specificity）**：同維度隨機正交子空間（其餘條件相同）的 gap 縮減 ≤ 主臂
  縮減量的 25%。
- **H3（task preservation / 無全局推注）**：full-strength 介入不產生全局 buy/sell
  偏移（64-prompt mean margin 位移 ≤ 0.15 nats），且不扰动 identity-stripped
  baseline（anonymous prompt margin 位移 ≤ 0.10 nats）。

## 2. 名詞定義

- **margin M(p)**：prompt p 在 `final_position` 的 fixed answer-token margin
  `log p("buy") − log p("sell")`（nats，buy 為正），依
  `core/continuation_scoring` 的 FP32 next-token log-prob 計分；不生成文本。
- **entity contrast margin**：公司 c 的 4-variant 平均 margin 減去 in-run
  identity-stripped（anonymous）prompt 的 margin，`E_c = M̄_c − M_anon`。
- **group gap G**：`mean(M̄_c, c ∈ {NSC, BLK}) − mean(M̄_c, c ∈ {IT, BDX})`
  （TOP/BOTTOM 群繼承 entity-to-dial frozen 定義）。clean 存檔值 1.028 nats。
- **spread S**：16 公司 `M̄_c` 的 IQR（clean 存檔值 0.5697）。
- **entity-difference subspace**：e-01 持久化的 state-difference SVD 右奇異向量
  的前 8 個（`V₈ ∈ R^{2560×8}`，正交歸一）。
- **entity-contrast center μ̄**：16 家 named 公司（每家取 rev0/ord0 variant）的
  L15 post-block 殘差於參考 instruction-span 格點上的逐位置平均向量
  `μ̄[p] ∈ R^{2560}`（calibration 階段 in-run 計算，不持久化張量，只存 digest）。
- **subspace removal transform**：對 L15 post-block 殘差 `h`，於指定位置集合 P
  施加 `h[p] ← h[p] − α·V Vᵀ(h[p] − μ[p])`（`p ∈ P`，其餘位置不變）；
  `α ∈ {0.25, 0.5, 0.75, 1.0}` 為 **strength**（roadmap 術語）。
  V 為 entity-difference subspace（主臂，k=8）或 frozen-seed 隨機正交 8 子空間
  （specificity 對照）；μ 為 entity-contrast center（primary）、0 或
  identity-stripped center（anonymous prompt 的 L15 狀態）。
- **position scope**：`instruction`（primary，依 entity-to-dial 定位）或
  `full`（全序列對照）。
- **layer control**：以 L15-fitted 子空間於其他層（L13/L14/L16/L17）施加同型
  transform，測試層級特異性。
- **dial probe**：在 clean 與 α=1 主臂下各施加 L15/n8490 `mlp_addition` ±4
  native-unit push，量 ΔM，檢查介入後決策旋鈕仍可控（描述性）。

## 3. Frozen 設計常數

| 項目 | 值 | 來源 |
|---|---|---|
| model path | `.cache/models/qwen3.5-4b`（bf16） | 上游 provenance |
| population | 2A `phase2a-gpu-bf16-01/prepare/prompts.jsonl`（64 = 16 公司 × 2 reverse × 2 order） | 上游 prepare |
| archived margin 參照 | 2A `forward/results.jsonl`（in-run clean forward 必須 bit-exact 復現） | 上游 forward |
| basis | e-01 `analyze/summary.json`：`pca_basis_vectors`（[16, 2560] rows）前 8 行；`load_pca_basis` fail-closed 驗證（orthonormal/finite/遞減） | 上游 analyze |
| 介入層 | L15 post-block（primary）；L13/L14/L16/L17（layer control） | entity-to-dial 收線定位 |
| position scope | instruction span（primary）；full sequence（對照） | entity-to-dial 收線定位 |
| strength grid | α ∈ {0.25, 0.5, 0.75, 1.0}；α = 0 為 structural no-op（不掛 hook） | 本提案 |
| primary centering | entity-contrast center μ̄（calibration in-run 計算） | 本提案 |
| centering 對照 | μ = 0；identity-stripped center（anonymous） | 本提案 |
| reference row（μ̄ 格點） | 2A prepare 中第一個 `(reverse=False, order=0)` row | 本提案 |
| random control | seed 20260913，Gaussian(2560×8) → QR → 正交列；in-run 由 seed 重算並存 digest | 本提案 |
| calibration population | 16 家 × rev0/ord0（每家 1 sequence）+ 1 anonymous sequence | 本提案 |
| dial probe | 1 支 prompt（reference row），±4 native-unit，clean 與 α=1 各一次 | 本提案 |
| 計分 | fixed answer-token margin（FP32），不生成 | repo 語義邊界 |

## 4. 介入定義與實作約定

1. **transform**：`h' = h` 於 `P` 外 bit-exact 不變；於 `P` 內
   `h'[p] = h[p] − α·(V (Vᵀ (h[p] − μ[p])))`，投影運算以 FP32 執行後 cast 回
   tensor dtype（與 Phase E joint patch 的 FP32 鏈慣例一致）。
2. **掛鉤點**：`core/inference/residual_interventions(model, {layer: fn})`
   （post-block forward hook），`fn` 接收/回傳 `[batch, seq, d]`。
   α = 0 臂完全不掛 hook（structural no-op，clean 的 bit-exact 別名）。
3. **dial probe 組成**：subspace removal（residual hook）與 `mlp_addition`
   （MLP down-proj pre-hook）為不同模組 hook，可嵌套；strict single-fire
   與 exception-safe 清理同 Phase F dual-hook 慣例。
4. **calibration 擷取**：`record_block_states(model, ids, [15])` 取
   `states[15]["post"]`（transient，用完即棄）；μ̄ 於參考格點、μ_anon 於
   anonymous 自身格點；跨序列對齊用 `nearest_position_mapping`（Phase E 慣例）。
5. **不持久化**任何 raw 狀態/殘差/μ 張量；持久化者僅 digest（SHA-256 over FP32
   bytes）、count、digest 對應的 reference row ID、random seed。

## 5. Run 結構

`prepare → forward → analyze`（單一 run，`ArtifactRun` 生命週期）。

| 臂 | α | μ | 層 | scope | population | forwards |
|---|---|---|---|---|---|---|
| `clean`（in-run 參照＋ α=0 no-op） | 0 | — | — | — | 64 | 64 |
| `dose_{25,50,75,100}` | .25/.5/.75/1.0 | μ̄ | 15 | instruction | 64 | 256 |
| `center_zero` | 1.0 | 0 | 15 | instruction | 64 | 64 |
| `center_anon` | 1.0 | anon | 15 | instruction | 64 | 64 |
| `scope_fullseq` | 1.0 | μ̄ | 15 | full | 64 | 64 |
| `ctrl_random` | 1.0 | μ̄ | 15 | instruction | 64 | 64 |
| `ctrl_layer_{13,14,16,17}` | 1.0 | μ̄ | 各層 | instruction | 16（rev0/ord0） | 64 |
| `anon_probe`（＋ in-run clean anon） | 1.0 | μ̄ | 15 | instruction(anon) | 1 | 2 |
| `dial_probe_{pos4,neg4}`（clean 與 α=1） | 1.0 + ±4 | μ̄ | 15 | instruction | 1 | 4 |
| calibration 擷取（16 named + 1 anon，只 capture） | — | — | 15 | — | 17 | 17 |

Formal 總計 ≈ 663 forwards（單卡約 60–70 分鐘）。Smoke（fake model）同結構、
2 公司 × 4 variant（8 prompts）＋ 1 anon、strength grid {0.5, 1.0}、layer
control 只取 1 層、dial probe 用 fake neuron。

Artifact 布局（`artifacts/qwen3.5-4b/selective-intervention/runs/<run-id>/`）：
`prepare/{provenance.json, metadata.json}`、`forward/records.jsonl`
（每 prompt × 臂：id/ticker/sector/reverse/order/arm/alpha/centering/layer/
position_scope/margin/decision/noop）、`forward/metadata.json`（臂清單、
digest、seed、m_anon clean）、`analyze/summary.json`（gates＋描述統計）、
`manifest.json`。JSON key 不得命中 core 序列化器保留字（含遞迴 guard 測試）。

## 6. Pre-checks（prepare 階段，fail-closed）

1. 上游 manifest digest（2A、e-01）與 template 常數一致。
2. model/tokenizer identity 與 2A、e-01 provenance 一致。
3. basis 載入成功（`load_pca_basis`，k=16、d=2560；使用前 8 列）。
4. 64 行 instruction span 非空；每公司 4 variant 的序列長度與 instruction span
   一致（prompt_ids 可因 reverse 文案不同，長度必須相同）。
5. reference row 存在（第一個 rev0/ord0）。
6. **in-run bit-exact**：4 支 probe prompt（4 個 (reverse, order) 組合各取 1 家）
   clean forward margin 與 2A 存檔 margin 差異 0.0。

## 7. 分析與 gates

主臂 = `dose_100`（α=1.0、μ̄、L15、instruction）。所有 gate 用 in-run clean
值計算（clean 與 2A 存檔 bit-exact 已於 pre-check 6 確認）。

| Gate | 判定 | 門檻 |
|---|---|---|
| **G1a**（efficacy，primary） | `|G_α1| ≤ 0.5 · G_clean`（group gap 至少減半） | 0.5 |
| **G1b**（efficacy，generalization） | `S_α1 ≤ 0.5 · S_clean`（16 公司 IQR 至少減半） | 0.5 |
| **G2**（specificity） | 隨機子空間 gap 縮減 `≤ 0.25 ×` 主臂縮減，其中縮減 `= 1 − |G_int|/G_clean` | 0.25 |
| **G3**（無全局推注） | `\|mean_64(M_α1) − mean_64(M_clean)\| ≤ 0.15` nats | 0.15 |
| **G4**（identity-stripped 穩定） | `\|M_anon(α1) − M_anon(clean)\| ≤ 0.10` nats | 0.10 |

描述性（不 gate）：
- 完整 dose-response 曲線：每 α 的 G、S、mean margin、entity contrast 縮減、
  64-prompt sell→buy decision flip 數。
- per-company 表：clean vs α1 的 `M̄_c`、`E_c`、decision。
- centering 比較（μ̄ vs 0 vs anon，α=1）：G、mean shift。
- position scope 比較（instruction vs full）。
- layer control 曲線（L13/14/15/16/17，α=1）。
- order-consistency：per-company `(M̄_ord1 − M̄_ord0)` 於 clean vs α1 的最大位移。
- dial probe：clean 與 α1 下 ±4 push 的 ΔM（決策旋鈕可控性）。

## 8. 決策表

| 結果 | 解讀與後續 |
|---|---|
| G1a＋G1b＋G2＋G3＋G4 全 pass | V1 完成：subspace removal 在 frozen population 內有效、specific、無全局副作用；M5 主要宣稱成立，線可推進 M6（hold-out population / cross-model） |
| G1a pass、G1b fail（G2/G3/G4 pass） | efficacy 限於 4 家 calibrated 極端公司的 contrast；12 家擴散未被 8 維子空間完整解釋（與子空間以 4 家 direction 擬合一致）；記為 V1 邊界 |
| G1a fail | efficacy null：k=8 子空間移除不縮減 group gap。與 Phase E（transplant 有效）並列記錄——sufficient for transfer ≠ removable in behavior；線以 null 收線或依報告討論決定 follow-up（非 V1 範圍） |
| G2 fail | 縮減非子空間特異（一般性 L15 instruction-state 擾動即可達）；H2 被否決，宣稱降級 |
| G3 或 G4 fail | full-strength 有全局副作用；報告 dose 曲線，V1 於 full strength 為負結果（不預先註冊 dose 救援） |

## 9. 邊界與限制

1. **population 內 efficacy**：子空間以同一 16 公司 population 中 4 家極端公司
   的 8 direction 擬合；V1 宣稱限於 population 內。Hold-out population、
   cross-model 屬 M6。
2. **frozen balanced-evidence task only**：該 task 無 ground-truth 標籤
   （evidence 2 正 2 負、公司中立），legitimate entity information preservation
   無法直接以 accuracy 度量；以 G3/G4＋dial probe 作為操作性替代。
3. **μ̄ calibration 與評估 population 重疊**（16 家 named 公司）；已如實記錄。
4. V1 無 risk gating / per-prompt 適配（「selective」指 entity-contrast 特異
   子空間，非 adaptive）。
5. 計分為 fixed answer-token margin（不生成）；不宣稱生成端行為。
6. 單模型（Qwen3.5-4B）；bf16、bit-exact 可重現。

## 10. 實作與驗證

- 新 package `llm_bias/selective_intervention/`（`template.py` frozen 常數、
  `subspace.py` transform＋calibration＋random basis、`analysis.py` gates、
  `pipeline.py` 三階段 lifecycle）；operator `scripts/selective_intervention_v1.py`
  （`--smoke` / formal `--run-id`）。不得 import 其他 experiment package
  （2A rows 以存檔 JSONL 消費，非 import）。
- 測試 `tests/test_selective_intervention_*.py`（fake model：16 層、hidden 32、
  intermediate 64；2 公司 fake population 沿用 balanced_evidence_gap 測試基建；
  fake basis [32, 8]）：α=0 bit-exact、span 外位置 bit-exact、pure in-subspace
  分量於 α=1 精確移除、calibration 對齊與 fail-closed、random basis
  orthonormal＋seed 決定性、dial probe hook 組成與清理、gate 邊界值、
  summary schema 過 core 序列化器遞迴 guard、pipeline smoke lifecycle。
- 驗證命令：`uv run pytest -q`、`uv run python -m compileall -q llm_bias`、
  `uv build`、`uv lock --check`。
- Smoke 在 fake model 上執行（無 GPU）；real-model smoke 與 formal 依慣例
  逐次授權。Real-model smoke 的 population 為 4 家 frozen group 公司
  （NSC/BLK/IT/BDX）× 4 variants = 16 prompts、strength grid {0.5, 1.0}、
  layer control 只取 1 層：122 arm forwards ＋ 5 calibration forwards = 127
  （Rev 1.1）。Fake pipeline 測試以 16 層 fake model（L15 = final layer）
  跑同結構 smoke。

## 11. Revision history

- **Rev 1.5（2026-09-14，formal results＋implementation notes）**：
  - formal run `selective-intervention-v1-gpu-bf16-01` 完成（663 forwards、
    1130 s、manifest complete）：G1a pass（0.5863 ≤ 0.6348）、G1b pass
    （0.2800 ≤ 0.2820）、G2 pass（random 縮減 0.0049 vs 主臂 0.5382）、
    G3 fail（+0.3299 > 0.15）、G4 fail（−0.2689 > 0.10）；依 frozen 決策
    表 V1 於 full strength 為負結果（efficacy＋specificity 陽性）；完整
    報告見 [report-v1.md](report-v1.md)。
  - B6（實作修正，additive，不改變 frozen 設計或 gate 邏輯）：forward
    構造 orthonormal 檢查的 `torch.eye` device 不匹配（GPU-only）修正；
    forward metadata 新增 `n_clean_bit_exact`／`clean_max_abs_delta_m`
    審計欄位；summary 新增 `random_control` 描述欄位。
- **Rev 1.1（2026-09-13，implementation notes）**：frozen 後實作對齊筆記，
  不改變 frozen 設計：
  - B1：smoke population 由 2 家改為 4 家 frozen group 公司（group-gap
    描述統計在 smoke 仍可計算）；smoke 總 forwards = 127。
  - B2：calibration 的 full-sequence cloud grid 與 instruction-span grid
    共享同一批 named-sequence forwards（calibration 總計 17 forwards，§5
    預算不變）。
  - B3：dial probe 以 core `mlp_addition` 與 subspace removal 的 residual
    hook 嵌套（不同模組，strict single-fire 與 exception-safe 清理）。
  - B4：`load_e01_basis`、`nearest_position_mapping`、`anonymous_prompt` /
    `instruction_char_span` 為 self-contained 副本（experiment package 互不
    import 之邊界）；promote to core 為未來收斂事項。
  - B5：margin 減法语義對齊 2A 參照機制——`margin_from_log_probs` 先轉
    Python float（float64）再相減，與 `score_single_token_margin_fp32`
    bit-for-bit 一致（FP32 張量減法會引入 ≤ 1 float32 ulp 的圓整差異，
    首次 real-model smoke 於 clean 預檢以 5e-8 差異被擋下；純實作修正，
    不改變 frozen 設計）。附回歸測試鎖定該語義。
- **Rev 1（2026-09-13）**：初稿。基於 entity-to-dial 收線（k=8 子空間、L15
  定位、F1/F2 結果）與 roadmap M5 要求（low-rank baseline、layer × strength、
  random-direction control、dose-response、side-effect analysis）起草；2026-09-13
  用戶按 Rev 1 原樣 frozen（完整臂，formal ~65 min 預算）。
