# Balanced Evidence Gap — Phase 3：Entity-specific MLP coordinate 因果驗證協議

> **狀態：Frozen（Rev 2，2026-09-10 使用者確認凍結）**。本協議是 [Phase 2](proposal-phase2.md)
> 2C 發現（L19/L20/L26 三個 entity-specific MLP channel）的 causal
> validation 版本。凍結後任何 direction source、primary outcome、
> controls 或 gate 變更都必須建新版本（見 [experiment
> versioning](../documentation-system.md#experiment-versioning)），不回填。

## 1. 背景與待解問題

Phase 2 的 2C（[report-phase2.md](report-phase2.md)）在 handoff 區間
L12–31 做 MLP down-projection 的 local first-order attribution
（|∂M/∂a|·|a|，entity position，9216 維），找到三個通過 gate 2C 的
channel（run `phase2c-gpu-bf16-05`，gate 重評
`phase2c-gate-reanalysis-01`）：

| Coordinate | 2C ρ（attribution vs pure entity margin，16 tickers） |
|---|---|
| L19 / n6334 | −0.897 |
| L20 / n6520 | +0.894 |
| L26 / n2394 | −0.859 |

2C 的證據是**關聯性**（跨 ticker 的一階敏感度與 margin 的相關，勝過
10 個 matched controls）。未驗證的問題：**對這些 coordinate 做 additive
干預，decision margin 是否真的移動，且效果勝過 matched controls？**
本協議回答這個問題。範式同 [financial-soundness causal
validation](../financial-soundness-causal-validation/proposal.md) 的
candidate-vs-controls 結構；干預語義複用 investment-dial 的
`mlp_addition`（additive Δ，all token positions，native units）。

## 2. 定義（本協議新增術語）

- **candidate coordinate**：§4.1 凍結的三個 (layer, neuron) MLP
  down-projection coordinate。
- **predicted direction**：由 2C ρ 符號先驗決定的 margin 移動方向
  （§4.1）：ρ>0 → δ>0 應使 margin 上升（buy 方向）；ρ<0 → δ>0 應使
  margin 下降（sell 方向）。依據：2C attribution 量測「該 channel 的
  影響幅度」與 ticker 的 buy-lean 程度相關；predicted direction 假設
  高影響幅度對應「推向該 ticker 自身 margin 方向」，即 channel 編碼
  buy-lean/sell-lean coordinate。此假設由 pilot 的 local derivative
  診斷（§4.4，descriptive）對照，但不改變 gate 方向。
- **gate δ**：每個 candidate 單個、先驗決定的干預點（§4.5）：predicted
  direction 為 buy 方向者取 +4s，sell 方向者取 −4s（s 定義見 §4.5）。
- **control coordinate**：該 candidate 同層的 2C matched random
  channels（seed 42+layer，10 個/層，§4.2）。
- **ΔM**：M(δ) − M(0)，logit margin（buy−sell）差值，與 2A/2B/2C
  同一定義（fixed answer token logit margin，非 top-1 probability 差）。

## 3. 假說

- **H1（primary，causal existence + direction）**：至少 2/3 個
  candidate coordinate 的 additive 干預使 margin 朝 predicted direction
  移動，跨 ticker 一致性通過 Holm 調整的 sign-flip test，且 |mean ΔM|
  勝過同層全部 10 個 control coordinates。
- **H2（descriptive，linearity）**：observed ΔM(δ) 近似線性於 δ（與
  一階 attribution 的預測對照），用於檢查干預幅度是否越出 local
  linear regime。不 gate。

## 4. 實驗設計

### 4.1 Candidate coordinates 與 predicted directions（frozen after Rev-2 2C gate）

來源：`phase2c-gate-reanalysis-01/analyze/summary.json`（gate 2C
passing layers；provenance 含 source run 的 SHA-256）。

| Coordinate | 2C ρ | predicted direction（δ>0） | gate δ 符號 |
|---|---|---|---|
| (19, 6334) | −0.897 | margin 下降（sell） | −4s |
| (20, 6520) | +0.894 | margin 上升（buy） | +4s |
| (26, 2394) | −0.859 | margin 下降（sell） | −4s |

### 4.2 Control coordinates（frozen）

沿用 2C 的 matched-control 抽樣規則（`random.Random(42 + layer).sample(
range(9216), 10)`），index 可從 seed 規則確定性重導（2C records 未
持久化 index，只有 ρ 值；重導結果已驗證不含該層 top neuron）：

| 層 | 10 個 control neurons（sorted） |
|---|---|
| L19 | 454, 2977, 3551, 4805, 5252, 5258, 5856, 7965, 8101, 9120 |
| L20 | 1067, 2646, 2803, 2834, 3065, 3901, 5077, 5828, 7614, 8323 |
| L26 | 1818, 3987, 4349, 5396, 7075, 7178, 7640, 7687, 8204, 9064 |

run 的 prepare stage 將這 30 個 index 以同一 RNG 規則重導並持久化。
**（Rev 2 校準）**pilot 的 2C 重導驗證容差從 Rev 1 的 bit-exact（1e-9）
改為經實測校準的 jitter tolerance（Rev 2，2026-09-10）：

- 30 個 controls：每個 |Δρ| ≤ 0.05；
- 3 個 candidate coordinates：|ρ_recomputed − ρ_2c| ≤ 0.05；
- top neuron：重導 argmax |ρ| ≥ 2C top |ρ| − 0.05（層級發現強度保留），
  取代 Rev 1 的 strict argmax 匹配。

依據（診斷 run，16 tickers × 3 層重算）：L26 全部 11 個 ρ bit-exact；
L19/L20 的 control ρ 最大跨 run 差異 0.0412、candidate ρ 最大 0.0177；
L19 的 2C top（n6334，−0.8971）與 n5233（+0.8794）重算後 |ρ| 完全相等
（0.8794），屬 bf16 backward jitter 下的刀鋒鏡像對——strict argmax 匹配
在此模型上原理上不可達成。wrong construct/position/index 會造成系統性
大偏離（≫0.05），fail-closed 语义保留。prepare 另從模型斷言每個
candidate 層的 `dense_down_projection.in_features == 9216`（sample space
與 2C runtime 一致），不符即 fail-closed。

### 4.3 刺激與 baseline

- 16 tickers（§template 同 2A/2C）× canonical variant
  （reverse=False, order=0）= 16 條 2A frozen-template prompt。
- baseline M(0) = 2A run `phase2a-gpu-bf16-01` 的 pure entity margin
  （確定性 logit scoring，從 run records 重算；與 2A summary 對照
  fail-closed）。16/16 為 sell（margin 全負）。

### 4.4 Pilot stage（descriptive，定 grid scale 與 derivative 診斷）

對 33 個 coordinates（3 candidate + 30 control）× 16 prompts 各一次
forward（可合併為 16 次 multi-hook forward）：

1. **Activation scale**：每個 coordinate 的 s = 90th percentile of
   |a|，a = MLP down-projection input 在該 (layer, neuron) 的 value，
   跨全部 prompts 與全部 positions。s 持久化於 pilot summary（grid 的
   唯一輸入）。
2. **Local derivative 診斷**（`mlp_summed_derivatives`，descriptive）：
   sign(∂M/∂a) 於 3 個 candidates，跨 tickers 的平均。若與 predicted
   direction 矛盾，在 report 標記 diagnostic flag；**gate 方向不變**
   （方向先驗凍結於 §4.1，pilot 不調整 gate）。

### 4.5 干預與 δ grid（pre-registered rule）

- 干預 primitive：`llm_bias/core/inference/mlp_addition`（additive Δ，
  all token positions，native units；同 investment-dial A-curve 語義）。
  單 step continuation scoring（buy/sell answer token），無 generation。
- Grid rule（frozen）：candidate c 的 grid = {0, ±s_c, ±2s_c, ±4s_c}
  （s_c 來自 pilot §4.4.1）。0 點重用 §4.3 的 M(0)，不 forward。
- 干預點：candidates 跑全部 6 個非零 δ（curve，descriptive）；
  **gate 只評估 gate δ**（§2 定義）。controls 只跑其層的
  {−4s_c, +4s_c} 兩點（取 max |mean ΔM|，direction-agnostic）。

### 4.6 Gate 3A（pre-registered）

每個 candidate 需同時滿足（3 條）：

1. **Direction**：mean_c(ΔM at gate δ) 的符號 = predicted direction。
2. **Consistency**：16 個 tickers 中 sign(ΔM(t) at gate δ) =
   predicted direction 的比例 k/16；exact binomial one-sided test
   p_c；Holm 調整跨 3 candidates；adjusted p < 0.05。
   （n=16 下 equivalent threshold：≥13/16 同向。）
3. **Control superiority**：|mean_c(ΔM at gate δ)| > max over 該層
   10 個 controls 的 |mean(ΔM)|（controls 取其兩點中較大者）。

**Line-level verdict**：confirmed = 通過全部 3 條的 candidate 數；
**gate 3A pass = confirmed ≥ 2/3**。per-candidate 結果全部報告，
不論 verdict。

### 4.7 附加測量（descriptive，不 gate）

- **Decision flips**：baseline 16/16 sell；L20（buy 方向）在各 δ 的
  sell→buy flip 數；sell 方向 candidates 的 margin 距 flip 門檻的
  距離變化。
- **Sector 一致性**：4 sectors × 4 tickers 的 mean ΔM 方向（同 2C
  sector agreement 口徑）。
- **Linearity（H2）**：ΔM(δ) 對 δ 的 slope 與 pilot local derivative
  的對照（一階預測 vs observed）。
- **Dial 對照**（descriptive）：同一批 prompts 對 L15/n8490 的
  mlp_addition 效果（investment-dial 已知為 model-level prior），用於
  區分「entity-specific channel」與「通用 stance coordinate」的效應
  量級差異。

## 5. 實現與 artifact

- 新模組 `llm_bias/balanced_evidence_gap/neuron_causal.py`：pilot
  stage（activation scale + derivative 診斷）、intervene stage
  （`mlp_addition` 循環 + margin scoring）、analyze stage（gate 3A）。
  複用 `template.py` 常量、`analysis.py` 的 test helpers、
  `llm_bias/core/inference/mlp_addition.py`。不 import 其他 experiment
  package。
- Operator：`scripts/balanced_evidence_gap_phase3.py`。
- Run：`phase3-gpu-bf16-01`（formal）；先 fake-model smoke
  （`phase3-smoke-01`）與小型 GPU smoke。
- Stages：`prepare`（provenance：2A summary SHA-256、2C re-analysis
  summary SHA-256、control index 重導 + 2C ρ 對照）→ `pilot` →
  `intervene` → `analyze` → finalize。
- 持久化（compact，無 raw activation）：
  - `prepare/provenance.json`
  - `pilot/summary.json`（s_c、derivative 診斷、2C ρ 對照結果）
  - `intervene/records.jsonl`（per (coordinate, δ, ticker)：margin、
    ΔM、decision；每行一條）
  - `analyze/summary.json`（per-candidate 3 條 criterion、Holm p、
    control 分佈、verdict；descriptive 區塊）
- 不保存 raw activations/residuals/hidden states；pilot 只存分位數
  統計量與符號。

## 6. 成本估算

| Stage | forwards | 估計時間（GPU 0） |
|---|---|---|
| pilot | 16（multi-hook） | ~1 min |
| intervene（candidates） | 3 × 6 δ × 16 = 288 | ~6 min |
| intervene（controls） | 30 × 2 δ × 16 = 960 | ~20 min |
| **total** | **~1,264** | **~25–30 min** |

## 7. 邊界與非目標

- 只做 causal validation；2C 發現本身 frozen，不因本線結果重評。
- 16 companies（2A population）；對 427 家宇宙（investment-dial
  dataset）或 test split 其餘 70 家的一般化聲明**不在本線範圍**
  （若本線 pass，另開 version 做 population 擴展）。
- All-positions 干預語義（dial convention）；entity-position-only
  masking 不在本版（若需要，新版本）。
- 不做 3 個 coordinates 的 joint（同時）干預（若需要，新版本）。
- predicted direction 基於 2C 的 unsigned attribution 符號解讀（§2）；
  若 pilot derivative 診斷矛盾，報告 flag，gate 方向不調整。
- Gate 只在單一 gate δ 評估；curve 其餘點 descriptive。
- 本線 pass 不等於「這三個 neuron 是 entity 表示的完整載體」；只
  確認「這三個 coordinate 對 16-company decision margin 有超過
  controls 的定向因果效應」。

## 8. Revision record

| Rev | 日期 | 變更 |
|---|---|---|
| 0（Draft） | 2026-09-10 | 初稿：3 candidate coordinates、predicted directions、control set、grid rule、gate 3A。 |
| 1（Frozen） | 2026-09-10 | 使用者確認凍結。§4.2 明確化：ρ 對照 tolerance = bit-exact（容許 1e-9 浮點表示誤差）；prepare 新增 `dense_down_projection.in_features == 9216` 斷言。其餘 §4 數值與 gate 未變。 |
| 2（Frozen） | 2026-09-10 | 使用者確認凍結。僅改 §4.2 的 2C 重導驗證 tolerance（bit-exact → 實測校準的 jitter tolerance：controls \|Δρ\|≤0.05、candidate \|Δρρ\|≤0.05、層級發現強度 argmax\|ρ\| ≥ 2C top − 0.05）。原因：formal run `phase3-gpu-bf16-01` 在 bit-exact 檢查 fail-closed；診斷顯示 hybrid 模型 bf16 backward 跨 run jitter（L19/L20 control ρ 最大差 0.0412；L19 top 為 n6334/n5233 完美鏡像對）。candidate coordinates、predicted directions、grid、gate 3A 全部不變。 |
