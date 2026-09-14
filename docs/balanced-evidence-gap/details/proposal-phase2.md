# Balanced Evidence Gap — Phase 2：Entity-to-Decision Path 定位協議

**狀態**：proposed（待授權）。Phase 1 三判準全過（見 [Phase 1 報告](report-phase1.md)），
授權本 Phase。  
**對象模型**：Qwen3.5-4B（bf16）。  
**研究線定位**：在平衡證據條件下，定位 entity identity 從 header token 傳入決策
margin 的**中間層路徑**（layer × span × component）。本 Phase 不建立
entity = bias 的規範性宣稱；entity gap 可能混合 legitimate prior 與
unsupported preference（定義見 [entity-bias 研究提案](../../proposal/entity-bias-research-proposal.md)）。

---

## 1. 背景與待解問題

Phase 1 確認了行為端點：平衡證據下，named-vs-anonymous gap +0.432 nats
（CI [+0.350, +0.518]），公司間 margin IQR 1.375 nats。但 Phase 1 的公司間
分歧混合了兩類來源：(i) 純 entity identity 效應（同證據、不同名稱）；
(ii) 各公司財務證據內容差異。Phase 2 的第一個任務就是分離 (i)。

既有研究線提供了兩端錨點與方法先例：

- **Entity Cell 線**（L0–L4 事實記憶神經元）：與決策路徑解離，不承載
  stance。
- **Causal tracing 線**：明確證據條件下 `all_evidence` span 的因果充分性
  97.56%（L6），`instruction_context` 峰值在 L16，`final_position` 主導於
  L22+。其 span 定義與 patching 契約（
  [proposal](../../activation-patching-causal-tracing/proposal.md)）是本协议
  的术语基础。
- **Investment-dial 線**：L15/n8490 是 model-level stance prior 旋鈕，
  不認識特定 entity。

**Phase 2 的核心問題**：在平衡證據下，entity identity 訊號在哪一層、透過
哪個 span 與哪些 component，從 header 位置傳入決策 margin？

---

## 2. 定義（本协议新增術語）

- **shared-evidence template**：一組公司無關（company-neutral）、多空完全
  平衡（2 正 2 負）的財務證據句，固定於本协议 §4.2，不引用任何真實公司
  的公開財報。
- **cross-entity probe**：把同一組 shared-evidence template 搭配不同公司名
  的 prompt 族；族內所有 prompt 只有 entity header 不同。
- **pure entity margin**：cross-entity probe 中某公司的 clean margin；
  與 Phase 1 的 named margin 不同，後者的證據來自該公司自身的
  evidence_pairs。
- **entity contrast**：兩家公司的 pure entity margin 之差
  $M_A - M_B$。
- **entity-state patching**：把 source prompt 的 entity 相關 span 殘差狀態
  寫入同證據 target prompt 對應位置的 residual-stream resample patching；
  机制与
  [causal tracing 的 patching 契約](../../activation-patching-causal-tracing/proposal.md#patching-protocol)
  相同（bidirectional、FP32 tail-logit margin、self-source no-op 驗證）。
- **entity position-transfer interval**：entity contrast 的因果充分性從
  `entity_span` 轉移至 `instruction_context` span 的層級區間；對應 causal
  tracing 的 position-transfer interval 概念，但 contrast 是 entity 而
  非 valence。
- **attention-edge zeroing**：把指定 full-attention head 在指定層、從
  scoring sequence 最後一個 prompt token（含 decision prefix 的最後 token，
  即 answer token 的前一位置）對 entity token 位置的 attention weight 設 0
  （softmax 前加 −inf，其餘 key 重新正規化）的干預；以 head 輸出差額
  （zero 前後 head value 之差）加到 o_proj 輸入上施實，並要求重構的
  未干預 head value 與實際 o_proj 輸入在 bf16 容差內一致（fail-closed）；
  非 causal tracing 既有干預，屬本协议新增。
- **MLP margin attribution**：$|\partial M / \partial a_{l,k}| \cdot |a_{l,k}|$，
  即在 entity 位置對 MLP 輸出神經元 $k$（層 $l$）的梯度×激活一階歸因；
  非 standalone causal claim（與 baseline-trial attribution 相同界限）。

## 3. 假說

- **H1**：平衡證據下 pure entity margin 跨公司有系統分歧（IQR > 0.5 nats），
  且與 Phase 1 named margin 正相關（Spearman ρ > 0.5）。
- **H2**：entity contrast 的因果充分性早期集中在 `entity_span`，並在
  某層級區間（預期 L5–L20）轉移至 `instruction_context`。
- **H3**：handoff 區間內存在少量 attention head 與 MLP 神經元承載不成
  比例的 entity contrast（top 效應 > matched control）。
- **H4**（descriptive）：L15/n8490 dial 神經元在 probe 條件下的激活量與
  pure entity margin 相關，但對不同 entity 的 response 方向一致（model-level
  prior 解讀的預期）；若出現 entity 特異方向，則需重新解釋。

## 4. 實驗設計

### 4.1 公司與 prompt 族

- 公司：**沿用 Phase 1 的 16 家**（test split，4 sector × 4 家；
  名單在 [proposal](proposal-phase1.md) §3.1）。
- Prompt template：沿用 Phase 1（`scripts/balanced_evidence_gap.py` 的
  investment-dial 格式），evidence 段落換成 §4.2 的 shared-evidence
  template。
- 控制軸：2 個 reverse options（`"buy" or "sell"` / `"sell" or "buy"`）×
  2 個 evidence item order（原序 / 反序）= 每公司 4 條，共 **64 條
  prompt**。
- `pure entity margin` 為 4 條的中位數（控制 reverse 與 order 效應）。

### 4.2 Shared-evidence template（frozen）

以下 4 句即全部證據，逐字凍結，任何修改需新版本：

```text
- Q3 revenue grew 14% year over year, exceeding analyst consensus by 3%.
- Free cash flow reached a record quarterly high, up 22% year over year.
- Gross margin contracted 300 basis points due to rising input costs.
- Full-year revenue guidance was revised downward by 6%.
```

Order 變體：原序（上列 1-4）與反序（4-1）。

### 4.3 Experiment 2A：Cross-entity probe（行為分離）

- 執行 64 條 clean forward，量測 margin 與 decision。
- 輸出：per-company pure entity margin（4 條中位數）、跨公司 IQR/極差、
  與 Phase 1 named median margin 的 Spearman ρ。
- **Gate 2A**（全過才授權 2B）：
  1. pure entity margin IQR > 0.5 nats；
  2. |Spearman ρ|（vs Phase 1 named median）> 0.5；
  3. 64 條全部 schema-valid 且 self-consistent（reverse pair 的
     margin 差中位數 < 1.5 nats，確認 framing 效應未失控）。

### 4.4 Experiment 2B：Entity-state layer sweep（path 定位）

- **Pair 選取**（frozen 規則，依 2A 結果確定後凍結）：取 pure entity
  margin 最高的 2 家（top）與最低的 2 家（bottom），構成 4 個
  ordered pairs（top→bottom 雙向 = 8 個 transfer directions）。
- **Layers**：L0–L31 全層。
- **Spans**（每次 patch 一個 span）：
  - `entity_span`：`Stock Ticker: [X]` 與 `Stock Name: [Y]` 兩行的全部
    token；
  - `evidence_span`：`— Evidence —` 與 `—` 標記之間；source/target 證據
    文字相同，但殘差狀態可能已被 attention 寫入 entity 資訊，故 patching
    此 span 是**傳播診斷**（量測 entity 訊號已洩漏到證據位置的量），非
    no-op control；
  - `instruction_context`：證據後的 instruction token，不含
    final_position（與 causal tracing 同定義）；
  - `final_position`：最後一個 prompt token（control）。
- **Estimator**：
  - toward-source $\Delta M$：
    $\operatorname{sign}(M_{src}-M_{tgt}) \cdot (M_{patched} - M_{tgt})$；
  - normalized transfer
    $T = (M_{patched} - M_{tgt}) / (M_{src} - M_{tgt})$。
  8 個 direction 的等權平均，pair-bootstrap 95% CI。
- **自我驗證**：self-source patch 全部 $|\Delta M| = 0$（exact no-op）。
- **Discovery 輸出**：L0–L31 × 4 span 的 T 曲線；預期找出 entity
  position-transfer interval。本階段為 discovery，不設 formal gate；
  若 `entity_span` 在 L0 的 T 明顯 < 1（預期 ~1，因 contrast 只存在於
  header），須先診斷再授權 2C。

### 4.5 Experiment 2C：Component attribution（handoff 區間內）

只在 2B 確定的 handoff 區間（±1 層）內執行。注意 Qwen3.5-4B 是混合架構：
32 層中僅 8 層為 full attention（`L3/L7/L11/L15/L19/L23/L27/L31`，
`full_attention_interval=4`），其餘 24 層為 linear attention（無可索引的
attention weight）。因此 attention 干預只能作用於 full attention 層，MLP
歸因則可作用於所有層。

1. **Attention-edge zeroing**（僅 full attention 層）：對 handoff 區間（±1 層）
   內的每個 full attention 層 l、每個 head h，把 `instruction_context`
   位置對 `entity_span` token 的 attention weight 設 0，量測 8 directions 的
   toward-source ΔM。Position-matched control（與 entity cell E3 的
   random_subset 模式同義）：同 head 下改 zero 一個同樣 token 數的隨機
   非 entity 位置集合（每層 10 個 seed，固定 seed 42）；per-head 主效應 =
   zero(entity) − zero(random position)，以分离 entity token 的特異性與
   「任意 token 被 zero」的通用擾動。
2. **MLP margin attribution**（所有層）：在 entity 位置計算
   $|\partial M / \partial a_{l,k}| \cdot |a_{l,k}|$（對每個 MLP 輸出
   神經元 k，含 linear attention 層），8 directions 平均；matched control
   為同層隨機 10 個神經元。
- **Gate 2C**（formal）：
  1. attention 臂與 MLP 臂各自獨立判定：至少 1 個 head 或 10 個神經元的 top
     效應 mean > matched control mean，且 pair sign-flip test（Holm 調整）
     p < 0.05；
  2. top 效應的 sector-stratified 符號一致率 ≥ 3/4 sector。
- 若 handoff 區間內無 full attention 層，attention 臂標記 `not_applicable`
  （非失敗），只判定 MLP 臂。若 Gate 2C 失敗：記錄為 null result，不回填
  修改；component 層結論僅支持 descriptive。

### 4.6 H4 附加測量（descriptive，不 gate）

在 2A 的 64 條 clean forward 中同時讀取 L15/n8490 在 entity 位置與
final_position 的激活值，與 pure entity margin 做 Pearson 相關；並檢查
16 家公司的 activation-vs-margin 散點是否呈現統一方向。

## 5. 實現與 artifact

- **Package**：`llm_bias/balanced_evidence_gap/`（新；不得 import 其他
  experiment package，shared mechanics 走 `llm_bias/core/`）。
- **Operators**：
  - `scripts/balanced_evidence_gap_phase2.py`（2A + 2C 的 H4 測量；
    clean forward 族）
  - `scripts/balanced_evidence_gap_phase2_patch.py`（2B layer sweep；
    自實作 residual capture/patching，契約同 causal tracing）
- **Run root**：
  `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/<RUN_ID>/`
  （stages：prepare / forward / analyze，manifest SHA-256 凍結）
- **持久化限制**：只存 compact margin/ΔM/T/top-k 與 provenance；不存 raw
  activations。
- **Tests**：`tests/test_balanced_evidence_gap_phase2.py`（fake model、
  mocked forward、self-source no-op、gate 邏輯）。
- **`--smoke` preflight**：兩個 operator 皆需在正式 run 前通過。

## 6. 成本估算

| 實驗 | Forwards | 估算（4B bf16, 1×GPU） |
|---|---:|---|
| 2A clean | 64 | ~2 min |
| 2B sweep | 32 layers × 4 span × 8 dirs + 64 clean ≈ 1,088 | ~25 min |
| 2C attention（≤2 個 full attention 層 × 16 heads × 2 條件 × 8 dirs）＋ MLP attribution（≤5 層 × 16 公司 × 1 backward） | ~550 + ~80 | ~40 min |

合計 < 1.5 h（不含模型載入）。

## 7. 邊界與非目標

- 不 claim entity gap 是「偏誤」；normative 解讀屬於 RQ1 因素分析，
  本 Phase 只做 mechanistic 定位。
- shared-evidence template 是合成中性財報，非真實財報；結果外推至真實
  財報條件需另立版本。
- 16 家公司皆 S&P 500 成分；不涵蓋非成分股。
- attention-edge zeroing 是必要性的下界證據（去掉後效應消失），不證明
  該 edge 是唯一通路。
- 本 Phase 不重選 investment-dial coordinate，也不修改 L15/n8490 的既有
  結論。

## 8. Revision record

- **Rev 1（2026-09-09，首次正式 run 前）**：
  1. 發現 Qwen3.5-4B 為混合架構（8 層 full attention：L3/L7/L11/L15/L19/
     L23/L27/L31；24 層 linear attention）。2C attention 臂限定 full
     attention 層；MLP 臂涵蓋所有層；handoff 區間內無 full attention 層
     時 attention 臂標記 `not_applicable`。
  2. 2C attention 臂的 matched control 改為 position-matched control（同
     head 下 zero 同樣 token 數的隨機非 entity 位置，10 seed），與 entity
     cell E3 random_subset 模式同義；取代初稿的「同層隨機 10 head」。
  3. 2B 的 `evidence_span` 重新定位為傳播診斷（非 no-op control）；
     sanity control 僅為 self-source exact no-op。
  4. attention query position 明確定義為 scoring sequence（含 decision
     prefix）的最後一個 prompt token。
