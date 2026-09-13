# Balanced Evidence Gap: Entity-Induced Decision Gap 行為確認實驗

**狀態**：proposed  
**對象模型**：Qwen3.5-4B（bf16）  
**研究線定位**：Phase 1 行為確認。在多空完全對稱的財務證據下，測量不同公司名稱是否誘發可量測的決策分歧（Decision Gap）。確認 gap 存在後，Phase 2 另立研究線進行中間層 Path Patching。

---

## 1. 背景與動機

本研究線試圖回答的問題是：**entity identity 如何透過中間層神經元觸發決策旋鈕（investment dial），造成兩家公司在相同財務證據下出現不同的買賣決策？**

現有研究線已確認兩端：

- **上游**（Entity Cell，L0–L4）：壓制 JNJ、BAC 等公司的事實記憶神經元，導致事實失憶但**買賣決策翻轉 0 次**。事實記憶層與決策路徑功能解離（dissociated）。
- **下游**（Investment Dial，L15/n8490）：單神經元加性干預能以 ≤0.08 誤差精確遷移全域立場先驗，但該神經元是 **model-level prior**，對不同公司完全相同地施加平移，不認識特定 entity。
- **中游**（Causal Tracing，L14–L15 轉移區）：在明確財報條件下，決策充分性 97.56% 來自 `all_evidence` span，entity header 效應衰退至 ~0。
- **先前嘗試（J-space V2 Prior Probe）**：零證據條件下 entity-specific margin contrast 全部 ≤ 0.4% residual norm；模型強烈偏 sell 且對所有公司一致，entity identity 無可量測分歧。

三條線共同指向同一核心缺口：**only 在「多空對稱、誰都可能對或錯」的邊界條件下，entity prior 才有機會接管剩餘決策空間**。本實驗驗證這個條件。

---

## 2. 研究問題

**主問題（Phase 1 Behavioral Confirmation）**：  
在多空完全對稱（positive_count = 2，共 4 條財務證據）的提示詞下，以相同 4 條證據搭配不同公司名稱，是否能量測到：
1. **公司間 margin 分歧**（Entity-Induced Decision Gap）？
2. **公司名 vs. 匿名基線的系統性 margin 偏移**（Named vs. Anonymous Gap）？

若兩者皆成立且效應量可量測，Phase 2 再設計中間層 Path Patching 研究線。

---

## 3. 實驗設計

### 3.1 公司名單（16 家，來自 investment-dial test split）

全部來自 `data/baseline/investment-dial/exploratory-v1.json` 的 `test` split，未曾出現在 investment-dial A/B 校準實驗中。每家公司均備有 4 個 `evidence_pairs`（各含 positive / negative 兩面陳述）。

| Sector | Ticker | Company |
|---|---|---|
| Information Technology | AMAT | Applied Materials |
| Information Technology | GLW | Corning Inc. |
| Information Technology | HPE | Hewlett Packard Enterprise |
| Information Technology | IT | Gartner |
| Financials | AXP | American Express |
| Financials | BLK | BlackRock |
| Financials | C | Citigroup |
| Financials | GS | Goldman Sachs |
| Health Care | ABT | Abbott Laboratories |
| Health Care | BDX | Becton Dickinson |
| Health Care | DHR | Danaher Corporation |
| Health Care | SYK | Stryker Corporation |
| Industrials | CSX | CSX Corporation |
| Industrials | DE | Deere & Company |
| Industrials | HON | Honeywell Technologies |
| Industrials | NSC | Norfolk Southern |

### 3.2 Prompt 結構

沿用 `investment-dial` prompt 格式（`llm_bias/investment_dial/prompts.py`），固定以下參數：

- `positive_count = 2`：4 條證據中固定恰好 2 正 2 負，確保多空完全平衡
- `repeats = 4`：每家公司使用 4 組不同的隨機 evidence 抽樣（seed 固定）
- `reverse_options` 兩向皆測（`False` 和 `True`）

每家公司因此產生 **4 repeats × 2 reverse = 8 條 prompt**，16 家共 **128 條 named prompt**。

匿名化控制組（Anonymous）：對同一批 128 條 prompt，將公司名稱與 ticker 替換為 `[Company X]` 與 `[TICKER]`，產生 **128 條 anonymous prompt**。

共 **256 條 prompt**，前向推論一次即可完成所有測量。

### 3.3 測量指標

每條 prompt 的核心輸出：

- `margin = log P(buy) - log P(sell)`（FP32 tail logit scoring）
- `decision`：margin > 0 則 buy，否則 sell
- Named vs. Anonymous margin 差（`named_margin - anon_margin`）

**聚合層次**（analysis 階段計算）：

1. **per-company named margin**：8 條 named prompt 的中位數 margin
2. **cross-company margin spread**：16 家公司 named margin 的四分位距（IQR）與極差
3. **named vs. anonymous gap**：每個 (company, repeat, reverse) triplet 的 `named_margin - anon_margin`，再對稱平均（排除 reverse_options 效應）
4. **sector-level margin 分布**：4 個 sector 各自的 named margin 均值

### 3.4 驗證門檻（Calibration Gate，Phase 1）

Phase 1 不設 formal gate；以下為 success 的描述性判準，供 Phase 2 決策：

| 判準 | 說明 |
|---|---|
| Cross-company margin IQR > 0.5 nats | 公司間 margin 分歧可量測 |
| Named vs. Anonymous gap 中位數 95% CI 不含 0 | entity name 對 margin 有系統性效應 |
| ≥ 2 家公司 named decision 與 anonymous decision 不同 | 至少存在名稱誘發的決策翻轉 |

---

## 4. Operator 設計

**Script**：`scripts/balanced_evidence_gap.py`  
**Input**：`data/baseline/investment-dial/exploratory-v1.json`（read-only，現有資料）  
**Model**：`--model .cache/models/qwen3.5-4b`  
**Output root**：`artifacts/qwen3.5-4b/balanced-evidence-gap/runs/<RUN_ID>/`

Stage：
1. `prepare`：從 input data 生成 256 條 prompt，存 `prepare/prompts.jsonl`
2. `forward`：對全部 256 條跑前向推論，存 `forward/results.jsonl`
3. `analyze`：計算 per-company margin、named vs. anon gap、sector 分布，存 `analyze/summary.json`

`--smoke`：執行 2 條 prompt（named + anon 各 1）驗證 pipeline，不寫入正式 output。

---

## 5. 邊界與限制

- Phase 1 不做中間層分析；gap 存在只證明行為效應，不建立因果機制。
- 4 條 evidence 來自同一公司的 `evidence_pairs`，各公司的財務情境不同，故 named vs. anonymous gap 混合了 entity identity 效應與 evidence content 差異。若需純粹的 entity effect，Phase 2 應設計跨公司共享同一組 evidence 的 cross-entity probe。
- `test` split 公司未曾用於 investment-dial 校準，但仍來自同一宇宙（S&P 500 2020–2025），entity 資訊存在於模型訓練資料中。
- 本實驗不 claim sector-specific bias；sector 分組僅供探索性描述，不構成 formal sector comparison。
