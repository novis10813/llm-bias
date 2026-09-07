# Entity Cell Decision-Level Suppression Probe Report (proposed, non-frozen)

**Status:** 5 組 proposed probe runs 完成。主要結論：單一 entity cell 壓制**不產生
buy/sell 決策翻轉**；AMZN entity cell `(L0, N1476)` 在決策層只移除 entity prior
的一小部分（<5%），決策層 entity prior 是分散的。NVDA top-1 `(L2, N1786)` 無法
壓制 NVDA 的 entity 訊號（與其 gate fail 一致）。本報告非 frozen protocol。

**Owning experiment:** entity-cell localization。前序：
[HFM discovery report](report-hfm-discovery.md)（E1 V2 HFM localization +
factual amnesia probe）。Operator:
`scripts/entity_cell_factual_recall_preflight.py` 同系列之
`scripts/entity_cell_decision_flip_probe.py`（generic suppression/flip probe，
含可選 fact block）。

## Motivation

Factual amnesia probe（[report-hfm-discovery §4](report-hfm-discovery.md)）已確認
`(L0, N1476)` 壓制造成 Amazon HQ 事實回想的特異性、劑量依賴崩塌（-1.059 nats）。
本報告回答決策層的問題：**壓制 entity cell 會不會讓 buy/sell 決策 output 反轉？**
E1 V2 的 amnesia endpoint gate 量測的是 margin 朝 anonymous 基線的移動
（`anonymous_progress`），不是翻轉；本 probe 直接搜尋「clean 與 anonymous 方向
相反」的財務 prompt，測試壓制是否造成跨零翻轉。

## Instrument

- **Margin**：frozen amnesia-endpoint 定義，`logP(buy) − logP(sell)`，接在
  `DECISION_PREFIX`（`{\n  "decision": "`）之後，v2 FP32 final-norm tail 儀器
  （`score_single_token_margin_fp32`）。
- **Observed output**：greedy next token（display-only 診斷，另附 8-token greedy
  JSON 續寫）。
- **Dose**：frozen `ALPHA_GRID = (1.0, 0.0, -1.0, -2.0, -3.0)`，
  `all_positions` 與 `header_only` 兩種 scope。
- **對照**：wrong-entity cell、deterministic matched-random（由 discovery run 的
  generic baseline stats 選取）、跨 ticker 控制（同 cell 壓制其他 ticker）。
- **Device**：GPU 0 bfloat16。
- **財務 prompt**：header 與 JSON 指令結構與 `entity-cell-prepare-hfm-v3` 相同；
  本報告所有證據行皆為 placeholder（non-8-K），僅控制證據極性/強度。

## Runs on record

Outputs 在 `artifacts/qwen3.5-4b/entity-cell-localization/`：

| output | 內容 | 主要結果 |
|---|---|---|
| `decision_flip_probe_amzn_neutral.json` | 18 個中性 placeholder prompt（6 tickers × 3）方向掃描 + AMZN 翻轉測試 | 17/18 同向（全 sell）；唯一相反組合在 NVDA；AMZN 無翻轉 |
| `suppression_nvda_n1786.json` | NVDA top-1 `(L2, N1786)` fact block（HQ frame）+ NVDA 3 prompts 翻轉測試 + 跨 ticker | 事實無崩塌（-0.035）；相反 combo 不翻；無特異性 |
| `suppression_amzn_positive.json`（inputs: `inputs_amzn_positive.jsonl`, sha256 b2cccd78…） | 3 個強正面證據 prompt | clean 與 anon 皆 buy；無相反組合；N1476 微降 buy 偏置 |
| `suppression_amzn_ladder.json`（inputs: `inputs_amzn_ladder.jsonl`, sha256 39c68ede…） | 6 個溫和→中等正面證據 ladder | 邊界位於 L3（全 sell）與 L4（全 buy，anon +0.50）之間 |
| `suppression_amzn_band.json`（inputs: `inputs_amzn_band.jsonl`, sha256 4f57d17a…） | 6 個邊界帶 prompt | **找到相反組合**（`record_bndd0`）；翻轉測試：**不翻** |

## 1. 方向掃描：generic prior 是 sell，entity prior 是多頭偏移

18 個中性證據 prompt 的 anonymous 端全部為 sell（margin -2.3 ~ -4.3）——模型對
中性財務證據有強烈的 generic SELL prior。AMZN 的 entity prior 把 margin 往 buy
推 +1.1 ~ +3.3 nats（clean − anon gap），但不足以跨零。全套 18 個 prompt 唯一的
相反組合（clean buy vs anon sell）在 **NVDA**（+1.287 vs -2.334，gap -3.62，
全套最大 entity 偏移）。

## 2. 證據極性搜尋：buy/sell 邊界很陡，相反組合存在於窄帶

強正面證據下 anonymous 也變 buy（+4.5 ~ +5.6）——證據強度壓過 entity 差異。
6 級溫和→中等正面 ladder 顯示決策邊界極陡：

| Ladder | 證據 | clean AMZN | anon |
|---|---|---|---|
| L3 | revenue grew modestly / expectations met | -1.95 sell | -1.90 sell |
| L4 | revenue +15% / margins expanded modestly | +2.16 buy | +0.50 buy（剛過零） |

邊界帶細化（6 個 prompt）找到**真正的相反組合** `record_bndd0`（證據：
「slightly exceeded revenue expectations / margins stable」）：

| | margin | greedy |
|---|---|---|
| clean AMZN | **+1.176** | **buy** |
| clean anonymous | **-0.023** | **sell** |

同一證據、唯一差別是 header 的 AMZN 身份：模型對 Amazon 說 buy、對匿名公司說
sell。該 prompt 的 entity prior 幅度 = clean − anon ≈ **+1.20 nats**。

## 3. 翻轉測試：N1476 壓制不移除 entity prior（無翻轉）

對 `record_bndd0` 的 N1476 dose sweep（all_positions）：

| α | +1.0 | 0.0 | -1.0 | -2.0 | -3.0 |
|---|---|---|---|---|---|
| margin | +1.176 | +1.161 | +1.156 | +1.144 | **+1.125** |

α=-3 完整壓制只移除 **0.051 nats**（entity prior 的 ~4%）；token 全程 buy。
要翻轉需移除 1.176 nats。對照組（wrong-entity `(L2, N770)` +1.105、
matched-random `(L0, N5865)` +1.151）移動量與 target 同級——target 甚至不比
控制組更強。header-only scope 無方向性效應。**結論：這個 prompt 上驅動
「AMZN 特別偏 buy」的 entity prior 不在 N1476 裡。**

## 4. NVDA `(L2, N1786)`：壓制不了 entity

NVDA 是唯一現成相反組合的 ticker，但其 top-1 cell 在 discovery 已 fail
form-robust 與 amnesia endpoint。本 probe 直接驗證：

- **事實層**（HQ frame，gold ` Santa Clara,`）：clean -0.799；target α=-3
  -0.834（Δ-0.035）；name-only -0.831；wrong-cell（N1476）-0.778；
  matched-random -0.811。無崩塌、無特異性。
- **決策層**（NVDA 相反 combo，clean +1.287 buy）：target α=-3 → +1.163
  （buy，ap +0.034）；wrong-cell +1.327；matched-random +1.153——random 的
  移動比 target 還大。無翻轉、無特異性。

N1786 不載體 NVDA 的 entity 訊號（事實或決策層皆然），與其 gate fail 判定一致。

## 5. 判讀

1. **entity cell ≠ 決策驅動單元**。`(L0, N1476)` 是真實的 AMZN entity-specific
   representation（factual amnesia：HQ 事實 -1.059 nats 特異性崩塌；localization
   score 3386.7、form/template-robust），但 buy/sell 決策層的 entity prior
   （+1.2 nats @ boundary prompt）不由它承载；完整壓制只移除 ~4%。
2. **決策 entity prior 是分散的**。沒有單一 neuron 承載足以翻轉決策的 entity
   偏置；這是模型層面的結論（Qwen3.5-4B，單一 task family），不是定位方法的
   失敗——定位出的 cell 在其載體的內容（entity 事實）上表現出完整因果特異性。
3. **對 frozen gate 設計的印證**：E1 V2 amnesia endpoint 用
   `anonymous_progress`（margin 朝 anonymous 的移動比例）而非翻轉做門檻，與
   本結果一致——單單元壓制產生 graded margin shift，不產生離散決策翻轉。
4. **模型行為結構**：buy/sell 決策由證據強度主導（邊界 ~3 nats 寬），entity
   prior 是疊在證據之上的 +1.2 ~ +2.5 nats 偏移（幅度隨證據條件變異）。

## 6. 限制

- 證據行皆為 placeholder（non-8-K）；真實 8-K 證據下 entity prior 幅度可能不同。
- 單一 token margin（buy/sell）；未量 reason 欄位的 entity 依存。
- 相反組合只有 1 個（boundary 上、anon margin -0.023 貼零）；樣本量不足以
  對「entity prior 分散性」做統計推論，僅為描述性。
- Proposed probe，非 frozen protocol；未 pre-register。

## 7. Next steps（建議）

1. 本報告與 [report-hfm-discovery](report-hfm-discovery.md) 一起作為
   「entity cell = entity-specific factual representation（非決策驅動單元）」
   的 discovery 證據結案。
2. 若要決策層 claim，需多 cell 協同壓制（N1476 + 其他候選）或 population-level
   分析，超出單 cell 實驗範圍。
3. calibration / held-out test 仍未執行（同 HFM discovery）。
