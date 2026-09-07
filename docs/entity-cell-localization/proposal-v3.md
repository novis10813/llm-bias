# Entity Cell Localization: Proposal V3 (Fact-Level Amnesia Gate)

**Document status:** frozen。本文件依 V2
[proposal-v2 §4 版本分立觸發條件](proposal-v2.md) 第 4 條（變更四道門檻的
及格邏輯）另立新版；V2 維持 frozen 不動，既有 V2 run 的有效性不受影響。
凍結依 [§10](#10-凍結條件與驗證計畫) 完成：calibration（11 家）+ hold-out
（11 家新 entity，`entity-cell-e1-v3-holdout-v1`）皆通過驗收。V3 run 自此
成為 entity cell 確認的 formal 判定；版本分立觸發條件見
[§11](#11-版本分立觸發條件version-break-triggers)。

**Model for run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)

**Depends on（校準證據）:**
- E1 V2 discovery runs: `entity-cell-e1-hfm-discovery-v1`（AMZN 等 6 家）、
  `entity-cell-e1-hfm2-discovery-v1`（11 家）
- Fact-level amnesia probe outputs: `artifacts/qwen3.5-4b/entity-cell-localization/`
  下的 `factual_probe_{jnj,jpm,ko,orcl,uber}.json`、`fact-probe-batch2/`
  （13 個 targets）、`suppression_nvda_n1786.json`（proposed probe operator，
  見 [research scripts reference](../research-scripts.md)）

---

## 1. 動機：V2 Gate 4 的量測層被實證推翻

V2 的 Gate 4（`amnesia_endpoint_pass`）在 α = −3.0 下量測財務題 buy/sell
決策 margin 是否朝 anonymous 方向移動。後續兩組實驗顯示此量測層與 entity
cell 的因果功能正交：

1. **決策翻轉測試（0 翻轉）**：跨 4 個 entity（AMZN、NVDA、JNJ、JPM）、3 種
   證據極性（中性 / 強正面 / 邊界帶）、2 個真正的 clean-vs-anonymous 相反
   組合，壓制 entity cell 從未翻轉 buy/sell 決策；JNJ/JPM 的 margin 在壓制下
   微幅偏 buy（**遠離** anonymous，與 Gate 4 的方向假設相反）。見
   [decision probe report](report-decision-probe.md) 與
   [JNJ/JPM battery report](report-jnj-jpm-battery.md)。
2. **事實層有乾淨的因果效應**：同一批 cell 在事實 cloze 上有量級明確、
   控制組乾淨的崩塌（JNJ −8.51 nats、INTC −2.79、Visa −2.50、AMZN −1.06、
   JPM −0.43~−0.58），而決策層沒有。

同時，HFM-2 的 13 個 fact probe targets 發現第二種神經元類型：
**(L0, N4485) 一顆神經元承載五家不同公司（INTC、QCOM、Visa、WMT、XOM，
另有 KO 部分）的 HQ/ticker 參數記憶**，對 AMZN 無效應。V2 的 collision
caveat 只能標記「共享 top-1 需轉 E3」，無法對這種單元下分類結論。

V3 因此把 Gate 4 的量測層從決策層改為**事實層**，並把共享性檢查從
「collision 標記」升級為**明確的分類規則**。

## 2. 與 V2 的關係

| 項目 | V2 | V3 |
|---|---|---|
| Localization 句框（12 個 v2-frames） | 有 | **不變** |
| Gate 1（held_variant_overlap） | 有 | **不變** |
| Gate 2（form_robust） | 有 | **不變** |
| Gate 3（template_robust） | 有 | **不變** |
| Gate 4 | 決策 margin endpoint（財務題） | **改為 fact-level amnesia gate（§4）** |
| Fact frames（F0/F2/F3 cloze） | 無 | **新增**（§3） |
| Shared fact channel 分類 | collision caveat | **明確規則**（§5） |
| Pipeline stage | e1-baseline / e1-localization / e1-amnesia / analyze | 新增 `e1-fact-amnesia`（e1-amnesia 保留為 auxiliary 診斷，不再計入門檻） |

決策 margin（amnesia endpoint）保留為 **auxiliary 診斷輸出**，不再是門檻。

## 3. Fact frames 與 gold sequence

### Fact frames

三個固定 raw-text cloze 句框（無 chat template；公司名填入 surface form，
正式名稱，須通過 frozen `contained=True` span 規則）：

| frame | 模板 | 量測的事實 |
|---|---|---|
| F0 | `The headquarters of {name} is located in` | 總部所在地 |
| F2 | `The stock ticker of {name} is` | 股票代號 |
| F3 | `{name} was founded in` | 成立年份 |

### Gold sequence

Gold 序列 = 模型在 **clean（無干預）** 狀態下對該 frame greedy 生成的前
3 個 token。Gold 序列不是外部事實庫：它固定的是「模型自己乾淨時的回答」，
使崩塌量測的是壓制造成的改變，而非 prompt 的難易度。

**Gold validity（人驗）**：probe 執行時由操作者檢查 gold 解碼文本是否為
**正確的具體事實**（例如 Intel 的 F0 gold 必須是 HQ 城市）。通過的 frame
標記 `gold_verified: true` 並記入 output provenance。下列情形標記
`gold_verified: false` 並剔除該 frame：

- 內容錯誤（與公開事實不符）；
- 內容泛化（如 `the city of`、`the United States`）；
- 拒答或格式破損（如 `___`）。

ORCL 的 F0（gold `the city of`）即以此規則剔除。

## 4. V3 四道門檻

### Gate 1–3

繼承 V2 不變（`held_variant_overlap >= 1`、`form_robust == True`、
`template_robust == True`），定義與 fail-closed 行為見 V2 §3。

### Gate 4（V3）：fact-level amnesia gate

**設定。** 對**每家 ticker 的 top-5 localization candidates 逐一評估**（不以前三道
門檻為前置——門檻 1–3 是 entity-cell 分類條件，見 §5）：

1. 對 F0/F2/F3 三個 frame：先取 clean gold 序列（greedy 3 tokens）並人驗。
2. 在 **α = −3.0、all_positions** 下壓制候選神經元，量測 gold 序列的
   teacher-forced 聯合 log-probability（nats）。
   `collapse(frame) = logp(suppressed) − logp(clean)`（負值 = 忘記）。
3. 同一組 frame 上跑兩個控制組（終端劑量 α = −3.0）：
   - **wrong-entity control**：繼承 V2 §3.2 的確定性規則——字母序下一家
     ticker（wrap-around）；取其候選清單依 rank 次序第一個與候選神經元不同的
     神經元；若五顆皆相同，標記 `degraded_control: true` 且該 frame 只比對
     matched-random。共享 top-1（如 N4485）會頻繁觸發此遞移，屬預期行為。
   - **matched-random control**：同層、激活統計配對的確定性隨機神經元
     （`select_matched_random_neuron`，由 generic baseline stats 產生）。

**通過條件（三條全要，fail-closed）：**

1. **Effect**：至少一個 `gold_verified: true` 的 frame 之
   `collapse <= −0.5` nats。
2. **Specificity**：該 frame 上 wrong-entity 與 matched-random 的
   `|collapse| <= 0.3` nats。
3. **Gold validity**：該 frame 的 gold 通過人驗（`gold_verified: true`）。

若所有 frame 皆 `gold_verified: false`，該候選標記
`fact_gate_not_applicable`（不等於 fail，但不計入 eligible——無法驗證的事實
記憶不構成 entity cell 證據）。

**Ticker 的 V3 分類**依 §5；`eligible: true` 僅在分類為 `entity_cell` 時。

## 5. Shared fact channel 分類與 ticker 分類

**定義。** **Shared fact channel**：對 batch 內 ≥ 2 家不同公司的 fact frame
皆產生 Gate 4 級崩塌（`collapse <= −0.5`）的單一神經元——承載多實體共享的
參數記憶，**不是** entity-specific cell。

**分類程序**（對每個 fact gate 通過的候選執行；cross 檢查在 stage 內對全部
top-5 候選無條件執行，分類時才使用）：

1. 對 batch 內每家**其他** ticker 的 F0 frame：取 clean gold（同 §3 人驗
   規則）並在該候選 α = −3.0 下量測 collapse。
2. 任一其他 ticker 的 F0 `collapse <= −0.5` 且其 gold 通過人驗 → 該候選為
   shared fact channel 成員；否則為 entity-specific。

**Ticker 級分類（fail-closed 優先序）：**

| 分類 | 條件 |
|---|---|
| `shared_fact_channel` | 任一候選 fact gate 通過且為 shared（無論 Gates 1–3） |
| `entity_cell` | 有 fact gate 通過候選、無 shared、且 Gates 1–3 全過；trusted cell = 通過者中 localization rank 最高者 |
| `fact_carrier_unrobust` | 有 fact gate 通過候選、非 shared、Gates 1–3 未過——真實事實載體，但定位穩健性未建立 |
| `fact_gate_not_applicable` | 無任何 gold 通過人驗 |
| `not_eligible` | 有人驗 gold 但無候選通過 fact gate |
| `fact_gate_unavailable` | 無 fact 量測（如 surface form span 失敗） |

**校準觀察**：(L0, N4485) 對 INTC/QCOM/Visa/WMT/XOM 的 F0/F2 皆
`<= −0.5`，對 AMZN（自有 cell `(L0, N1476)`）≈ 0——分類規則可把 N4485
歸為 shared fact channel，把 AMZN/JNJ/JPM 的自有 cell 歸為 entity cell
（JPM 因 Gate 1 未過而為 fact_carrier_unrobust）。
JNJ 的次候選 (L4, N4047) 對 JNJ F0 −0.68、對 AMZN ≈ 0 → entity cell
（次通道）。

## 6. Input / Output 契約

### Input 契約（prepare 新增）

- `fact_frames.jsonl`：每家 ticker 的 F0/F2/F3（正式名稱 surface form）
  與 cross 用其他 ticker 的 F0。tokenization span 必須通過 frozen
  `contained=True` 規則（正式名稱；prepare 階段預檢）。

### Output 契約

- `e1/fact_amnesia.jsonl`：每筆 (ticker, candidate, frame)：gold tokens/text、
  `gold_verified`、clean logp、suppressed logp、`collapse`、兩控制組 collapse、
  `degraded_control`；cross 檢查每筆 (ticker, candidate, other_ticker) 的
  F0 collapse。
- `analyze/summary.json`：新增 `v3_candidate_eligibility`（Gate 1–4 逐項、
  `fact_gate_not_applicable` 標記、`classification`）。
- 禁令與數值約束同 V2（不存原始激活；finite float）。

## 7. CLI 契約（版本化變更）

```bash
entity-cell prepare \
  --input <input.csv> --split-manifest <manifest> \
  --baseline <baseline> --baseline-identity <identity> \
  --model .cache/models/qwen3.5-4b \
  --run-id <run_id> --sector <sector-label> \
  --localization-family v2-frames          # 新增：自動生成 fact_frames.jsonl

entity-cell run-localization \
  --prepared-dir <prepared_dir> \
  --model .cache/models/qwen3.5-4b \
  --run-id <run_id> \
  --artifact-root artifacts \
  --localization-family v2-frames \
  --stages e1-baseline e1-localization e1-fact-amnesia analyze
```

V2 的 `e1-amnesia` stage 保留（auxiliary 診斷輸出），V3 run 不要求執行。

## 8. 邊界情況與防禦性行為

1. **Gold 不通過人驗**：frame 剔除並記錄（§3）；全 frame 剔除 →
   `fact_gate_not_applicable`。
2. **Surface form tokenization 失敗**（`contained=True` 不過）：該 ticker
   標記 `fact_gate_unavailable`，不計入 eligible；prepare 階段應以正式名稱
   預檢（HFM-2 已驗證 11/11 通過）。**已知陷阱（hold-out 發現）**：單字正式
   名稱（Pfizer、Chevron、Boeing、Nike）的首詞 BPE token 含前導空白，落在
   name 字元範圍之外，`contained=True` 必然 fail-closed；對策是使用多詞正式
   形式（Pfizer Inc.、Chevron Corp.、Boeing Co.、Nike Inc.），與 HFM-2 的
   命名慣例一致。
3. **Wrong-entity 退化**：繼承 V2 §3.2（字母序遞移；全同則
   `degraded_control` 只比對 matched-random）。N4485 這類共享 top-1 會
   **頻繁**觸發遞移，屬預期行為。
4. **候選與控制撞車**：matched-random 由確定性配對排除候選自身；wrong-entity
   遞移排除同神經元。
5. **Fail-closed 預設**：任何缺失輸入、非有限數值、未記錄的 gold 驗證 →
   該候選 not eligible。

## 9. 校準數據與閾值依據

14 家 entity 的 fact probe population（proposed probe operator 產出；
正式 run 須依 §10 重現）：

> **候選來源註記。** 本表中 INTC/QCOM/Visa/WMT/XOM 的候選取自
> `header_family_candidates`（V1 header 家族的相容欄位）的 top-1，即 N4485
> 是 V1 header 家族神經元；這五家的 **V2 frame localization** top-5 不含
> N4485（見 §10 calibration 結果）。本表量測的是被測 cell 的因果效應
>（作為 shared channel 證據有效），不是 V2 對這些 ticker 的提名候選。

| entity（候選神經元） | F0 | F2 | F3 | 控制組 max | 判定 |
|---|---|---|---|---|---|
| JNJ (L4, N7676) | −8.51 | −0.74 | −4.49 | ≤0.05 | 通過（entity cell） |
| KO (L0, N4485) | −0.12 | −2.86 | −0.62 | ≤0.1 | 通過（shared channel 成員） |
| INTC (L0, N4485) | −1.07 | −2.79 | −0.06 | 0.14 | 通過（shared channel 成員） |
| Visa (L0, N4485) | −1.09 | −2.50 | −0.71 | 0.10 | 通過（shared channel 成員） |
| AMZN (L0, N1476) | −1.06 | ≈0 | ≈0 | ≤0.05 | 通過（entity cell） |
| WMT (L0, N4485) | −0.34 | −0.87 | −0.17 | 0.12 | 通過（shared channel 成員） |
| QCOM (L0, N4485) | −0.73 | −0.54 | −0.11 | 0.07 | 通過（shared channel 成員） |
| XOM (L0, N4485) | −0.55 | −0.25 | −0.34 | 0.06 | 通過（shared channel 成員） |
| JPM (L0, N9025) | −0.58 | −0.53 | −0.43 | ≤0.05 | 通過（entity cell） |
| PLTR (L1, N6529) | −0.15 | −0.31 | +0.01 | 0.11 | 否決（灰區，max < 0.5） |
| NVDA (L2, N1786) | −0.04 | ≈0 | ≈0 | ≤0.05 | 否決 |
| UBER (L0, N8900) | ≈0 | ≈0 | ≈0 | ≤0.05 | 否決 |
| ORCL (L0, N4485) | 剔除（gold 泛化） | 控制組同量級 | — | — | 否決（specificity + gold） |
| FTNT | preflight NO-GO（clean 無參數事實） | — | — | — | not applicable |

**閾值選擇：**

- Effect floor **0.5 nats**：最弱通過者 XOM −0.55 與灰區 PLTR max −0.31
  之間的最大間隔中點偏保守側；0.3 會讓 PLTR 灰區漏入。
- Specificity ceiling **0.3 nats**：population 控制組實測上限 0.14
  （INTC wrong-entity），2× 裕度。
- 分類 threshold 同 effect floor（0.5）：N4485 的跨實體 F0 崩塌
  （0.34~1.09，其中 ≥0.5 者 4 家）與 entity-specific 細胞的跨實體
  崩塌（≤0.08）分離清晰。
- 崩塌 profile 觀察（非門檻）：N4485 型通道 F2 > F0 > F3（founded-year
  相對 spared）；entity cell 型 JNJ 為 F0 > F3 > F2。

## 10. 凍結條件與驗證計畫

凍結前須全部完成（依序）：

1. **實作** `e1-fact-amnesia` stage、`verify-fact-gold` 子命令與
   `v3_candidate_eligibility`（`llm_bias/entity_cell/` 內，不改 V2 既有
   stage 行為；已實作，regression tests 見 `tests/test_entity_cell_v3_fact_gate.py`）。
2. **Calibration 正式 run**（已完成：`entity-cell-prepare-hfm2-v2` +
   `entity-cell-e1-v3-hfm2-calibration-v1`，cells.jsonl 与原 hfm2 discovery
   byte-identical；gold 人驗 33 項，2 項 F3 因模型年份錯誤被剔除）：
   - `entity_cell`：**JNJ** trusted (L4, N7676) rank 1；**PLTR** trusted
     (L2, N5003) rank 3（calibration 意外發現：ad-hoc probe 測的
     (L0, N8900)/(L1, N6529) 未通過，V2 提名的 (L2, N5003) 通過）；
   - `fact_carrier_unrobust`：JPM（top-1 (L0, N9025) 自特異但 Gate 1 未過）、
     V、XOM；
   - `shared_fact_channel`：KO（(L0, N4485) 成員 INTC/QCOM/V/XOM；另
     (L0, N3) 成員 9 家）、ORCL（(L0, N4485)、(L2, N8500)）、WMT（(L0, N104)）；
   - `not_eligible`：INTC、QCOM、UBER（其 V2 top-5 候選未通過 fact gate；
     INTC/QCOM 的事實記憶由 header-family 的 N4485 承載，未被 V2 提名）。
   - Shared channel 網絡（正式量測）：(L0, N3) 9 家、(L0, N4485) 4 家、
     (L4, N4047) JNJ↔JPM、(L3, N7403) INTC↔QCOM、(L0, N104) WMT↔V、
     (L2, N8500) INTC↔XOM。
3. **Hold-out validation**（已完成：`entity-cell-prepare-holdout-v1` +
   `entity-cell-e1-v3-holdout-v1`；input 已持久化至
   `data/entity-cell/holdout-v1-input.csv` + `holdout-v1-split-manifest.json`）：
   11 家 calibration 未見的新 entity（LLY/PFE/UNH 醫療、BAC/WFC 金融、
   CVX/COP 能源、BA/CAT 工業、PG/NKE 消費）。驗收結果：
   - `entity_cell`：**BAC** trusted (L0, N7801) rank 1（F0 −7.77 / F2 −7.84
     nats、控制 ≤0.1、cross max 0.14 非共享、F3 年份保留）；**CAT** trusted
     (L2, N7997) rank 1（F2 −3.49；單 frame 通過）；
   - `fact_carrier_unrobust`：PG（(L0, N4) F2）；
   - `shared_fact_channel`：LLY（(L3, N5456) ↔ BAC）、WFC（(L2, N5322) ↔ BAC）；
     另 (L4, N4184)（calibration 已知通道 JPM↔V）在 BAC 亦通過 fact gate
     （跨 batch 通道延續，descriptive）；
   - `not_eligible`：BA、CVX、NKE、PFE；`fact_gate_not_applicable`：COP、UNH
     （三框 gold 皆未過人驗）。
   - **驗收判定**：控制組分佈與 §9 一致（matched-random max 0.142→0.153、
     wrong-entity 0.101→0.116，皆遠低於 0.3 上限）；新 shared channel 皆為
     2 成員對且非控制組型假陽性。**gold 人驗通過率 18/33（55%）**，低於
     calibration 的 30/33，原因為模型行為（6 家 F0 以測驗格式拒答、5 家 F2
     續出錯誤 ticker），protocol 正確 fail-closed（zero-gold →
     `fact_gate_not_applicable`，無偽通過）；記錄為 entity 依賴性質，不構成
     protocol 缺陷。

## 11. 版本分立觸發條件（Version Break Triggers）

以下任一變更必須另立新版（`proposal-v4.md`），禁止原地修改本文件：

1. 修改 F0/F2/F3 句框文本或 gold 長度（3 tokens）；
2. 變更 Gate 4 任一閾值（0.5 / 0.3）或三條通過條件的邏輯；
3. 變更 shared fact channel 的分類規則、cross 檢查範圍，或 §5 的 ticker
   級分類狀態/優先序；
4. 變更 gold validity 的人驗規則；
5. 把 e1-amnesia（決策 margin）重新納入門檻。
