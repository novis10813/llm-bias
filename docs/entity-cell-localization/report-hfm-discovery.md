# Entity Cell Localization: High-Factual-Memory Discovery Report (E1 V2 HFM)

**Status:** HFM discovery complete. 1/6 trusted candidate entity cell
(AMZN, `(L0, N1476)`). Factual amnesia probe (proposed, non-frozen) confirms a
specific, dose-dependent collapse of Amazon's HQ fact recall under target
suppression. Calibration and held-out test not run.

**Protocol:** [proposal-v2](proposal-v2.md)（E1 V2 frozen protocol，同一套
four gates 與 frame family）。Version index: [README](README.md)。

**Motivation.** E1 V2 原研究（35 Technology tickers）的唯一 trusted candidate
FTNT `(L0, N104)` 在 v2 儀器下通過了全部四道門檻，但 factual recall preflight
（見 [§6](#6-factual-recall-preflight-ftntadi-muftv)）顯示 Qwen3.5-4B 對
Fortinet 的參數化事實記憶極弱（HQ 拒答、ticker 答錯、founded year 錯）——
即原論文（Barzilay et al. / ROME 一系）驗證閉環的步驟 3（壓制後回到事實句子
度量遺忘）**無法對 FTNT 驗證**。本 discovery 改以「模型確實記得事實」的高頻
entity 重跑同一 frozen E1 V2 protocol，使步驟 3 可驗證。

## Runs on record

All under `artifacts/qwen3.5-4b/entity-cell-localization/` unless noted:

| run / output | state | note |
|---|---|---|
| `factual_scan_broad.json` | complete | 10 家高頻公司 × 4 factual-cloze frames 的 clean 事實記憶掃描（inline operator，raw-text framing） |
| `factual_recall_preflight.json` | complete | FTNT/ADI/MU/FTV 原 4 家 × 4 frames + Anonymous 對照（`entity_cell_factual_recall_preflight.py`）：NO-GO，FTNT 事實記憶極弱 |
| `runs/entity-cell-prepare-hfm-v2` | superseded | formal-name prepared inputs；model identity 與 CLI 路徑不一致，provenance 檢查失敗 |
| `runs/entity-cell-prepare-hfm-v3` | complete | 同上 inputs，model identity = `.cache/models/qwen3.5-4b`（與 CLI 一致） |
| `runs/entity-cell-e1-hfm-discovery-v1` | **complete** | E1 V2 全四階段（6 tickers；GPU 0 bf16，8m54s） |
| `factual_amnesia_probe_amzn.json` | complete | Factual amnesia probe（`entity_cell_factual_amnesia_probe.py`，proposed）：`(L0, N1476)` dose curve + controls + cross-ticker |

**Provenance notes.**

1. HFM prepare 的 financial prompts 為**中立的 placeholder 證據**（non-8-K；
   canonical 兩行 header + 中性 evidence + frozen JSON 指令結構），只因
   localization 階段不消費它們。這影響 amnesia endpoint gate（財務決策
   margin）數值的解讀，不影響 frame localization 與 factual amnesia probe。
2. 公司名 surface form 必須通過 frozen `_frame_name_span` 的 contained-span
   規則：單字名（`Nvidia`、`Apple`…）在 Qwen3.5 tokenizer 下尾部空格併入
   名稱 token，違反規則；改採 formal name（`Nvidia Corp.`、`Apple Inc.`、
   `Microsoft Corp.`、`Alphabet Inc.`、`Amazon.com`、`Tesla Inc.`），全部
   12 個 frozen frame 通過。正式名稱的事實記憶保留狀況見 [§3](#3-surface-form-與事實記憶複核)。
3. Broad scan 與 formal-name 複核為 inline ad-hoc runs（`uv run python -c`，
   非註冊 operator）；數值記錄於本報告與 `factual_scan_broad.json`。

## 1. 高事實記憶 entity 篩選（broad scan）

10 家高頻公司 × 4 個 factual-cloze frames（F0 HQ / F2 ticker / F3 founded /
F8 CEO），clean 模型、raw-text framing（無 chat template；chat template 下
模型 echo prompt，不可用於 factual readout——詳見
[README「Proposed: factual recall preflight」](README.md)）。

| Ticker | HQ | Ticker | Founded | CEO | 分數 |
|---|---|---|---|---|---|
| TSLA (Tesla) | Austin ✓ | TSLA ✓ | 2003 ✓ | **Elon Musk ✓** | **4/4** |
| NVDA (Nvidia) | **Santa Clara ✓** (top1 -0.42) | NVDA ✓ | 1993 ✓ | a billionaire（泛化） | 3.5/4 |
| WMT (Walmart) | **Bentonville ✓** (-0.19) | WMT ✓ | 1962 ✓ | 泛化 | 3/4 |
| JPM (JPMorgan) | New York ✓ | JPM ✓ | 1799 ✓ | 泛化 | 3/4 |
| AAPL (Apple) | Cupertino ✓ | AAPL ✓ | 1976 ✓ | 泛化 | 3/4 |
| MSFT (Microsoft) | Redmond ✓ | MSFT ✓ | 1975 ✓ | 泛化 | 3/4 |
| GOOGL (Google) | Mountain View ✓ | GOOG ✓ | 1998 ✓ | 泛化 | 3/4 |
| AMZN (Amazon) | Seattle ✓ (-1.30) | AMZN ✓ | 1994 ✓ | 泛化 | 3/4 |
| META (Meta) | Menlo Park ✓ | 拒答 | 2004 ✓ | 泛化 | 2/4 |
| BRK (Berkshire) | 拒答 | BRK.A ✓ | 1839 ✗ | **Warren Buffett ✓** | 2/4 |

選入 discovery 的 6 家：3+ 分且屬 Technology sector（實驗 sector 框架）——
**NVDA, AAPL, MSFT, GOOGL, AMZN, TSLA**（WMT 零售、JPM 金融，分數同為 3 但
非 Technology sector，未選入）。

## 2. E1 V2 HFM discovery（`entity-cell-e1-hfm-discovery-v1`）

同一 frozen E1 V2 protocol（12 frames、template control、399 題 generic
baseline、four gates）。Input：6 tickers × formal names
（`entity-cell-prepare-hfm-v3`）。

### 結果：1/6 trusted candidate（AMZN）

| Ticker | Top-1 cell | Stability score | 四道門檻 |
|---|---|---|---|
| **AMZN** | **(L0, N1476)** | **3386.7** | **全部通過（trusted candidate entity cell）** |
| NVDA | (L2, N1786) | 650.1 | fail：form-robust、amnesia endpoint（2/3 → gate 需 ≥2 且… 見下）、held overlap 0 |
| TSLA | (L3, N0) | 563.8 | fail：form-robust、amnesia endpoint |
| AAPL | (L2, N770) | 538.0 | fail：form-robust、amnesia endpoint |
| GOOGL | (L0, N104) | 449.7 | fail：form-robust、amnesia endpoint |
| MSFT | (L2, N770) | 428.6 | fail：form-robust、amnesia endpoint、held overlap 0 |

AMZN 詳細：

- **Top-5**：`(L0, N1476)` 3386.7、`(L0, N6219)` 1373.5、`(L1, N4927)` 1185.0、
  `(L1, N122)` 900.7、`(L0, N2372)` 894.0。Top-1 與 top-2 分數比 > 2.4×。
- **Held-variant**：top-5 overlap 1（N1476 於 held 句框 rank 3）。
- **Form-robust**：anonymous / name-form surface control top-5 overlap 皆為 0。
- **Template-robust**：不在 template signature。
- **Amnesia endpoint**（financial decision margin，placeholder 證據）：
  2/3 prompts 通過，均值 $A_p(+1.0 \to -3.0) = +0.0656$；wrong-entity 對照
  均值 ≈ -0.161、matched-random ≈ -0.037（皆負）。

**結構觀察（與原 35 家研究一致）**：`(L0, N104)`（原 FTNT cell）在本 6 家
集合中再度出現為 GOOGL 的 top-1 與 AAPL/MSFT/NVDA 的 top-5——共享槽位結構
重現；`(L2, N770)` 為 AAPL/MSFT 共享 top-1。但 `(L0, N1476)` 為 AMZN
**專屬** top-1（不在其他 5 家 top-5），且 stability score 3386.7 為原研究
唯一 trusted cell（FTNT 363.4）的約 9.3×。

## 3. Surface form 與事實記憶複核

Formal name（localization 使用的 surface form）下，clean 模型的事實記憶
（raw-text，greedy 前 3 token）：

| Ticker | Name（frozen surface form） | HQ | Ticker | Founded |
|---|---|---|---|---|
| NVDA | Nvidia Corp. | Santa Clara, California ✓ | 975（Nasdaq 代號，非 ticker） | 199…（1993 前綴）✓ |
| AAPL | Apple Inc. | Cupertino, California ✓ | 弱（005…） | 197…（1976 前綴）✓ |
| MSFT | Microsoft Corp. | Redmond, Washington ✓ | 弱（100…） | 197…（1975 前綴）✓ |
| GOOGL | Alphabet Inc. | Mountain View, California ✓ | **GOOG** ✓ | 201…（2015，Alphabet 成立年）✓ |
| AMZN | Amazon.com | **Seattle, Washington ✓** | 弱（10…） | 199…（1994 前綴）✓ |
| TSLA | Tesla Inc. | Austin, Texas ✓ | **TSLA** ✓ | 200…（2003 前綴）✓ |

HQ 與 founded-year 記憶在 formal name 下完整保留（localization 訊號的基礎）；
部分 ticker 記憶在 formal name 下變弱（不影響本報告結論，factual probe 的
gold 序列以 clean greedy 為準）。

## 4. Factual amnesia probe（proposed，非 frozen protocol）

**設計**（`scripts/entity_cell_factual_amnesia_probe.py`）：

- **標靶**：AMZN `(L0, N1476)`（frozen 自 `entity-cell-e1-hfm-discovery-v1`）。
- **Frames**（raw-text factual cloze，`Amazon.com` surface form）：
  F0 HQ、F2 ticker、F3 founded。
- **Gold 序列**：clean 模型 greedy 前 3 token（frozen，人類側驗證為正確事實）：
  F0 = ` Seattle, Washington`、F2 = ` 10`（模型 ticker 回想弱，見下）、
  F3 = ` 19`（1994 前綴）。
- **量測**：gold 序列的 joint log-probability（teacher forcing）＋ top-1
  診斷。條件：clean；target dose ∈ {+1.0, +0.5, 0.0, -1.0, -2.0, -3.0}
  （all positions）；target α=-3（僅名稱 token 位置）；wrong-entity 對照
  `(L2, N770)`（AAPL/MSFT top-1）α=-3；matched-random `(L0, N5865)`（由
  run 的 generic baseline stats 決定性選取）α=-3；跨 ticker 對照（同一
  N1476 壓制於 NVDA/AAPL/TSLA 的 HQ frame）。
- **Device**：GPU 0 bf16。

### 結果

**F0（HQ「Seattle, Washington」）— 特異性成立：**

| 條件（α=-3 除註記） | gold logp | Δ vs clean |
|---|---|---|
| clean | -1.274 | — |
| **target N1476，all positions** | **-2.333** | **-1.059 nats（機率 0.28 → 0.10）** |
| target N1476，僅名稱位置 | -2.413 | -1.139 nats |
| wrong-entity (L2, N770) | -1.288 | -0.014 |
| matched-random (L0, N5865) | -1.282 | -0.008 |
| 跨 ticker：NVDA / AAPL / TSLA HQ | — | +0.021 / +0.008 / +0.013 |

**劑量反應（F0 target，單調）**：

| α | +1.0 | +0.5 | 0.0 | -1.0 | -2.0 | -3.0 |
|---|---|---|---|---|---|---|
| gold logp | -1.274 | -1.303 | -1.448 | -1.704 | -1.891 | -2.333 |

**F2（ticker）／F3（founded year）**：clean 端 gold logp -5.833 / -0.494；
α=-3 時 Δ = +0.029 / +0.070（無崩塌）。F2 的 clean greedy 為 ` 10`
（模型對 `Amazon.com` 的 ticker 回想本來就弱且非 AMZN），該 frame 判讀為
inconclusive；F3（1994）為乾淨的負結果。

### 判讀

1. **閉環成立**：(L0, N1476) 是 AMZN 專屬 entity cell——壓制它**選擇性、
   劑量依賴地**破壞 Amazon 的 HQ 事實回想（-1.06 nats @ α=-3），wrong-cell、
   matched-random 與跨 ticker 對照全部平穩。這是原 FTNT 線因模型缺乏 FTNT
   事實記憶而無法完成的驗證步驟。
2. **效應形態**：confidence collapse（α=-3 下 top-1 仍為 ` Seattle`，但
   logp -1.001 → -1.866），非答案翻轉。
3. **事實選擇性**：HQ 有效應、founded year 無效應 → 單一 cell 非 Amazon
   全部事實的儲存地；事實記憶分散，N1476 為其一個載體。
4. **名稱位置敏感**：僅壓制名稱 token 位置（-1.139）≥ all-positions
   （-1.059），與「cell 由實體名稱觸發」的預期機制一致。

## 5. 與原 FTNT 線的對照

| | FTNT `(L0, N104)`（原 35 家） | AMZN `(L0, N1476)`（HFM 6 家） |
|---|---|---|
| Stability score | 363.4 | 3386.7（≈9.3×） |
| 模型對該實體的事實記憶 | 極弱（preflight NO-GO） | 強（HQ / founded year 乾淨） |
| 壓制 → 事實回想崩塌 | 無法驗證 | **-1.059 nats（劑量依賴、特異）** |
| 壓制 → 財務決策 margin | 有（$A_p=+0.097$，v2 儀器） | 有（amnesia endpoint 2/3，placeholder 證據） |
| 跨 ticker 撞車 | 17/35 共享 top-1 | 6 家內 AMZN 專屬 top-1 |

## 6. Factual recall preflight（FTNT/ADI/MU/FTV）

`scripts/entity_cell_factual_recall_preflight.py` 對原 E3 四家（simple name，
raw-text）的 clean 事實記憶結果：**NO-GO**。

- FTNT：HQ 拒答（`___? A.`）、ticker 錯（FORT）、founded 錯（2000 vs 1999）、
  CEO 拒答。
- ADI：founded 1965 ✓，其餘拒答/泛化。
- MU：HQ **Boise, Idaho ✓**（唯一乾淨的 entity-specific 事實）、founded 錯
  （1968 vs 1978）。
- FTV：全數拒答/泛化。
- Anonymous 對照不產生任何公司事實（確認非 generic prior）。

此結果是改以高記憶 entity 重做 discovery 的直接依據。

## 7. Status 與 next steps

1. **Discovery 結案**：HFM 6 家的 E1 V2 localization 與 factual amnesia probe
   完成；AMZN `(L0, N1476)` 為本線第一個完成「定位 → 事實記憶存在 → 壓制 →
   事實崩塌 → 特異性控制」完整閉環的 entity cell。
2. **Factual amnesia probe 為 proposed**：尚未凍結為 protocol（collapse target
   定義、gate 閾值、樣本量皆未 pre-register）。任何正式 claim 需先立版本文件。
3. **決策層延伸**（已完成，proposed）：decision-level suppression/flip probe
   顯示壓制 `(L0, N1476)` 不產生 buy/sell 決策翻轉（邊界 prompt 上只移除
   entity prior 的 ~4%），決策層 entity prior 分散；見
   [decision probe 報告](report-decision-probe.md)。
4. **未做**：calibration / held-out test；更多事實框（product / sector /
   competitor）的 cell-事實映射；其他 5 家的 probe；HFM financial prompts
   換成真實 8-K 證據後重跑 amnesia endpoint gate。
