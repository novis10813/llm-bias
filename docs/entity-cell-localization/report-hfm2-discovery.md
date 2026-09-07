# Entity Cell Localization: Batch 2 (Broad) Discovery Report (E1 V2 HFM-2)

**Status:** Batch 2 discovery complete — 20 家新候選（含跨 sector）× formal-name
surface forms。E1 V2 四門檻 **0/11 trusted**（9/11 被 amnesia endpoint gate
擋下）。Factual amnesia probe 額外確認 **2 個 entity cell**：JNJ `(L4, N7676)`
（全研究線最強，HQ 事實崩塌 -8.51 nats、完整壓制下模型拒答 `___?`）與 JPM
`(L0, N9025)`（三框皆特異、中等效應）；KO `(L0, N4485)` 部分特異（ticker /
founded 框）。共享槽位結構跨 sector 重現。Calibration / held-out test 未執行。

**Protocol:** [proposal-v2](proposal-v2.md)（同一套 frozen four gates 與
frame family）。前序：[HFM discovery report](report-hfm-discovery.md)（6 家
Technology）、[decision-level probe report](report-decision-probe.md)
（決策層 entity prior 分散的結論）。Version index: [README](README.md)。

**Motivation.** 把 HFM 線（模型確實記得事實的 entity）從 6 家 Technology 擴充
到更多高頻公司（含跨 sector），觀察 entity cell 的普遍性與共享槽位結構。

## Runs on record

`artifacts/qwen3.5-4b/entity-cell-localization/` 下：

| run / output | state | note |
|---|---|---|
| `factual_screen_batch2.json` | complete | 20 tickers × 31 surface variants × 4 factual-cloze frames 的 clean 事實記憶掃描（`entity_cell_factual_recall_preflight.py --entities`） |
| `runs/entity-cell-prepare-hfm2-v1` | complete | 11 家 prepared inputs（scope label `HighFactualMemory`；neutral placeholder 證據；`--localization-family v2-frames`） |
| `runs/entity-cell-e1-hfm2-discovery-v1` | **complete** | E1 V2 全四階段（11 tickers；GPU 0 bf16，14m36s） |
| `factual_probe_jnj.json` / `_orcl` / `_jpm` / `_uber` / `_ko` | complete | Factual amnesia probe（`entity_cell_factual_amnesia_probe.py`，泛化 CLI）：5 個專屬候選 cell 的 dose curve + controls + cross-entity |

**Provenance notes.**

1. Scope label `HighFactualMemory`：prepare 的 `--sector` 是單一 scope 過濾值；
   本 batch 跨 GICS sector（Healthcare / Financials / Consumer / Energy /
   Technology），統一以選取準入的 scope label 標示，非 GICS sector。
2. 財務證據仍為 neutral placeholder（non-8-K）；endpoint gate 數值解讀同
   [HFM discovery 報告](report-hfm-discovery.md)。
3. Surface forms 全部先通過 frozen contained-span tokenization 規則（12/12
   frames）；裸名（PayPal / Netflix / Visa）失敗，改用 Inc. 版本。

## 1. 事實記憶篩選（20 家新候選）

`factual_screen_batch2.json`：20 tickers、31 個 surface variants × 4 frames
（F0 HQ / F2 ticker / F3 founded / H0 CEO），clean 模型 raw-text greedy，
人類側對照 web-grounded 事實核對。

**選入 localization（≥3/4，11 家）：**

| Ticker | Surface form（localization 用） | HQ | Ticker | Founded | CEO | 分數 |
|---|---|---|---|---|---|---|
| JNJ | Johnson & Johnson | New Brunswick ✓ | JNJ ✓ | 1886 ✓ | 泛化 | 3/4 |
| JPM | JPMorgan Chase | New York ✓ | JPM ✓ | 1799 ✓ | 泛化 | 3/4 |
| KO | The Coca-Cola Company | Atlanta ✓ | KO ✓ | 1886 ✓ | 泛化 | 3/4 |
| ORCL | Oracle Corporation | Redwood Shores ✓ | ORCL ✓ | 1977 ✓ | 泛化 | 3/4 |
| PLTR | Palantir Technologies | San Francisco ✓ | PLTR ✓ | 2003 ✓ | 泛化 | 3/4 |
| QCOM | Qualcomm Inc. | San Diego ✓ | QCOM ✓ | 1985 ✓ | 泛化 | 3/4 |
| UBER | Uber Technologies | San Francisco ✓ | UBER ✓ | 2009 ✓ | **Travis Kalanick ✓** | 3.5/4 |
| V | Visa Inc. | San Francisco ✓ | V ✓ | 1958 ✓ | 泛化 | 3/4 |
| WMT | Walmart Inc. | Bentonville ✓ | WMT ✓ | 1962 ✓ | 泛化 | 3/4 |
| XOM | Exxon Mobil Corp. | Irving, TX ✓ | XOM ✓ | 1909 ✓ | 泛化 | 3/4 |
| INTC | Intel Corp. | Santa Clara ✓ | INTC ✓ | 1971 ✓ | 泛化 | 3/4 |

**淘汰**：ADBE / CRM / CSCO / PEP（2/4）；DIS / NFLX / MCD / PYPL（2.5/4，
ticker 框錯或 partial）；AMD（1/4，HQ 拒答、ticker 答 AMAT）。匿名控制組
不產生任何公司事實（行為正常）。

## 2. E1 V2 HFM-2 discovery（`entity-cell-e1-hfm2-discovery-v1`）

11 tickers × 12 frames + 399 generic baseline + template control + surface
controls + 3 個 financial prompts / ticker（placeholder 證據）。

### 結果：0/11 通過四門檻

| Ticker | Top-1 cell | Score | 排除原因 |
|---|---|---|---|
| JNJ | **(L4, N7676)** | **10994** | endpoint only |
| ORCL | (L0, N5101) | 7759 | endpoint, template, form-robust |
| PLTR | (L0, N8900) | 6679 | endpoint only |
| JPM | (L0, N9025) | 4354 | endpoint, held-overlap 0 |
| UBER | (L0, N8900) | 4236 | endpoint only |
| KO | (L0, N4485) | 3977 | endpoint, template, form-robust |
| QCOM | (L0, N104) | 507 | endpoint, held-overlap 0, form-robust |
| WMT | (L1, N7264) | 595 | endpoint, form-robust |
| INTC | (L2, N1786) | 345 | endpoint, form-robust |
| XOM | (L0, N104) | 398 | endpoint, form-robust |
| V | (L3, N228) | 437 | form-robust only |

**9/11 被 `endpoint_control_gate_below_2` 擋下**——這與
[decision-level probe 報告](report-decision-probe.md) 的結論一致：本模型的
buy/sell 決策 prior 由證據強度主導、entity 效應分散，單 cell 壓制在決策層
只產生小量 margin 移動，「target 顯著大於兩個控制組 ≥2 prompts」的條件天然
難滿足。JNJ / PLTR / UBER 其餘三關（form-robust、held-overlap、
template-robust）全過。

### 共享槽位結構（跨 sector 重現）

| Cell | 出現於 top-5 的 tickers |
|---|---|
| **(L4, N4047)** | JNJ, JPM, ORCL, PLTR, UBER（5/11，新共享槽位） |
| **(L0, N104)**（原 FTNT cell） | QCOM, V, WMT, XOM（4/11，跨 sector） |
| **(L2, N770)** | INTC, QCOM, XOM（3/11） |
| **(L1, N575)** | 3/11 |
| **(L0, N8900)** | PLTR top-1 + UBER top-1（2 家共享 top-1） |
| **(L0, N4485)** | KO top-1 + ORCL #2 |
| **(L3, N228) / (L3, N0)** | 各 2/11 |
| **(L2, N1786)**（NVDA batch top-1） | INTC top-1（跨 batch 重現） |

共享槽位在跨 sector 集合中重現，且 `(L0, N104)` 原 FTNT cell 出現在
Financials / Energy / Retail 的 top-5——與原 35 家研究的 17/35 共享 top-1
現象同構。

## 3. Factual amnesia probe（5 個專屬候選）

`entity_cell_factual_amnesia_probe.py`（泛化 CLI）：3 個 factual cloze frames
（F0 HQ / F2 ticker / F3 founded）、6 點 dose curve（+1.0 → -3.0）、
name-only scope、wrong-entity 對照（統一用已驗證的 AMZN cell `(0, 1476)`）、
matched-random（各 run 的 generic baseline stats 決定性選取）、3 個
cross-entity HQ 對照。Gold = clean greedy 前 3 token（人類側核對正確）。

### JNJ `(L4, N7676)` — 全研究線最強的 entity cell

| F0（HQ " New Brunswick,"） | gold logp |
|---|---|
| clean / α=+1.0 | -0.922 |
| α=+0.5 / 0.0 | -0.977 / -1.212 |
| α=-1.0 | **-4.619**（top-1 翻成 ` ___`） |
| α=-2.0 / -3.0 | -8.969 / **-9.433** |
| α=-3.0 name-only | -9.480 |
| wrong-entity (L0, N1476) α=-3 | -0.920（Δ-0.002） |
| matched-random (L4, N9197) α=-3 | -0.873（Δ+0.049） |
| cross：JPM / KO / WMT HQ | Δ-0.003 / +0.001 / -0.023 |

- **HQ 崩塌 -8.51 nats**（AMZN N1476 的 8 倍）、劑量單調、name-position
  敏感（name-only ≥ all-positions）。
- **行為層面**：α=-3 下 greedy 從 ` New Brunswick,` 變成 **` ___?`**——模型
  以拒答模式表明它「不知道」了。
- F3（founded " 18"〔1886〕）崩塌 -4.485 nats（controls 平穩）；F2（ticker
  " JNJ."）-0.741。
- wrong-entity 與 matched-random 完全平穩 → 特異性乾淨。
- JNJ 在 formal gate 中唯獨 fail endpoint（決策層）——與決策層結論一致。

### JPM `(L0, N9025)` — 中等效應、三框皆特異

F0 -0.581（wrong +0.105 / random +0.018）、F2 -0.525（+0.079 / +0.101）、
F3 -0.426（-0.007 / -0.017）。三個事實框皆顯示特異性移動，方向一致。
Formal gate 中 held-overlap 0 使其未達 trusted；fact-level 證據支持它是一個
真實但效應量中等的 JPM 專屬 cell。

### KO `(L0, N4485)` — 部分特異

F2（ticker " **KO**."）崩塌 **-2.86**（wrong +0.007 / random -0.019，特異）；
F3（founded " 18"）-0.617（controls 平穩）；F0（HQ）僅 -0.121。Cross 對照中
WMT -0.343 / XOM -0.552 有小幅移動（非完全平穩）——部分特異，弱於 JNJ/JPM。

### ORCL `(L0, N5101)` / UBER `(L0, N8900)` — 否決

- ORCL：F0 -0.467（gold 前 3 token 為泛化的 ` the city of`，entity 特異部分
  不在 gold 內）；F2 -0.155 但 wrong-entity 同級（-0.112）→ 無特異性。
- UBER：F0 +0.137、F2 -0.025、F3 -0.010 → 無效應（與 PLTR 共享 top-1 一致，
  非 UBER 專屬事實載體）。

## 4. 判讀

1. **Entity cell 是存在的，且跨 sector**：JNJ（Healthcare）、JPM
   （Financials）在 fact-level 驗證下確認為 entity-specific 因果載體，加上
   AMZN（Consumer Discretionary），三個不同 sector 都有 entity cell。
2. **強度分層**：JNJ (L4, N7676)（-8.5 nats、拒答行為）> AMZN (L0, N1476)
   （-1.06 nats、confidence collapse）> JPM (L0, N9025)（-0.4 ~ -0.6 nats）>
   KO 部分 > 其餘（無）。效應量與 factual memory 強度、localization score
   大體相關（JNJ 10994 最高、效應最大）。
3. **Layer 分佈**：AMZN/JPM/KO/ORCL/PLTR/UBER 的候選在 L0，JNJ 的在
   **L4**——entity cell 不限定單一層。
4. **共享槽位**：(L4, N4047) 5/11、(L0, N104) 4/11（跨 sector）、(L2, N770)
   3/11——多個 entity 共享同一批高穩定性 neuron，這些槽位不是單一 entity 的
   專屬載體（與 JNJ/JPM 的專屬 cell 形成對照）。
5. **Endpoint gate 的定位**：決策層門檻在本模型下是四道門檻中最弱的一關
   （決策 prior 分散、證據主導）；fact-level amnesia probe 是對 entity cell
   更直接的驗證（已對 AMZN/JNJ/JPM/KO 完成）。

## 5. 限制

- 0/11 formal trusted：本報告的「entity cell 確認」基於 fact-level proposed
  probe，非 frozen gate 判定；正式 claim 需先凍結 fact-level 門檻（collapse
  target、控制組差異閾值、樣本量）。
- ORCL 的 F0 gold 質量弱（泛化前綴）；JPM held-overlap 0；KO 有跨 entity
  小移動。
- 單一模型（Qwen3.5-4B）；placeholder 證據；calibration / held-out 未執行。

## 6. Next steps（建議）

1. **凍結 fact-level amnesia 門檻**（以 JNJ/AMZN/JPM 的效應量與控制組分佈
   為依據立版本文件），把「entity cell 確認」從 proposed probe 升級為 frozen
   判定。
2. 對 JNJ `(L4, N7676)` 做 E3 式深度解剖（downstream attribution / patching
   源頭分析），它是目前線內最強標靶。
3. 擴充事实框（product / sector / competitors / CEO 指名）畫 cell-事實映射。
4. Calibration / held-out test。
