# Entity Cell: Proposal E5（Fact Coverage Profile Probe）

**Document status:** proposed（非 frozen protocol）。本文件不修改任何 frozen
protocol（E1 V3、E2、E3、E4 皆維持原狀），只為 4 個 frozen V3 entity cells 新增
一個輸出層的事實覆蓋診斷實驗。實驗以 proposed probe script 執行
（`entity_cell_fact_coverage_probe.py`，待實作，見
[research scripts reference](../../research-scripts.md)），重用 frozen
`e1-fact-amnesia` 的計分路徑（`llm_bias/entity_cell/fact_amnesia`：3-token gold、
teacher-forced joint log-prob、α = −3.0、all_positions）與
`select_matched_random_neuron`（`llm_bias/entity_cell/mlp_cells`）。

**Model for run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)

**Depends on:**
- Frozen V3 entity cells（calibration + hold-out 雙驗證）：JNJ `(L4, N7676)`、
  PLTR `(L2, N5003)`（calibration）、BAC `(L0, N7801)`、CAT `(L2, N7997)`
  （hold-out），見 [E1 V3 proposal](proposal-v3.md) 與
  [收線報告](../report.md)
- E5 clean preflight（兩輪，已完成）：
  `artifacts/qwen3.5-4b/entity-cell-localization/e5-preflight/`
  下 `e5_fact_coverage_preflight_v1.json`（9 框初版）與
  `e5_fact_coverage_preflight_v2_revised_frames.json`（6 框修正版），由
  `entity_cell_factual_recall_preflight.py --frames` 產出
- 計分與干預代碼：frozen `fact_amnesia` / `mlp_cells` 模組（不修改）

---

## 1. 術語定義

**Phase E5（Fact coverage profile probe）**：對單一 frozen entity cell 施加
壓抑（α = −3.0、all_positions），在多個事實類別的 fact frame 上量測 clean gold
序列（3 token）的 teacher-forced joint log-probability 變化（collapse），據此
描述該 cell 在該 entity 上承載的事實類別集合。輸出層因果量測：壓抑是干預、
collapse 是讀出；**事實覆蓋輪廓（fact coverage profile）本身是描述性特徵，
不構成新的 frozen gate**，也不建立決策因果或 chain-of-thought 證據。

**事實覆蓋輪廓（fact coverage profile）**：單一 entity 上，該 cell 在参考
門檻（collapse ≤ −0.5 nats）達標的事實類別集合。本詞在本文件定義，文件未
引用前只可使用完整中文描述句。

與既有量的關係：V3 Gate 4 與 E5 的計分層完全相同（輸出層 gold joint
log-prob collapse）；差異在三處——(a) frame 集擴展到 F0/F2/F3 以外的事實
類別；(b) 對象是已認證 cell，不是 localization 候選；(c) 輸出是覆蓋輪廓
描述，不是 pass/fail 判定。

## 2. 動機

1. **V3 只測了三類事實，且每顆 cell 的崩塌模式不同**：JNJ 為 F0（總部）
   −8.51 + F3（年份）−4.49、F2（代號）−0.74 近乎保留；BAC 為 F0 −7.77 +
   F2 −7.84、F3 保留；CAT/PLTR 只有 F2 過關。即每顆 cell 承載的事實子集
   不同：既非「整家公司」也非「只有地理」（代號是非地理事實，BAC/PLTR 上
   同樣崩塌，部分反駁地理假說）。
2. **E4 已確立輸出層失憶是下限**：BAC F3 與 PLTR F0 在終端未崩（+0.44 /
   +0.15 nats），但中間層表徵位移達 6.25 / 1.50。因此 E5 的 null 只能解讀
   為「輸出層不受影響」，不是「內部無影響」。
3. **未回答的問題**：各 cell 的事實覆蓋邊界——除總部/代號/年份外，它還
   承載哪些類別（總部第二句框、上市交易所、競爭對手）？這直接回答
   「entity cell 是公司身分錨點，還是單一事實通道」的區分問題。

## 3. 實驗設計

### 3.1 Targets 與 surface forms

4 個 frozen entity cells；surface form 沿用各 source run 的命名慣例
（多詞正式名稱，與 V3 §8 的 tokenization 陷阱對策一致）：

| Entity | Cell | Source run | Surface form |
|---|---|---|---|
| JNJ | (L4, N7676) | `entity-cell-e1-v3-hfm2-calibration-v1` | Johnson & Johnson |
| BAC | (L0, N7801) | `entity-cell-e1-v3-holdout-v1` | Bank of America |
| CAT | (L2, N7997) | `entity-cell-e1-v3-holdout-v1` | Caterpillar |
| PLTR | (L2, N5003) | `entity-cell-e1-v3-hfm2-calibration-v1` | Palantir Technologies |

### 3.2 Frame 表

frozen frame ID 照用；新增框使用 E5 命名空間的 proposed ID（模板文字為
identity 來源，ID 只是標籤）：

| Frame | 模板 | 事實類別 | 來源 |
|---|---|---|---|
| F0 | `The headquarters of {name} is located in` | 總部（HQ） | frozen V3 |
| 20 | `{name} is headquartered in` | 總部（第二句框） | preflight v1 |
| F2 | `The stock ticker of {name} is` | 股票代號 | frozen V3 |
| F3 | `{name} was founded in` | 成立年份 | frozen V3 |
| 23 | `{name} is listed on` | 上市交易所 | preflight v1 |
| 24s | `A major competitor of {name} is` | 競爭對手（單數句框） | preflight v1 |
| 24p | `The main competitors of {name} are` | 競爭對手（複數句框） | preflight v2 |

### 3.3 (Entity, frame) 矩陣與 gold 綁定（18 pairs）

由兩輪 clean preflight 的 greedy 3-token 輸出經人驗（fail-closed：內容錯誤、
泛化、拒答或格式破損者剔除）決定。Gold 綁定：probe 執行時重算 clean greedy
前 3 token，必須與下表 **完全一致**，否則該 pair fail-closed 中止（與 E4
§3.1 的 gold 綁定同規則）。所有 gold 已對照公開事實人驗通過。

| Entity | Frame | Gold token ids | Gold text | 人驗註記 |
|---|---|---|---|---|
| JNJ | F0 | `[1478, 58122, 11]` | ` New Brunswick,` | 總部城市，正確 |
| JNJ | 20 | `[1478, 58122, 11]` | ` New Brunswick,` | 與 F0 同 slot、第二句框 |
| JNJ | F2 | `[604, 83986, 13]` | ` JNJ.` | 代號，正確 |
| JNJ | F3 | `[220, 16, 23]` | ` 18` | 年份前兩碼（完整 1886，token 限制） |
| JNJ | 23 | `[279, 1478, 4121]` | ` the New York` | NYSE，正確 |
| JNJ | 24s | `[93510, 13, 198]` | ` Pfizer.\n` | 競爭對手，正確；token 3 為換行（後接測驗格式殘留，見 §3.4 註記） |
| BAC | F0 | `[27449, 11, 4634]` | ` Charlotte, North` | 總部城市，正確 |
| BAC | 20 | `[27449, 11, 4634]` | ` Charlotte, North` | 與 F0 同 slot、第二句框 |
| BAC | F2 | `[417, 1646, 13]` | ` BAC.` | 代號，正確 |
| BAC | 23 | `[279, 1478, 4121]` | ` the New York` | NYSE，正確 |
| BAC | 24p | `[604, 8530, 8202]` | ` JPMorgan` | 競爭對手（JPMorgan Chase），正確 |
| CAT | F2 | `[42525, 13, 561]` | ` CAT. The` | 代號，正確；token 3 為續寫殘留 |
| CAT | 20 | `[4974, 10465, 11]` | ` Peoria,` | 總部城市，正確（F0 句框拒答，此句框成功） |
| CAT | 23 | `[279, 1478, 4121]` | ` the New York` | NYSE，正確 |
| PLTR | F0 | `[5655, 12510, 11]` | ` San Francisco,` | 總部城市，正確 |
| PLTR | F2 | `[10009, 2301, 13]` | ` PLTR.` | 代號，正確 |
| PLTR | F3 | `[220, 17, 15]` | ` 20` | 年份前兩碼（完整 2003，token 限制） |
| PLTR | 23 | `[279, 15613, 60764]` | ` the NASDAQ` | 交易所，正確（唯一非 NYSE） |

**Gold 內容註記（與 frozen 3-token 定義一致）**：3-token gold 固定的是模型
自己的 clean 回答，第三 token 可能含續寫或格式殘留（如 JNJ 24s 的換行、
CAT F2 的 `The`）；這是 frozen fact gate 計分定義的既有行為，不因殘留而
改動 gold 長度。年份 gold 只有前兩碼（1886 → ` 18`、2003 → ` 20`），人驗以
完整年份為準。

### 3.4 剔除的 frame 與原因（not_applicable 記錄）

以下 frame 在 preflight 中未建立可用 gold，依 fail-closed 規則不進入矩陣，
原因記錄如下（證據見兩份 preflight JSON 的 `greedy_text` 欄）：

| 類別 / frame | 剔除原因 |
|---|---|
| CEO（`The CEO of {name} is`、`{name} is led by`） | 4/4 泛化答（"a member of the board"、"a board of directors that"）——模型拒絕給 CEO 名，類別整體不可用 |
| 產業（`{name} is a company in the`、`{name} operates in the`、`The industry of {name} is`） | JNJ/BAC/CAT 皆答地理（"United States..."）而非產業；僅 PLTR 答出 "data analytics and artificial intelligence"，不足以建立類別 gold |
| 產品（`The main business of {name} is`、`The main product of {name} is`、`The main products of {name} are`） | JNJ "Baby powder"、CAT "Tractors" 內容為真實產品，但被測驗格式污染（`A.` / `:\nA.` 前綴），依格式破損規則剔除 |
| F3（BAC） | gold "1998" 與公開事實不符（1923），內容錯誤 |
| F3（CAT） | gold "1836" 與公開事實不符（1925），內容錯誤 |
| F0（CAT） | 拒答 `___?\nA.` |
| 20（PLTR） | 只答到國家層級（"the United States, with"），泛化 |
| 24s（BAC/CAT/PLTR）、24p（JNJ/CAT/PLTR） | 交叉句框皆為格式破損或泛化（見 preflight 記錄） |

### 3.5 條件（4 個；劑量與 fact gate 相同：−3.0、all_positions）

| 條件 | 干預 | 期望 |
|---|---|---|
| clean | 無 | baseline |
| target | 該 entity 自己的 cell | 達標類別的 collapse 顯著為負 |
| wrong_entity | 配對的另一個 frozen cell，固定配對 **JNJ↔PLTR、BAC↔CAT**（與 E4 §3.2 一致：JNJ 題壓 PLTR `(L2, N5003)`、PLTR 題壓 JNJ `(L4, N7676)`、BAC 題壓 CAT `(L2, N7997)`、CAT 題壓 BAC `(L0, N7801)`） | \|collapse\| ≤ 0.3 nats |
| matched_random | 同層 matched neuron（`select_matched_random_neuron`；統計由 probe 以 frozen `collect_generic_stats` 路徑、explicit versioned generic baseline 語料生成，corpus provenance 記入輸出） | \|collapse\| ≤ 0.3 nats |

matched_random 統計的語料來源：優先使用與 frozen V3 run 相同的 generic
baseline 語料；若該語料不在本機，重新生成 4B versioned generic baseline 並
完整記錄 provenance。控制組對語料 identity 的敏感度低於 gold 綁定（見
§5 限制 4）。

### 3.6 計分

- 沿用 frozen 計分路徑：raw text prompt、`add_special_tokens=True`、無 chat
  template；clean greedy 3-token gold（綁定 §3.3）；`score_token_ids`
  teacher-forced joint log-probability（nats，FP32 tail）。
- `collapse(cond) = logp(cond) − logp(clean)`（負值 = 忘記）。
- **一致性錨點**：F0/F2/F3 pairs 重測的是 frozen V3 已記錄的量。target
  collapse 與 frozen V3 記錄（JNJ F0 −8.51 / F3 −4.49 / F2 −0.74；BAC
  F0 −7.77 / F2 −7.84；CAT F2 −3.49）的 diff 必須 ≤ 0.1 nats；0.1 < diff
  ≤ 0.3 標記 `instrument_drift: true`（結果可用但標註）；diff > 0.3 中止
  並視為儀器失敗。

### 3.7 Output 契約（compact）

- 每 (entity, frame, condition)：gold token ids/text、`logp`、`collapse`、
  `degraded_control` 標記（若 matched_random 選取撞車）。
- summary：每 cell 的覆蓋矩陣（entity × category × condition）、控制組
  max |collapse|、錨點 diff、provenance。
- 禁令同 V3：不持久化任何 raw activation / residual / hidden state；只輸出
  compact 派生量與 provenance；finite float。

### 3.8 Provenance

輸出 JSON 的 provenance 包含：model、device、dtype、劑量與門檻、兩份
preflight artifact 的 path + sha256（gold 綁定來源）、generic baseline
corpus provenance、source run 的 cell 確認記錄（V3 §5.2 名冊）、frame 表
版本（E5 frame set v1）、錨點對照結果。

## 4. 解讀框架（描述性，非 gate）

- **参考門檻**（沿用 V3 常數，用於分類而非 pass/fail）：
  - `collapse ≤ −0.5` nats → 該類別標記 **carried**（承載）；
  - `−0.5 < collapse ≤ −0.3` → gray zone；
  - `collapse > −0.3` → **not carried at output**（輸出層未承載）；
  - 該 pair 任一控制組 |collapse| > 0.3 → 標記 `unreliable_control`，
    target 讀數不作解讀。
- **假說（描述性比較，不做統計檢定）**：
  - **H_geo**：只有 HQ 類別（F0/20）崩塌 → cell 是地理事實通道；
  - **H_identity**：多數類別皆 carried → cell 是公司身分錨點；
  - **H_subset**：cell 專屬的事實子集（V3 模式的延伸，每顆 cell 子集
    不同）。
- **預期錨點**：JNJ 的 F0/F3、BAC 的 F0/F2、CAT/PLTR 的 F2 應重現 frozen
  記錄（§3.6）；新增類別（20、23、24s/24p）是有資訊量的增量。

## 5. 解讀限制

1. **輸出層是下限**（E4）：null 不等於內部無影響；BAC F3 / PLTR F0 式的
   「終端不變、中層位移」在 E5 的新 frame 上可能重現，E5 本身不量測中層。
2. **弱先驗 frame**：F23 的 NYSE 答案在 3/4 entity 與 anonymous 對照的
   預設值相同（preflight：anonymous "the New York Stock Exchange" top-1），
   先前驗主導。該類別的 null 只能結論「輸出層無效應」，不能結論「未
   承載」；PLTR 的 NASDAQ 才有鑑別力。
3. **未測類別不可下結論**：CEO、產業、產品因 gold 不可用而未入矩陣
   （§3.4）；對這些類別不能宣稱 cell「未承載」。
4. **matched_random 語料 identity**：若 V3 原始 generic baseline 語料不在
   本機而重新生成，matched_random 的配對基準與 V3 不同；控制組門檻
   （0.3 nats）遠寬於預期效應，影響有限，但 summary 必須標記語料來源。
5. **單一劑量與固定 surface form**：α = −3.0 全量壓抑、3-token gold、
   固定 surface form；句框形式敏感（如 CAT 的 F0 拒答但 20 成功）是本線
   已知的模型行為來源，不因單一句框的 null 推翻類別結論。
6. **輪廓的邊界**：fact coverage profile 描述單一 cell 干預下的輸出層
   效應輪廓，不等於「該公司的知識只存在於這些類別」；完整失憶需要多載體
   組合干預（本協議範圍外，屬後續工作候選）。

## 6. 狀態與 promotion 條件

- **proposed**：實作 `scripts/entity_cell_fact_coverage_probe.py`（待實作；
  重用 `fact_amnesia` 計分與 `mlp_cells` 干預/配對代碼，不修改 frozen
  模組），產出 compact JSON + 報告（`report-e5-fact-coverage.md`）。
  Run id 慣例：`entity-cell-e5-fact-coverage-v1`。
- 計算量估算：18 pairs × 4 條件（≈ 72 組 teacher-forced 計分）+ clean gold
  重算 + generic baseline 統計 pass，與 preflight 同量級（~10 min，
  RTX 3060 / bf16）。
- **若結果顯示穩定且控制組乾淨的 per-cell 覆蓋輪廓**（每 cell 的 carried
  集合跨 frame 句框一致、控制組 ≤ 0.3、錨點 diff ≤ 0.1），可討論提升為
  frozen protocol；依 [experiment
  versioning](../../documentation-system.md#experiment-versioning) 另立
  版本編號與 frozen 文件，不把 proposed 結果回填成 frozen 判定。

## 7. 版本分立觸發條件（Version Break Triggers）

以下任一變更必須另立新版（`proposal-e5-v2.md`），禁止原地修改本文件：

1. 修改任何 frame 模板文字或 gold 長度；
2. 變更 §3.3 的 (entity, frame) 矩陣或任何 gold 綁定值；
3. 變更 wrong_entity 配對、劑量或 all_positions 範圍；
4. 變更参考門檻（0.5 / 0.3）或分類邏輯；
5. 新增事實類別（新 frame）——須另立版本或正式 addendum，不得靜態擴充。
