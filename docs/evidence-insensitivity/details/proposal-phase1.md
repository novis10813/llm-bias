# Evidence-insensitivity Phase 1：S&P 500 證據敏感度行為篩選（frozen protocol）

**狀態**：frozen（Rev 2；2026-09-16 pilot 通過＋用戶批准 formal run）
**對象模型**：Qwen3.5-4B（bf16，32 層，hidden 2560；`.cache/models/qwen3.5-4b`）
**研究線定位**：全線開場行為篩選。以共用 company-agnostic 證據控制設計，在 S&P 500
全人口（2024 名單 503 家）上量測每家公司在零證據 prior 與證據淨極性梯度下的決策
行為，做預先註冊的 between-company 分組，並建立 base-rate 控制。本 phase 的輸出是
frozen 公司分組表，供 Phase 2（L15 組間對比）與 Phase 3（上游因果定位）使用。
本 phase 不介入、不建立任何 causal claim。

---

## 1. 核心假說與文獻邊界

**研究問題**：在證據完全相同的條件下，哪些公司的 buy/sell 輸出跟隨證據淨極性
（evidence-responsive），哪些公司的輸出固定不隨極性改變（evidence-insensitive）？
兩組的人口規模、sector 結構與零證據 prior 分佈為何？

**動機**（上游線留下的缺口）：

- [balanced-evidence-gap](../../balanced-evidence-gap/report.md)：共享平衡證據下 16 家
  64/64 sell、公司間 margin spread ~1.5 nats；只測了 entity 主效果，未測證據極性。
- [entity-to-dial](../../entity-to-dial/report.md)：決策形成定位在 L15 instruction span。
- [entity-concept-decision](../../entity-concept-decision/report.md)：L15 無可分離公司
  概念；決策是微小公司間差的高增益非線性讀出；stance 是 1D 讀數方向。
- [jspace-token-experiments V2 prior probe](../../jspace-token-experiments/report.md)：
  零證據模板 6 家全 sell（M ∈ [−4.93, −3.39]）、identity 無可量測 association——人口
  太小，無法定義「證據不敏感」群組。

**文獻原始設定 vs 本線適應（Adaptations）**：

| 來源設定 | 本線適應 | 引入的風險／邊界 |
|---|---|---|
| Park et al. (2026) "Your AI, On a Dial"：平衡多空證據、全域立場校準 | 共用 company-agnostic 證據＋極性梯度；不做 dial 校準 | 本線量行為分佈，不宣稱立場控制 |
| FELAB-UNIST LLM bias in finance（trial plan 來源）：per-company LLM 生成證據 | 人寫共用證據（全部公司相同），保留 trial plan 的 5/6/8/10/15% 百分比梯子與 JSON 輸出格式 | 失去公司特定證據內容；組間對比變純 entity 對比（設計目標），外部效度限於 generic 證據情境 |
| jspace V2 prior probe：6 tickers 零證據 header-only | 503 家零證據；模板幾乎相同（去掉 Sector 行，對齊 trial plan header） | 模板細節差異屬於 prompt family 變更，已於本協議定義為新 family |
| 既有白箱線：fixed answer-token margin 為 primary | generation（JSON decision + reason）為 primary、decision-position margin 為 auxiliary | margin 依賴 canonical JSON 前綴（pilot 驗證）；generation 帶來 parse 邊界情況 |

## 2. 名詞定義

- **margin M(p)**：prompt p 在 decision 位置的 fixed answer-token margin
  `log p("buy") − log p("sell")`（nats，buy 為正；FP32 tail logit scoring，沿用
  `core/continuation_scoring` 的 FP32 計分與 float64 減法語義）。
- **decision-position margin**：JSON 模板下，輸入為 `prompt + decision position prefix`
  （即 `prompt + '{\n  "decision": "'`，模型 greedy 實際決策位置，pilot 實測 259/259
  記錄前綴 byte-identical；沿用 entity_cell 與 jspace_intervention 線的既有
  `DECISION_PREFIX` 慣例），於輸入最終位置計 buy/sell 兩 token 的 margin。
  不等同行尾 `final_position` 計分。
- **shared company-agnostic evidence**：全部公司共用、不指向任何特定公司的 2 條
  證據項（1 正 1 負，百分比佔位）。證據內容跨公司完全相同，只有百分比數值與
  極性配置變動。
- **net polarity（淨極性）與 polarity score**：一條 prompt 中優勢證據項的極性與其
  相對強度。本線以 `(advantage%, disadvantage%)` 定義 8 個條件，polarity score
  `v ∈ {−4, −3, −2, −1, +1, +2, +3, +4}`（負 = 負證據項優勢，正 = 正證據項優勢，
  絕對值為強度層級 6/8/10/15% 對應 1/2/3/4）。
- **response contrast C_c**：公司 c 的 `M(P15) − M(N15)`（最強極性格的 margin 差，
  輔助量；continuous，不分組時的主 estimand）。
- **evidence-responsive（跟隨組）**：生成 decision 滿足 `D(N15) = sell 且 D(P15) = buy`
  的公司。
- **evidence-insensitive（不敏感組）**：生成 decision 在兩極性格不變的公司；
  **fixed-sell** = `D(N15) = D(P15) = sell`，**fixed-buy** = `D(N15) = D(P15) = buy`。
- **mixed**：其餘 decision 組合（描述性，排除出主對比）。
- **identity-stripped prompt**：header 公司身分替換為 `Stock Ticker: [TICKER]`、
  `Stock Name: [Company X]` 的匿名版本（沿用 selective-intervention 慣例）。
- **prior-consistent / prior-reversed**：不敏感組的零證據 decision 與固定方向相同
  （consistent）或相反（reversed）的描述性子標籤。
- **generation endpoint**：greedy 生成（`do_sample=False`）的完整 JSON，解析
  第一個 JSON 物件的 `"decision"` 欄位（buy/sell）作為 primary 行為端點；同時
  持久化 reason 文字（descriptive 分析用）。解析規則：自生成文字第一個 `{` 起用
  `json.JSONDecoder().raw_decode` 提取第一個 JSON 物件，忽略尾端內容（模型會在
  物件後附加 stop marker 文字，pilot 實測）；提取失敗或 `decision` 非 buy/sell
  即 parse fail。
- **hold-out 公司**：Phase 1 分組前以 seed 分出的 20% 公司集；Phase 2/3 的機制
  選擇只能使用 discovery split（其餘 80%），hold-out 留作 confirmation。

## 3. 預設 Input / Output 契約

### 3.1 Input

| 項目 | 路徑／值 | 契約 |
|---|---|---|
| 公司名單 | `data/all_constituents_2020_2025.csv` | 取 `index_name == "S&P 500"` 且 `year == "2024"` 的 503 行；欄位 `ticker`、`company_name`、`gics_sector`；prepare fail-closed 檢查 503 筆、ticker 唯一、name 非空 |
| 證據文字 | 本協議 §4 表（人寫，凍結於本文件） | SHA-256 記於 provenance；文字變更 = prompt family 變更（version break） |
| model | `.cache/models/qwen3.5-4b`（bf16） | forward 前驗證 model identity 與既有 run 慣例一致 |
| tokenizer | 與 model 同目錄 | greedy、`enable_thinking` 不啟用（沿用 baseline trial 慣例） |
| hold-out seed | 20260916 | stratified by `gics_sector`，20%（100–101 家） |
| order-swap 子樣本 seed | 20260916 | 100 家，stratified by `gics_sector`，自全 503 抽出 |

### 3.2 Output（compact，不存 raw activations / residuals / KV cache）

`artifacts/qwen3.5-4b/evidence-insensitivity/runs/<run-id>/`：

- `prepare/prompts.jsonl`：每 prompt 一筆：`prompt_id`（sha 前 12）、`ticker`、
  `company_name`、`gics_sector`、`split`（discovery/hold-out）、`arm`
  （primary / order_swap / anon）、`condition`（zero / N6..N15 / P6..P15）、
  `polarity_score`、`prompt_text`、`evidence_sha`、span token 邊界
  （`header_span`、`evidence_span`、`instruction_span`，互斥且 100% 覆蓋 prompt）。
- `prepare/provenance.json`：population digest（503 tickers 排序後 SHA-256）、
  證據文字 SHA-256、seeds、split 計數、model identity。
- `forward/records.jsonl`：每 prompt 一筆：`prompt_id`、`arm`、`condition`、
  `margin`（finite float）、`decision`（buy/sell/null）、`generated_text`
  （完整生成，短 JSON）、`n_new_tokens`。
- `forward/metadata.json`：generation 參數（greedy、max_new_tokens=128）、
  model/tokenizer identity、run 時間、determinism re-run 結果（20 prompts 的
  max|ΔM| 與 text mismatch 數）。
- `analyze/summary.json`：gate 值、分組表（503 家 × group/sub-label/split）、
  描述統計（見 §8）。所有數值欄位 finite float。
- `manifest.json`：`ArtifactRun` lifecycle。

### 3.3 CLI 契約（1:1 綁定子命令）

```bash
uv run python scripts/evidence_insensitivity_phase1.py pilot    # real-model 50 家 pilot（§9.2），輸出 runs/<pilot-id>/，status=pilot
uv run python scripts/evidence_insensitivity_phase1.py prepare --run-id <RUN_ID>
uv run python scripts/evidence_insensitivity_phase1.py forward --run-id <RUN_ID>
uv run python scripts/evidence_insensitivity_phase1.py analyze --run-id <RUN_ID>
uv run python -m pytest tests/test_evidence_insensitivity_phase1.py -q   # fake-model 回歸（無 GPU）
```

`pilot` 同時滿足 formal run 前的強制端到端 preflight（docs/AGENTS.md §5）；
pilot 後若 pipeline 有任何變更，formal 前必須再跑 1 筆 real-model 1-prompt smoke
（`prepare --run-id <smoke-id> --one-prompt` 等價路徑，於 forward 前驗證 schema）。

## 4. Frozen 設計常數

| 項目 | 值 | 來源 |
|---|---|---|
| population | S&P 500 2024，503 家（§3.1） | 本提案 |
| POS_ITEM | `According to analysts, the company recently secured a major long-term partnership that is expected to significantly expand its revenue base, providing a catalyst for a {p}% increase in the stock price over the coming quarter.` | 本提案（人寫，review 後 freeze） |
| NEG_ITEM | `Recent operational challenges in the company's main business division have raised concerns about execution timelines, with multiple initiatives experiencing delays and cost overruns that could compress margins by approximately {n}% over the coming quarters, potentially driving the stock price lower.` | 本提案 |
| 條件矩陣（primary arm，item 1 = POS、item 2 = NEG） | zero（`1. No evidence provided.`）＋ N6/N8/N10/N15（POS {5,5,5,5}%、NEG {6,8,10,15}%、v=−1..−4）＋ P6/P8/P10/P15（POS {6,8,10,15}%、NEG {5,5,5,5}%、v=+1..+4） | 本提案（梯子沿用 trial plan 設計） |
| order-swap arm | 8 極性格、item 順序反轉（item 1 = NEG）；100 家子樣本 | 本提案 |
| anonymous arm | identity-stripped header；zero + 8 極性格 = 9 prompts | 本提案 |
| prompt 模板 | `Refer to the evidence below to make a final investment decision.\nStock Ticker: [{ticker}]\nStock Name: [{name}]\n--- Evidence ---\n1. {item1}\n2. {item2}\n---\nRespond with one valid JSON object containing only the keys "decision" (buy \| sell) and "reason" (brief justification). Do not choose hold.`（zero 條件證據段為 `1. No evidence provided.`） | 本提案（對齊 trial plan / prior probe family） |
| decision position prefix | `'{\n  "decision": "'`（margin 計分位置；模型實際 greedy 決策位置，pilot 259/259 byte-identical；與 entity_cell / jspace_intervention 的 `DECISION_PREFIX` 慣例一致） | 本提案（pilot 驗證） |
| generation | greedy、`max_new_tokens=128` | 本提案 |
| 計分 | decision-position margin（FP32）；margin 減法 float64（與 2A 參照機制一致） | repo 語義邊界 |
| forward 總計 | 503×9（4527）＋ 100×8（800）＋ 9（anon）= **5336**；另 20 筆 determinism re-run（post-check） | 本提案 |

## 5. 分組規則（pre-registered；看內部狀態前凍結）

對每家公司（N15 與 P15 的 generation 皆 parse 成功者）：

1. `D(N15) = sell 且 D(P15) = buy` → **evidence-responsive**
2. `D(N15) = sell 且 D(P15) = sell` → **fixed-sell**
3. `D(N15) = buy 且 D(P15) = buy` → **fixed-buy**
4. 其餘 → **mixed**（描述性）

描述性子標籤（不影響分組）：零證據 `D0` 與 `M0`；不敏感組的 prior-consistent /
prior-reversed。parse 失敗的公司在該條件上 `decision=null`；N15 或 P15 任一
parse fail 的公司不參與分組、計入 G-P1 分母，並描述性報告其 margin 軌跡。

**穩健性（不 gate）**：6/8/10% 極性格的 decision 與 margin 曲線；若 15% 格分組
與 10% 格分組（`D(N10)`、`D(P10)` 同規則）的組別一致率 < 80%，於報告標註
「分組對強度敏感」，不事後改規則。

## 6. Gates（fail-closed）

| Gate | 判定 | 門檻 |
|---|---|---|
| **G-P1**（parse） | 全 5336 prompts 的 decision parse 成功率 | ≥ 0.95 |
| **G-P2**（符號一致） | parse 成功者中 `sign(M) == (decision == buy)` 比例（M=0 記 mismatch） | ≥ 0.90 |
| **G-P3**（確定性） | 20 筆 seed 20260916 抽樣 re-run：生成文字 0 mismatch 且 max\|ΔM\| = 0 | 0 / 0 |
| **G-P4**（組 power，branching） | `\|evidence-responsive\| ≥ 10` 且 `\|fixed-sell ∪ fixed-buy\| ≥ 10` | 過 → between-company 主對比；未過 → 預設 fallback |

**G-P4 fallback（pre-registered，非事後調整）**：主 estimand 改為 continuous
response contrast C_c（503 家全量 rank，tertile 分層供 Phase 2 設計）；分組降為
描述性。Phase 2 協議依 fallback 結果另訂。

## 7. 邊界情況與防禦性行為

- **parse fail**：`decision=null`；不參與分組；計入 G-P1。連續 parse fail > 5%
  的單一 condition 觸發 preflight 級診斷（模板/前綴問題），formal 無效、修後
  新 run ID。
- **控制組缺失**：anonymous arm 任一 prompt 執行失敗 → run 無效（base rate 解讀
  依賴 anonymous）；不允許以公司組均值代替。
- **退化條件**：某 condition 的 parse 成功 decision 中 buy 比例 < 5% 或 > 95%
  時，該 condition 標記 degenerate；分組規則仍按 §5 執行（規則不依分佈改動），
  描述性報告標註。
- **span 覆蓋**：prepare 為每 prompt 記錄 `header_span`（模板首行至 header 末）、
  `evidence_span`（`--- Evidence ---` 至其後 `---`，不含界線 token）、
  `instruction_span`（末段指令）的 token 邊界；三者互斥且與模板結構 100% 對應
  （prepare 單測以 tokenizer 斷言）。
- **數值容差**：margin 為 FP32；G-P3 要求 bit-identical（max|ΔM| = 0），因同
  model/dtype/greedy 下推論確定；intervention 相關容差（bf16 jitter 帶 0.05 nats
  慣例）不適用於本 phase（無介入）。

## 8. 描述統計（analyze 輸出，不 gate）

1. **anonymous base-rate 曲線**：9 條件的 margin 與 decision（模板＋證據的全域
   sell/buy 傾向；分離「模型對所有人忽略證據」與「模型對這些公司忽略證據」）。
2. **group × gics_sector 表**：各組在 11 個 GICS sector 的計數。
3. **prior 分佈**：零證據 margin 分位數、D0 計數、prior-consistent/reversed 拆分。
4. **population 極性曲線**：每 condition 的 mean/IQR margin 與 buy 比例。
5. **order-swap 效應**：100 家子樣本的 primary vs order-swap 的 decision 差異率
   與 ΔM（item 位置效應控制）。
6. **reason 文本簽章**：人工 heuristic 分類（entity-prior 語言 vs 證據引用語言），
   per-group 比例；descriptive、non-causal。
7. **C_c 分佈**：response contrast 的 quantile 與 rank 表（fallback 用）。
8. **generation 前綴一致率**：生成文字匹配 regex `^\s*\{\s*"decision"\s*:\s*"`
   （第一個 JSON key 為 decision、容忍空白格式化）的比例（pilot 判定依據）。

## 9. Run 結構

### 9.1 Formal run

`prepare → forward → analyze`（單一 run，`ArtifactRun` lifecycle，無 resume）。

| 階段 | 內容 | 產出 |
|---|---|---|
| prepare | population 載入＋fail-closed 檢查（§3.1）、hold-out 與 order-swap split、prompt 構築 5336 支、span 邊界、provenance | `prepare/*` |
| forward | 每 prompt 1 次 forward（greedy generation + decision-position margin，單過程）；完成後 20 筆 re-run determinism check | `forward/*`（5336 records） |
| analyze | parse、分組、gates、描述統計 | `analyze/summary.json`、`manifest.json` |

### 9.2 Pilot（formal 前，mandatory preflight）

`pilot`：50 家（seed 20260916、stratified by `gics_sector`）×（zero + N6 + P6 +
N15 + P15 = 5 條件）＋ anonymous 9 = **259 forwards**。目的：(a) 生成格式
（首 key 為 decision 的一致率 ≥ 95%，regex 定義見 §8 第 8 項）；(b) parse
率、符號一致率；(c) 跟隨 vs 固定組的初步比例（G-P4 預訊）；(d) 端到端 schema
驗證。pilot 輸出 `status=pilot`，不寫入正式分組表；pilot 通過後
方可 formal。

### 9.3 決策表

| 結果 | 解讀與後續 |
|---|---|
| G-P1..G-P4 全 pass | Phase 1 complete：分組表 frozen（503 家 × group/split）；開 Phase 2 協議（L15 組間對比，discovery split） |
| G-P4 fallback 觸發 | continuous C_c 主 estimand 分支；分組描述性；Phase 2 依 tertile 設計 |
| G-P1 或 G-P2 fail | 模板/前綴/parse 實作問題；診斷＋修復＋1-prompt real-model smoke＋新 run ID；不得以 sub-threshold 結果出組表 |
| G-P3 fail | 非確定性 bug（deterministic flag / bf16 配置）；修復後重跑 |

## 10. 版本分立觸發條件

任一變更即建立 `proposal-phase1-revN.md`（或新 phase 文件），禁止原地修改：

1. prompt 構造或 prompt family（模板、證據文字、條件矩陣、canonical 前綴）；
2. primary endpoint（generation vs margin 的取捨）或 estimand（分組規則、C_c 定義）；
3. gate 門檻或分組判據；
4. 控制組構造（anonymous、order-swap、hold-out split）。

## 11. 邊界與限制

- 無介入、無 causal claim；行為端點限於受測模板、greedy、英文。
- 證據為人寫 generic 文字（1 正 1 負）；百分比梯子 5/6/8/10/15% 沿用 trial plan
  設計；不宣稱該梯度的強度刻度線性。
- 分組定義於 ±15% 最強極性格；6/8/10% 格僅穩健性。
- 無 marketcap 欄位（constituents CSV 不提供）；sector 為唯一共變量。
- 分組為 discovery split 上的標籤；機制相（Phase 2/3）只用 discovery split 選
  座標，hold-out 20% 留 confirmation。
- 單模型（Qwen3.5-4B）。

## 12. 實作與驗證

- 新 package `llm_bias/evidence_insensitivity/`：`template.py`（frozen 證據文字、
  prompt builder、條件矩陣、span 邊界）、`population.py`（constituents 載入、
  split）、`screening.py`（generation + margin + parse）、`analysis.py`（分組、
  gates、描述統計）、`pipeline.py`（lifecycle）。
- operator：`scripts/evidence_insensitivity_phase1.py`（subcommands：
  `pilot` / `prepare` / `forward` / `analyze`）。
- 重用 `core/prompt_input`、`core/inference`（forward＋generation）、
  `core/analysis`、`core/artifacts`；不 import 其他 experiment package。
- 測試 `tests/test_evidence_insensitivity_phase1*.py`（fake model，無 GPU）：
  prompt builder digest 與 span 覆蓋斷言、條件矩陣完整性（5336 計數）、
  parse 邏輯（valid/invalid JSON、decision=null 路徑）、分組規則邊界值
  （4 組合 + parse-fail 排除）、gate 邊界值（0.95/0.90 門檻）、population
  loader（503 唯一 ticker fail-closed）、determinism check 記錄格式、
  summary schema 過 core 序列化器遞迴 guard、pipeline smoke lifecycle。
- 驗證命令：`uv run pytest -q`、`uv run python -m compileall -q llm_bias`、
  `uv build`、`uv lock --check`。

## 13. Revision history

- **Rev 1（2026-09-16）**：初稿。基於 2026-09-16 brainstorming（artifact
  `art_1789532306838`，用戶批准方向 A＋三項修正：共用 company-agnostic 證據
  控制、generation 為 primary endpoint、S&P 500 全人口；雙模型與匹配集延後）
  起草。待 pilot 與用戶批准後 frozen。
- **Rev 1.1（2026-09-16）**：第一次 pilot（`pilot-20260916T075257Z`，259
  forwards 完成）發現模型 greedy 輸出為 pretty-printed JSON（`{\n  "decision":`，
  非逐字 canonical 前綴）且在 JSON 物件後附加 stop marker 文字，致整串
  `json.loads` parse 率 0.0（G-P1 全 fail 為實作問題，非模板問題）。對齊修正
  （非版本 break：prompt family、endpoint、gate 門檻、控制組皆未變）：
  (a) parse 規則改為第一 `{` 起 `raw_decode`（§2 定義補明）；
  (b) pilot 判定 (a) 由「逐字前綴一致」改為「首 key 為 decision」regex
  一致率（§8 第 8 項、§9.2）；(c) canonical JSON 前綴重新定位為 margin 計分
  位置常數，非生成格式約束（§4 表）。pilot 重跑。
- **Rev 1.2（2026-09-16）**：第二次 pilot（`pilot-20260916T085540Z`）G-P1
  1.0、G-P3 pass、G-P2 0.869（34/259 筆 sell 決策 margin>0，無 buy 決策
  margin<0）：margin 計分位置 `prompt + '{"decision": "'` 不在模型實際 greedy
  軌跡上（模型先發 `{'、新行、空白才到決策 token），前綴 detour 後 stance 轉
  sell。修正：decision position 改定義為模型實際決策位置 `prompt +
  '{\n  "decision": "'`（pilot 259/259 前綴 byte-identical，含 anon arm；與
  entity_cell / jspace_intervention 既有 `DECISION_PREFIX` 慣例一致），§2 與
  §4 表同步更新。非版本 break：endpoint、分組規則、gate 門檻、控制組皆未變
  （margin 仍是 auxiliary，primary 仍是 generated decision）。pilot 重跑。
- **Rev 2（2026-09-16，frozen）**：第三次 pilot（`pilot-20260916T090440Z`，
  259 forwards）判定：G-P1 1.0、G-P2 0.981（剩 5 筆 near-tie，|M| ≤ 0.22
  nats）、G-P3 bit-exact（20 筆 re-run 0 mismatch）、生成格式 259/259 首 key
  為 decision；G-P4 在 50 家 pilot 規模為 3 responsive / 47 fixed-sell（預訊，
  formal 以 503 家重評）。pilot 行為預訊：50/50 zero-evidence sell、P15 僅
  3/50 翻 buy、全樣本 C_c > 0。用戶 2026-09-16 批准 formal run；協議 frozen，
  formal run ID `phase1-gpu-bf16-01`。Phase 2 設計輸入（用戶討論）：L15 狀態
  反應曲線（per-company stance 投影 × polarity 梯級）、offset vs gain 分解、
  行為組 × P15 狀態方向 2×2 交叉表；皆為 analyze 層描述，不新增 forward。
- **Rev 2.1（2026-09-16，cross-model 延伸）**：用戶指示主線統一使用
  Qwen＋Gemma 雙模型。本協議設計（prompt family、條件矩陣、分組規則、
  gates）不變，僅對象模型擴充：Gemma-4-E2B-it 以 50 家 probe
  （`cross-model-gemma4-e2b-01`，259 forwards）為 pilot 記錄，formal run
  `phase1-gpu-bf16-01`（503 家，同 seed 同 split）執行中。Llama 3.2
  1B/3B 對照見 [cross-model
  diagnostic](diagnostic-cross-model-probe.md)（不屬本線主體，僅 prompt-vs-model
  裁決用）。
