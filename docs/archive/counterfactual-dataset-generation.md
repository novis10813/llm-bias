# 8-K counterfactual entity dataset

**Status:** protocol and generation workflow implemented; validated dataset review
and promotion remain pending.

這份 workflow 把已清理的 8-K earnings events 轉成「內容與正確答案不變、只改
entity identity」的 span-patching pairs。實作位於
`llm_bias/counterfactual_data/`，CLI 是 `prepare-counterfactual-data`。

## 研究契約

V1 的 outcome 是 filing 文字所表達的語意正負向，不是事件後股價反應。所有 pair
共用同一份 context template 與 `expected_outcome`；entity 替換不得改變正確答案。
模型評分固定為：

```text
margin = logit(positive) - logit(negative)
```

因此 bias pair 不使用 factual pair 的 `source_answer → target_answer`
normalized transfer。Patching runner 另外輸出；完整的 patch execution、artifact
與 dashboard contract 見 [Counterfactual residual activation patching](counterfactual-patching.md)：

- `direct_entity_effect = target_margin - source_margin`
- `causal_patch_effect = patched_margin - source_margin`
- `control_patch_effect = control_margin - source_margin`
- 依 gold outcome 調整方向的 source、target、patched expected margin

CIK 是穩定 entity id。公司名稱使用 event date 當下可見的最近歷史名稱，禁止
lookahead，最大 staleness 為 550 天。V1 不要求 ticker，也不把目前名稱回填到過去。

## 資料來源

- 清理後事件：`artifacts/edgar_8k/cleaned/`
- 歷史 filing metadata：
  `../10-k/edgar-crawler/datasets/FILINGS_METADATA.csv`
- EDGAR quarterly indices：
  `../10-k/edgar-crawler/datasets/INDICES/`

`FILINGS_METADATA.csv` 提供 CIK、歷史公司名、filing date、form 與 SIC。INDICES
只做 provenance signature cross-check，不用來覆寫 metadata。

## 執行順序

```bash
# 1. 建立 point-in-time company history
uv run prepare-counterfactual-data entities

# 2. 固定 seed 的 500 筆 pilot
uv run prepare-counterfactual-data sample --count 500 --seed 20260730

# 3. 本地 llama.cpp + LangExtract；這是唯一需要 extraction extra 的階段
uv run --extra extraction prepare-counterfactual-data annotate \
  --model qwen3.5-mtp \
  --base-url http://127.0.0.1:11433/v1

# 4. 產生 200 筆人工審核材料
uv run prepare-counterfactual-data review-bundle --count 200

# 5. 編輯 review_template.jsonl 後，通過 gate 才能 promote
uv run prepare-counterfactual-data promote \
  --review artifacts/counterfactual_data/8k_earnings_v1/review/reviewed.jsonl

# 6. 建立四種 condition families、五種 pairing strategies 與雙向 pair
uv run prepare-counterfactual-data build-pairs

# 7. 針對每個實驗模型產生 tokenizer-specific spans
uv run prepare-counterfactual-data render \
  --model .cache/models/llama-3.2-1b-instruct
uv run prepare-counterfactual-data render \
  --model .cache/models/qwen3.5-4b

# 8. 驗證目前已有的 stage
uv run prepare-counterfactual-data validate
```

`annotate` 每完成一筆就原子性更新 draft JSONL，預設會 resume。先用
`--max-events 1` 做 endpoint smoke test；若要從頭重跑則加 `--no-resume`。

## Sampling

候選必須同時符合：

- `event_family=financial_results`
- `candidate_status=candidate`
- `announcement_only=false`
- `has_numeric_fact=true`
- `analysis_text` 介於 200–5,000 characters
- cleaner 找到 registrant mention

抽樣以 5-year bucket × SIC 2-digit strata round-robin，stratum 內用
SHA-256(seed, section_id) 排序；每個 source CIK 最多兩筆。SIC 是從
`filings.jsonl` join 進 section，不能把缺值當成產業類別。

## LangExtract annotation

本地模型為 OpenAI-compatible provider：

```text
model       qwen3.5-mtp
base URL    http://127.0.0.1:11433/v1
temperature 0
workers     1
seed        20260730
```

目前 annotator version 為 `langextract-hybrid-v2`。輸入切為最多 1,800
characters 的 chunks，單次 structured output 上限 2,048 tokens。Semantic
entity extraction 執行兩 passes；event facts 執行一 pass。

以下欄位由 metadata／regex 決定，不送給 LLM：

```text
registrant_name, registrant_alias, ticker, security_identifier,
exact_calendar_date, exact_amount, exact_share_count, exact_percentage
```

LangExtract 只抽取需要語意判斷的 classes：

```text
subsidiary, product, brand, business_segment, person, person_role,
counterparty, identifying_location
```

`rare_transaction_detail` 保留在 dataset taxonomy，但 hybrid-v2 不自動生成，需由
人工 review 補充。這避免一段交易敘述被模型展開成大量、難以稽核的 JSON。

JSON resolver 使用 `suppress_parse_errors=false`；任何 malformed/truncated JSON
會讓該 row 成為 `annotation_status=failed`，不能再偽裝成 complete。Active
extractions 的 `extraction_text` 必須逐字等於 `analysis_text[char_interval]`。
大小寫差異可直接 canonicalize 回原文；其他微小改寫只在 0.90 LCS fuzzy
alignment 能找到原文 span 時接受，並保留 `model_extraction_text` 與
`grounding_canonicalized` audit 欄位。無法 grounded 的 extraction 會讓該 row
失敗，後續 resume 會重試。Manifest 分別記錄 total、complete、failed counts；
review bundle 與 promotion 只接受 current annotator version 的 complete rows。

Event fact 必須是含 metric 與 direction 的 grounded span。程式只用固定 ontology
決定正負向：例如 revenue 上升為 positive、loss 上升為 negative。facts 混合、
方向不明或 metric 不在 ontology 時，`expected_outcome=null`，promotion 時排除。
LLM 不直接決定最終 label。

Promotion 只保留包含 event facts 的原文句子作為 `filing_excerpt`，依原始順序
組合並限制為 1,200 characters；這避免把整份 section 當 prompt。Render 階段
另強制 source/target 各不超過 512 tokens，超限會明確失敗，不會靜默 truncate。

## Templates 與 specificity holdout

主 `context_template`：

- registrant name/alias/ticker → `{ENTITY}`，全文恰好一個 slot
- 其他 entity → `[PRODUCT]`、`[PERSON]` 等 role marker
- 日期與精確數字 → `[DATE]`、`[VALUE]`
- 後續 registrant references → `the company`

`specificity_template` 使用同一份 entity redaction，但保留日期與數字，可用於純
數字抽取或 event classification holdout。它標記 `specificity_flag=true`；V1
不為它展開完整 condition families 與 pairing strategies。

## Review gate

所有 annotation 一律先寫成 `dataset_status=draft`。至少 200 筆 review JSONL
必須每列填入 reviewer 與 JSON boolean，且整體達到：

| 指標 | 門檻 |
|---|---:|
| registrant/ticker recall | 100% |
| other entity precision | ≥95% |
| other entity recall | ≥95% |
| grounded span rate | 100% |
| semantic outcome accuracy | ≥90% |
| identity leakage | 0 |

其他 entity 的 precision/recall 由每列
`other_entity_true_positive_count`、`other_entity_false_positive_count`、
`other_entity_false_negative_count` 聚合計算；其餘檢查使用 JSON boolean。
Review 可以提供 `corrected_entities`、`corrected_event_facts` 與
`corrected_expected_outcome`。任何缺欄、字串假布林、未達門檻或 template 仍含
identity 都會使 promotion 拒絕；未通過 gate 的資料不會生成 validated pairs。

## Pairing

每個 validated content 最多建立四種 condition families、五種 pairing strategies，並 materialize forward/reverse：

| Condition | Strategy | 用途 |
|---|---|---|
| real vs real | same industry, matched exposure | 測 reputation transfer |
| real vs real | cross-industry neutral/stress | 測 context 不合理時是否硬套記憶 |
| real vs anonymous | identity removal | no-entity baseline |
| real vs synthetic | memorized identity | 真實記憶與名稱形式 |
| synthetic vs synthetic | name-form baseline | tokenizer/name-form control |

同產業先找 SIC 4-digit，沒有才 fallback SIC 2-digit。這裡的「matched」不是市值：
V1 使用 event date 以前的 total filing、8-K、trailing-3-year 8-K 與上市歷史長度
計算 log-distance，欄位名稱固定為 `matched_exposure`，避免誤稱 size match。
每個 target CIK 最多使用五次。

Cross-industry stress 只在高精度 cue（例如 same-store sales、credit loss、FFO、
production、subscribers）存在時標記；其餘是 neutral。無可用 target 時不偷偷
放寬限制，而是寫入 `pair_omissions.jsonl`。

Synthetic names 由 content id 的 deterministic hash 組合音節，並對本地歷史公司
名稱與 validated pilot excerpts 做 collision scan。這只表示「未在本地資料撞
名」，不能宣稱模型從未見過該字串。

## Artifacts

預設根目錄為 `artifacts/counterfactual_data/8k_earnings_v1/`：

```text
company_history.jsonl
entities_manifest.json
sampled_events.jsonl
sample_manifest.json
draft_annotations.jsonl
annotation_manifest.json
review/review_template.jsonl
review/review_bundle.html
validated_content.jsonl
promotion_rejections.jsonl
pairs_unrendered.jsonl
pair_omissions.jsonl
rendered/<model>/pairs.jsonl
validation_report.json
```

Artifacts 被 git ignore；程式、測試、文件與 lockfile 才納入 repository。

## 2026-07-30 initial run

| Stage | Result |
|---|---:|
| CIK histories | 2,732 |
| point-in-time metadata observations | 266,549 |
| quarterly index files checked | 132 |
| index 8-K rows / matching local metadata signatures | 2,126,939 / 219,233 |
| eligible earnings events | 1,501 |
| fixed-seed pilot sample | 500 |
| hybrid-v2 one-row smoke test | passed, zero warnings |
| hybrid-v2 ten-row smoke test | 9 complete, 1 explicit grounding failure |

INDICES 涵蓋的 EDGAR universe 大於本地 crawler metadata，因此 10.3% signature
match 不能解讀為 crawler error rate；它只證明本地 metadata rows 可回查至 raw
quarterly indices。

舊 hybrid-v1 run 在 421 rows 時累積 74 次 suppressed JSON parse warnings，原因是
把大量日期、金額與百分比也交給 LangExtract，輸出約在 12–15KB 被截斷；所有
rows 卻仍被標成 complete。該 run 已停止並封存，不能用於 promotion。新版進度
可用：

```bash
wc -l artifacts/counterfactual_data/8k_earnings_v1/draft_annotations.jsonl
tail -f artifacts/counterfactual_data/8k_earnings_v1/logs/annotation.log
```
