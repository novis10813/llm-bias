# EDGAR 8-K 清理與事件候選資料

本文件說明 `prepare-edgar-8k` 的資料來源、清理規則、輸出 schema，以及
2026-07-30 全量執行的結果。這個 workflow 的目的，是把 edgar-crawler 已抽取的
8-K JSON 轉成可稽核、可重現、適合後續 event-frame 與 counterfactual pair
建構的 staging corpus。

## 範圍與資料流

輸入預設為：

```text
../10-k/edgar-crawler/datasets/EXTRACTED_FILINGS/8-K
```

程式只讀取該目錄的 JSON，不修改 edgar-crawler、不讀取其 Python 程式、不連網，
也不補抓 filing 或 exhibits。處理流程為：

```text
extracted filing JSON
  → filing metadata validation
  → non-empty item sections
  → normalized_text
  → analysis_text + removed spans
  → SEC item taxonomy
  → entity mentions and quality signals
  → JSONL corpus + manifest + quality report
```

實作位於：

- `llm_bias/edgar_preparation/pipeline.py`：streaming cleaner、schema 與 validator。
- `llm_bias/edgar_preparation/taxonomy.py`：新舊 8-K item taxonomy。
- `llm_bias/edgar_preparation/cli.py`：`clean` 與 `validate` CLI。

## 執行方式

全量清理：

```bash
uv run prepare-edgar-8k clean \
  --input ../10-k/edgar-crawler/datasets/EXTRACTED_FILINGS/8-K \
  --output artifacts/edgar_8k/cleaned
```

小型 smoke run：

```bash
uv run prepare-edgar-8k clean \
  --input ../10-k/edgar-crawler/datasets/EXTRACTED_FILINGS/8-K \
  --output artifacts/edgar_8k/smoke_100 \
  --max-files 100
```

驗證正式產物：

```bash
uv run prepare-edgar-8k validate \
  --input artifacts/edgar_8k/cleaned
```

為避免把部分完成的 corpus 當成正式資料，清理時會先寫 sibling temporary
directory，全部成功後才原子性發布。若 output directory 已存在，程式會拒絕
覆寫。

## 文字清理

每個非空 section 同時保存兩種文字：

- `normalized_text`：HTML entity decode、Unicode NFKC、換行、控制字元與空白
  正規化；不修改大小寫、數字或語意。
- `analysis_text`：從 `normalized_text` 移除 item heading、純 exhibit reference、
  `deemed filed` 與 incorporation-by-reference 等 legal boilerplate。

每個移除區段都以 `kind`、`start`、`end` 記錄，offset 指向
`normalized_text`。單一移除區段上限為 2,500 characters，避免 regex 跨越長篇
實質內容。全量 sanity audit 中，實際最大移除區段為 1,636 characters。

上游 edgar-crawler 可能已移除數字表格，而且多數 extracted filings 不包含
Exhibit 99 等附件。因此：

- `analysis_text` 缺少數字，不代表原始 submission 沒有數字。
- `announcement_only=true` 常表示 section 只說「發布財報，詳見 Exhibit」。
- 這一階段不生成摘要，也不推論 filing 沒有直接提供的資訊。

## Event family 判定

`event_family` 只由 SEC item code 決定，不是文字分類或 LLM 推論。JSON key
`item_2.02` 會轉成 item code `2.02`；有小數點的 code 使用新版 taxonomy，整數
code 使用 2004 年以前的 legacy taxonomy。

| Event family | 新版 item | Legacy item |
|---|---|---|
| `material_agreement` | 1.01, 1.02 | — |
| `bankruptcy` | 1.03 | 3 |
| `mine_safety` | 1.04 | — |
| `cybersecurity` | 1.05 | — |
| `acquisition_disposition` | 2.01 | 2 |
| `financial_results` | 2.02 | 12 |
| `financing` | 2.03, 2.04 | — |
| `restructuring` | 2.05 | — |
| `impairment` | 2.06 | — |
| `listing_status` | 3.01 | — |
| `equity_issuance` | 3.02 | — |
| `security_holder_rights` | 3.03 | — |
| `auditor_change` | 4.01 | 4 |
| `financial_restatement` | 4.02 | — |
| `control_change` | 5.01 | 1 |
| `management_change` | 5.02 | 6 |
| `governance_change` | 5.03 | — |
| `benefit_plan_trading` | 5.04 | 11 |
| `code_of_ethics` | 5.05 | 10 |
| `shell_company_status` | 5.06 | — |
| `shareholder_vote` | 5.07 | — |
| `shareholder_nomination` | 5.08 | — |
| `asset_backed_securities` | 6.01–6.05 | — |
| `regulation_fd` | 7.01 | 9 |
| `other_event` | 8.01 | 5 |
| `fiscal_year_change` | — | 8 |
| `supporting_material` | 9.01 | 7 |

`sections_by_event_family` 是所有非空 sections 的加總，不只計算
`candidate_status=candidate`。例如全量結果的 `financial_results=61,221`：

```text
item_2.02  55,789
legacy 12   5,432
total      61,221
```

## Candidate status 與品質訊號

狀態判定依下列優先順序：

1. 清理後只有 `Not applicable`、`None` 或 `N/A`：
   `candidate_status=not_applicable`。
2. 新版 9.01 或 legacy 7：
   `candidate_status=supporting_only`。
3. `analysis_text` 少於 80 個 non-space characters，或少於 12 個英文詞：
   `candidate_status=insufficient_content`。
4. 其餘：
   `candidate_status=candidate`。

另外保存以下獨立訊號，供後續篩選：

- `has_numeric_fact`
- `has_currency`
- `has_percent`
- `has_exhibit_reference`
- `announcement_only`
- `analysis_char_count`
- `analysis_alpha_token_count`

`announcement_only` 不會自動排除 section。它表示文字同時提到 press release
與 exhibit，但沒有辨識到數字事實或比較方向；後續 event-frame extraction 應把
它當成低 evidence-strength 樣本。

## Entity mention 範圍

這一版不是 general NER，只標註能從 filing metadata 確認的 registrant references：

- legal company name
- 移除 `/DE/` 等州標記後的名稱
- 移除 `Inc.`, `Corp.`, `Ltd.` 等後綴的 short name
- `the Company`、`the Registrant`、`Registrant`

spans 使用 longest-match、non-overlapping 規則，offset 指向
`normalized_text`。Ticker、產品、子公司、交易對手與高層姓名仍需後續
entity-resolution；schema 以
`entity_annotation_scope=registrant_aliases_only` 明確標示此限制。

## 輸出 schema

### `filings.jsonl`

每份 filing 一列，主要欄位包括：

- `filing_id`, `accession`, `source_file`, `source_sha256`
- `cik`, `company`, `filing_type`
- `filing_date`, `period_of_report`, `sic`
- SEC source links 與州／會計年度 metadata
- `all_item_codes`, `nonempty_item_codes`
- `section_count`, `candidate_count`, `quality_flags`

### `sections.jsonl`

每個非空 item section 一列，主要欄位包括：

- `section_id`, `filing_id`, `item_code`, `item_schema`
- `item_title`, `event_family`, `taxonomy_version`
- `candidate_status`, `rejection_reasons`
- `normalized_text`, `analysis_text`, `removed_blocks`
- `entity_mentions`, `entity_annotation_scope`
- 長度、數字、貨幣、百分比、exhibit 與 announcement-only 訊號

### `manifest.json`

記錄 schema/taxonomy/cleaner version、CLI 參數、來源 fingerprint、輸出 hashes、
筆數、thresholds、執行時間和上游資料限制。

### `quality_report.json`

記錄年份、item code、event family、candidate status、quality reason、CIK 與公司
名稱數量的 aggregate statistics。

## 2026-07-30 全量結果

| 指標 | 數量 |
|---|---:|
| Filings | 217,800 |
| Non-empty sections | 433,423 |
| Event candidates | 225,612 |
| Supporting-only sections | 165,724 |
| Insufficient-content sections | 39,361 |
| Not-applicable sections | 2,726 |
| Announcement-only flags | 91,317 |
| Unique CIKs | 1,703 |
| Unique company strings | 2,377 |

主要 event families：

| Event family | Sections |
|---|---:|
| `supporting_material` | 165,857 |
| `other_event` | 66,726 |
| `financial_results` | 61,221 |
| `regulation_fd` | 41,081 |
| `management_change` | 30,282 |
| `material_agreement` | 29,575 |
| `financing` | 8,156 |
| `shareholder_vote` | 6,035 |
| `governance_change` | 5,734 |
| `acquisition_disposition` | 5,554 |

正式資料位於 `artifacts/edgar_8k/cleaned/`，約 1.7 GB。`artifacts/` 被 Git
忽略，因此 repository 只保存程式、測試與本文件，不提交大型 corpus。

全量產物已通過：

```text
schema version     edgar-8k-clean-v1
filings            217800
sections           433423
candidates         225612
validator result   valid=true
```

## 後續使用邊界

這批資料適合作為：

- event-frame extraction 的輸入。
- event type、數字與 entity mention 的 candidate corpus。
- counterfactual template 建構前的 auditable evidence layer。
- 依年份、item、event family 與品質訊號進行分層抽樣。

目前尚不應直接視為：

- 完整的 Exhibit 99／earnings release corpus。
- 已驗證的金融情緒或 abnormal-return dataset。
- 已移除所有 entity-specific terms 的 counterfactual template。
- source/target `Pair` 或 activation-patching input。
