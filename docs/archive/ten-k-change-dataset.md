# 10-K metadata-change window CSV

`prepare-10k-change-data` 從已處理的 EDGAR 10-K JSON 建立一份可自行組裝 prompt 的 CSV。它不固定問題模板、不載入或呼叫模型，也不推論答案。

輸入預設為：

```text
../10-k/edgar-crawler/datasets/EXTRACTED_FILINGS/10-K
```

## 資料範圍

流程先以 `CIK + period_of_report` 去重：選擇較晚的 `filing_date`，若仍相同則以檔名決定 canonical filing。接著在同一 CIK 內依 `period_of_report` 比較相鄰 canonical observations。

只有下列欄位的前後值皆非空且不同時才建立變動事件：

```text
company, state_location, state_of_inc, sic
```

每個事件只保留事件年度前後兩個 fiscal years 內的實際 canonical filings。`fiscal year` 取自 `period_of_report`；`filing_date` 不參與年度窗口判定。缺少某一年時不合成資料列。

## 唯一 CSV

輸出目錄中的唯一 CSV 是 `change_window_items.csv`，header 固定為：

```text
year,cik,item
```

- `year`：該窗口 filing 的 fiscal year。
- `cik`：該 filing 的 CIK。
- `item`：這個事件中有改動的欄位，以及該年份 filing 記錄的實際值，格式為 `field=value`。

例如：

```csv
year,cik,item
2019,320193,company=ACME INC
2019,320193,state_location=CA
2020,320193,company=NEW ACME INC
2020,320193,state_location=NY
```

同一 filing 若位於多個變動事件的窗口、或同一事件改動多個欄位，可能出現重複或同年多列；這些列保留事件窗口範圍。CSV 不保存 prompt，後續實驗可自行以 `year`、`cik` 和 `item` 組裝問題。

## 稽核檔案

除唯一 CSV 外，artifact 包含不供 prompt 使用的稽核檔案：

- `change_events.jsonl`：CIK、變動前後值、窗口 filings、缺少年與來源 provenance。
- `canonical_exclusions.jsonl`：相同 CIK/period 未被選中的 duplicate filing。
- `input_issues.jsonl`：空檔、無效 JSON 或 schema 問題的來源。
- `manifest.json` 和 `validation_report.json`：schema、hashes、counts 與驗證結果。

有效來源仍會在其他來源有問題時發布；manifest status 會是 `complete_with_input_issues`。`--fail-on-input-issues` 會在 artifact 發布後以非零狀態結束。若沒有任何有效來源，流程 fail closed。

## 執行

```bash
uv run prepare-10k-change-data build \
  --input ../10-k/edgar-crawler/datasets/EXTRACTED_FILINGS/10-K \
  --output artifacts/ten_k_change_windows/v1
uv run prepare-10k-change-data validate \
  --input artifacts/ten_k_change_windows/v1
```

## 10-K metadata QA generation

`generate` 讀取已驗證的 `change_window_items.csv`，將 `item=field=value` 的欄位名稱映射為：

- `company` → `company name`
- `state_location` → `state/location`
- `state_of_inc` → `state of incorporation`
- `sic` → `SIC code`

每列使用固定 prompt：

```text
In year {year}, what is the {item_name} of the company with CIK code {cik}? Answer without explanation
```

這個階段不將 CSV 中的答案值拼入問題；原始 `item` 與解析後的值會保留在 generated-output record 中供稽核。使用本地 Qwen 3.5 4B：

```bash
uv run prepare-10k-change-data generate \
  --input artifacts/ten_k_change_windows/v1/change_window_items.csv \
  --model .cache/models/qwen3.5-4b \
  --output artifacts/qwen3.5-4b/ten-k-change-windows-v1/runs/001/forward \
  --max-new-tokens 16 \
  --max-seq-len 256 \
  --temperature 0
```

輸出為 `forward/generated_outputs.jsonl` 與 `forward/metadata.json`；metadata 會記錄 source CSV hash、model、prompt template、mapping version 與 row count。

### Schema-constrained generation

若本機有 OpenAI-compatible llama.cpp server，可使用 `generate-structured`。它透過
`response_format=json_schema` 強制模型輸出唯一的 `answer` 字串欄位，不依賴自然語言中的
「不要解釋」指示：

```bash
uv run prepare-10k-change-data generate-structured \
  --input artifacts/ten_k_change_windows/v1/change_window_items.csv \
  --output artifacts/qwen3.5-9b-mtp/ten-k-change-windows-v1/runs/003 \
  --base-url http://127.0.0.1:11433/v1 \
  --model qwen3.5-9b-mtp \
  --max-tokens 64
```

輸出為 `structured_answers.jsonl` 與 `metadata.json`。每筆 record 同時保存
`generated_text`、解析後的 `parsed_answer`、`parse_status`、`finish_reason` 與來源 row metadata。
