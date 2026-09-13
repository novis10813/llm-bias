# Technology identity-header span sensitivity: proposal

## 目的

這個 pilot 先回答一個行為層問題：在 evidence 與回答格式固定時，明確出現在
prompt header 的 ticker／company name 改變多少固定 Buy/Sell continuation margin。
它在新的 residual intervention 或 lens localization 前篩選值得追查的 identity spans。

主要分數為

\[
M=\log P(\text{buy})-\log P(\text{sell}),
\qquad
\Delta M_c=M_c-M_{original}.
\]

Workflow 不比較不同 top-1 tokens 的機率，也不把 Jacobian-lens readout 當成
causal effect。

## Conditions

每個 source prompt 產生七個 paired conditions：

| Condition | Header ticker | Header name |
|---|---|---|
| `original` | 原 ticker | 原公司名 |
| `anonymous_ticker` | `ANON` | 原公司名 |
| `anonymous_name` | 原 ticker | `Anonymous Company` |
| `anonymous_identity` | `ANON` | `Anonymous Company` |
| `same_sector_swap` | 同 split 的 Technology peer | 同一 peer 公司名 |
| `constructed_identity` | frozen constructed ticker | frozen constructed name |
| `name_form_control` | 原 ticker 的 ROT13 surface form | 原公司名的 ROT13 surface form |

Same-sector peer 只代表相同 CSV sector label；它沒有匹配 market cap、corpus exposure
或 business similarity。Constructed identities 只保證來自 workflow 的 frozen list，不代表模型
從未見過相同字串。Name-form control 保留字母大小寫、標點與字元長度，但不保證
model-token count 相同；artifact 會保存 source/replacement token counts 與差值。

## Mutation scope

V1 只修改兩行 canonical bracketed headers：

```text
Stock Ticker: [...]
Stock Name: [...]
```

Evidence body 完全不變。Workflow 會記錄原 ticker 與完整公司名在 header 外的 exact
occurrence counts。因此這個 pilot 測量的是「在既有 evidence 上額外顯示 identity
header 的邊際敏感度」，不是完整 anonymization 或 entity-only substitution。Evidence
中的簡稱、產品名或其他 alias 也可能保留 identity information。

如果 V1 找到 held-out 可重現的 header effect，下一步才建立 reviewed alias mapping，
執行完整 identity replacement，接著對穩定 spans 做 Jacobian sensitivity ranking 與
residual patching。

## Split 與統計

Workflow 使用既有 ticker-level split manifest。每個 condition 先在同一 prompt 內減去
`original` margin，再於 ticker 內平均所有 prompts。報告使用 equal-ticker mean、median、
ticker bootstrap 95% interval、ticker-mean sign-flip test，以及六個 non-original
conditions 的 Holm correction。Prompt rows 不視為獨立公司樣本。

Discovery、calibration 與 test 必須分開執行。不要根據 test 結果調整 condition、prompt
column 或 replacement identity。

## Technology discovery pilot

先使用一個 prompt family 的一個 occurrence，涵蓋 35 個 discovery tickers：

```bash
uv run span-sensitivity run \
  --input data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv \
  --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
  --model .cache/models/qwen3.5-4b \
  --sector Technology \
  --split discovery \
  --prompt-column prompt_with_context_attribute_0 \
  --max-records 35 \
  --max-seq-len 1024 \
  --seed 20260827 \
  --run-id tech-header-discovery-$(date -u +%Y%m%d)
```

輸出位於：

```text
artifacts/qwen3.5-4b/technology-header-span-sensitivity/runs/<run-id>/
├── prepare/prepared_prompts.jsonl
├── prepare/metadata.json
├── forward/margin_results.jsonl
├── forward/metadata.json
├── analysis/summary.json
└── manifest.json
```

Artifacts 只保存 prompts、token IDs/counts、candidate log probabilities、margins、paired
effects 與 provenance；不保存 activations、residuals、hidden states、gradients、Jacobians
或 KV caches。

## Go/no-go 判準

Discovery 用來選擇後續 span 與 prompt family，不提出 confirmatory claim。進入 causal
localization 前至少需要：

1. `anonymous_name`、`anonymous_ticker` 或 `same_sector_swap` 在 calibration 呈現同方向
   effect，且 ticker-bootstrap interval 不只是由少數公司驅動；
2. effect 大於 `name_form_control` 可解釋的 surface-form variation；
3. held-out test 的 condition、metric、prompt columns 與統計方法在 inference 前凍結；
4. 完整 identity replacement 另有 alias review，不能把 V1 header-only 結果改稱
   entity-only causal effect。

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**

- [jspace-sector-intervention](../jspace-sector-intervention/proposal.md)（資料／產物依賴）：共用 split manifest；以 header 表面替換建立行為基準，不依賴介入成功。

**後續**：尚無已立案的後續研究；不把報告中的建議視為已授權實驗。

最終／最新結果見本研究的 [report](report.md)。
