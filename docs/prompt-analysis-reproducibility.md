# Prompt-analysis 實驗重現指南

這份文件說明如何以任意相容 decoder model，從同一份 prompt CSV 重跑
forward-only 的 Jacobian-lens readout 與 Temperature Scope 不確定性資料；只有
明確啟用 backpropagation stage 時，才會額外產生 generated-token attribution，並
可進一步執行 validation、圖表與互動式 dashboard。

整體分成三個獨立入口：

- `fit-jacobian-lens`：建立可重用的 lens artifact。
- `prompt-analysis`：執行 readout、optional attribution、validation 與 visualization。
- `scripts/run_prompt_analysis.sh`：只 orchestration prompt experiment，不會 fitting
  lens；`RUN_ATTRIBUTION=0` 是預設，只有 `RUN_ATTRIBUTION=1` 才會啟動 attribution
  backpropagation。

以下範例使用 Qwen 3.5-4B，但 Python API 與 CLI 名稱均不依賴特定模型。

## 前置條件

```bash
uv sync
test -d third_party/jacobian-lens
test -d third_party/jspace-viz
test -d .cache/models/qwen3.5-4b
test -f sp500_r1k_r2k_entityBiasPrompt.csv
```

## 1. 準備 model-specific canonical Jacobian lens

Canonical lens 必須保留每個 intermediate layer。Qwen3.5-4B 已使用 English-only、
Simplified-Chinese-only 與 mixed 各 128 題完成 controlled selection；完整設計、
重跑方式與結果見
[Qwen3.5-4B Jacobian-lens calibration 與候選選擇](qwen-jacobian-lens-selection.md)。

一般新模型可以先使用 standalone fitter 建立 experimental lens，但不要用小型
smoke calibration 直接覆寫已有的 validated canonical artifact：

```bash
uv run fit-jacobian-lens \
  --model /path/to/new-model \
  --output artifacts/candidate_lenses/new-model/smoke/jacobian_lens.pt \
  --calibration-prompts 16 \
  --dim-batch 16 \
  --max-seq-len 128
```

Qwen production workflow 使用：

```bash
bash scripts/run_qwen_lens_candidates.sh
```

選出的 active artifact 為：

```text
artifacts/lenses/qwen3.5-4b/
├── jacobian_lens.pt
├── jacobian_lens.pt.metadata.json
└── selection.json
```

Metadata 記錄模型、hidden width、layers、fitting 參數、calibration digest 與
jlens version。Checkpoint 與 partial/experimental lens 不放在 active model lens
folder；若 fitting 中斷，以完全相同的 model/output 與 calibration 重跑即可
繼續。

也可以使用外部 calibration prompts。純文字格式為每行一個 prompt；JSONL
則預設讀取 `text` 欄位：

```bash
uv run fit-jacobian-lens \
  --model /path/to/model \
  --calibration-file calibration.jsonl \
  --calibration-field text \
  --calibration-prompts 32 \
  --output artifacts/candidate_lenses/my-model/experiment/jacobian_lens.pt
```

## 2. 執行 prompt-analysis

預設 shell 設定為：

```text
MODEL=.cache/models/qwen3.5-4b
LENS=artifacts/lenses/qwen3.5-4b/jacobian_lens.pt
RUN_ROOT=artifacts/prompt_analysis/qwen3.5-4b
```

Lens 必須已存在；runner 不會自動 fitting。Runner 預設為 forward-only，
`RUN_ATTRIBUTION=0` 不會執行生成或 backpropagation：

```bash
RUN_ATTRIBUTION=0 bash scripts/run_prompt_analysis.sh
```

也可以省略變數，因為 `RUN_ATTRIBUTION=0` 是預設值：

```bash
bash scripts/run_prompt_analysis.sh
```

若需要 generated-token attribution，必須明確啟用 attribution backpropagation
stage；這不是 readout 的 fallback：

```bash
RUN_ATTRIBUTION=1 bash scripts/run_prompt_analysis.sh
```

預設建立 detached tmux session：

```bash
tmux attach -t prompt_analysis
tail -f artifacts/prompt_analysis/qwen3.5-4b/run.log
```

若要在目前終端執行：

```bash
RUN_IN_TMUX=0 bash scripts/run_prompt_analysis.sh
```

使用其他模型或獨立 run：

```bash
MODEL=/path/to/model \
LENS=artifacts/lenses/my-model/jacobian_lens.pt \
RUN_ROOT=artifacts/prompt_analysis/my-model/repro-001 \
SESSION=prompt_analysis_repro_001 \
bash scripts/run_prompt_analysis.sh
```

Runner 執行一個 forward stage，以及一個預設關閉的 optional attribution stage：

1. `RUN_READOUT=1` 時，對六個 `with/without × index` prompt columns 做逐日期、逐層
   Jacobian Lens readout，產生 output top-k 與 Temperature Scope uncertainty；此
   stage 不做 generation backpropagation。
2. `RUN_ATTRIBUTION=1` 時，才對每個 condition 均勻抽樣日期，生成 output tokens，並
   以 `--backprop` 計算 generated-token attribution。`RUN_ATTRIBUTION=0` 時刻意跳過
   整個 stage，不會留下空的或 placeholder attribution artifact。

條件式輸出結構如下：

```text
${RUN_ROOT}/
├── per_date/                         # RUN_READOUT=1 時產生
│   ├── prompt_layer_uncertainty.jsonl
│   ├── prompt_layer_topk.jsonl
│   └── metadata.json                 # 記錄 backpropagation=false
├── generated_attribution/            # 僅 RUN_ATTRIBUTION=1 時產生
│   ├── generated_token_attribution.jsonl
│   └── metadata.json
└── run.log
```

因此 forward-only run 的 artifact contract 只有 `per_date/` readout 與 uncertainty
結果；它刻意不產生 `generated_attribution/`。需要 generated-token attribution 的
validation 或 visualization 必須改用 `RUN_ATTRIBUTION=1` 產生的完整 artifact，不新增
fallback 或以 forward-only 結果冒充 attribution。

### Runner 環境變數

| 變數 | 預設值 |
|---|---|
| `MODEL` | `.cache/models/qwen3.5-4b` |
| `LENS` | `artifacts/lenses/qwen3.5-4b/jacobian_lens.pt` |
| `INPUT_CSV` | `sp500_r1k_r2k_entityBiasPrompt.csv` |
| `RUN_ROOT` | `artifacts/prompt_analysis/qwen3.5-4b` |
| `READOUT_BATCH_SIZE` | `32` |
| `READOUT_MAX_SEQ_LEN` | legacy/auto：`256`；明確設定 `DATASET_FORMAT=return-pairs`：`512` |
| `TOP_K` | `15` |
| `ATTR_SAMPLE_PER_CONDITION` | `32` |
| `ATTR_MAX_NEW_TOKENS` | `64` |
| `ATTR_RUNS` | `1` |
| `ATTR_TEMPERATURE` | `0`（greedy） |
| `ATTR_SEED` | empty（不固定 seed） |
| `ATTR_TOP_P` | `1.0` |
| `ATTR_TOP_K` | `0` |
| `ATTR_OUTPUT_DIR` | `${RUN_ROOT}/generated_attribution` |
| `RUN_READOUT` | `1` |
| `RUN_ATTRIBUTION` | `0`；設為 `1` 才執行 generated-token attribution backpropagation |
| `SESSION` | `prompt_analysis` |
| `RUN_IN_TMUX` | `1` |

Return-pair 的 system instruction 與 user question 合併後通常超過 256 tokens；runner 在明確設定
`DATASET_FORMAT=return-pairs` 時使用 512，避免右側 truncation 移除 user question 或 assistant
generation marker。Legacy/auto workflow 維持原本的 256 預設。

只有 `RUN_ATTRIBUTION=1` 時，`ATTR_RUNS`、`ATTR_TEMPERATURE`、`ATTR_SEED`、`ATTR_TOP_P`
與 `ATTR_TOP_K` 才會控制 generated-token stage。預設 `ATTR_RUNS=1` 且
`ATTR_TEMPERATURE=0` 使用 deterministic greedy generation。若要在同一批 shared dates
上重複 sampling，可設定 `ATTR_RUNS=30`、`ATTR_TEMPERATURE=0.7`、固定 `ATTR_SEED`，並
把 `ATTR_OUTPUT_DIR` 指向新的目錄。`runs > 1` 會建立 `run_000/` 到 `run_029/`，每個 run
各自保存 JSONL 與 metadata，避免覆蓋既有結果；不要把多個 run 的 raw JSONL 直接交給只
接受單一 run 的 visualizer。

`ATTR_TEMPERATURE` 是 generation sampling temperature；它與 readout 的 uncertainty
中 `effective_temperature` 不同。後者由 final-normalized residual 的 L2 norm 推導，
不是 generation 的 sampling 設定。

## 3. Attribution validation（需要 attribution-enabled artifact）

這個 validation 只接受 `RUN_ATTRIBUTION=1` 產生的
`generated_token_attribution.jsonl`。Forward-only run 刻意沒有 generated
attribution artifact，因此不能執行此命令；不要新增 fallback，也不要把 readout 或
uncertainty artifact 當成 attribution 輸入。

```bash
uv run prompt-analysis validate-attribution \
  --model .cache/models/qwen3.5-4b \
  --attribution artifacts/prompt_analysis/qwen3.5-4b/generated_attribution/generated_token_attribution.jsonl \
  --output-dir artifacts/prompt_analysis/qwen3.5-4b/attribution_validation
```

## 4. 建立視覺化（需要完整 attribution-enabled run）

`visualize_prompt_analysis.sh` 需要同一個 run 的 per-date uncertainty 與 generated
attribution；forward-only artifact 可以用第 6 節的 uncertainty distribution 命令，
但不能建立 attribution dashboard。缺少 generated attribution 時，visualizer 應直接
失敗，不使用 fallback。

```bash
bash scripts/visualize_prompt_analysis.sh
```

自訂 run：

```bash
RUN_ROOT=artifacts/prompt_analysis/my-model/repro-001 \
TOKENIZER=/path/to/model \
bash scripts/visualize_prompt_analysis.sh
```

Visualizer 尋找：

1. `${RUN_ROOT}/per_date/prompt_layer_uncertainty.jsonl`（必要）
2. `${RUN_ROOT}/generated_attribution/generated_token_attribution.jsonl`（必要，必須來自
   attribution-enabled run）
3. optional `${RUN_ROOT}/attribution_validation/semantic_scope_aopc.jsonl`

輸出為：

```text
${RUN_ROOT}/visualization/
├── final_layer_effective_temperature_with_context.png
├── final_layer_effective_temperature_without_context.png
├── final_layer_entropy_with_context.png
├── final_layer_entropy_without_context.png
├── final_layer_uncertainty.csv
├── final_layer_entropy.csv
└── attribution_dashboard.html
```

直接在瀏覽器開啟 `attribution_dashboard.html` 即可。

## 5. Multi-run price distribution 研究圖

完整 multi-run sampling artifact 可以直接輸出三個 index 的 close-price 研究圖：

```bash
uv run prompt-analysis plot-price-distributions \
  --sampling-root artifacts/prompt_analysis/qwen3.5-4b/sp500_uncertainty/generated_attribution_sampling_t0.7_r30 \
  --prices sp500_r1k_r2k_entityBiasPrompt.csv \
  --output-dir artifacts/prompt_analysis/qwen3.5-4b/sp500_uncertainty/generated_attribution_sampling_t0.7_r30/price_distribution
```

命令會先驗證 `manifest.json`、所有宣告的 `run_NNN/` 目錄、每個 run 的 record
數量，以及每個 date/index/context condition 是否完整。Incomplete artifact 不會被靜默當成
完整實驗。

每個市場輸出一張 300-DPI PNG：上方兩個共享 price scale 的 panels 分別顯示 without
context 與 with context，包含 actual close、LLM median、25–75% band 與 5–95% band；
下方 error panel 比較兩個 conditions 每日期的 median absolute percentage error（MdAPE）：

```text
abs(generated price - actual close) / actual close × 100
```

輸出為：

```text
${OUTPUT_DIR}/
├── sp500_price_distribution.png
├── russell1000_price_distribution.png
├── russell2000_price_distribution.png
├── price_distribution_samples.csv
├── price_distribution_summary.csv
└── price_distribution_metadata.json
```

`price_distribution_samples.csv` 保留所有 run 的 normalized records，包括 raw generated
text、parse status、actual close 與 sample-level error。只有 finite numeric `answer` 進入
quantile 與 MdAPE；`null`、string、malformed 或 non-finite answers 不會被轉成 0。
`price_distribution_metadata.json` 保存輸入 SHA-256、generation config、quantile method、
error formula 與 valid/invalid counts。圖中的 bands 是 valid generated prices 的中央 90% 與
50%，描述的是 sampling variability 與 prediction error，不是 causal effect。

## 6. Final-layer uncertainty distribution 研究圖

既有 readout artifact 已保存每個日期與 condition 的 final-layer uncertainty，因此可以直接
建立跨日期分布，不需重新載入 model/lens 或執行 inference：

```bash
uv run prompt-analysis plot-uncertainty-distributions \
  --uncertainty-root artifacts/qwen3.5-4b/qwen3.5_temperature_scope_per_date \
  --output-dir artifacts/qwen3.5-4b/qwen3.5_uncertainty_distribution
```

Raw distribution 使用 ECDF，比較每個市場的 with-context 與 without-context：

- `entropy_nats`：final-layer full-vocabulary softmax 的 Shannon entropy。
- `effective_temperature`：final-normalized residual L2 norm 的倒數；不是 generation
  sampling temperature。

Paired distribution 會在每個市場內，以同日期配對後計算：

```text
with context − without context
```

Russell 1000 若有缺少 with-context 的日期，只會從 Russell 1000 的 paired set 排除，
不影響另外兩個市場。Paired differences 是 descriptive associations，不是 causal effects。

輸出為：

```text
${OUTPUT_DIR}/
├── final_layer_entropy_raw_ecdf.png
├── final_layer_entropy_paired_delta_violin.png
├── final_layer_effective_temperature_raw_ecdf.png
├── final_layer_effective_temperature_paired_delta_violin.png
├── final_layer_uncertainty_distribution_raw.csv
├── final_layer_uncertainty_paired_delta.csv
├── final_layer_uncertainty_distribution_summary.csv
└── final_layer_uncertainty_distribution_metadata.json
```

Metadata 保存 source SHA-256、condition/date counts、各市場 paired 與 unmatched counts、
quantile method、metric definitions 與 non-causal interpretation。Entropy 與 effective
temperature 使用不同 figures，不使用 dual axis。

## Artifact 與研究限制

- 保存的是 compact readout/attribution 結果：top-k、rank、統計量、token IDs/text、
  probabilities、generation 設定與 provenance；不保存完整 raw residual、embedding 或
  gradient activation。需要重新分析時，必須重新執行對應 stage，不可假設有 raw activation
  可回填。
- Readout 的 aggregate 必須先對每個 condition 的完整 vocabulary softmax 做平均，再
  選 top-k。Temperature Scope uncertainty 是 residual-space 的 readout measure；
  `ATTR_TEMPERATURE` 則是 generation sampling 設定，兩者不可互換或解讀成同一量。
- Generated-token attribution 是局部的一階 gradient sensitivity/semantic-scope
  readout，不是 attention map、chain-of-thought、離散 reasoning path，也不是
  standalone causal proof。Attribution validation 的 ablation 結果只能提供額外的
  validation evidence，不能把單次 attribution 直接宣稱為一般化 causal effect。
- 因此 forward-only 與 attribution-enabled run 必須以 artifact provenance、metadata
  的 backpropagation 設定及各自的前置條件分開保存與解讀；不以缺少 attribution 的
  forward-only 結果做 fallback。

## Legacy MAG7/S&P500 compatibility checklist

在替換或擴充 prompt-analysis workflow 前，請確認下列既有 contract 仍成立：

- **CSV schema**：`Date` 與名稱完全符合
  `prompt_with_context_<ticker>` / `prompt_without_context_<ticker>` 的欄位會被載入；
  不符合 pattern 的額外欄位可保留但不會被當成 condition。BOM、quoted multiline
  prompt、空 prompt 與任意 ticker 名稱都必須可讀；空白或空 prompt 會被跳過並在
  `condition_counts` 記錄 `skipped_empty_prompts`。
- **Date sampling**：generated attribution 的自動 sampling 先取所有 condition
  都有非空 prompt 的共同 `Date`，再以 deterministic spread 選日期；每個 condition
  使用同一組日期，不可各自抽樣造成 paired conditions 日期不一致。明確傳入
  `--date` 時則只使用指定日期。
- **Prompt assembly**：預設使用 tokenizer 的 user-only chat template，且不啟用
  thinking；`readout --raw-prompt` 才改用原始 prompt。attribution 的 scope 仍是
  chat-formatted prompt 中的 raw user message tokens。
- **Legacy artifacts**：readout 的核心檔名仍為
  `prompt_layer_topk.jsonl`、`prompt_layer_uncertainty.jsonl`、
  `average_layer_topk.jsonl`、`average_layer_topk.csv` 與 `metadata.json`；
  multi-run attribution 仍使用 `manifest.json`、`run_NNN/` 與每個 run 的
  `generated_token_attribution.jsonl`。JSONL/manifest 的既有欄位（日期、condition、
  `run_index`、`sample_index`、generation config、condition counts 與 record counts）
  必須保留。
- **Price answer parser**：只接受 JSON object 中 finite numeric `answer`；整數與
  浮點數都會轉成 numeric price。`confidence` 是 optional numeric metadata，不會把
  有效 answer 變成 invalid；null、字串、布林、NaN、malformed JSON 與缺少 answer
  必須標為 invalid，且不可靜默轉成 0。
- **CLI defaults**：現有 `prompt-analysis readout`、`attribute` 與 visualization
  的 input/output、sampling、generation、chat-template 與 token-length defaults
  必須維持。不要在本 checklist 宣稱尚未落地的命令或 workflow。

## 完成檢查

Forward-only run（`RUN_ATTRIBUTION=0`）至少應具備 lens、readout 與 uncertainty，且不應
有 generated attribution artifact：

```bash
test -f artifacts/lenses/qwen3.5-4b/jacobian_lens.pt
test -f artifacts/prompt_analysis/qwen3.5-4b/per_date/prompt_layer_uncertainty.jsonl
test ! -e artifacts/prompt_analysis/qwen3.5-4b/generated_attribution/generated_token_attribution.jsonl
```

只有 attribution-enabled run（`RUN_ATTRIBUTION=1`）才能檢查 attribution validation
與完整 visualization：

```bash
test -f artifacts/prompt_analysis/qwen3.5-4b/generated_attribution/generated_token_attribution.jsonl
test -f artifacts/prompt_analysis/qwen3.5-4b/visualization/attribution_dashboard.html
```
