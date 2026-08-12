# Prompt-analysis 實驗重現指南

這份文件說明如何以任意相容 decoder model，從同一份 prompt CSV 執行
Jacobian-lens readout、generation 與 generated-token attribution stages，並依 stage
前置條件執行 validation、圖表與互動式 dashboard。

整體分成三個獨立入口：

- `jacobian-lens fit`：建立可重用的 lens artifact。
- `prompt-analysis`：提供 `readout`、`generate`、`attribute-generated`、validation
  與 visualization commands。
- `scripts/run_prompt_analysis.sh`：只 orchestration prompt experiment，不會 fitting
  lens；各 stage 由 `RUN_READOUT`、`RUN_GENERATION` 與 `RUN_ATTRIBUTION` 控制。

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

Canonical lens 必須保留每個 intermediate layer。Qwen3.5-4B 使用 model-specific
residual width/layer count；完整設計、候選選擇與 promotion 流程見
[Qwen3.5-4B Jacobian-lens calibration 與候選選擇](qwen-jacobian-lens-selection.md)。

一般新模型可以先使用 standalone fitter 建立 experimental lens，輸出到候選目錄；
active canonical lens 的位置固定為：

```text
artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt
```

fitting checkpoints 的位置固定為：

```text
artifacts/archive/<model-slug>/jacobian-lens/checkpoints/
```

選出的 Qwen active artifact 例：

```text
artifacts/qwen3.5-4b/jacobian-lens/
├── jacobian_lens.pt
├── jacobian_lens.pt.metadata.json
└── selection.json
```

Metadata 記錄模型 identity、hidden width、layer coverage、fitting 參數、calibration
digest 與 jlens version。完整 active lens 不得由小型 smoke fit 覆寫。

## 2. 執行 prompt-analysis

目前 runner 與 CLI 的 stage contract 是：

- `readout`：讀取 CSV、載入既有 lens，輸出 `${RUN_ROOT}/readout/` 的逐層
  vocabulary readout、uncertainty、aggregate top-k 與 metadata。
- `generate`：對輸入 conditions 產生並保存 `${RUN_ROOT}/forward/generated_outputs.jsonl`
  與 `forward/metadata.json`；它是唯一執行 generation 的 stage。
- `attribute-generated`：只讀取既有 forward artifact，驗證 model identity 與
  parent SHA-256，再把 gradient attribution 寫入 `${RUN_ROOT}/backward/`；它不會
  再次 generation。

預設 shell 設定為：

```text
MODEL=.cache/models/qwen3.5-4b
LENS=artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt
RUN_ROOT=artifacts/qwen3.5-4b/<dataset-slug>/runs/<run-id>
RUN_READOUT=1
RUN_GENERATION=0
RUN_ATTRIBUTION=0
GEN_SAMPLE_PER_CONDITION=32
```

Runner 不會 fitting lens。完整 MAG7 8-K return-pairs runner 的固定設定另見
[本文件的 MAG7 completion contract](#dataset-specific-completion-contracts)。

### Runner 環境變數

| 變數 | 預設值 |
|---|---|
| `MODEL` | `.cache/models/qwen3.5-4b` |
| `LENS` | `artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt` |
| `INPUT_CSV` | `sp500_r1k_r2k_entityBiasPrompt.csv` |
| `DATASET_FORMAT` | `auto` |
| `DATASET_SLUG` | input filename 的安全 slug |
| `RUN_ID` | UTC timestamp |
| `RUN_ROOT` | `artifacts/<model-slug>/<dataset-slug>/runs/<run-id>` |
| `READOUT_BATCH_SIZE` | `32` |
| `READOUT_MAX_SEQ_LEN` | `256`；`return-pairs` 時為 `512` |
| `TOP_K` | `15` |
| `MAX_ROWS` | empty（不限制） |
| `GEN_SAMPLE_PER_CONDITION` | `32`；`0` 代表傳入 `--full-generation` |
| `GEN_MAX_NEW_TOKENS` | `64` |
| `GEN_TEMPERATURE` | `0`（greedy） |
| `GEN_SEED` | empty |
| `GEN_TOP_P` | `1.0` |
| `GEN_TOP_K` | `0` |
| `RUN_READOUT` | `1` |
| `RUN_GENERATION` | `0` |
| `RUN_ATTRIBUTION` | `0` |
| `FORWARD_ARTIFACT` | attribution-only execution 時的既有 forward JSONL |
| `BACKWARD_INPUT_TOP_K` | empty |
| `RUN_IN_TMUX` | `1` |
| `SESSION` | `prompt_analysis` |

`RUN_GENERATION=1` 執行 `generate`；`RUN_ATTRIBUTION=1` 執行
`attribute-generated`。若 generation 關閉而 attribution 開啟，必須提供
`FORWARD_ARTIFACT`；該 artifact 必須先存在。Stage outputs 只保存 compact records、
token IDs/text、scores、generation config 與 metadata。

## 3. Attribution validation（需要 attribution-enabled artifact）

這個 validation 只接受 `RUN_ATTRIBUTION=1` 產生的
`backward/generated_token_attribution.jsonl`。`readout` 與 `forward` artifact 不是
attribution input，必須先完成 `attribute-generated` stage。

```bash
uv run prompt-analysis validate-attribution \
  --model .cache/models/qwen3.5-4b \
  --attribution artifacts/qwen3.5-4b/<dataset-slug>/runs/<run-id>/backward/generated_token_attribution.jsonl \
  --output-dir artifacts/qwen3.5-4b/<dataset-slug>/runs/<run-id>/attribution_validation
```

## 4. 建立視覺化

`visualize_prompt_analysis.sh` 需要同一個 run 的 `readout` uncertainty 與 `forward`
generated outputs；存在 `backward` artifact 時才會啟用 attribution panel。Script 不猜測
run ID，必須明確指定 canonical run root：

```bash
RUN_ROOT=artifacts/<model-slug>/<dataset-slug>/runs/<run-id> \
TOKENIZER=/path/to/model \
bash scripts/visualize_prompt_analysis.sh
```

Visualizer 尋找：

1. `${RUN_ROOT}/readout/prompt_layer_uncertainty.jsonl`（必要）
2. `${RUN_ROOT}/forward/generated_outputs.jsonl`（必要）
3. `${RUN_ROOT}/backward/generated_token_attribution.jsonl`（optional；存在時啟用
   attribution panel）
4. optional `${RUN_ROOT}/attribution_validation/semantic_scope_aopc.jsonl`

輸出為：

```text
${RUN_ROOT}/visualization/
├── final_layer_effective_temperature_with_context.png
├── final_layer_effective_temperature_without_context.png
├── final_layer_entropy_with_context.png
├── final_layer_entropy_without_context.png
├── final_layer_uncertainty.csv
└── attribution_dashboard.html
```

直接在瀏覽器開啟 `attribution_dashboard.html` 即可。

## 5. Multi-run price distribution 研究圖

Multi-run price sampling 是獨立的 forward artifact family，不是單一 `RunManifest` lifecycle run，
也不需要 generated-token backward attribution。先建立空的 sampling root：

```bash
uv run prompt-analysis generate \
  --model .cache/models/qwen3.5-4b \
  --input sp500_r1k_r2k_entityBiasPrompt.csv \
  --output artifacts/qwen3.5-4b/sp500-price-sampling/t0.7-r30 \
  --runs 30 \
  --temperature 0.7 \
  --seed 123
```

此 command 產生 `sampling_manifest.json` 與每個
`run_NNN/forward/generated_outputs.jsonl`。接著可輸出三個 index 的 close-price 研究圖：

```bash
uv run prompt-analysis plot-price-distributions \
  --sampling-root artifacts/qwen3.5-4b/sp500-price-sampling/t0.7-r30 \
  --prices sp500_r1k_r2k_entityBiasPrompt.csv \
  --output-dir artifacts/qwen3.5-4b/sp500-price-sampling/t0.7-r30/price_distribution
```

命令會先驗證 `sampling_manifest.json`、所有宣告的 `run_NNN/forward/generated_outputs.jsonl`、
forward SHA-256、每個 run 的 record 數量，以及每個 date/index/context condition 是否完整。
Incomplete artifact 不會被靜默當成完整實驗；backward attribution 不是此圖的輸入。

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
  --uncertainty-root artifacts/qwen3.5-4b/sp500_uncertainty/runs/readout \
  --output-dir artifacts/qwen3.5-4b/sp500_uncertainty/uncertainty_distribution
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

## Stage artifact contract

The runner writes one canonical run tree:

```text
artifacts/<model-slug>/<dataset-slug>/runs/<run-id>/
├── manifest.json
├── readout/                         # when RUN_READOUT=1
│   ├── prompt_layer_topk.jsonl
│   ├── prompt_layer_uncertainty.jsonl
│   ├── average_layer_topk.jsonl
│   ├── average_layer_topk.csv
│   ├── output_topk_distribution.png
│   └── metadata.json
├── forward/                         # when RUN_GENERATION=1
│   ├── generated_outputs.jsonl
│   └── metadata.json
└── backward/                        # when RUN_ATTRIBUTION=1
    ├── generated_token_attribution.jsonl
    └── metadata.json
```

`manifest.json` is schema version `1` and is produced by `RunManifest`. Its canonical
object contains:

- `schema_version`, `model`, `model_slug`, `dataset`, `dataset_slug`, `run_id` and
  `run_root`;
- lifecycle `status` (`created`, `running`, `complete` or `failed`), timestamps, and
  optional `error`;
- `artifacts`, plus the role-indexed `input_refs`, `lens_refs` and `output_refs`;
- `record_counts`, keyed by registered `artifact_type`; and
- `stages`, keyed by enabled stage name, whose status is `created`, `running`, `complete`
  or `failed`, with lifecycle timestamps.

Each artifact reference records `artifact_type`, `stage`, `status`, `role`, a path, a
lowercase SHA-256 digest, and an optional JSONL `record_count` and producer metadata.
The runner registers the input CSV and configured lens during initialization. As stages
finish it registers the files that actually exist, computes their SHA-256 values, and
infers JSONL record counts. The root manifest is marked `complete` only after the runner
has finished all enabled stages; a failed stage leaves the run `failed` with an error.
The stored hashes, counts, and stage states support independent completion checks without
using stale file existence as the completion signal.

`attribute-generated` writes backward metadata with the model identity, parent forward
path, parent forward SHA-256 (also exposed as `parent_forward_hash` and
`parent_artifact`), output SHA-256, record counts, and coverage counts. It verifies the
forward model identity, parent hash, and per-record generated-token coverage. The
metadata does not claim a dataset/run binding that the producer does not implement.

## Dataset-specific completion contracts

- **Legacy-wide**: `generate` defaults to 32 deterministically spread dates per condition.
  The same selected date set is used for each prompt condition.
- **MAG7 8-K return-pairs**: the dedicated runner sets `DATASET_FORMAT=return-pairs`,
  `READOUT_MAX_SEQ_LEN=512`, `RUN_GENERATION=1`, `RUN_ATTRIBUTION=0`, and
  `GEN_SAMPLE_PER_CONDITION=0`. The runner translates zero into `generate --full-generation`.
  The input contains 710 unique pairs, so forward generation writes 1,420 condition
  records: one `original` and one `counterfactual` record per pair.
- **Stage ordering**: `readout` and `generate` complete before `attribute-generated`.
  Backward consumes the persisted generated token IDs and does not perform generation.

## Artifact 與研究限制

- 保存的是 compact readout/attribution 結果：top-k、rank、統計量、token IDs/text、
  probabilities、generation 設定與 provenance；不保存完整 raw residual、embedding 或
  gradient activation。
- Readout aggregate 先對每個 condition 的完整 vocabulary softmax 做平均，再選 top-k。
  `effective_temperature` 是 residual-space 的 readout measure；`GEN_TEMPERATURE` 是
  generation sampling 設定，兩者不可互換或解讀成同一量。
- Generated-token attribution 是局部的一階 gradient sensitivity/semantic-scope
  readout，不是 attention map、chain-of-thought、離散 reasoning path，也不是
  standalone causal proof。Attribution validation 的 ablation 結果只能提供額外的
  validation evidence，不能把單次 attribution 直接宣稱為一般化 causal effect。

## 完成檢查

使用 tiny JSONL 與 temporary directory，至少驗證：

1. run root 使用 `<model-slug>/<dataset-slug>/<run-id>`，manifest identity 與 stage
   metadata identity 相符；
2. enabled stages have `complete` status and their registered files exist;
3. manifest and stage metadata SHA-256 values can be recomputed;
4. backward metadata parent path/hash match the supplied forward artifact, and coverage
   counts agree with persisted generated token IDs;
5. run tree 沒有 raw activation suffix 或完整 activation 欄位。

這些 checks 不需要載入 checkpoint。整合 gate 另外執行 `uv run pytest -q`、
`uv run python -m compileall -q llm_bias`、`uv lock --check` 與 `uv build`；不要在
unit smoke 中執行 710-pair inference。
