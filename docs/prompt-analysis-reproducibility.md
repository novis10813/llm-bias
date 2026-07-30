# Prompt-analysis 實驗重現指南

這份文件說明如何以任意相容 decoder model，從同一份 prompt CSV 重跑
Jacobian-lens readout、Temperature Scope 不確定性資料、generated-token
attribution，以及最終圖表與互動式 dashboard。

整體分成三個獨立入口：

- `fit-jacobian-lens`：建立可重用的 lens artifact。
- `prompt-analysis`：執行 readout、attribution、validation 與 visualization。
- `scripts/run_prompt_analysis.sh`：只 orchestration prompt experiment，不會 fitting
  lens。

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

Lens 必須已存在；runner 不會自動 fitting：

```bash
bash scripts/run_prompt_analysis.sh
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

Runner 執行兩個階段：

1. 對六個 `with/without × index` prompt columns 做逐日期、逐層 readout。
2. 對每個 condition 均勻抽樣日期，生成 output tokens 並計算 attribution。

輸出結構：

```text
${RUN_ROOT}/
├── per_date/
│   ├── prompt_layer_uncertainty.jsonl
│   ├── prompt_layer_topk.jsonl
│   └── metadata.json
├── generated_attribution/
│   ├── generated_token_attribution.jsonl
│   └── metadata.json
└── run.log
```

### Runner 環境變數

| 變數 | 預設值 |
|---|---|
| `MODEL` | `.cache/models/qwen3.5-4b` |
| `LENS` | `artifacts/lenses/qwen3.5-4b/jacobian_lens.pt` |
| `INPUT_CSV` | `sp500_r1k_r2k_entityBiasPrompt.csv` |
| `RUN_ROOT` | `artifacts/prompt_analysis/qwen3.5-4b` |
| `READOUT_BATCH_SIZE` | `32` |
| `READOUT_MAX_SEQ_LEN` | `256` |
| `TOP_K` | `15` |
| `ATTR_SAMPLE_PER_CONDITION` | `32` |
| `ATTR_MAX_NEW_TOKENS` | `64` |
| `SESSION` | `prompt_analysis` |
| `RUN_IN_TMUX` | `1` |

## 3. Optional attribution validation

```bash
uv run prompt-analysis validate-attribution \
  --model .cache/models/qwen3.5-4b \
  --attribution artifacts/prompt_analysis/qwen3.5-4b/generated_attribution/generated_token_attribution.jsonl \
  --output-dir artifacts/prompt_analysis/qwen3.5-4b/attribution_validation
```

## 4. 建立視覺化

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

1. `${RUN_ROOT}/per_date/prompt_layer_uncertainty.jsonl`
2. `${RUN_ROOT}/generated_attribution/generated_token_attribution.jsonl`
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

## 完成檢查

```bash
test -f artifacts/lenses/qwen3.5-4b/jacobian_lens.pt
test -f artifacts/prompt_analysis/qwen3.5-4b/per_date/prompt_layer_uncertainty.jsonl
test -f artifacts/prompt_analysis/qwen3.5-4b/generated_attribution/generated_token_attribution.jsonl
test -f artifacts/prompt_analysis/qwen3.5-4b/visualization/attribution_dashboard.html
```
