# Entity-level causal representation experiments

This repository studies when an entity-sensitive representation forms inside a
decoder language model and whether it causally changes the answer distribution.
The repository contains two independent experiments:

- `counterfactual-patching`: aligned entity-span swapping and causal residual
  activation patching.
- `prompt-analysis`: per-layer prompt readout, uncertainty, generated-token
  attribution, validation, and result visualization.
- `prepare-edgar-8k`: auditable staging-data preparation for extracted 8-K
  filings.
- `prepare-counterfactual-data`: reviewed entity-only counterfactual generation
  from staged 8-K earnings events.

Jacobian-lens fitting is a third, standalone tool. Both experiments consume a
fitted lens but never fit one themselves. `jlens` readouts are transported
representations, not direct decoders of hidden chain-of-thought.

## Preparing extracted 8-K filings

The EDGAR preparation workflow reads the external crawler dataset without
modifying it. It streams every extracted filing into JSONL artifacts under the
git-ignored `artifacts/` directory:

```bash
uv run prepare-edgar-8k clean \
  --input ../10-k/edgar-crawler/datasets/EXTRACTED_FILINGS/8-K \
  --output artifacts/edgar_8k/cleaned

uv run prepare-edgar-8k validate \
  --input artifacts/edgar_8k/cleaned
```

Use `--max-files 100` for a smoke run. The cleaner writes `filings.jsonl`,
`sections.jsonl`, `quality_report.json`, and `manifest.json`. Each non-empty
section retains lightly normalized source text alongside an analysis version
with recorded boilerplate spans removed. Event-family and candidate flags are
deterministic; no model-generated summary, counterfactual pair, or market label
is introduced at this stage.

The upstream extracted filings may omit exhibits and may have removed numerical
tables. These limitations are recorded in the manifest, and absent extracted
text must not be interpreted as an absent company disclosure.

完整的清理規則、event-family mapping、JSONL schema 與全量執行統計見
[EDGAR 8-K 清理與事件候選資料](docs/edgar-8k-preparation.md)。

## Building entity-only counterfactual data

The counterfactual-data workflow builds point-in-time CIK/name histories,
samples a deterministic 500-event earnings pilot, uses the local
OpenAI-compatible llama.cpp endpoint through LangExtract, and enforces a
200-item manual review gate before any row is marked validated:

```bash
uv run prepare-counterfactual-data entities
uv run prepare-counterfactual-data sample --count 500 --seed 20260730
uv run --extra extraction prepare-counterfactual-data annotate
uv run prepare-counterfactual-data review-bundle --count 200
```

No ticker or current market cap is required in V1. Same-industry targets are
matched by historical filing exposure and are explicitly named
`matched_exposure`, not size-matched. Full schemas, review thresholds, pairing
rules, promotion, model-specific rendering, and validation are documented in
[8-K counterfactual entity dataset](docs/counterfactual-dataset-generation.md).

## Setup

Dependencies are managed with `uv`. The root project uses the local
`third_party/jacobian-lens/` and `third_party/jspace-viz/` as editable workspace
members. The external checkouts are intentionally ignored by the root Git
repository; their exact commits are recorded in `artifacts/visualization/` when
the integration server is started.

On a fresh checkout, restore the ignored external worktrees before syncing:

```bash
mkdir -p third_party
git clone https://github.com/anthropics/jacobian-lens.git third_party/jacobian-lens
git clone https://github.com/Festyve/jspace-viz.git third_party/jspace-viz
uv sync
```

The implementation targets `unsloth/Llama-3.2-1B-Instruct`. Start the model
download in the background with:

```bash
mkdir -p artifacts .cache/models
nohup uv run hf download unsloth/Llama-3.2-1B-Instruct \
  --local-dir .cache/models/llama-3.2-1b-instruct \
  > artifacts/llama_download.log 2>&1 &
```

Qwen3.5-4B is also available as a local model. It is a multimodal
conditional-generation checkpoint, so the project loader selects the matching
Transformers class automatically:

```bash
uv run hf download Qwen/Qwen3.5-4B \
  --local-dir .cache/models/qwen3.5-4b \
  > artifacts/qwen3.5_download.log 2>&1
```

Qwen requires its own lens because its residual width and layer count differ
from Llama. Use the controlled three-candidate workflow below instead of
overwriting its canonical lens with the standalone fitter's small built-in
smoke corpus. The winner is promoted to the model-specific canonical path
`artifacts/lenses/qwen3.5-4b/jacobian_lens.pt`:

```bash
bash scripts/run_qwen_lens_candidates.sh
uv run counterfactual-patching run \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/lenses/qwen3.5-4b/jacobian_lens.pt \
  --output artifacts/counterfactual_patching/qwen3.5-4b/patch_results.jsonl
uv run counterfactual-patching serve \
  --model .cache/models/qwen3.5-4b \
  --pairs artifacts/counterfactual_patching/pairs.jsonl
```

## Run

```bash
uv sync
uv run pytest
uv run fit-jacobian-lens \
  --model .cache/models/llama-3.2-1b-instruct \
  --calibration-prompts 16
uv run counterfactual-patching prepare-data
uv run counterfactual-patching run \
  --lens artifacts/lenses/llama-3.2-1b-instruct/jacobian_lens.pt
uv run counterfactual-patching visualize
uv run counterfactual-patching serve
```

The last command starts a local interactive counterfactual dashboard at
`http://127.0.0.1:8321`. It runs source, target, and patched forwards for the
selected pair and displays their J-space top-1 readouts. The server writes the
two ignored external checkout revisions to
`artifacts/visualization/dependencies.json`.

The smoke run can use `--max-pairs 4` and two calibration prompts. The full
current run produced 170 aligned pairs across countries, months, animals, and
numbers. Results are written under `artifacts/counterfactual_patching/`.

The visualization command writes the corrected transfer curve,
`patch_transfer_heatmap.png`, and `jacobian_readout_heatmap.png`.

## Qwen bilingual Jacobian-lens selection

完整 calibration design、fitting/recovery、holdout metric、2026-07-30 結果、
uncertainty 與 promotion provenance 見
[Qwen3.5-4B Jacobian-lens calibration 與候選選擇](docs/qwen-jacobian-lens-selection.md)。

Qwen3.5-4B uses a controlled three-candidate calibration experiment:

- `english`: 128 English passages.
- `chinese_simplified`: 128 Simplified-Chinese passages.
- `mixed`: 64 English and 64 Simplified-Chinese passages.

The conditions share the same 16 domains × 8 discourse styles, keeping content
composition and Qwen token lengths closely matched. Regenerate and verify the
tracked inputs with:

```bash
uv run python scripts/prepare_qwen_calibration.py
uv run python scripts/prepare_qwen_lens_eval.py
```

Fit all three complete 31-source-layer candidates in a resumable tmux session:

```bash
bash scripts/run_qwen_lens_candidates.sh
tmux attach -t qwen_lens_candidates
```

Candidates use the Qwen chat template with thinking disabled, `skip_first=16`,
and checkpoints keyed by the hash of the exact formatted calibration corpus.
Selection uses a separate 32-pair/64-prompt bilingual holdout. The
preregistered primary metric is balanced English/Chinese mean log10 rank of
one canonical native-language token over layers L11–L23 (leading-space token
for English, raw token for Chinese); the corresponding canonical cross-lingual
token rank is the tie-break. After all fits finish, the runner evaluates the candidates,
archives the prior canonical Qwen lens, and promotes the winner to
`artifacts/lenses/qwen3.5-4b/jacobian_lens.pt`.

The evaluation also reports paired uncertainty across the 32 semantic pairs:
a deterministic 10,000-resample bootstrap confidence interval and a
one-sided sign-flip permutation p-value for the selected candidate against
each alternative. These are descriptive, not confirmatory, because the same
holdout is used both to select the winner and to quantify its advantage.

The completed run selected `chinese_simplified` under the fixed primary metric:
balanced native mean log10 ranks were 3.826620 (Chinese-only), 3.881337
(mixed), and 3.889848 (English-only). Both paired 95% intervals for the
winner's advantage cross zero, so this is an operational selection, not
evidence that Chinese-only calibration is generally superior.

## Interactive Prompt Lens Dashboard

任意 prompt 的完整逐層 readout 與模型實際 greedy response 操作、API schema 和
研究解讀限制見
[Interactive Prompt Lens Dashboard](docs/interactive-prompt-lens-dashboard.md)。

```bash
uv run prompt-analysis serve \
  --model .cache/models/qwen3.5-4b \
  --host 0.0.0.0 \
  --port 8322
```

The server resolves the canonical model lens automatically, rejects
wrong-model or incomplete lens artifacts, and never saves raw activations.

## Average prompt-output distributions

`prompt-analysis readout` reads every `prompt_with_context_*` and
`prompt_without_context_*` column in
`sp500_r1k_r2k_entityBiasPrompt.csv`. For each prompt it reads the next-token
distribution at the final, non-padding prompt position. Fitted intermediate
layers use the Jacobian transport; the final layer is the model's actual
output distribution. The default `k` is 15. CSV prompts are wrapped as a user
message with the tokenizer's chat template. Thinking mode is disabled by
default for tokenizers that support it; use `--enable-thinking` to restore it
or `--raw-prompt` for an unformatted base-model read.

The aggregate is computed by averaging the complete vocabulary softmax for
each condition before selecting its top-k. This is different from averaging
only the per-prompt top-k entries, which would bias the result. The canonical
lens retains every intermediate layer:

```bash
uv run prompt-analysis readout \
  --input sp500_r1k_r2k_entityBiasPrompt.csv \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/lenses/qwen3.5-4b/jacobian_lens.pt \
  --top-k 15 \
  --batch-size 32 \
  --attribution-batch-size 8 \
  --attribution-output-top-k 1 \
  --output-dir artifacts/prompt_analysis/qwen3.5-4b/per_date
```

For a quick integration check, add `--max-rows 2`. The command writes:

- `output_topk_distribution.png`: six output-layer average distributions
  (three indices by with/without context).
- `average_layer_topk.jsonl` and `.csv`: average top-k text and probabilities
  for every fitted layer and the output layer.
- `prompt_layer_topk.jsonl`: compact per-prompt/per-layer top-k text; omit it
  with `--no-save-prompt-topk`.
- `prompt_layer_uncertainty.jsonl`: per-date/per-layer Temperature Scope
  effective inverse temperature, effective temperature, plus entropy,
  normalized entropy, perplexity, top-1 probability, and top-k probability
  mass; omit it with `--no-save-prompt-uncertainty`.
- `input_token_attribution.png` and `.jsonl`: for every mean output top-k
  token, the most influential input token identities and end-aligned input
  positions.
- `metadata.json`: exact aggregation semantics, layer coverage, empty-prompt
  counts, input hash, and third-party checkout revisions.

Only token IDs, decoded text, probabilities, and provenance are saved; raw
activations are never written.

Input influence uses the absolute gradient × input-embedding attribution of
each output token's log probability. Scores are normalized across positions
within each prompt, aligned from the final prompt token, then averaged by
condition. This is a local sensitivity attribution, not an attention map or a
claim of standalone causal effect. Attribution is the expensive part of the
run; `--attribution-output-top-k 1` is a quick focus on the mean top-1 output,
and `--attribution-max-rows 128 --attribution-batch-size 1` gives a
deterministic date-spread sample that fits large models. Use
`--no-input-attribution` to skip it entirely.

## Portable prompt-analysis runner

日後重跑與視覺化的完整操作說明見
[Prompt-analysis 實驗重現指南](docs/prompt-analysis-reproducibility.md)。

The complete workflow is also available as a tmux-backed shell runner:

```bash
bash scripts/run_prompt_analysis.sh
```

The runner and visualization script also accept arbitrary ticker-style prompt
columns. For example, a MAG7 table with `Date`, `aapl`, ..., `tsla`, and
`prompt_{with,without}_context_<ticker>` columns can be run with:

```bash
INPUT_CSV=mag7_entityBiasPrompt.csv \
RUN_ROOT=artifacts/prompt_analysis/qwen3.5-4b/mag7 \
bash scripts/run_prompt_analysis.sh

INPUT_CSV=mag7_entityBiasPrompt.csv \
RUN_ROOT=artifacts/prompt_analysis/qwen3.5-4b/mag7 \
bash scripts/visualize_prompt_analysis.sh
```

Condition names, price columns, uncertainty plots, and dashboard labels are
discovered from the artifacts instead of being restricted to the three legacy
index names. Attribution sampling uses shared dates across non-empty prompt
conditions so missing ticker observations do not prevent date selection.

It requires a fitted lens, saves per-date/per-layer top-k and uncertainty for
all selected prompt columns, and runs sampled generated-token attribution. Lens
fitting is deliberately separate so the runner cannot unexpectedly start an
expensive fitting job. Override settings with environment variables:

```bash
MODEL=/path/to/another-model \
LENS=artifacts/lenses/another-model/jacobian_lens.pt \
RUN_ROOT=artifacts/prompt_analysis/another-model \
ATTR_SAMPLE_PER_CONDITION=32 \
bash scripts/run_prompt_analysis.sh
```

The default detached session is `prompt_analysis`; attach with
`tmux attach -t prompt_analysis`. Set `RUN_IN_TMUX=0` to run in the foreground.

## Prompt-analysis result visualizations

After the six per-date uncertainty files and sampled generated-token
attribution have been produced, create the final-layer uncertainty plots and the
standalone interactive attribution dashboard:

```bash
bash scripts/visualize_prompt_analysis.sh
```

The script uses `artifacts/prompt_analysis/qwen3.5-4b/` by default and
automatically selects its attribution and optional validation outputs. For a
separate run directory, set `RUN_ROOT`; custom inputs can be supplied with
`INPUT_CSV`, `TOKENIZER`, `ATTRIBUTION`, `VALIDATION`, `UNCERTAINTY_ROOT`, or
`OUTPUT_DIR`.

The equivalent direct command is:

```bash
uv run prompt-analysis visualize \
  --uncertainty-root artifacts/prompt_analysis/qwen3.5-4b \
  --attribution artifacts/prompt_analysis/qwen3.5-4b/generated_attribution/generated_token_attribution.jsonl \
  --prices sp500_r1k_r2k_entityBiasPrompt.csv \
  --tokenizer .cache/models/qwen3.5-4b \
  --output-dir artifacts/prompt_analysis/qwen3.5-4b/visualization
```

The command writes two final-layer Temperature Scope uncertainty time-series
PNGs (`final_layer_effective_temperature_*.png`). Each PNG has one independent
panel per index so the three series are not overlaid, plus entropy comparison
PNGs, `final_layer_uncertainty.csv` containing both measures, and
`attribution_dashboard.html`. Open the HTML
directly in a browser; it has no server or model inference dependency, although
the tokenizer is loaded once while constructing the HTML so every actual CSV
prompt token can be displayed. Choose one of the selected dates, then hover a
generated output token to color the complete SP500 input prompt by its attribution.
The generated-token attribution uses the paper's Semantic Scope objective:
for each generated token, it differentiates that token's target logit with
respect to each input embedding and uses the gradient L2 norm as the token
influence. It is a local first-order sensitivity score; it is not attention or
a standalone causal claim. The default is 64 new tokens so the JSON answer and
confidence/evidence fields are included when the model emits them.

For a complete multi-run sampling artifact, plot the actual close against the
LLM price distribution and add a per-date error panel:

```bash
uv run prompt-analysis plot-price-distributions \
  --sampling-root artifacts/prompt_analysis/qwen3.5-4b/sp500_uncertainty/generated_attribution_sampling_t0.7_r30 \
  --prices sp500_r1k_r2k_entityBiasPrompt.csv \
  --output-dir artifacts/prompt_analysis/qwen3.5-4b/sp500_uncertainty/generated_attribution_sampling_t0.7_r30/price_distribution
```

This command requires a complete manifest and all declared run directories. It
writes one 300-DPI figure per index, with separate without-context and
with-context price panels plus a shared median absolute percentage error (MdAPE)
panel. The price panels show the actual close, LLM median, 25–75% band, and
5–95% band. Invalid generated answers are excluded from numeric summaries but
retained in `price_distribution_samples.csv`; the summary CSV and metadata JSON
record the exact quantile method, error formula, input hashes, and valid/invalid
counts. These figures describe sampling variability and prediction error, not a
causal effect.

Existing final-layer readout artifacts can also be summarized as cross-date
uncertainty distributions without rerunning the model:

```bash
uv run prompt-analysis plot-uncertainty-distributions \
  --uncertainty-root artifacts/qwen3.5-4b/qwen3.5_temperature_scope_per_date \
  --output-dir artifacts/qwen3.5-4b/qwen3.5_uncertainty_distribution
```

The command writes separate entropy and effective-temperature ECDFs for the
with-context and without-context conditions, plus violin plots of the paired
same-date `with − without` differences. It also preserves raw, paired, and
summary CSV tables and input hashes in metadata. Entropy is the Shannon entropy
of the final-layer full-vocabulary softmax. Effective temperature is the
reciprocal L2 norm of the final-normalized residual; it is not generation
sampling temperature. Paired differences are descriptive associations, not
causal estimates.

For a selected-date dashboard with longer generated JSON outputs, first run
attribution only for the dates used by the dashboard:

```bash
uv run prompt-analysis attribute \
  --input sp500_r1k_r2k_entityBiasPrompt.csv \
  --model .cache/models/qwen3.5-4b \
  --date 2010-07-16 \
  --date 2014-04-10 \
  --date 2024-05-30 \
  --max-new-tokens 64 \
  --output-dir artifacts/prompt_analysis/qwen3.5-4b/generated_attribution_selected
```

Then pass
`artifacts/prompt_analysis/qwen3.5-4b/generated_attribution_selected/generated_token_attribution.jsonl`
as `--attribution` to the visualization command.

When `--input-top-k` is omitted, the generated-token Semantic Scope run saves
the score for every raw input-prompt token. Providing `--input-top-k` enables a
smaller storage-only top-k export.

To validate the Semantic Scope ranking with zero-vector input ablations and a
deterministic random baseline:

```bash
uv run prompt-analysis validate-attribution \
  --attribution artifacts/prompt_analysis/qwen3.5-4b/generated_attribution_selected/generated_token_attribution.jsonl \
  --model .cache/models/qwen3.5-4b \
  --output-dir artifacts/prompt_analysis/qwen3.5-4b/attribution_validation
```

Pass the resulting `semantic_scope_aopc.jsonl` with `--validation` when
rebuilding the dashboard to show per-output-token AOPC values.
## Interactive prompt lens explorer

輸入任意 prompt，查看每個 fitted Jacobian-lens layer 與 token position 的
top-k readout、entropy、kurtosis 與 layer diagnostics：

```bash
uv run prompt-analysis serve \
  --model .cache/models/llama-3.2-1b-instruct \
  --host 0.0.0.0 \
  --port 8322
```

開啟 `http://localhost:8322`。Dashboard 預設會使用同一個 model 與 chat
template 顯示 deterministic greedy response；`Output tokens` 可控制回答長度，
也可關閉「顯示模型回答」以降低延遲。互動 API 只回傳 compact top-k／統計資料
與生成文字，不保存 raw activations。
