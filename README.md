# Entity-level causal representation experiments

This repository studies when an entity-sensitive representation forms inside a
decoder language model and whether it causally changes the answer distribution.
The repository contains two independent experiments:

- `counterfactual-patching`: aligned entity-span swapping and causal residual
  activation patching.
- `prompt-analysis`: per-layer prompt readout, uncertainty, generated-token
  attribution, validation, and result visualization.

Jacobian-lens fitting is a third, standalone tool. Both experiments consume a
fitted lens but never fit one themselves. `jlens` readouts are transported
representations, not direct decoders of hidden chain-of-thought.

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

Fit a separate lens for Qwen because its residual width and layer count differ
from Llama, then pass the same model and Qwen lens to the patch/dashboard
commands:

```bash
uv run fit-jacobian-lens \
  --model .cache/models/qwen3.5-4b \
  --output artifacts/lenses/qwen3.5-4b/stride2/jacobian_lens.pt
uv run counterfactual-patching run \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/lenses/qwen3.5-4b/stride2/jacobian_lens.pt \
  --output artifacts/counterfactual_patching/qwen3.5-4b/patch_results.jsonl
uv run counterfactual-patching serve \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/lenses/qwen3.5-4b/stride2/jacobian_lens.pt \
  --pairs artifacts/counterfactual_patching/pairs.jsonl
```

## Run

```bash
uv sync
uv run pytest
uv run fit-jacobian-lens \
  --model .cache/models/llama-3.2-1b-instruct \
  --output artifacts/lenses/llama-3.2-1b-instruct/stride4/jacobian_lens.pt \
  --calibration-prompts 16 \
  --layer-stride 4
uv run counterfactual-patching prepare-data
uv run counterfactual-patching run \
  --lens artifacts/lenses/llama-3.2-1b-instruct/stride4/jacobian_lens.pt
uv run counterfactual-patching visualize
uv run counterfactual-patching serve \
  --lens artifacts/lenses/llama-3.2-1b-instruct/stride4/jacobian_lens.pt
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
only the per-prompt top-k entries, which would bias the result. Use the
stride-1 lens to retain every intermediate layer:

```bash
uv run prompt-analysis readout \
  --input sp500_r1k_r2k_entityBiasPrompt.csv \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/lenses/qwen3.5-4b/stride1/jacobian_lens.pt \
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

It requires a fitted lens, saves per-date/per-layer top-k and uncertainty for
all six prompt columns, and runs sampled generated-token attribution. Lens
fitting is deliberately separate so the runner cannot unexpectedly start an
expensive fitting job. Override settings with environment variables:

```bash
MODEL=/path/to/another-model \
LENS=artifacts/lenses/another-model/stride1/jacobian_lens.pt \
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
