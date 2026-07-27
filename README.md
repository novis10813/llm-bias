# Entity-level causal representation experiment

This repository studies when an entity-sensitive representation forms inside a
decoder language model and whether it causally changes the answer distribution.
The first experiment uses aligned factual counterfactual pairs and residual
activation patching. `jlens` is used as a Jacobian-lens readout; it is not
treated as a direct decoder of hidden chain-of-thought.

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
uv run python -m llm_bias fit-lens \
  --model .cache/models/qwen3.5-4b \
  --output artifacts/qwen3.5_entity_control/jacobian_lens.pt
uv run python -m llm_bias run-patch \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/qwen3.5_entity_control/jacobian_lens.pt \
  --output artifacts/qwen3.5_entity_control/patch_results.jsonl
uv run python -m llm_bias serve-viz \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/qwen3.5_entity_control/jacobian_lens.pt \
  --pairs artifacts/entity_control/pairs.jsonl
```

## Run

```bash
uv sync
uv run pytest
uv run python -m llm_bias prepare-data
uv run python -m llm_bias fit-lens --calibration-prompts 16 --layer-stride 4
uv run python -m llm_bias run-patch --lens artifacts/entity_control/jacobian_lens.pt
uv run python -m llm_bias visualize
uv run python -m llm_bias analyze-prompt-outputs
uv run python -m llm_bias serve-viz --lens artifacts/entity_control/jacobian_lens.pt
```

The last command starts a local interactive counterfactual dashboard at
`http://127.0.0.1:8321`. It runs source, target, and patched forwards for the
selected pair and displays their J-space top-1 readouts. The server writes the
two ignored external checkout revisions to
`artifacts/visualization/dependencies.json`.

The smoke run can use `--max-pairs 4` and two calibration prompts. The full
current run produced 170 aligned pairs across countries, months, animals, and
numbers. Results are written under `artifacts/entity_control/`.

The visualization command writes the corrected transfer curve,
`patch_transfer_heatmap.png`, and `jacobian_readout_heatmap.png`.

## Average prompt-output distributions

`analyze-prompt-outputs` reads every `prompt_with_context_*` and
`prompt_without_context_*` column in
`sp500_r1k_r2k_entityBiasPrompt.csv`. For each prompt it reads the next-token
distribution at the final, non-padding prompt position. Fitted intermediate
layers use the Jacobian transport; the final layer is the model's actual
output distribution. The default `k` is 15. CSV prompts are wrapped as a user
message with the tokenizer's chat template. Qwen thinking mode is disabled by
default so the requested JSON output is the first generated distribution; use
`--enable-thinking` to restore it or `--raw-prompt` for an unformatted
base-model read.

The aggregate is computed by averaging the complete vocabulary softmax for
each condition before selecting its top-k. This is different from averaging
only the per-prompt top-k entries, which would bias the result. Use the
stride-1 lens to retain every intermediate layer:

```bash
uv run python -m llm_bias analyze-prompt-outputs \
  --input sp500_r1k_r2k_entityBiasPrompt.csv \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/qwen3.5_entity_control/stride1/jacobian_lens.pt \
  --top-k 15 \
  --batch-size 32 \
  --attribution-batch-size 8 \
  --attribution-output-top-k 1 \
  --output-dir artifacts/sp500_r1k_r2k_jspace
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
deterministic date-spread sample that fits large Qwen models. Use
`--no-input-attribution` to skip it entirely.

## Portable Qwen runner

The complete workflow is also available as a tmux-backed shell runner:

```bash
bash scripts/run_qwen_jspace_experiment.sh
```

It fits a stride-1 lens, saves per-date/per-layer top-k and uncertainty for all
six prompt columns, and runs sampled generated-token attribution. Override
settings with environment variables when moving to another model or machine:

```bash
MODEL=.cache/models/qwen3.5-27b \
RUN_ROOT=artifacts/qwen27b_jspace \
FIT_DIM_BATCH=2 \
ATTR_SAMPLE_PER_CONDITION=32 \
bash scripts/run_qwen_jspace_experiment.sh
```

The default detached session is `qwen_jspace_experiment`; attach with
`tmux attach -t qwen_jspace_experiment`. Set `RUN_IN_TMUX=0` to run in the
foreground. The script resumes lens fitting from its checkpoint if interrupted.
For very large models, choose `FIT_DIM_BATCH` conservatively for available
GPU memory; the model loader still needs to be adapted separately if the model
does not fit on the selected device.

## Qwen result visualizations

After the six per-date uncertainty files and sampled generated-token
attribution have been produced, create the final-layer uncertainty plots and the
standalone interactive attribution dashboard:

```bash
uv run python -m llm_bias visualize-qwen-results \
  --uncertainty-root artifacts \
  --attribution artifacts/qwen_generated_attribution_semantic_scope_full_selected/generated_token_attribution.jsonl \
  --prices sp500_r1k_r2k_entityBiasPrompt.csv \
  --output-dir artifacts/qwen_result_visualization
```

The command writes two final-layer Temperature Scope uncertainty time-series
PNGs (`final_layer_effective_temperature_*.png`). Each PNG has one independent
panel per index so the three series are not overlaid, plus entropy comparison
PNGs, `final_layer_uncertainty.csv` containing both measures, and
`attribution_dashboard.html`. Open the HTML
directly in a browser; it has no server or model inference dependency, although
the tokenizer is loaded once while constructing the HTML so every actual CSV
prompt token can be displayed. Choose one of the selected dates, then hover a
Qwen output token to color the complete SP500 input prompt by its attribution.
The generated-token attribution uses the paper's Semantic Scope objective:
for each generated token, it differentiates that token's target logit with
respect to each input embedding and uses the gradient L2 norm as the token
influence. It is a local first-order sensitivity score; it is not attention or
a standalone causal claim. The default is 64 new tokens so the JSON answer and
confidence/evidence fields are included when Qwen emits them.

For a selected-date dashboard with longer generated JSON outputs, first run
attribution only for the dates used by the dashboard:

```bash
uv run python -m llm_bias analyze-generated-attribution \
  --input sp500_r1k_r2k_entityBiasPrompt.csv \
  --model .cache/models/qwen3.5-4b \
  --date 2010-07-16 \
  --date 2014-04-10 \
  --date 2024-05-30 \
  --max-new-tokens 64 \
  --output-dir artifacts/qwen_generated_attribution_semantic_scope_full_selected
```

Then pass
`artifacts/qwen_generated_attribution_semantic_scope_full_selected/generated_token_attribution.jsonl`
as `--attribution` to the visualization command.

When `--input-top-k` is omitted, the generated-token Semantic Scope run saves
the score for every raw input-prompt token. Providing `--input-top-k` enables a
smaller storage-only top-k export.

To validate the Semantic Scope ranking with zero-vector input ablations and a
deterministic random baseline:

```bash
uv run python -m llm_bias validate-semantic-scope \
  --attribution artifacts/qwen_generated_attribution_semantic_scope_full_selected/generated_token_attribution.jsonl \
  --model .cache/models/qwen3.5-4b \
  --output-dir artifacts/qwen_semantic_scope_validation_selected
```

Pass the resulting `semantic_scope_aopc.jsonl` with `--validation` when
rebuilding the dashboard to show per-output-token AOPC values.
