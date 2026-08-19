# Entity-level causal representation experiments

This repository studies when an entity-sensitive representation forms inside a
decoder language model and whether replacing a source entity activation with a
target entity activation causally changes the answer distribution.

## Workflows

- `baseline-trial`: run the `data/baseline/` trial-plan prompts through the
  prompt-analysis stages, with a lens-forward per-layer readout stage.
- `prompt-analysis`: per-layer prompt readout, uncertainty, generated-token
  attribution, validation, and result visualization.
- `jacobian-lens fit`: standalone Jacobian-lens fitting; experiment workflows
  consume fitted lenses and never fit one implicitly.

Archived workflows (`counterfactual-patching`, `prepare-edgar-8k`,
`prepare-counterfactual-data`, `synthetic-entity-bias`,
`prepare-10k-change-data`) live in [`archive/`](archive/README.md); they are
not installed entry points.

`jlens` readouts are transported representations, not direct decoders of hidden
chain-of-thought or discrete reasoning paths.

## Documentation map

### Research design and planning

- [Research proposals and planning](docs/proposal/README.md)
- [Entity-bias research proposal](docs/proposal/entity-bias-research-proposal.md)
- [Entity-bias roadmap](docs/proposal/entity-bias-roadmap.md)
- [J-space evaluation design](docs/j-space-evaluation.md) — optional,
  proposed, non-runnable synthetic task-local J-space-candidate preflight for
  comparing compatible local models; not a proposal component or entity-bias
  evidence gate.

### Workflow operations

- [Baseline trial plan prompts](docs/baseline-trial-plan-prompts.md)
- [Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md)
- [Prompt-analysis reproducibility](docs/prompt-analysis-reproducibility.md)
- [Interactive prompt-lens dashboard](docs/interactive-prompt-lens-dashboard.md)
- [Archived workflow operations](docs/archive/) — counterfactual patching,
  8-K/10-K dataset preparation, synthetic entity-bias pilot

## Setup

Dependencies are managed with `uv`. The root project uses the local
`third_party/jacobian-lens/` and `third_party/jspace-viz/` as editable workspace
members. The external checkouts are intentionally ignored by the root Git
repository.

On a fresh checkout, restore the ignored external worktrees before syncing:

```bash
mkdir -p third_party
git clone https://github.com/anthropics/jacobian-lens.git third_party/jacobian-lens
git clone https://github.com/Festyve/jspace-viz.git third_party/jspace-viz
uv sync
```

The implementation targets `unsloth/Llama-3.2-1B-Instruct`. Download the model
into the ignored `.cache/` directory when needed:

```bash
mkdir -p artifacts .cache/models
uv run hf download unsloth/Llama-3.2-1B-Instruct \
  --local-dir .cache/models/llama-3.2-1b-instruct
```

Qwen3.5-4B uses a model-specific lens because its residual width and layer count
differ from Llama. For checkpoints whose `config.json` proves the exact base
identity `Qwen/Qwen3.5-4B`, install the pinned Neuronpedia artifact explicitly:

```bash
uv run jacobian-lens install \
  --model .cache/models/qwen3.5-4b \
  --base-model Qwen/Qwen3.5-4B
```

The command downloads from a pinned Hugging Face revision, verifies the binary,
source config, model shape, complete layer coverage, and local schema-v2
metadata, then installs an offline canonical artifact. It never downloads at
experiment runtime and does not fuzzy-match instruct or differently shaped
checkpoints. If no exact registry entry exists, use `jacobian-lens fit` or the
controlled local candidate-selection alternative in
[Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md); do not
replace a canonical lens with a small smoke fit.

## Minimal smoke workflow

```bash
uv lock --check
uv run pytest -q
uv run jacobian-lens fit \
  --model .cache/models/llama-3.2-1b-instruct \
  --calibration-prompts 16
uv run baseline-trial inspect \
  --input data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv
```

For the complete baseline-trial stage sequence, see
[Baseline trial plan prompts](docs/baseline-trial-plan-prompts.md). The
archived counterfactual-patching quickstart remains in
[docs/archive/counterfactual-patching.md](docs/archive/counterfactual-patching.md).

## Data and artifact boundaries

- Models, lens binaries, experiment outputs, and external checkouts remain in
  `.cache/`, `artifacts/`, and `third_party/`; they are not committed to root
  Git.
- Model-scoped lenses live under `artifacts/<model-slug>/jacobian-lens/`.
  Prompt-analysis runs are isolated under
  `artifacts/<model-slug>/<dataset-slug>/runs/<run-id>/`, with `manifest.json`
  and compact `forward/`, `readout/`, and optional `backward/` stage artifacts.
- The manifest records model/dataset identity, input and artifact SHA-256 values,
  record counts, and stage status. Backward attribution must reference the exact
  forward artifact hash and never regenerate its tokens. A run is consumable only
  after all required stages are `complete`.
- Legacy-wide generated-token sampling remains 32 shared dates by default. MAG7
  8-K return-pairs full-generation instead covers every unique pair and writes both
  `original` and `counterfactual` records; it is not the legacy 32 sample.
- No workflow writes complete raw activations to tracked artifacts.
- The archived 8-K counterfactual workflow required manual review and
  promotion before a row was considered validated. See the [dataset
  protocol](docs/archive/counterfactual-dataset-generation.md).
- Jacobian-lens attribution is a local transported-readout or sensitivity
  diagnostic; it is not by itself a causal claim.

## Standard verification

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
node --check llm_bias/static/prompt_readout.js
```
