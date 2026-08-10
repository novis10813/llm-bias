# Entity-level causal representation experiments

This repository studies when an entity-sensitive representation forms inside a
decoder language model and whether replacing a source entity activation with a
target entity activation causally changes the answer distribution.

## Workflows

- `counterfactual-patching`: residual activation patching for factual smoke pairs
  and reviewed entity-only entity-bias pairs.
- `prompt-analysis`: per-layer prompt readout, uncertainty, generated-token
  attribution, validation, and result visualization.
- `prepare-edgar-8k`: auditable staging-data preparation for extracted 8-K filings.
- `prepare-counterfactual-data`: point-in-time entity histories, reviewed
  entity-only counterfactual generation, model-specific rendering, and validation.
- `prepare-10k-change-data`: auditable, prompt-agnostic `year,cik,item` CSV
  for extracted 10-K metadata-change windows.
- `fit-jacobian-lens`: standalone Jacobian-lens fitting; experiment workflows
  consume fitted lenses and never fit one implicitly.

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

- [Counterfactual patching](docs/counterfactual-patching.md)
- [8-K counterfactual entity dataset](docs/counterfactual-dataset-generation.md)
- [EDGAR 8-K preparation](docs/edgar-8k-preparation.md)
- [10-K metadata-change LLM dataset](docs/ten-k-change-dataset.md)
- [Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md)
- [Prompt-analysis reproducibility](docs/prompt-analysis-reproducibility.md)
- [Interactive prompt-lens dashboard](docs/interactive-prompt-lens-dashboard.md)
- [Synthetic entity-bias pilot](docs/synthetic-entity-bias.md)

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
differ from Llama. Follow the controlled candidate-selection workflow in
[Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md); do not
replace its canonical lens with a small smoke fit.

## Minimal smoke workflow

```bash
uv lock --check
uv run pytest -q
uv run fit-jacobian-lens \
  --model .cache/models/llama-3.2-1b-instruct \
  --calibration-prompts 16
uv run counterfactual-patching prepare-data \
  --model .cache/models/llama-3.2-1b-instruct \
  --max-pairs 4
uv run counterfactual-patching run \
  --model .cache/models/llama-3.2-1b-instruct \
  --lens artifacts/llama-3.2-1b-instruct/jacobian-lens/jacobian_lens.pt \
  --max-pairs 4
```

For the complete patching sequence, including model-scoped pair paths,
metrics, artifacts, and dashboard behavior, use
[Counterfactual patching](docs/counterfactual-patching.md).

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
- The 8-K counterfactual workflow requires manual review and promotion before a
  row is considered validated. See the [dataset protocol](docs/counterfactual-dataset-generation.md).
- This prompt-analysis layout does not claim that counterfactual-patching has
  completed a repository-wide artifact migration.
- Jacobian-lens attribution is a local transported-readout or sensitivity
  diagnostic; it is not by itself a causal claim.

## Standard verification

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
node --check llm_bias/static/counterfactual.js
```
