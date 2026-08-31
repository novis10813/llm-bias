# Entity-level causal representation experiments

This repository studies when an entity-sensitive representation forms inside a
decoder language model and whether replacing a source entity activation with a
target entity activation causally changes the answer distribution.

## Workflows

- `baseline-trial`: run the `data/baseline/` trial-plan prompts through the
  prompt-analysis stages, with a lens-forward per-layer readout stage.
- `prompt-analysis`: per-layer prompt readout, uncertainty, generated-token
  attribution, validation, and result visualization.
- `span-sensitivity`: single-sector behavioral screen over explicit ticker/name
  header spans using fixed Buy/Sell continuation margins.
- `jspace-intervention`: sector-coordinate swap/gain, valence vocabulary readout,
  and matched-random V1 token causal screening over a validated canonical lens.
  Outcome-conditioned decision-flip V2 is implemented against the frozen Draft 1
  protocol. Its first formal pipeline completed with `success=false`: Buy steering
  passed, but the sell-direction Holm gate lacked enough eligible test tickers.
  The V2 doc also defines a readout-only zero-evidence header-only prior probe
  and a Jacobian-lens direction decode; both have a recorded first formal run.
  The same package implements hierarchical activation patching over prepared
  positive/negative valence pairs; discovery found a layer-wise sufficiency
  shift from evidence positions to post-evidence context and then the final
  position. The unchanged held-out confirmation passed every frozen gate with
  `success=true`. The package also implements three sector/context follow-ups:
  A V1 cross-sector header-state patching, B V1 negative-evidence context
  overriding, and C V1 L16 instruction-context readout. Their discovery runs
  are recorded, but calibration/test gates are not frozen.
- `jacobian-lens fit`: standalone Jacobian-lens fitting; experiment workflows
  consume fitted lenses and never fit one implicitly.

Archived workflows (`counterfactual-patching`, `prepare-edgar-8k`,
`prepare-counterfactual-data`, `synthetic-entity-bias`,
`prepare-10k-change-data`) live in [`archive/`](archive/README.md); they are
not installed entry points.

`jlens` readouts are transported representations, not direct decoders of hidden
chain-of-thought or discrete reasoning paths.

## Documentation map

- [Documentation and instruction system](docs/documentation-system.md)
- [Shared experiment core](docs/shared-experiment-core.md)
- [Artifact identity and run manifest contract](docs/artifact-contract.md)
- [Research scripts reference](docs/research-scripts.md)

### Research design and planning

- [Research proposals and planning](docs/proposal/README.md)
- [Entity-bias research proposal](docs/proposal/entity-bias-research-proposal.md)
- [Entity-bias roadmap](docs/proposal/entity-bias-roadmap.md)
- [J-space evaluation design](docs/j-space-evaluation/proposal.md) — optional,
  proposed, non-runnable synthetic task-local J-space-candidate preflight for
  comparing compatible local models; not a proposal component or entity-bias
  evidence gate.

### Experiment proposals and reports

- [J-space sector intervention proposal](docs/jspace-sector-intervention/proposal.md) and [held-out report](docs/jspace-sector-intervention/report.md)
- [J-space valence vocabulary readout proposal](docs/jspace-valence-readout/proposal.md) and [Technology discovery report](docs/jspace-valence-readout/report-technology-discovery.md) —
  positive-vs-negative J-space vocabulary readout that nominates representation
  candidates for later signed steering/gain/swap; transported-representation
  evidence, not causal evidence
- [J-space token experiment versions](docs/jspace-token-experiments/README.md) —
  separates completed V1 vocabulary-direction margin screening from the
  implemented (Draft 1) V2 outcome-gradient Buy/Sell decision-flip testing
  - [V1 proposal](docs/jspace-token-experiments/proposal-v1.md) and [report](docs/jspace-token-experiments/report-v1.md) —
    completed discovery screen; no candidate passed the frozen shortlist gate
  - [V2 proposal](docs/jspace-token-experiments/proposal-v2.md) and [report](docs/jspace-token-experiments/report-v2.md) —
    Draft 1 protocol frozen and implemented; the first formal pipeline returned
    `success=false` because the sell-direction Holm gate had only two eligible test
    tickers; final-position control also reproduced the steering effect; includes
    first formal runs for the zero-evidence header-only prior probe and the
    Jacobian-lens decode of the frozen V2 outcome direction
  - [V2 outcome direction geometric projection](docs/jspace-token-experiments/report-v2-geometry.md) —
    auxiliary descriptive geometry: per-layer Technology minus Financial Services
    sector state difference projected onto the frozen V2 outcome directions
    (parallel/perpendicular decomposition; not causal evidence)
- [Activation patching proposal](docs/activation-patching-causal-tracing/proposal.md) and [experiment report](docs/activation-patching-causal-tracing/report.md) —
  implemented Draft 1 workflow that patches model-produced residual states between
  positive and negative valence prompts to scan layer and prompt-span sufficiency;
  completed evidence-to-context-to-final layer localization; frozen
  calibration and unchanged held-out confirmation both returned `success=true`
- [Sector and context follow-up proposal](docs/sector-context-followup/proposal.md) and [discovery report](docs/sector-context-followup/report-discovery.md) —
  A: cross-sector identity-header state patching, B: cross-sector context overriding
  under negative evidence, C: L16 instruction-context Jacobian-lens readout; protocols
  frozen with discovery runs recorded for all three; no calibration or test verdict yet

### Workflow operations

- [Baseline trial proposal](docs/baseline-trial/proposal.md) and [prompt-analysis reproducibility report](docs/baseline-trial/report-reproducibility.md)
- [Technology header-span sensitivity proposal](docs/span-sensitivity/proposal.md) and [report status](docs/span-sensitivity/report-status.md)
- [Qwen Jacobian-lens selection proposal](docs/jacobian-lens-selection/proposal.md) and [selection report](docs/jacobian-lens-selection/report.md)
- [Interactive prompt-lens dashboard](docs/interactive-prompt-lens-dashboard.md)
- [Archived workflow operations](docs/archive/README.md) — counterfactual patching,
  8-K/10-K dataset preparation, synthetic entity-bias pilot, and easy-bias feasibility

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

The lightweight smoke example uses `unsloth/Llama-3.2-1B-Instruct`; current
Technology J-space experiments use Qwen3.5-4B with its model-specific canonical lens.
Download the smoke model into the ignored `.cache/` directory when needed:

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
[Qwen Jacobian-lens selection](docs/jacobian-lens-selection/proposal.md); do not
replace a canonical lens with a small smoke fit.

## Minimal smoke workflow

```bash
uv lock --check
uv run pytest -q
uv run jacobian-lens fit \
  --model .cache/models/llama-3.2-1b-instruct \
  --calibration-prompts 16
uv run baseline-trial inspect-input \
  --input data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv
```

For the complete baseline-trial stage sequence, see
[Baseline trial plan prompts](docs/baseline-trial/proposal.md). The
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
node --check llm_bias/static/attribution_dashboard.js
```
