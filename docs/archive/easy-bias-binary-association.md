# Easy-bias binary-association feasibility workflow

**Status:** exploratory feasibility workflow. It is not the reviewed financial
entity-bias dataset and does not replace the M1–M5 financial protocol.

This workflow re-runs the small Traditional Chinese career prompt from
`../easy-bias` with a local, gradient-accessible model. The first target is
Qwen3.5-4B with its model-specific canonical Jacobian lens. The experiment asks
whether the following complete chain is technically and statistically
observable:

```text
career association
  -> residual entity-span mediation
  -> train-only residual direction
  -> held-out directional steering
```

A Jacobian-lens readout is a transported representation diagnostic. It is not a
reasoning trace, attention map, global workspace result, or causal proof. A
steering result is association control evidence, not a claim about real parental
outcomes or population gender ratios.

## Data contract

The source career list is `../easy-bias/expanded_careers.json`. Prompt constants
are extracted from `inference.py` and `compare_option_order.py` using Python AST
literal evaluation; the easy-bias LangChain/API runtime is never imported.

Each career is assigned once to `train`, `calibration`, or `confirmation` using
the SHA-256 hash of `seed || career`. Each career has two prompt orders:

- `dad_first`: `爸爸還是媽媽`;
- `mom_first`: `媽媽還是爸爸`.

The renderer preserves the exact system/user prompts, applies the selected
model's chat template, maps both career occurrences to tokenizer-specific spans,
and records candidate continuation token IDs. Candidate scores use:

```text
log P(candidate | prompt)
  = sum_i log P(candidate_token_i | prompt, previous candidate tokens)
```

This supports multi-token candidates. The fixed association margin is:

```text
logP(媽媽 continuation | prompt) - logP(爸爸 continuation | prompt)
```

A rendered pair is `task_type=binary_association`, not
`task_type=entity_bias`. It has no `expected_outcome` field because this prompt
has no externally verified gold parent.

## CLI

The commands are exposed through the existing entry point:

```bash
uv run counterfactual-patching prepare-binary-association \
  --model <qwen3.5-4b-path-or-id> \
  --output-dir artifacts/qwen3.5-4b/easy-bias-zh-tw-binary-v1/prepare \
  --seed 0
```

The preparation stage writes:

```text
prepare/
  careers.jsonl
  rendered_prompts.jsonl
  pairs.jsonl
  omissions.jsonl
  prepare_metadata.json
```

Any failed render is recorded in `omissions.jsonl`; it is never silently
removed. The pair builder emits deterministic source-to-target and reverse
pairs, with two career spans on each side. Pair validation rejects mismatched
career replacement, missing occurrences, inconsistent token spans, wrong
candidate specification, or incompatible task semantics.

Before using Jacobian-lens diagnostics, validate the model-specific canonical
lens without fitting a new lens:

```bash
uv run counterfactual-patching validate-binary-lens \\
  --model <qwen3.5-4b-path-or-id> \\
  --lens artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt \\
  --output <run>/lens_validation.json
```

This checks model width, layer coverage, model identity, binary/metadata hashes,
and complete canonical provenance. A failed check is a compatibility failure;
it must not trigger implicit lens fitting.

Run the baseline scorer before any layer selection:

```bash
uv run counterfactual-patching baseline-binary-association \
  --model <qwen3.5-4b-path-or-id> \
  --rendered artifacts/qwen3.5-4b/easy-bias-zh-tw-binary-v1/prepare/rendered_prompts.jsonl \
  --output artifacts/qwen3.5-4b/easy-bias-zh-tw-binary-v1/baseline_scores.jsonl
```

The baseline artifact contains compact margins, candidate token IDs and counts,
input IDs, spans, split, order, and provenance. It does not contain raw hidden
states.

A single-layer multi-occurrence patch smoke run is available as:

```bash
uv run counterfactual-patching run-binary-patch \
  --model <qwen3.5-4b-path-or-id> \
  --pairs <prepare>/pairs.jsonl \
  --layer <train-selected-layer> \
  --output <run>/patch_results.jsonl
```

The patch primitive uses normalized-nearest mapping independently for each
career occurrence. It does not insert/delete sequence tokens, average
activations, or synthesize replacement states.

Fit and evaluate the train-derived steering direction with:

```bash
uv run counterfactual-patching fit-binary-direction \\
  --model <qwen3.5-4b-path-or-id> \\
  --baseline <run>/baseline_scores.jsonl \\
  --layer <calibration-selected-layer> \\
  --output <run>/direction.pt

uv run counterfactual-patching run-binary-steering \\
  --model <qwen3.5-4b-path-or-id> \\
  --baseline <run>/baseline_scores.jsonl \\
  --direction <run>/direction.pt \\
  --alpha -2 -1 0 1 2 \\
  --split confirmation \\
  --output <run>/steering_results.jsonl
```

Use `--direction-variant random` or `--direction-variant permuted` for the
norm-matched control runs, and `--span-policy final-non-entity` for the
position-injection control. The default `fitted`/`entity` combination consumes
only the train-derived direction artifact at the locked entity spans. All
variants retain the same locked layer and split.

Create order-stratified baseline, patch, and steering summaries with:

```bash
uv run counterfactual-patching summarize-binary-association \\
  --baseline <run>/baseline_scores.jsonl \\
  --patch <run>/patch_results.jsonl \\
  --steering <run>/steering_results.jsonl \\
  --output <run>/binary_summary.json
```

The summary uses career-level paired units for option-order comparisons and
reports bootstrap intervals plus sign-flip tests; it does not pool prompt-order
rows as independent observations.

## Pre-registered gates

### G0 — rendering

The locked career population must render both orders, map both career
occurrences, and round-trip both candidate continuations. Model, tokenizer,
template, input and output identities must be recorded. If the locked population
cannot be retained, stop and declare a rendering failure rather than redefining
the population after inspection.

### G1 — association rather than option position

On confirmation, report each order separately and pooled: margin distribution,
order effect, career-level same-sign agreement, and margin correlation with a
paired bootstrap interval. If the effect is almost entirely caused by option
position, the result is labelled `option_order_sensitivity`, not a stable career
association.

### G2 — residual causal localisation

Layer scanning is restricted to train. Calibration locks one layer/span policy
before confirmation. The patch runner emits entity-span patching, final
non-entity, matched random non-entity, and permuted-target controls. The
statistical unit is the unordered career pair; layer rows and order rows are not
independent observations.

If entity patching does not move the margin in the direction predicted by the
source/target direct effect and exceed controls, do not fit or evaluate a
steering direction. Report behavioural association without residual mediation
evidence.

### G3 — directional steering

Fit a normalized `mother-associated - father-associated` residual direction from
train records only. Calibration may choose layer, direction variant, occurrence
policy, and a fixed strength grid. Confirmation may not refit or reselect any of
these. Compare `+alpha` and `-alpha` against norm-matched random directions,
permuted directions, and non-entity-position injection. Report the full dose
response and greedy output-format preservation.

The direction artifact contains only the aggregate vector and compact metadata:
source record IDs/hash, dimension, layer, quantile thresholds, model/tokenizer
identity, and provenance. It must not contain example-level activations or
gradients.

## Artifact layout and claim boundary

A recommended run root is:

```text
artifacts/<model-slug>/easy-bias-zh-tw-binary-v1/runs/<run-id>/
  manifest.json
  prepare/
  baseline/
  patch/
  direction/
  steering/
  summaries/
```

Pass the same `--manifest <run-root>/manifest.json` to each binary stage. Each
stage registers input/output hashes and record counts while leaving the run in
`running` state. After the final summary, close the lifecycle explicitly:

```bash
uv run counterfactual-patching finalize-binary-manifest \\
  --manifest <run-root>/manifest.json
```

The current CLI stages can also be run with explicit paths without a manifest.
Do not merge these artifacts with reviewed 8-K `entity_bias` results or use
factual normalized transfer for this task.

The strongest possible positive result is:

> Under the declared model, tokenizer, prompt template, career population,
and holdout split, a career-associated binary margin is reproducible across
option order; a matched residual span mediates part of the source/target
movement; and a train-derived residual direction predictably controls that
margin under held-out, norm-matched controls.

It is not evidence that the model's association is factually correct, that a
specific occupation is intrinsically parental, or that a general de-biasing
intervention has been demonstrated.
