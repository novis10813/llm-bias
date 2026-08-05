# Entity-bias research roadmap

**Status:** active execution roadmap for the entity-bias research program.

This roadmap is the single planning source for the current entity-bias work. It
separates research design from implementation documentation and separates
`code/protocol implemented` from `validated artifact/evidence complete`.
The conceptual rationale is in the [research proposal](entity-bias-research-proposal.md);
commands, schemas, and artifact contracts remain in the linked execution docs.

## Scope and dependencies

The active path is:

```text
reviewed counterfactual data
  -> model-specific rendered pairs
  -> direct / representation / causal baselines
  -> controls and paired statistics
  -> selective intervention
  -> cross-model and cross-task evaluation
```

The separate [J-space evaluation design](../j-space-evaluation.md) is an
optional auxiliary preflight derived from the working-space concept in the
Jacobian-lens literature. It evaluates synthetic task-local J-space-candidate
evidence and may make compatibility checks and cross-model protocol diagnostics
faster. It is not a substitute for entity-only data or domain-specific causal
patching, and it is not a blocker or evidence gate for the M1–M6 milestones.

## Schedule

The dates below follow the current backward-planned schedule supplied in the
research proposal. They are planning targets, not completed results or official
conference deadlines. M1 and M2 may overlap because review/promotion and
model-specific rendering are both part of the dataset-construction window.

| Milestone | Planned window | Expected completion |
|---|---|---|
| M1 — Review and promotion closure | 2026-08-04 – 2026-08-10 | Promoted, provenance-complete `validated_content.jsonl` |
| M2 — Rendered pair readiness | 2026-08-11 – 2026-08-17 | Validated model-specific pair manifests and smoke artifacts |
| M3 — Mechanistic analysis baseline | 2026-08-18 – 2026-08-31 | Layer-level Jacobian and causal-localisation results |
| M4 — Controls and confirmatory statistics | 2026-08-18 – 2026-08-31 | Control-complete effect, uncertainty, and significance tables |
| M5 — Selective intervention evaluation | 2026-09-01 – 2026-09-07 | Guidance-inspired intervention implementation and dose-response results |
| M6 — Cross-model and cross-task generalisation | 2026-09-08 – 2026-09-14 | Comparable model/task experiment results and manifests |
| Analysis and consolidation | 2026-09-15 – 2026-09-21 | Statistical analysis, error analysis, and final figures |
| Draft and advisor review | 2026-09-22 – 2026-09-28 | Full paper draft and first advisor feedback cycle |
| Revision and submission preparation | 2026-09-29 – 2026-10-11 | Revised manuscript and submission-ready package |

## Current status

| Workstream | Code/protocol status | Validated evidence status |
|---|---|---|
| Model-specific Jacobian lenses | Fitting, metadata, validation, and Llama/Qwen workflows exist | Cross-model conclusions are not established |
| Residual span patching | Core mapping and hooks exist; variable-length support is under integration validation | Full model smoke coverage and all acceptance artifacts remain pending |
| Entity-only counterfactual data | Annotation, review bundle, promotion, four families, five pairing strategies, rendering, and validation code exist | Draft rows still require review/promotion; no unpromoted draft is research-ready |
| Representation readout | Compact transported readouts and outcome margins exist | Formal residual-distance/divergence evidence is pending |
| Causal controls/statistics | Basic non-entity control and exploratory summaries exist | Bidirectional controls, independent sampling, paired tests, and correction are pending |
| Selective intervention | Research design only | Not implemented or evaluated |
| Cross-model/task evaluation | Model loaders and model-specific lens paths exist | No standardised cross-model/task result exists |
| Optional J-space-candidate preflight | Proposed non-runnable design exists | `jspace_eval` package and CLI are not implemented |

## Data terminology

The counterfactual dataset has **four condition families**:

- `real_vs_real`
- `real_vs_anonymous`
- `real_vs_synthetic`
- `synthetic_vs_synthetic`

V1 materialises **five pairing strategies** because `real_vs_real` has two
strategies:

1. same-industry `matched_exposure`;
2. cross-industry neutral/stress;
3. identity removal;
4. memorised identity; and
5. name-form baseline.

`matched_exposure` is based on historical filing exposure and is not market-cap
or size matching. The authoritative schema, review gate, and omission behavior
are documented in [8-K counterfactual entity dataset](../counterfactual-dataset-generation.md).

## Milestones

### M1 — Close review and promotion (2026-08-04 – 2026-08-10)

- Complete the review bundle and required reviewer fields.
- Meet the registrant recall, entity precision/recall, grounding, semantic
  outcome, and identity-leakage gates.
- Promote only rows that pass the gate and preserve correction provenance.
- Record rejected and omitted rows instead of silently relaxing constraints.

**Exit evidence:** a promoted `validated_content.jsonl`, a review manifest, and a
provenance record showing that the dataset is not merely an annotation draft.

### M2 — Render and validate model-specific pairs (2026-08-11 – 2026-08-17)

- Build the four condition families and five V1 pairing strategies from promoted
  content.
- Materialise forward and reverse pairs.
- Render tokenizer-specific entity spans for each target model.
- Validate span boundaries, prompt alignment, identity leakage, sequence limits,
  and compact artifact contents.
- Complete deterministic tests for 1→1, 1→2, 2→1, and 2→3 span mappings, plus
  model-backed smoke artifacts where available.

**Exit evidence:** validated rendered pair manifests for each model, with source
and target spans, position mappings, omissions, and no raw activations.

### M3 — Establish the entity-bias baseline (2026-08-18 – 2026-08-31)

- Use a fixed outcome margin such as `logit(positive) - logit(negative)`.
- Report `direct_entity_effect`, representation signal, and
  `causal_patch_effect` as separate quantities.
- Compare raw, anonymous, real-swap, synthetic, and patched conditions under
  the same context and expected outcome.
- Run layer-wise Jacobian readouts as transported representation evidence, not
  as causal claims.

**Exit evidence:** compact per-pair/per-layer results and a reproducible report
that never mixes factual answer-transfer metrics with bias-pair margins.

### M4 — Add controls and confirmatory statistics (2026-08-18 – 2026-08-31)

- Run source→target and target→source patches.
- Add unrelated/random entity, matched synthetic, non-entity position, residual
  interpolation, and norm-matched controls where supported.
- Sample at the independent content level, not by treating layer rows as
  independent observations.
- Add paired bootstrap and sign-flip/permutation tests, effect sizes,
  confidence intervals, and multiple-comparison correction for scans.
- Separate exploratory layer selection from held-out confirmation.

**Exit evidence:** control-complete tables and uncertainty summaries that support
an appropriately scoped causal statement.

### M5 — Implement and evaluate selective intervention (2026-09-01 – 2026-09-07)

Only enter this milestone after M1–M4 show a stable entity-specific causal
signal that is not explained by token form, arbitrary residual perturbation, or
uncontrolled factual answer changes.

- Implement mean-difference and low-rank direction baselines.
- Select candidate layers from causal localisation, then run a layer × strength
  sweep.
- Add adaptive risk gating only after a held-out calibration design exists.
- Evaluate steering efficacy, dose response, task preservation, specificity,
  calibration, coherence, latency, and memory overhead.
- Compare against unguided entity substitution and random-direction controls.

**Exit evidence:** intervention artifacts with direction provenance, held-out
calibration, full dose-response curves, and side-effect analysis.

### M6 — Cross-model and cross-task generalisation (2026-09-08 – 2026-09-14)

- Fit and validate a separate lens for every model.
- Use normalised layer depth rather than raw layer number.
- Compare at least two model scales and, where feasible, multiple families.
- Complete one non-financial entity-sensitive task only after the financial
  protocol is stable.
- Report common mechanisms separately from model-specific results.

**Exit evidence:** a shared protocol, model-specific manifests, and complete
data→patch→statistics results for the selected model/task set.

## Optional analysis design: J-space-candidate preflight

The separate [J-space evaluation design](../j-space-evaluation.md) describes a
future, non-runnable `jspace_eval` tool that may:

- check whether a local decoder exposes the residual, unembedding, gradient, and
  hook capabilities required by the analysis;
- assess lens identifiability and baseline task competence;
- measure readability, causal necessity, and cross-operator transfer as
  synthetic task-local candidate evidence;
- segment candidate evidence into comparable layer bands; and
- emit machine-readable diagnostics that distinguish stable candidates,
  task-dependent candidates, multiple candidate bands, no detectable candidate
  evidence, unidentifiable lenses, insufficient competence, and incompatible
  models.

This is an optional model-comparison aid and preflight diagnostic. It neither
gates M1–M6 nor qualifies an entity-bias claim. Every cross-model entity-bias
result still requires that model's complete entity-only causal protocol,
controls, and statistics. Its conceptual implementation phases are not merged
into the active roadmap.

## Research-ready gate

A result may support a primary entity-bias claim only when all of the following
are true:

- source and target differ only in entity identity and share the expected
  outcome;
- the rows passed the manual review and promotion gates;
- direct entity effect, representation signal, and causal transfer are reported
  separately;
- bidirectional and matched control conditions are available;
- uncertainty is estimated with content-level paired statistics;
- model-specific lens, tokenizer, data, and configuration provenance is
  complete;
- the result is reproduced in the declared model set; and
- artifacts contain compact metrics and provenance, not raw activations or model
  weights.

Until these gates are met, results should be labelled exploratory, operational,
or draft rather than validated evidence of harmful entity bias.

## Canonical execution documents

- [Counterfactual patching](../counterfactual-patching.md)
- [8-K counterfactual entity dataset](../counterfactual-dataset-generation.md)
- [Qwen Jacobian-lens selection](../qwen-jacobian-lens-selection.md)
- [EDGAR 8-K preparation](../edgar-8k-preparation.md)
- [Prompt-analysis reproducibility](../prompt-analysis-reproducibility.md)
- [Repository constraints and verification commands](../../CLAUDE.md)
