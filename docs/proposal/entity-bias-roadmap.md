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

The separate [J-space evaluation design](../j-space-evaluation/proposal.md) is an
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
| [M1 — Review and promotion closure](#m1--close-review-and-promotion-2026-08-04--2026-08-10) | 2026-08-04 – 2026-08-10 | Promoted, provenance-complete `validated_content.jsonl` |
| [M2 — Rendered pair readiness](#m2--render-and-validate-model-specific-pairs-2026-08-11--2026-08-17) | 2026-08-11 – 2026-08-17 | Validated model-specific pair manifests and smoke artifacts |
| [M3 — Mechanistic analysis baseline](#m3--establish-the-entity-bias-baseline-2026-08-18--2026-08-31) | 2026-08-18 – 2026-08-31 | Layer-level Jacobian and causal-localisation results |
| [M4 — Controls and confirmatory statistics](#m4--add-controls-and-confirmatory-statistics-2026-08-18--2026-08-31) | 2026-08-18 – 2026-08-31 | Control-complete effect, uncertainty, and significance tables |
| [M5 — Selective intervention evaluation](#m5--implement-and-evaluate-selective-intervention-2026-09-01--2026-09-07) | 2026-09-01 – 2026-09-07 | Guidance-inspired intervention implementation and dose-response results |
| [M6 — Cross-model and cross-task generalisation](#m6--cross-model-and-cross-task-generalisation-2026-09-08--2026-09-14) | 2026-09-08 – 2026-09-14 | Comparable model/task experiment results and manifests |
| Analysis and consolidation | 2026-09-15 – 2026-09-21 | Consolidate the [activation patching](../activation-patching-causal-tracing/report.md), [sector/context](../sector-context-followup/report-discovery.md), and [span-sensitivity](../span-sensitivity/report-status.md) results |
| Draft and advisor review | 2026-09-22 – 2026-09-28 | Draft against the [canonical research proposal](entity-bias-research-proposal.md) and current evidence-readiness table |
| Revision and submission preparation | 2026-09-29 – 2026-10-11 | Apply the [research-ready gate](#research-ready-gate) before retaining primary claims |

## Current status

| Workstream | Code/protocol status | Validated evidence status |
|---|---|---|
| Model-specific Jacobian lenses | Qwen fitting, selection, promotion, and pinned-registry workflows exist; Llama usage remains in the archived patching smoke path | Cross-model conclusions and alternative-lens sensitivity/robustness are not established |
| Residual span patching | Batch mapping, hooks, and variable-length normalized-nearest mapping are implemented under `archive/llm_bias/counterfactual_patching/`; the archived interactive dashboard still rejects differing encoded shapes | Full model smoke coverage and all acceptance artifacts remain pending |
| Entity-only counterfactual data | Annotation, review bundle, promotion, four families, five pairing strategies, rendering, and validation code are frozen under `archive/llm_bias/counterfactual_data/` | Draft rows still require review/promotion; no unpromoted draft is research-ready |
| External entity covariates | Market capitalisation, media/corpus exposure, and other continuous attributes are part of the research design | No date-aligned, provenance-complete continuous covariate linkage or coverage/missingness validation exists; current synthetic evidence uses only coarse tier/sector groupings |
| Representation readout | Compact transported readouts and outcome margins exist | Formal residual-distance/divergence evidence is pending |
| Causal controls/statistics | Basic non-entity control and exploratory summaries exist; the entity-only patching protocol and code paths are available | Reviewed/promoted pairs, full entity-only patching runs, bidirectional and unrelated/random controls, independent sampling, paired uncertainty, and correction are pending |
| Entity-specific selective intervention | Research design only | Not implemented or evaluated |
| J-space sector intervention | Active `jspace-intervention` workflow implements swap/gain, dose-matched controls, valence readout, V1 token causal screening, and V2 outcome-conditioned decision-flip (Draft 1 frozen and implemented; first formal pipeline completed 2026-08-28) | Qwen3.5-4B sector held-out specificity was not supported; V1 token shortlist is empty; V2 Test run 1 returned `success=false`: Buy steering passed, while the sell-direction Holm gate could not pass with only two eligible test tickers; final-position control reproduced both directions, so evidence-position specificity was not established |
| Activation patching causal tracing | Draft 1 hierarchical residual resample patching and frozen confirmation analysis are implemented in `jspace-intervention` | Discovery localized an evidence → instruction-context → final-position sufficiency shift; the unchanged Qwen3.5-4B held-out confirmation returned `success=true` |
| Sector and context follow-up | A V1 cross-sector header-state patching, B V1 negative-evidence context overriding, C V1 L16 instruction-context readout, and B V1 frozen confirmation analysis are implemented with separate artifacts | A/B/C discovery runs are recorded. B V1 calibration reproduced the L16 context effect but failed the same-sector peer specificity gate (`success=false`, `test_authorized=false`); held-out test was not run, so no formal sector-conditioned claim is established |
| Technology header-span sensitivity | V1 header-only condition preparation, fixed Buy/Sell margin scoring, paired analysis, artifact lifecycle, and CLI are implemented | Technology discovery completed on 35 tickers. `same_sector_swap` exceeded the name-form control and is the calibration primary condition; calibration/test protocol remains unfrozen |
| Entity cell localization and downstream attribution | Proposed; not yet frozen, implemented, or run | No runs; proposal defines E1 stability-score MLP localization, E2 DLA head classification, and E3 cell-suppression causal test for Qwen3.5-4B financial header prompts |
| Cross-model/task evaluation | Model loaders and model-specific lens paths exist | No standardised cross-model/task result exists |
| Optional J-space-candidate preflight | Proposed non-runnable design exists | `jspace_eval` package and CLI are not implemented |

## Main-line alignment assessment

The current experiments follow the proposal's mechanistic-analysis method order:
behavioural sensitivity → residual causal localisation → transported readout →
matched-control confirmation. They provide strong M3 evidence and partial M4
evidence for Qwen3.5-4B:

- [Activation patching causal tracing](../activation-patching-causal-tracing/proposal.md)
  has a held-out `success=true` result for the evidence → instruction-context →
  final-position sufficiency shift.
- [Sector/context follow-up B V1](../sector-context-followup/proposal.md) reproduced
  the L16 context effect in calibration, but its
  [confirmation report](../sector-context-followup/report-confirmation.md) records
  `success=false` because the same-sector peer specificity gate failed.
- [Technology header-span sensitivity](../span-sensitivity/proposal.md) found a
  behavioural `same_sector_swap` effect in discovery, but calibration/test remain
  unfrozen.

These experiments therefore fit the mechanistic-analysis main line, but they do
not complete the primary entity-bias claim. M1/M2 reviewed entity-only data are
still missing, sector specificity was not confirmed, and the results have not
been reproduced across the declared model set. The current defensible claim is
Qwen3.5-4B identity-conditioned context-state sensitivity under the active
fixed-evidence task, not validated harmful entity bias or a sector-specific
mechanism.

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
are documented in [archived 8-K counterfactual entity dataset](../archive/counterfactual-dataset-generation.md).

## Milestones

### M1 — Close review and promotion (2026-08-04 – 2026-08-10)

**Related protocols:** [archived counterfactual dataset protocol](../archive/counterfactual-dataset-generation.md)
and [archived EDGAR preparation protocol](../archive/edgar-8k-preparation.md).

- Complete the review bundle and required reviewer fields.
- Meet the registrant recall, entity precision/recall, grounding, semantic
  outcome, and identity-leakage gates.
- Promote only rows that pass the gate and preserve correction provenance.
- Record rejected and omitted rows instead of silently relaxing constraints.

**Exit evidence:** a promoted `validated_content.jsonl`, a review manifest, and a
provenance record showing that the dataset is not merely an annotation draft.

### M2 — Render and validate model-specific pairs (2026-08-11 – 2026-08-17)

**Related protocols:** [archived counterfactual patching](../archive/counterfactual-patching.md),
[archived counterfactual dataset rendering](../archive/counterfactual-dataset-generation.md),
and [artifact identity contract](../artifact-contract.md).

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

**Related experiments:** [baseline trial](../baseline-trial/proposal.md),
[Technology header-span sensitivity](../span-sensitivity/proposal.md),
[Jacobian-lens selection](../jacobian-lens-selection/proposal.md),
[J-space valence readout](../jspace-valence-readout/proposal.md),
[activation patching causal tracing](../activation-patching-causal-tracing/proposal.md),
and [sector/context follow-up](../sector-context-followup/proposal.md).

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

**Related reports:** [activation patching held-out confirmation](../activation-patching-causal-tracing/report.md),
[sector/context discovery](../sector-context-followup/report-discovery.md),
[B V1 confirmation](../sector-context-followup/report-confirmation.md),
[J-space sector intervention held-out evaluation](../jspace-sector-intervention/report.md),
and [Technology header-span sensitivity discovery](../span-sensitivity/report-status.md).

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

**Related experiments:** [J-space sector intervention](../jspace-sector-intervention/proposal.md),
[J-space token experiment index](../jspace-token-experiments/README.md),
[V1 token causal screen](../jspace-token-experiments/proposal-v1.md), and
[V2 outcome-conditioned decision flip](../jspace-token-experiments/proposal-v2.md).

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

**Related protocols:** [model-specific Jacobian-lens selection](../jacobian-lens-selection/proposal.md),
[baseline trial model workflow](../baseline-trial/proposal.md), and the optional
[J-space candidate preflight](../j-space-evaluation/proposal.md). No completed
cross-model/task experiment currently satisfies this milestone.

- Fit and validate a separate lens for every model.
- Use normalised layer depth rather than raw layer number.
- Compare at least two model scales and, where feasible, multiple families.
- Complete one non-financial entity-sensitive task only after the financial
  protocol is stable.
- Report common mechanisms separately from model-specific results.
- Repeat the synthetic localization protocol with at least one justified alternative
  lens condition and report peak-depth, sign, and metric-agreement sensitivity before
  treating a localization pattern as lens-robust.

**Exit evidence:** a shared protocol, model-specific manifests, and complete
data→patch→statistics results for the selected model/task set.

## Optional analysis design: J-space-candidate preflight

The separate [J-space evaluation design](../j-space-evaluation/proposal.md) describes a
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

- [Counterfactual patching (archived)](../archive/counterfactual-patching.md)
- [8-K counterfactual entity dataset (archived)](../archive/counterfactual-dataset-generation.md)
- [Qwen Jacobian-lens selection](../jacobian-lens-selection/proposal.md)
- [EDGAR 8-K preparation (archived)](../archive/edgar-8k-preparation.md)
- [J-space sector intervention proposal](../jspace-sector-intervention/proposal.md) and [report](../jspace-sector-intervention/report.md)
- [J-space valence vocabulary readout proposal](../jspace-valence-readout/proposal.md) and [report](../jspace-valence-readout/report-technology-discovery.md)
- [J-space token experiment versions](../jspace-token-experiments/README.md)
- [J-space token causal screen V1 proposal](../jspace-token-experiments/proposal-v1.md) and [report](../jspace-token-experiments/report-v1.md)
- [J-space outcome-conditioned decision-flip V2 proposal](../jspace-token-experiments/proposal-v2.md) and [report](../jspace-token-experiments/report-v2.md)
- [Activation patching causal tracing proposal](../activation-patching-causal-tracing/proposal.md) and [report](../activation-patching-causal-tracing/report.md)
- [Sector and context follow-up proposal](../sector-context-followup/proposal.md), [discovery report](../sector-context-followup/report-discovery.md), and [B V1 confirmation report](../sector-context-followup/report-confirmation.md)
- [Technology header-span sensitivity proposal](../span-sensitivity/proposal.md) and [discovery report](../span-sensitivity/report-status.md)
- [Entity cell localization and downstream attribution proposal](../entity-cell-localization/proposal.md) (proposed; not yet implemented)
- [Prompt-analysis reproducibility](../baseline-trial/report-reproducibility.md)
- [Shared experiment core](../shared-experiment-core.md)
- [Artifact identity and run manifest contract](../artifact-contract.md)
- [Research scripts reference](../research-scripts.md)
- [Repository constraints and verification commands](../../CLAUDE.md)
