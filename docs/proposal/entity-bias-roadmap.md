# Entity-influence research roadmap

**Status:** revised execution roadmap; program incomplete. Behavioural confirmation and mechanistic analysis are the current main deliverables. Cross-model and cross-task validation are both required. Selective intervention is a separate follow-up and does not gate this round.

The [research proposal](entity-bias-research-proposal.md) defines the questions and claim limits; the [research directory](../README.md) links all experiment protocols and results. This revision changes program scope, not historical experiment gates or authorizations.

## Original milestones remain traceable without being relabelled as completed

Early work retained the original method order: behavioural analysis, transported readout, causal localisation and matched controls. Later questions expanded into factual-memory/decision differences, global stance modulation, entity-gap localisation and instruction-state subspaces.

| Original milestone and planned window | Actual evidence or divergence | Revised planning role |
|---|---|---|
| M1: 8-K review/promotion, 2026-08-04–08-10 | The [counterfactual dataset workflow](../archive/counterfactual-dataset-generation.md) is archived; promotion is not established by the active baseline runs. | Historical, not completed. Active data provenance and contrast validity replace restoration as the current requirement. |
| M2: model-specific counterfactual rendering, 2026-08-11–08-17 | Active experiments use their own prepared baseline prompts, tokenization and spans, not the archived four-family dataset. | Historical contract retained in archive; new model/task preparation still needs its own validation. |
| M3: mechanistic baseline, 2026-08-18–08-31 | Readout and activation patching followed the plan; Entity Cell, Balanced Evidence Gap and Entity-to-Dial extended the mechanism questions. | Substantial Qwen3.5-4B evidence exists, with experiment-specific limits; not completion of the original entity-only contract. |
| M4: controls and confirmation, 2026-08-18–08-31 | Activation patching held-out confirmation passes; sector/context specificity and selected-neuron hypotheses fail; several scans remain discovery. | Preserve each verdict and require confirmation appropriate to any retained primary claim. |
| M5: selective intervention, 2026-09-01–09-07 | Sector/token steering did not establish entity selectivity; Investment-dial supports global stance calibration. [Selective-intervention V1](../selective-intervention/report.md) now completes a removal test: G1a/G1b/G2 pass, G3/G4 fail; full-strength gate fail. | Moved to independent follow-up, not marked passed and not required for this round. |
| M6: cross-model/task generalisation, 2026-09-08–09-14 | No completed standardised comparison establishes the current mechanism claims across models and tasks. | Required and incomplete; dedicated protocols and runs remain to be planned. |

The original consolidation (2026-09-15–09-21), draft/advisor review (09-22–09-28) and revision/submission preparation (09-29–10-11) windows are historical assumptions, not current completion promises. NAACL 2027 remains the original planning target, not a verified deadline. No replacement schedule is assigned by this revision.

## Current evidence separates completed experiments from established claims

| Research line | Evidence status | Role and limit |
|---|---|---|
| [Baseline trial](../baseline-trial/report.md) and [lens selection](../jacobian-lens-selection/report.md) | Reproducibility and instrument reports exist; completion is dataset/run-specific. | Foundation, not entity-bias certification. Lens choice alone does not establish lens robustness. |
| [Span sensitivity](../span-sensitivity/report.md) | V1 Technology discovery completed; calibration/test protocol unfrozen. | Behavioural header sensitivity, not held-out confirmation. |
| [Valence readout](../jspace-valence-readout/report.md) | Technology discovery completed. | Representation candidates, not causal proof. |
| [Sector intervention](../jspace-sector-intervention/report.md) and [J-space token V1/V2](../jspace-token-experiments/proposal.md) | Sector held-out specificity unsupported; V1 shortlist empty; V2 formal success=false, position specificity unestablished. | Limits of these direction/control choices, not proof that all selective intervention is impossible. |
| [Activation patching](../activation-patching-causal-tracing/report.md) | Draft 1 unchanged held-out confirmation success=true. | Valence-driven evidence→instruction context→final sufficiency shift, not entity-specific validation. |
| [Sector/context follow-up](../sector-context-followup/report.md) | A/B/C discovery completed; B V1 calibration success=false, held-out test not run. | Context-state effect does not establish sector specificity. |
| [Entity Cell](../entity-cell-localization/report.md) | Main line closed; V3 four factual cells confirmed in calibration/hold-out; E4 probe completed but proposed. | Factual memory certification and decision probes are separate evidence, not an entity-control success. |
| [Financial-soundness localisation](../financial-soundness-localization/report.md) and [causal validation](../financial-soundness-causal-validation/report.md) | Exploratory V1 completed; no formal certification. | Related measurement branch; no basis for declaring financial neurons absent. |
| [Investment-dial](../investment-dial/report.md) | Closed; V2 calibration pass, B reevaluation, not fresh held-out validation. | Global stance modulation; independent method replication, not entity-specific control. |
| [Balanced Evidence Gap](../balanced-evidence-gap/report.md) | Phase 1 behaviour confirmed; Phase 2 discovery completed; Phase 3 causal gate fails, 0/3 confirmed. | Entity gap and layer-band evidence; negative result limited to selected coordinates and interventions. |
| [Entity-to-Dial](../entity-to-dial/report.md) | Phase A–F completed and closed; L15 k=8 effect ratio 0.983 in tested directions; F1 additivity fails. | Instruction-state subspace transfer, not held-out generalisation, complete circuit recovery or selective gap reduction. |
| [Selective-intervention V1](../selective-intervention/report.md) | Formal `selective-intervention-v1-gpu-bf16-01` completed; G1a/G1b/G2 pass, G3/G4 fail. | Gap reduction and random-control specificity in the frozen population, but full-strength global/anonymous shifts fail preservation gates; not successful selective control. |
| [J-space evaluation](../j-space-evaluation/proposal.md) | Optional, proposed, non-runnable. | Auxiliary design only; not a required milestone. |

Use the [Balanced Evidence Gap Phase 2 report](../balanced-evidence-gap/details/report-phase2.md) for its estimands and revision history: Phase 1 compares named/anonymous prompts within company-specific evidence; Phase 2A uses shared evidence. Rev 1 failure remains a failure after Rev 2 reevaluation. Formal run completion does not promote discovery results into independent confirmation.

## Remaining work is ordered by dependencies, not the old dates

### Consolidate the Qwen evidence before choosing transfer claims

For each proposed primary claim, bind its contrast, population, prompt family, fixed-answer metric, intervention, controls, run ID and evidence status. Separate entity effects from evidence-valence transfer and factual recall. Check summary claims against phase reports, retaining protocol revisions and null results. Record which findings still need independent confirmation; do not reopen closed experiments by editing their gates.

**Completion evidence:** a claim-to-run comparison using existing reports, with no unsupported promotion of discovery or normative harmful-bias claims.

### Define comparable model and task protocols before new runs

Select at least two model scales and, where feasible, different families, plus at least one non-financial entity-sensitive task. Models, tasks, samples, splits and decision thresholds are not selected here. Establish task competence, identity contrasts, fixed-answer definitions, data provenance, independent sampling and analogous intervention operations. Validate tokenization/spans and conduct required model-backed smoke tests before formal runs.

Use normalised depth for layer comparisons. Fit/install and validate separate canonical lenses only for transported-readout arms; test alternative lens conditions before claiming lens robustness. Neither the Qwen layer/channel coordinates nor k=8 is a universal target.

**Completion evidence:** approved independent protocols, validated prepared inputs and instrument checks. This roadmap itself is not run authorization.

### Complete both cross-model and cross-task validation

Run the selected behavioural contrasts and causal tests with matched controls, paired uncertainty and correction where scanning is involved. Separate discovery selections from confirmation. Each comparison needs task-appropriate model competence and comparable evidence conditions; record failures and exclusions rather than silently changing thresholds.

**Completion evidence:** model/task-specific manifests, compact results and reports stating which patterns replicate, differ or are unresolved. A valid difference or null result can satisfy the evaluation requirement; identical mechanisms are not required. Neither model loaders nor a second task name alone satisfies this stage.

## Selective intervention remains a separate research line

The original M5 designs include attenuation, mean-difference subtraction, low-rank removal and Jacobian-weighted directions. Any future efficacy claim still requires dose response, specificity, preservation of task performance and legitimate entity information, and independent evaluation. Do not treat Investment-dial calibration or Entity-to-Dial transfer as satisfying those conditions.

[Selective-intervention V1](../selective-intervention/report.md) is independent follow-up work. Formal run `selective-intervention-v1-gpu-bf16-01` is now complete: G1a/G1b/G2 pass, but G3/G4 fail, so the frozen decision table yields a full-strength negative result. Its [report](../selective-intervention/report.md) preserves that verdict; this completed test does not mark M5 passed. Its success remains unnecessary for the present mechanism study, and this roadmap authorizes no further runs.

## Research-ready gate

The program is complete only when:

- primary claims specify evidence contrasts, populations and metrics, separating behaviour, causal transfer, factual recall and transported readout;
- datasets pass their owning protocol's validation, with identity-only differences where claimed; archived 8-K promotion is not silently counted as complete or imposed on unrelated active data;
- causal claims have the relevant bidirectional, no-op and matched controls, with scope consistent with the tested intervention;
- uncertainty uses appropriate independent units and scan correction, and discovery is not presented as untouched confirmation;
- required cross-model **and** cross-task evaluations are complete under their independent protocols, with positive, null and model-/task-specific findings reported;
- model/tokenizer/configuration/data provenance is complete, lens identity is bound where used, and outputs contain compact derived results rather than raw activations; and
- claims about harmful bias, useful-information preservation or selective control are omitted unless separately supported.

The current program does **not** meet this gate: cross-model and cross-task evidence remains missing. Individual Qwen lines may be closed without the overall study being complete. The gate assesses evidence completeness, not whether an intervention succeeds.
