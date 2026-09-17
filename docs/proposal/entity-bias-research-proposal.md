# Behavioural Confirmation and Mechanistic Analysis of Entity Influence in Large Language Models

**Status:** revised research program; cross-model and cross-task validation remain required and incomplete. The current contribution is behavioural confirmation and mechanistic analysis. Selective inference-time control is a separate follow-up, not a completion requirement. This program revision changes neither completed experiment protocols nor their gates, verdicts, or run authorizations.

## 1. Entity influence is the question; harmful bias requires additional evidence

Henry's *No Name, No Gain: Unpacking Entity Bias in LLM Financial Sentiment* motivates studying prediction changes when company identifiers are retained, anonymised, or replaced. Such changes may reflect factual memory, reputation, lexical/tokenization effects, temporal leakage, legitimate priors, or interactions with supplied evidence.

The core question for this study is: **when does entity identity change a fixed-answer decision, where does the relevant state affect computation, and which findings hold across models and tasks?**

An **entity effect** is a measured change under an identity contrast. **Entity bias** requires an interpretation beyond that contrast. An **unsupported preference** means an identity-conditioned preference not justified by the supplied evidence under the task's declared decision rule; an anonymous baseline or a margin difference alone does not establish that interpretation. The current experiments do not identify temporal leakage or establish harmful bias.

## 2. Objectives and research questions follow the evidence without presupposing success

| Objective | Question and intended output | Current boundary |
|---|---|---|
| 1. Confirm behaviour | RQ1: Under controlled evidence, when do named–anonymous and identity-swap contrasts change fixed-answer margins? Report robustness to the tested prompt forms and evidence ordering. | Market-cap, exposure, temporal and mention-type factor decomposition from the original proposal remains unestablished; it is not a promised completed contribution. |
| 2. Localise and test mechanisms | RQ2: Which positions, layers and components carry decision-relevant state? RQ3: Which controlled state transfers change the decision, and which candidate mechanisms fail their own controls? | Sufficiency, necessity, local sensitivity and representation readout must remain distinct. |
| 3. Distinguish mechanisms from controllability | RQ4: How do factual entity memory, entity-conditioned decision transfer and global stance modulation differ in the tested interventions? | RQ4 replaces the original selective-control deliverable; reducing entity influence while preserving useful information remains a follow-up question. |
| 4. Test generalisation | RQ5: Which behavioural and mechanistic findings are shared or model-/task-specific across model families/scales and a non-financial entity-sensitive task? | Both cross-model and cross-task validation are required. Qwen3.5-4B results alone cannot complete the program. |

## 3. The actual research path retains early methods and changes the later mechanism questions

Early baseline preparation, lens validation, transported readout, activation patching and matched controls followed the original analysis strategy. The original 8-K counterfactual implementation is now [archived](../archive/README.md); its uncompleted review/promotion gates are not retroactively satisfied by active `data/baseline/` experiments.

Later experiments distinguish three questions that the original roadmap did not separate sufficiently:

- **Factual memory versus decisions:** [Entity Cell](../entity-cell-localization/report.md) changed from header/frame candidate screening to V3 fact-level amnesia validation. Its decision probes and E4 readouts answer different questions from cell certification. The separate [financial-soundness localisation](../financial-soundness-localization/proposal.md) and [causal validation](../financial-soundness-causal-validation/proposal.md) remain exploratory.
- **Global stance versus entity differences:** [Investment-dial](../investment-dial/report.md) reproduced a global stance-calibration method. [Balanced Evidence Gap](../balanced-evidence-gap/report.md) then tested entity-induced decision gaps and their layer/component dependence.
- **Candidate neurons versus residual state:** [Entity-to-Dial](../entity-to-dial/report.md) tested paths suggested by the earlier results and examined L15 instruction-state differences. Its subspace transfer and failed compact-mechanism hypotheses constrain interpretation; they do not establish a selective debiasing method.

The [roadmap](entity-bias-roadmap.md) maps these lines to the original milestones. The [research directory](../README.md) records experiment relationships; individual protocols remain the source of operational definitions.

## 4. Each method answers a separate part of the mechanism question

### Behavioural contrasts require matched evidence and a fixed outcome

Balanced Evidence Gap Phase 1 compares named and anonymous versions within company-specific evidence. Its cross-company named margins contain evidence differences. Phase 2A instead holds shared evidence fixed and varies identity. Do not merge these estimands or describe Phase 1 as identical evidence across all companies; see the [Rev 2 protocol](../balanced-evidence-gap/details/proposal-phase2-rev2.md).

The archived four-family/five-strategy counterfactual design remains historical. It is not a required restoration step for the active program. New model/task datasets still require explicit provenance, identity-only contrasts where claimed, task-appropriate answer definitions and independent evaluation units.

### Controlled interventions test state transfer and candidate mechanisms

Residual span patching, block-contribution patching, neuron intervention and component attribution have now been exercised in separate experiments; they are no longer all future implementation work. Report exactly which operation was tested, with bidirectional conditions, self-source no-op checks and relevant matched controls. A null result for selected neurons does not exclude untested coordinates, positions or joint mechanisms. State-transfer sufficiency is not an exhaustive circuit decomposition.

### Jacobian methods interpret representations but do not establish causal paths

Token/layer gradients measure local first-order sensitivity to a fixed task score. Jacobian-lens readouts transport an intermediate representation into a vocabulary readout. Neither is an attention map, chain-of-thought trace, discrete reasoning path or standalone causal proof. Component nominations require separate causal validation; decoded semantics do not guarantee a controllable decision direction.

### Generalisation must compare mechanisms rather than identical coordinates

Retain the original cross-model ambition: compare at least two model scales and, where feasible, different families; complete at least one non-financial entity-sensitive task. Candidate models, task, sampling and acceptance criteria require dedicated protocols before runs. Compare normalised layer depth and analogous operations rather than requiring another model to reproduce L15/N8490 or an eight-dimensional subspace. A valid null or model-specific result is an outcome, not a failed research program.

Validate a model-specific canonical lens for transported readouts; do not make lens use a requirement for every causal intervention. The proposed [J-space evaluation](../j-space-evaluation/proposal.md) remains optional and non-runnable, not a generalisation gate.

## 5. Current evidence supports scoped contributions, not a completed program

| Evidence | Supported interpretation and limit |
|---|---|
| [Activation patching report](../activation-patching-causal-tracing/report.md) | Unchanged held-out confirmation supports the evidence→instruction-context→final sufficiency shift under valence contrasts, not an entity-specific or sector-specific mechanism. |
| [Entity Cell report](../entity-cell-localization/report.md) | V3 confirms four factual entity cells in calibration/hold-out; decision probes separately delimit their decision effects. E4 remains a proposed diagnostic, not a new causal certification. |
| [Balanced Evidence Gap Phase 2](../balanced-evidence-gap/details/report-phase2.md) and [Phase 3](../balanced-evidence-gap/details/report-phase3.md) | Behavioural gap confirmation and discovery localisation identify an early entity span and later instruction context. Three selected late MLP coordinates fail causal validation, 0/3 confirmed; this does not exclude all single-neuron mechanisms. |
| [Entity-to-Dial report](../entity-to-dial/report.md) | In the tested Qwen3.5-4B population/directions, L15 k=8 projection recovers a 0.983 effect ratio; Phase F additivity fails. Neither establishes held-out subspace generalisation or selective gap reduction. |
| [Investment-dial report](../investment-dial/report.md) | V2 calibration passes for global stance modulation; B is reevaluation of previously used companies, not fresh held-out entity-specific validation or numeric replication. |
| [J-space token V1/V2](../jspace-token-experiments/report.md), [sector intervention](../jspace-sector-intervention/report.md), [sector/context B V1](../sector-context-followup/report.md) | V1 shortlist is empty; V2 formal success=false and position specificity is unestablished; sector specificity was not confirmed. These limits remain part of the evidence. |

Formal execution does not make a discovery scan confirmatory. Numerical claims use their owning phase reports; the roadmap records remaining confirmation and generalisation work.

## 6. Expected contributions now centre on behaviour, mechanisms and their limits

1. A reproducible account of entity-induced decision gaps under explicit evidence and identity contrasts, without conflating factual recall, global stance and harmful bias.
2. Controlled localisation and comparison of decision-relevant positions, layer bands and residual/component interventions, including unsuccessful hypotheses and their tested scope.
3. An empirical account of the distinction between factual entity memory, entity-conditioned state transfer and global stance modulation; no universal neuron or selective-control success is presumed.
4. Cross-model and cross-task tests identifying shared findings and differences with model/task-specific provenance. This contribution is **required but not yet established**.

Selective inference-time control remains an independent follow-up. [Selective-intervention V1](../selective-intervention/report.md) has now tested low-rank removal in formal run `selective-intervention-v1-gpu-bf16-01`: gap-reduction and random-control specificity gates pass, but global/anonymous preservation gates fail, yielding a full-strength negative result. The original control ideas therefore remain unvalidated as selective-control methods. Any control claim still requires gap reduction, specificity, dose response and task/legitimate-information preservation; a strong transfer effect alone cannot substitute for them.

## 7. Completion depends on evidence quality and generalisation, not intervention success

Use paired independent sampling, effect sizes, uncertainty and scan-appropriate multiple-comparison correction. Separate discovery choices from confirmation. Freeze task definitions and controls before new model/task evaluation, retain negative results, and preserve compact artifacts and provenance without raw activations.

The [research-ready gate](entity-bias-roadmap.md#research-ready-gate) defines program completion. Closed Qwen experiment lines remain closed; subsequent confirmation/generalisation requires independent protocols, not revisions to their frozen gates.

NAACL 2027 remains the original venue planning target. Original dates are retained as historical scheduling assumptions in the roadmap; neither an official deadline nor a new completion date is asserted here.

## 8. References and method boundaries

- Henry, *No Name, No Gain: Unpacking Entity Bias in LLM Financial Sentiment*: motivating internal/related report; complete bibliographic metadata is unavailable in this repository.
- Gurnee et al. (2026), *Verbalizable Representations Form a Global Workspace in Language Models*, arXiv:2607.15495: Jacobian-lens motivation, not evidence that this project establishes a global workspace.
- Park et al. (2026), *Your AI, On a Dial: Controlling Investment Bias in LLMs with a Single Neuron*, arXiv:2608.22852: method provenance for the independent investment-dial line.
- Dhariwal and Nichol (2021), *Diffusion Models Beat GANs on Image Synthesis*, arXiv:2105.05233; Ho and Salimans (2021/2022), *Classifier-Free Diffusion Guidance*, arXiv:2207.12598: original control analogy retained only for follow-up design, not a claim that residual dynamics are diffusion dynamics.
