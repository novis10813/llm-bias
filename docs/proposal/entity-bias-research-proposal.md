# Mechanistic Analysis and Inference-Time Control of Entity Bias in Large Language Models

**Document status:** research proposal and design specification. The proposal
combines an implemented experimental foundation with future mechanistic and
intervention work. Claims below are hypotheses and planned evaluations unless
explicitly marked as current implementation.

## 1. Motivation and problem formulation

Entity information can influence a language-model prediction through several
mechanisms:

- memorisation of historical events associated with an entity;
- knowledge of outcomes that occur after the prediction timestamp;
- entity popularity or frequency in the pretraining corpus;
- stable positive or negative reputation associated with a company;
- relationships between companies, executives, products, and locations;
- legitimate prior information about the entity; and
- contextual interaction between the entity and the event described in the
  input.

In financial sentiment analysis, the same semantic content can produce a
different prediction when a company name is visible, anonymised, or replaced by
another company. Henry's report, *No Name, No Gain: Unpacking Entity Bias in LLM
Financial Sentiment*, reports that retaining company identifiers increases
long--short return predictability. It also reports that smaller models are more
sensitive to current-state events, while larger models show stronger revisions
for forward-looking events, and that entity effects become larger for large-cap
firms in larger models.

Anonymisation establishes that entity information matters, but it removes both
harmful leakage and potentially useful entity-dependent reasoning. The central
problem is therefore not simply whether to remove names. It is to determine how
entity information is represented, when it changes a prediction without support
from the prompt, and whether that influence can be selectively controlled at
inference time.

The central research question is:

> How does entity information affect model computation internally, and can the
> corresponding influence be selectively controlled without retraining the
> model or destroying legitimate entity-dependent reasoning?

The project distinguishes **entity effect** from **entity bias**. A raw versus
anonymous gap may combine legitimate prior information, historical reputation,
training-data memorisation, lexical/tokenization effects, temporal leakage, and
unsupported entity preferences. Behavioural differences alone are not enough to
assign a causal or normative interpretation.

## 2. Research objectives

### Objective 1: Identify factors associated with entity bias

Measure how the following variables change the raw--anonymous or raw--swap
prediction gap:

- entity popularity, media exposure, and market capitalisation;
- industry and historical entity reputation;
- model family and model size;
- mention type, including company, ticker, product, executive, and location;
- event category and polarity;
- current-state versus future-action language;
- temporal distance from the model knowledge cutoff; and
- direct versus indirect entity reference.

### Objective 2: Trace the internal mechanism

Use causal interpretability and transported representation readouts to identify:

- entity tokens and contextual tokens that influence the output;
- layers at which entity influence emerges or is amplified;
- attention heads, MLP blocks, and residual-stream components that transmit the
  signal; and
- whether the final prediction depends more on event semantics, entity
  identity, or stored entity-related information.

### Objective 3: Develop inference-time intervention

Develop a selective intervention that detects excessive or unsupported entity
influence, modifies the relevant internal representation, and preserves event
semantics and legitimate entity information without requiring full model
retraining.

### Objective 4: Evaluate generalisation

Evaluate whether the mechanism transfers across model families and scales, and
to at least one additional entity-sensitive task such as political stance,
product recommendation, institutional reputation, or news credibility.

## 3. Research questions

- **RQ1 — Factors:** Which entity, text, temporal, and model factors affect the
  raw--anonymous prediction gap?
- **RQ2 — Components:** Which tokens, layers, attention heads, MLP blocks, and
  residual representations transmit entity influence?
- **RQ3 — Causality:** Does replacing or patching an internal entity
  representation directly change the output toward an anonymised or
  counterfactual condition?
- **RQ4 — Control:** Can entity-induced prediction changes be reduced without
  reducing sentiment accuracy, useful entity information, calibration, or
  output stability?
- **RQ5 — Shared mechanisms:** Do small and large models, and different model
  families, use shared or distinct entity-sensitive pathways?

## 4. Evidence strategy and method roles

The project uses four evidence stages:

```text
behavioural decomposition
    -> causal localisation
    -> representation interpretation
    -> selective intervention
```

### 4.1 Behavioural counterfactuals

The primary financial dataset holds the filing context and expected outcome
constant while varying entity identity. The planned condition families are:

- `real_vs_real`;
- `real_vs_anonymous`;
- `real_vs_synthetic`; and
- `synthetic_vs_synthetic`.

The current V1 materialises five pairing strategies because `real_vs_real` has
both a same-industry `matched_exposure` strategy and a cross-industry
neutral/stress strategy. The exact data contract, review gates, and artifact
schema are maintained in [the counterfactual dataset protocol](../counterfactual-dataset-generation.md).

These contrasts help separate event semantics, entity priors, reputation
transfer, memorised identity, name-form effects, and temporal leakage. They do
not by themselves prove that an effect is harmful or unsupported.

### 4.2 Activation patching for causal localisation

For a fixed outcome margin, source and target forwards are compared with a
patched forward in which a selected residual representation is transferred from
source to target. The primary quantities are the direct entity effect, the
representation signal, and the causal patch effect. Bidirectional patching,
non-entity controls, unrelated or random controls, interpolation controls, and
paired statistics are required before a layer or component is treated as a
credible causal mediator.

The current implementation supports residual activation patching and
variable-length span alignment in the main workflow. Attention-output,
MLP-output, head-level, and path-level tracing remain future work.

### 4.3 Jacobian-based mechanistic analysis

For token representation `h_t` and a task score `s(x)`, token influence can be
summarised by:

```text
I_t = || d s(x) / d h_t ||_2
```

For entity positions `E` and contextual positions `C`, compare:

```text
I_entity  = sum(I_t for t in E)
I_context = sum(I_t for t in C)
R_entity  = I_entity / (I_entity + I_context + epsilon)
```

The same quantities can be evaluated at each layer. Layer-wise plots can reveal
where entity influence emerges, grows, or is suppressed. Raw--anonymised
residual differences provide a complementary representation-level signal.

Jacobian-lens readouts are **transported representations**: they map an
intermediate residual state into a vocabulary-space readout using an estimated
Jacobian. They are useful for interpreting candidate representations and
comparing raw, anonymous, swapped, and patched conditions. They are not a
chain-of-thought trace, a discrete reasoning path, an attention map, or
standalone causal evidence. Causal claims must come from controlled
interventions and their matched controls.

### 4.4 Guidance-inspired inference-time intervention

The intervention design is inspired by the controllability framing of classifier
guidance and classifier-free guidance in diffusion models: a direction and a
strength parameter provide a dose-response axis. The analogy is methodological,
not an assertion that transformer residual dynamics are diffusion dynamics.

Candidate intervention directions can be estimated from paired raw--anonymous
or entity-swap residual differences. Candidate forms include:

1. direct entity-token attenuation as a simple baseline;
2. removal of a contrastive mean direction;
3. low-rank projection/subtraction from an entity-difference subspace; and
4. Jacobian-guided weighting of directions that affect the current output.

An adaptive risk score may combine entity dominance, swap sensitivity,
raw--anonymous gap, exposure information, and temporal-leakage indicators. Any
risk score and intervention strength must be calibrated on held-out data rather
than selected from the evaluation examples.

Intervention layers should be selected using the causal localisation results,
then tested with a layer-by-strength sweep. A strong effect at a high-Jacobian
layer is useful causal corroboration, but does not make the Jacobian readout
itself causal.

## 5. Evaluation framework

Every intervention evaluation must report the following separately.

### Bias magnitude and steering efficacy

Measure the change in a fixed outcome probability or logit margin after
intervention. Report both absolute effect and reduction of the raw--anonymous or
raw--swap gap. Do not interpret differences between unrelated token top-1
probabilities as causal effects.

### Dose response

Sweep intervention strength `s` or `alpha` and test whether the effect changes
monotonically or predictably. Report the full curve, not only the strength with
the largest effect. Repeat the sweep across candidate layers and show a
layer-by-strength heatmap.

### Specificity and task preservation

Check sentiment accuracy, macro-F1 where applicable, calibration, confidence
stability, event-semantic retention, legitimate entity information, unrelated
financial questions, non-entity prompts, output format, coherence, latency,
and memory overhead.

### Baselines and controls

The original entity-substitution comparison is the unguided behavioural
baseline. It must be compared with simple attenuation, mean-difference
subtraction, random-direction controls, non-entity position controls, and
matched synthetic controls at comparable perturbation norms.

### Statistical discipline

Use content-level paired sampling, bootstrap or sign-flip/permutation tests,
effect sizes, confidence intervals, and multiple-comparison correction for
layer/component scans. Exploratory layer selection must be separated from
confirmatory evaluation data.

## 6. Current maturity and claim boundaries

### Implemented foundation

- model-specific Jacobian-lens fitting and artifact validation;
- residual activation patching with entity-span alignment;
- EDGAR 8-K staging and provenance-preserving cleaning; and
- entity-only counterfactual-data protocol, annotation workflow, review gate,
  pairing, model-specific rendering, and validation code.

### Active validation

- manual review and promotion of the counterfactual dataset;
- full variable-length patch integration coverage and model smoke artifacts;
- residual/representation metrics beyond the existing compact readouts; and
- paired controls and statistics suitable for confirmatory causal claims.

The counterfactual-data pipeline is implemented, but a draft annotation run is
not a validated research dataset. No bias conclusion should be based on rows
that have not passed the review and promotion gates.

### Future research

- attention, MLP, head, and path-level component tracing;
- guidance-inspired selective intervention and risk gating;
- systematic layer-by-strength intervention evaluation;
- cross-model and cross-task generalisation; and
- the separate [J-space evaluation design](../j-space-evaluation.md), which is
  an optional auxiliary preflight for synthetic task-local J-space-candidate
  comparison rather than part of this proposal.

## 7. Expected contributions

1. A controlled entity-only counterfactual protocol that separates four
   condition families and five V1 pairing strategies.
2. A causal and mechanistic analysis that distinguishes direct entity effects,
   representation signals, and causal transfer.
3. A layer-aware, guidance-inspired intervention framework with dose-response
   and specificity evaluation.
4. A reproducible approach to comparing entity-sensitive pathways across local
   models, supported by model-specific lens provenance and the same entity-only
   causal protocol, controls, and statistics for every model.

## 8. Timeline and target venue

The current planning target is **NAACL 2027**, with the following backward-planned
phases:

1. **Scope alignment:** finalise metrics, evidence roles, and the position of
   Henry's entity-swap findings as motivation and an unguided baseline.
2. **Dataset construction:** complete entity pairs, no-context/with-context
   probing, specificity holdouts, and external outcome data where required.
3. **Mechanistic analysis:** run baseline probing, extract Jacobian signals,
   localise causal layers, and establish controls.
4. **Intervention pipeline:** implement the selected direction and layer logic,
   then run strength sweeps.
5. **Full evaluation:** measure steering efficacy, dose response, specificity,
   layer-by-strength interactions, and baseline comparisons.
6. **Analysis and consolidation:** perform statistical analysis, error analysis,
   and figure consolidation.
7. **Draft and advisor review:** prepare the full paper and complete feedback
   cycles.
8. **Revision and submission:** finalise the paper and confirm the applicable ARR
   cycle and commitment deadline from the official conference sources.

The dates in the planning notes are scheduling assumptions and must be verified
before they are presented as official submission deadlines.

## 9. References

- Dhariwal, P. and Nichol, A. (2021). *Diffusion Models Beat GANs on Image
  Synthesis*. NeurIPS 2021. arXiv:2105.05233.
- Ho, J. and Salimans, T. (2021/2022). *Classifier-Free Diffusion Guidance*.
  NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications;
  arXiv:2207.12598.
- Henry's report: *No Name, No Gain: Unpacking Entity Bias in LLM Financial
  Sentiment*. Internal/related report; bibliographic metadata is not specified
  in the current repository.
- Gurnee et al. (2026). *Verbalizable Representations Form a Global Workspace
  in Language Models*. arXiv:2607.15495. This reference motivates the Jacobian
  lens readout and the working-space concept; it is not a claim that this
  project directly establishes a global workspace. A separate, optional,
  non-runnable [J-space evaluation design](../j-space-evaluation.md) proposes a
  synthetic task-local preflight for comparing candidate evidence across local
  models; it does not gate or replace the entity-only causal protocol.
