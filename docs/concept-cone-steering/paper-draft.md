# Where Investment Decision Steering Is Accessible: Layer-Localized Stance Axes and Multi-dimensional Operators in an LLM

## Abstract

**Placeholder:** Rewrite after the Introduction, study design, and result claims are finalized.
The abstract should mention entity-dependent investment behavior in one sentence, then focus on
inference-time representation steering, operator comparison, and the distinction between fixed-token
margin movement and greedy decision changes.

## 1. Introduction

Language models are increasingly used to summarize evidence, compare alternatives, and recommend
actions. In an investment prompt, changing only the named company can change the model's buy/sell
output even when the explicit evidence is held fixed. This entity-dependent behavior motivates a
more focused question: how can an LLM's investment stance be controlled at inference time, and
where in its residual representation does such control have leverage?

Prior work showed that a scalar intervention on one neuron can increase or decrease its activation
and shift a model's aggregate buy/sell stance; Park et al. (2026) refer to this control as an
investment dial. This result demonstrates that investment stance is controllable without changing
the prompt or model weights. It does not, however, establish which layer and prompt span provide a
useful intervention site, whether a multi-dimensional operator offers a broader control surface
than a single neuron, or whether a change in a fixed-token readout reliably changes the model's
generated decision.

We are motivated by the gap between observing that an LLM's investment stance is controllable in
its internal representation and knowing whether that control can be localized, extended beyond a
single neuron, and translated into generated decisions. The financial setting adds a further
requirement: a changed recommendation should be considered alongside its sensitivity to the
evidence provided in the prompt. A neuron-level intervention can reveal a control channel without
showing which evidence supports the resulting output. We therefore evaluate generated decisions
and evidence sensitivity alongside internal margin changes.

Recent refusal-steering work provides a methodological precedent for this question. Arditi et al.
(2024) showed that adding or removing a residual-stream refusal direction can bidirectionally
control generated behavior, while Wollschläger et al. (2025) argued that refusal may occupy a
multi-dimensional concept cone rather than a single functional direction. We transfer this
intervention perspective from safety refusal to investment stance. The transfer is non-trivial:
refusal is a safety behavior elicited by harmful or benign requests, whereas our target is a
template-conditioned buy/sell generation; our operators are constructed from company-level
decision-margin contrasts, and we evaluate both fixed-token readouts and complete greedy
decisions.

We study inference-time representation steering under a
fixed JSON decision template. We first use layer-wise
residual patching to identify where differences associated with the entity, evidence, instruction,
and final spans transfer a fixed buy/sell readout. We then compare three steering mechanisms at the
selected layer: the single-neuron scalar intervention described above; a token-wise
difference-in-means (DIM) direction formed by subtracting average residual states for low- and
high-margin companies; and a four-dimensional cone formed from sector-demeaned contrastive-SVD
directions.

This paper makes three contributions:

1. **Layer-localized steering analysis.** We identify the layer and span at which residual
   patching has the largest measured leverage on the fixed-token investment readout.
   will be analyzed with their own selected layers.
2. **Comparison of representation-steering operators.** We characterize the dose-response of a
   single-neuron scalar intervention, a one-dimensional DIM direction, and a multi-dimensional
   cone. The learned
   operators shift margins monotonically and flip greedy outputs in small construction-cohort
   pilots, while the recorded matched-norm random control produces little movement in one pilot.
3. **Decision-level steering boundaries.** We separately measure fixed-token margin movement and
   greedy JSON decisions. The two can diverge: strong adverse evidence can preserve a `sell`
   generation after the margin becomes positive, and a direction that moves a named prompt also
   moves an identity-stripped prompt.

Taken together, we frame investment stance as an inference-time steering problem. Our analysis asks
where residual interventions are effective, whether multi-dimensional operators provide a useful
control surface, and how changes in fixed-token readouts relate to generated decisions.
Entity-dependent behavior motivates this setting, while the main focus is how reliably an LLM's
decision stance can be controlled under a fixed decision protocol.

## 2. Related Work

### 2.1 Entity Sensitivity and Stance Control in Financial LLMs

Large language models deployed in economic and financial decision-making exhibit systematic
evaluation biases and pretraining memorization (Lopez-Lira et al., 2025; Kong et al., 2026),
as well as latent firm- and sector-level preferences (Lee et al., 2026).
Even when explicit evidence, task instructions, and decision schemas are held fixed, altering only
the named company can alter the model's recommendation due to pre-existing corporate and sectoral
biases (Hu & Zhao, 2026; Elbouanani et al., 2026). Prior work predominantly treats this behavior as an
auditing, benchmarking, or prompt-masking challenge. In contrast, Park et al. (2026) study the
same phenomenon from the control side, showing that a scalar intervention on a single neuron
(an "investment dial") can continuously calibrate an LLM's aggregate buy/sell prior at inference
time without modifying model weights or prompts.

While an investment dial demonstrates that investment stance is controllable via internal
representations, its intervention acts on an aggregate prior rather than a localized sequence span,
and it is not compared against multi-dimensional representations. Furthermore, in high-stakes
financial recommendation, an altered stance must be weighed against explicit prompt evidence.
Prior work documents that chain-of-thought rationales and post-hoc explanations in language models
frequently exhibit unfaithfulness, justifying decisions shaped by internal priors or superficial
cues rather than the provided facts (Turpin et al., 2023), especially when evaluating complex
decision-making under uncertain contexts (Jia et al., 2024). This motivates our evaluation
protocol: we examine both internal continuation margins and complete greedy JSON decisions alongside
evidence sensitivity, assessing when internal stance control genuinely alters downstream behavior
rather than merely producing rationalized readouts.

### 2.2 Operator Geometries: From Neurons to Multi-Dimensional Cones

Inference-time representation steering controls model behavior by adding targeted vectors to
intermediate hidden states during the forward pass (Turner et al., 2023; Zou et al., 2023;
Panickssery et al., 2024). In safety alignment, Arditi et al. (2024) demonstrate that refusal
behavior is largely mediated by a single linear direction in the residual stream, which can be added
or ablated to bidirectionally govern generated compliance. Such one-dimensional difference-in-means
(DIM) directions provide a lightweight intervention mechanism, but assume that the underlying
behavioral concept is adequately captured by a single functional axis.

Recent work reveals that complex model behaviors often exceed one-dimensional linear representations.
Wollschläger et al. (2025) show that refusal behavior occupies a multi-dimensional polyhedral
"concept cone," within which infinite valid steering directions exist. Crucially, they establish
that geometric orthogonality does not imply independence under intervention, introducing
*representational independence* to identify functionally decoupled directions and mitigate
non-linear saturation. Beyond linear vectors, multi-dimensional and invertible latent
transformations have also been proposed to achieve broader and more flexible behavioral control
(Nguyen & Le, 2026).

We transfer these geometric insights from safety refusal to structured, template-conditioned
investment decisions. Rather than examining a single operator in isolation, we conduct a controlled
cross-operator comparison: we evaluate a single-neuron scalar intervention (Park et al., 2026), a
one-dimensional token-wise DIM direction, and a four-dimensional concept cone constructed from
sector-demeaned contrastive SVD under matched-norm controls. This comparison tests whether
multi-dimensional polyhedral operators offer smoother, more monotonic control surfaces than
scalar or one-dimensional counterparts in financial reasoning.

### 2.3 Localizing Leverage: Weight Editing vs. Inference-Time Patching

A prominent branch of mechanistic interpretability uses activation patching and causal tracing to
localize factual associations before directly modifying model parameters (Meng et al., 2022, 2023).
In particular, ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023) identify critical MLP layers
that mediate subject-attribute recall and compute closed-form low-rank updates to rewrite the
corresponding MLP weight matrices ($W_{\text{proj}}$). Similarly, internal attention and feedforward
manipulations have been developed to diagnose and eliminate biases by suppressing specific internal
components (Zhou et al., 2024), while recent studies search for sparse, localized "entity cells" in
MLP layers that encode firm- or individual-level identities (Yona et al., 2026).

While parameter-editing methods permanently alter model weights, they risk collateral damage to
general capabilities and lack dynamic, context-dependent adjustability. In contrast, our approach
operates entirely at inference time on prefill residual representations, leaving all model parameters
untouched and allowing continuous, on-demand dosage tuning. Furthermore, whereas ROME and MEMIT
focus on single subject-token positions to update static factual memories, we execute a systematic
span-by-layer residual patching sweep across entity, evidence, instruction, and answer-prefix spans
to measure where the decision readout is most susceptible to steering.

Finally, while gradient-based attribution frameworks provide fine-grained token-level causal maps
(Liu et al., 2026), we interpret our localized intervention site (Layer 15 in Qwen3.5-4B) strictly
as an empirical locus of measured steering leverage over the fixed decision readout. We explicitly
refrain from claiming that this layer constitutes the historical origin of entity bias or a static
repository of financial beliefs.

## 3. Study Design

We formulate investment stance steering as an intervention on the residual representation during
prompt prefill. The current instantiation uses Qwen3.5-4B, a 32-layer decoder model with hidden
size 2560. The protocol selects the intervention layer separately for each model; the layer index
reported for Qwen3.5-4B should therefore be read as a model-specific result.

### 3.1 Task and measurements

Each prompt contains a company identity, a financial evidence block, and an instruction requiring
a single JSON object with a `buy` or `sell` decision and a short reason. We measure a continuation
margin at the answer prefix,

$$M = \log p(\text{buy}) - \log p(\text{sell}),$$

using the same fixed answer-token scorer for every condition. This margin measures the movement of
the fixed-token readout. We separately generate the complete response with greedy decoding
($T=0$, at most 48 new tokens), parse the JSON decision, and record whether the output is `buy`,
`sell`, or unparsed. A positive margin and a generated `buy` are therefore separate outcomes.

The canonical balanced prompt uses the frozen decision template. Evidence-condition probes vary
the evidence block while keeping the company and decision format fixed. Because the exploratory
scripts previously used both frozen and custom renderers, all scenario results require a single
frozen prompt renderer and a recorded prompt hash before they can be pooled in the paper.

### 3.2 Cohort and operator construction

Operator construction starts from the 200-company cohort in
`entity-to-dial-heldout-transfer-v1-01`. Companies are ranked by their clean continuation margin.
The token-wise DIM direction uses the ten highest- and ten lowest-margin companies. At each of the
100 instruction-span token positions, we subtract the mean residual state of the low-margin group
from the mean residual state of the high-margin group:

$$v_{\mathrm{DIM}}[p] = \mu_{\mathrm{high}}[p] - \mu_{\mathrm{low}}[p].$$

The multi-dimensional operator uses the twenty highest- and twenty lowest-margin companies. We
subtract the mean state of each sector before forming paired high-minus-low residual differences,
perform SVD independently at each instruction token, retain the leading four directions, and align
their signs with the positive contrast. The cone centroid is the normalized sum of these four
directions.

The current construction and pilot evaluation draw from the same 200-company cohort. They therefore
measure steering feasibility on the construction cohort; they do not establish held-out company
generalization. A generalization claim requires a company-disjoint evaluation split fixed before
operator construction.

### 3.3 Intervention and controls

For an operator $v[p]$, we add a scaled direction to the instruction-span residual states during
prefill,

$$h_{15,p} \leftarrow h_{15,p} + \alpha v[p].$$

The transform is disabled during one-token decode steps, so the intervention acts on the prompt
representation rather than repeatedly modifying the generated continuation. The operator study
compares a scalar intervention on one neuron, the one-dimensional DIM direction, and the
four-dimensional cone. The final comparison must use a common dose convention or matched
per-token intervention norm; raw alpha values alone are not comparable across operators.

The controls are matched-norm random directions and identity-stripped prompts. Random controls
test whether the learned direction has a larger effect than an arbitrary residual perturbation.
Identity-stripped prompts test whether the learned direction acts only on a company identifier or
also shifts the general investment stance. These controls do not by themselves establish semantic
independence or entity debiasing.

### 3.4 Analysis protocol

We report margin displacement, monotonicity over the dose sweep, greedy decision flips, JSON parse
rate, and the number of evaluated companies for every arm. Layer localization is selected before
the operator comparison and is evaluated with the same prompt and scoring protocol. Evidence
sensitivity is assessed by paired positive, negative, mixed, and zero-evidence conditions; generated
reasons are qualitative examples rather than a standalone measure of explanation faithfulness.

The primary behavioral claim is based on generated decision flips. Margin movement without a
corresponding greedy decision change is reported as a readout effect and is not counted as a
decision flip.

## 4. Layer-localized Steering Analysis

**Draft result structure:** Report a complete span × layer residual-patching map for entity,
evidence, instruction, and answer-prefix spans. Select the steering layer with a rule fixed before
the operator comparison, using fixed-token margin displacement as the primary localization metric
and greedy generation as a secondary check.

The existing Qwen3.5-4B development characterization makes L15 a candidate site: the stance
direction explains `R²=0.605` of the final margin variation in a 16-company layer scan, with L15
the highest value among the tested layers. This observation is not a substitute for the complete
span × layer map, and the final section should not claim that the selected layer is the origin of
entity-dependent behavior.

## 5. Representation-steering Operators

**Draft result structure:** Compare the single-neuron scalar intervention, DIM, and four-dimensional
cone on the same target set and dose convention. For each arm, report the complete dose-response
curve, intervention norm, margin displacement, greedy decision, parse rate, and matched-norm random
control.

The current frozen DIM pilot provides a decision-level feasibility result: MO, CNC, and FOXA move
from clean `sell` outputs toward `buy`, with observed flips at alpha 4, 5, and 4 respectively.
The same pilot contains one random-direction control and one identity-stripped prompt. These are
pilot observations until the multi-seed and multi-target confirmation run is stored as a compact
result artifact.

The cone construction artifact is available, but the current persisted cone evaluation is only a
smoke result at alpha 0. Claims about smoother scaling, reduced saturation, or an advantage over a
one-dimensional operator require a new output JSON with dimension ablations and matched-norm
controls. Generated reasons may illustrate different outputs, but they do not identify individual
cone axes with distinct financial concepts.

## 6. Decision-level Steering Boundaries

**Draft result structure:** Organize this section around the distinction between fixed-token margin
movement and generated decisions. For every evidence condition, report the clean and steered margin,
the generated JSON decision, the flip indicator, and parse status.

The first boundary is readout-to-generation divergence: a positive margin can coexist with a greedy
`sell`, as in the current DIM pilot for CNC. The second boundary is evidence sensitivity: strong
adverse evidence may delay or prevent a greedy flip even when the intervention moves the margin in
the opposite direction. This second result must be rerun with one frozen renderer because the
current note mixes frozen balanced prompts with custom scenario prompts.

Zero-evidence prompts and identity-stripped prompts can test whether the operator shifts a general
stance under missing company evidence. Macro-context probes are exploratory extensions and should
remain outside the primary claim unless they receive the same prompt, split, and compact-artifact
provenance as the main evaluation.

## 7. Discussion

**Placeholder:** Synthesize the result as a shared, layer-localized steering handle. Clarify that
the paper studies intervention accessibility and controllability rather than bias formation,
entity-selective debiasing, or the semantic identity of individual cone axes.

## 8. Limitations

**Placeholder:** Cover the single model, fixed template, small construction-cohort pilots, single
random control, greedy decoding, missing held-out generalization, missing full-cohort sweep,
missing dimension ablation, and lack of cross-model replication.

## 9. Ethical Considerations

**Placeholder:** Explain the risk of changing investment recommendations while preserving plausible
surface rationales. State clearly that the work is an interpretability and controllability study,
not financial advice or a trading strategy.

## 10. Conclusion

**Placeholder:** Restate that the paper transfers representation-steering methods from refusal and
related domains to investment stance, identifies L15 as a practical steering site for the tested
template, and characterizes both successful control and its decision-level boundaries.

## References

**Placeholder:** Add and verify references for Park et al., Arditi et al., Wollschläger et al.,
Meng et al., and the final representation-steering literature selected for the Related Work
section.

## Appendix A. Prompt and Measurement Details

**Placeholder:** Include the canonical prompt format, scenario-template distinction, token-span
definition, margin measurement, greedy parsing rule, and intervention timing.

## Appendix B. Reproducibility and Supplementary Materials

**Placeholder:** Describe the anonymized code/data supplement, compact derived artifacts, operator
provenance, and the analyses that remain outside the current paper claims.
