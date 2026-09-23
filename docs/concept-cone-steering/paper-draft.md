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

**Placeholder:** Position the paper at the intersection of:

- inference-time activation steering and representation engineering;
- refusal removal and refusal steering as the methodological lineage;
- single-direction versus multi-dimensional concept representations;
- causal tracing and layer-wise activation patching.

The final section should explain why this paper transfers those methods to investment stance rather
than claiming a new general-purpose steering algorithm.

## 3. Study Design

**Placeholder:** Define the model, company cohort, canonical and scenario prompt templates,
continuation margin, greedy JSON decision, layer-15 prefill intervention, DIM construction,
four-dimensional cone construction, and matched controls.

## 4. Layer-localized Steering Analysis

**Placeholder:** Report the span × layer residual-patching map and explain why the instruction-span
peak identifies a practical steering site without claiming that it is the origin of entity bias.

## 5. Representation-steering Operators

**Placeholder:** Compare the single-neuron dial, DIM direction, and multi-dimensional cone. Report
dose-response behavior, pilot greedy flips, anonymous-prompt behavior, random-direction control,
and the exploratory anti-saturation comparison.

## 6. Decision-level Steering Boundaries

**Placeholder:** Separate fixed-token margin movement from greedy decision changes. Report evidence
polarity, strong adverse evidence, zero-evidence prompts, sampled rationale behavior, and the
descriptive macro-context probe.

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
