# Entity Cell Localization and Downstream Attribution: Proposal

**Document status:** proposed; not yet frozen or implemented.
**Model for first run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)
**Depends on:** existing split manifest
`artifacts/qwen3.5-4b/jspace-intervention/splits.json`, active baseline prompts
`data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv`, canonical lens
`artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt`.

## Motivation

Activation patching ([proposal](../activation-patching-causal-tracing/proposal.md),
[report](../activation-patching-causal-tracing/report.md)) confirms that
Qwen3.5-4B's Buy/Sell margin depends on residual state at specific
layer/span combinations: evidence content dominates at L6, the
post-evidence `instruction_context` carries sufficient state at L16,
and the final decision position converges by L30.

Header-state patching ([sector/context follow-up B V1](../sector-context-followup/proposal.md))
reproduces the L16 context effect when company identity changes. Span
sensitivity discovery finds that same-sector peer swaps shift the margin
more than ROT13 name-form controls, suggesting the model encodes
something beyond surface character form.

What none of these experiments identify is **which internal units encode
a specific company and how their output reaches the decision-relevant
layer band**. Without that, we cannot design a guided suppression that
is precise enough to separate entity-conditioned influence from evidence-
driven reasoning.

Two published methods supply the missing mechanism steps:

- **Barzilay et al. (2026)**, *Friends and Grandmothers in Silico:
  Localizing Entity Cells in Language Models* (arXiv:2604.01404): a
  stability-score procedure that finds the single MLP neuron per entity
  that acts as a sparse canonical handle in early layers (L0–L5 in
  Qwen2.5-7B), validated by targeted amnesia ablation and placeholder
  injection.
- **Chughtai, Cooney, and Nanda (2024)**, *Summing Up the Facts:
  Additive Mechanisms Behind Factual Recall in LLMs* (arXiv:2402.07321,
  ICML 2024): a Direct Logit Attribution (DLA) decomposition by source
  token group that classifies attention heads into Subject Heads (read
  entity-position activations and write entity-correlated vocabulary
  logits), Relation Heads (read instruction-position activations), and
  Mixed Heads.

The experiment combines these two methods on our existing fixed-evidence
financial task. The central question is:

> Do early-layer MLP entity cells for specific companies connect causally
> to the downstream attention heads that write entity-correlated logits
> into the Buy/Sell decision?

Confirming that connection is the minimal mechanistic foundation needed
before designing a selective guidance intervention under M5.

## Research questions

- **RQ-E1 (localization):** Which MLP neuron(s) in L0–L5 have the
  highest stability score for each financial-task ticker under the
  current financial header prompts? Do entity cells generalize across
  prompt phrasings?
- **RQ-E2 (downstream attribution):** Which attention heads write
  entity-position-sourced logits into the Buy/Sell margin at L14–L17?
  Can they be classified as Subject, Relation, or Mixed by the DLA
  ratio gate?
- **RQ-E3 (causal link):** Does negatively ablating a ticker's entity
  cell reduce that ticker's Subject-Head DLA at downstream layers, and
  does the Buy/Sell margin shift toward the anonymised condition?

## Experiment design

### Input population

All three phases use the discovery tickers from the existing split
manifest (257 tickers across all sectors; 35 Technology discovery
tickers used in span sensitivity discovery form the primary pilot set).

Primary prompt source: `data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv`,
column `prompt_with_context_attribute_0`. Each prompt has a two-line
bracketed header:

```text
Stock Ticker: [TICKER]
Stock Name: [Company Name]
```

followed by a fixed evidence body and JSON instruction.

> **V1 scope note:** V1 uses only the financial header prompts. The
> stability scoring across `K` prompt variants (see Phase E1) samples
> from the available `prompt_with_context_attribute_*` columns for the
> same ticker, treating each attribute-variant as one draw from the
> entity's prompt distribution. A future version may augment with
> factual QA templates (e.g., `"The sector of [TICKER] is"`) to test
> whether entity cells are task-local or world-knowledge cells; that
> extension requires a separate frozen proposal before inference.

### Phase E1: Entity Cell Localization

**Method:** stability-score MLP neuron localization following
Barzilay et al. (2026), adapted to the financial header prompt family.

**Hook point:** the post-activation function scalar just before
`down_proj` (i.e., the gate-projected intermediate activation
$a_{\ell j}(x)$) at the last token of the entity header span. For
each prompt, the entity span covers the `[TICKER]` and `[Company Name]`
tokens; the last token of the span is the closing `]` of the name line.

**Generic baseline:** 399 non-financial, non-company prompts drawn from
a fixed seed sample of generic cloze templates. Used to compute
per-neuron mean $\mu_{\ell j}$ and standard deviation $\sigma_{\ell j}$
for normalization:

$$z_{\ell j}(x) = \frac{a_{\ell j}(x) - \mu_{\ell j}}{\sigma_{\ell j} + \varepsilon}$$

**Stability score:** for entity $e$ with $K$ prompt variants
$\{x_1, \ldots, x_K\}$ (drawn from `prompt_with_context_attribute_*`
columns, at most 10 per ticker):

$$S_{\ell j}(e) = \frac{\left(\mathbb{E}_i[z_{\ell j}(x_i)]\right)^2}{\operatorname{Std}_i[z_{\ell j}(x_i)] + \varepsilon}$$

The candidate entity cell is $(\ell^*, j^*) = \arg\max_{\ell \in \{0,\ldots,5\}, j} S_{\ell j}(e)$.
Report top-5 candidates per ticker alongside the top-1.

**Causal validation (amnesia test):** for the top-1 candidate, scale
the activation across all prompt tokens by $\alpha \in \{1, 0, -1, -2, -3\}$
and measure the change in Buy/Sell margin $M$ on the original financial
prompt. A credible entity cell shows margin shift when ablated for the
target ticker but not for a control ticker whose cell is at a different
neuron.

**Output per ticker:** candidate layer, neuron index, stability score,
top-5 table, amnesia $\Delta M$ curve, control ticker margin change.

**Artifact:** compact JSON per ticker; no raw activations, no hidden
states. Only neuron index, layer, score, and margin deltas.

### Phase E2: Downstream Attribution via Direct Logit Attribution

**Method:** DLA decomposition by source token group following Chughtai
et al. (2024), applied at each attention head in L10–L20.

**DLA by source group:** for head $h$ at layer $l$, decompose its
contribution to the final token's output logit into three source groups:

- **Entity group:** tokens in the entity header span (ticker + name).
- **Evidence group:** tokens in the evidence body.
- **Instruction group:** tokens in the post-evidence instruction and
  assistant-prefix spans.

The per-group DLA for head $(l, h)$ is:

$$\tilde{\pi}^{l,h,\text{group}} = \sum_{k \in \text{group}} \alpha_{l,h,k} \cdot \operatorname{LayerNorm}(z_k^{l-1} W_V W_O) W_U$$

where $\alpha_{l,h,k}$ is the attention probability from final position
to source position $k$, and $W_U$ is the unembedding matrix. Report the
DLA projected onto the Buy logit minus Sell logit axis.

**Head classification (following Chughtai et al. 2024):** aggregate
DLA ratios across all discovery prompts for the same ticker set:

- **Subject Head:** entity-group DLA / instruction-group DLA > 10.
- **Relation Head:** instruction-group DLA / entity-group DLA > 10.
- **Mixed Head:** intermediate ratio.

These labels describe the dominant source-token routing pattern; they
are not claims about head function in other tasks.

**Jacobian-lens cross-check:** for the same heads identified as Subject
or Mixed, apply the existing canonical Jacobian-lens readout to the
head's output vector. The transported vocabulary distribution shows
whether the head is writing entity-correlated tokens (company names,
sector terms, valence terms) into the residual stream. This is a
descriptive transported readout, not a causal claim.

**Output per head:** source-group DLA decomposition (entity, evidence,
instruction), Buy/Sell axis projection, head classification label,
Jacobian-lens top-10 tokens.

**Artifact:** compact JSON of head DLA tables and lens readout; no full
attention weight matrices or raw hidden states.

### Phase E3: Entity Cell Suppression and Causal Tracing

**Method:** targeted negative ablation of the located entity cell
followed by DLA re-measurement and margin change assessment.

**Suppression:** for a target ticker's entity cell $(\ell^*, j^*)$,
scale $a_{\ell^* j^*}(x)$ by $\alpha$ across entity header tokens only
(not all positions). Sweep $\alpha \in \{1, 0.5, 0, -1, -2, -3\}$.

**Margin measurement:** record $M$ after suppression; compute
$\Delta M_\alpha = M_\alpha - M_\text{original}$.

**DLA re-measurement:** at each $\alpha$, recompute the E2 DLA
decomposition for the Subject and Mixed heads identified in Phase E2.
Record the change in entity-group DLA for each suppression level.

**Primary causal contrast:** does Subject-Head entity-group DLA decrease
monotonically with $\alpha$? Does $\Delta M_\alpha$ move toward the
anonymised margin (from span sensitivity, `anonymous_identity` mean
$\Delta M \approx -0.59$)?

**Control conditions:**

- **Self-control:** suppress the same neuron for a different ticker
  whose entity cell is at a different position; confirm the other
  ticker's margin does not shift.
- **Wrong-neuron control:** suppress the top-2 stability-score neuron
  of a different ticker on the target ticker's prompt; confirm no
  amnesia.
- **Evidence-driven baseline:** on prompts where evidence strongly
  determines the decision (margin $|M| > 2$), confirm that cell
  suppression does not flip the decision.

**Output per (ticker, $\alpha$):** $\Delta M$, Subject-Head DLA delta,
control ticker $\Delta M$.

**Artifact:** compact JSON of suppression curves; no raw activations or
gradient tensors.

## Claim boundaries and interpretation limits

- Phase E1 identifies a **candidate entity cell** under financial header
  prompts. It does not prove the cell is necessary for all entity
  knowledge, nor that it is the unique encode point for company identity.
- Phase E2 DLA classification assumes approximate linearity through the
  unembedding. Non-linear LayerNorm interactions mean the decomposition
  is an approximation. Head labels are task-local and prompt-family
  local; they should not be generalized to other tasks without separate
  validation.
- Phase E3 ablation tests **causal sufficiency in the suppression
  direction**: if ablating the cell shifts the margin toward anonymous,
  the cell is a contributing encode point. Absence of shift does not
  prove the cell is irrelevant; distributed or redundant codes may
  compensate.
- Jacobian-lens readout is a transported representation readout, not a
  causal claim. Vocabulary tokens appearing in the top-k are plausible
  entity-correlated content; they do not establish that the model is
  reasoning about those tokens.
- All results are specific to Qwen3.5-4B and the current financial
  header prompt family. Cross-model and cross-task transfer requires
  separate experiments.

## Relation to M3/M4/M5 milestones

| Phase | Roadmap milestone | Evidence type |
|---|---|---|
| E1 (localization) | M3 — mechanistic baseline | Locates the early entity-encoding site; fills the gap between header-state patching effect and its internal origin |
| E2 (DLA) | M3/M4 — controls | Decomposes the entity-to-decision information route; classifies heads by source routing; cross-checks with Jacobian-lens transported readout |
| E3 (suppression) | M4 confirmatory + M5 prerequisite | Tests causal link from entity cell to decision margin; provides the entity-specific signal required before designing M5 guidance |

Phase E3 success (entity-specific margin shift with monotonic DLA
decrease, controls clean) is the minimal gate before entering M5
selective intervention. If E3 fails or produces distributed effects
across many neurons, M5 design must account for superposition and cannot
rely on single-neuron suppression.

## Discovery protocol

The discovery run uses the 35 Technology discovery tickers from the
existing split manifest. No calibration or test split data are used at
any phase of discovery.

Stability-score ranking, head classification thresholds, and
suppression $\alpha$ sweep values are fixed at the values specified
above before any model inference. Results may be used to refine the
protocol for a versioned V2, but they do not retroactively change the
V1 gates.

## Statistical reporting

- All ticker-level results reported as equal-ticker mean, 95% bootstrap
  CI (10,000 resample, ticker-level), and exact sign-flip test.
- Monotonicity of the $\alpha$ suppression curve assessed by Kendall's
  $\tau$ across the six $\alpha$ levels, reported per ticker and
  aggregated.
- No multiple-comparison correction is applied within Phase E1 neuron
  scoring (ranking, not hypothesis testing). Amnesia test $\Delta M$
  reported as a descriptive effect size; Holm correction applied only
  if a formal significance test is added in calibration.

## References

- Barzilay, D., Karasik, M., and Geva, M. (2026). *Friends and
  Grandmothers in Silico: Localizing Entity Cells in Language Models*.
  arXiv:2604.01404.
- Chughtai, B., Cooney, A., and Nanda, N. (2024). *Summing Up the
  Facts: Additive Mechanisms Behind Factual Recall in LLMs*. ICML 2024;
  arXiv:2402.07321.
- Activation patching causal tracing: [proposal](../activation-patching-causal-tracing/proposal.md)
  and [report](../activation-patching-causal-tracing/report.md).
- Sector and context follow-up: [proposal](../sector-context-followup/proposal.md),
  [discovery report](../sector-context-followup/report-discovery.md), and
  [B V1 confirmation report](../sector-context-followup/report-confirmation.md).
- Technology header-span sensitivity: [proposal](../span-sensitivity/proposal.md)
  and [discovery report](../span-sensitivity/report-status.md).
- Entity-bias research proposal: [proposal](../proposal/entity-bias-research-proposal.md).
- Entity-bias roadmap: [roadmap](../proposal/entity-bias-roadmap.md).
