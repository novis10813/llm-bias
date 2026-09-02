# Entity Cell Localization and Downstream Attribution: Proposal

**Document status:** frozen V1 implementation protocol; implemented on `feat/entity-cell-confirmation`; formal inference and calibration/test evidence not run.
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
- **RQ-E2 (downstream attribution):** Which standard full-attention
  components write entity-position-sourced updates into the Buy/Sell margin?
  How do those updates differ from evidence- and instruction-sourced updates?
- **RQ-E3 (causal link):** Does suppressing a ticker's candidate entity cell
  reduce the downstream entity-sourced contribution? Does selective
  attenuation of that downstream contribution move the decision toward the
  anonymous-identity condition while preserving evidence sensitivity?

## Experiment design

### Input population

All three phases use the 35 Technology discovery tickers from the existing
split manifest. The 12 calibration and 11 test tickers remain unseen until a
separate confirmation config freezes the discovery-selected selection rules,
components, contrasts, and gates.

The financial prompt source is
`data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv`. V1 preserves the
canonical bracketed identity header:

```text
Stock Ticker: [TICKER]
Stock Name: [Company Name]
```

E1 cannot treat the existing `prompt_with_context_attribute_*` columns as
localization variants at the entity position. Those columns share the same
prefix and differ only after the header, which a causal decoder cannot see at
the entity token. E1 therefore prepares a frozen set of 12 **header-prefix
variants**: each variant changes wording before the two canonical header lines
while preserving the ticker, company name, order, brackets, and text after the
header. Eight variants form the localization set and four form an unseen
within-discovery validation set.

E2 and E3 use the frozen baseline columns
`prompt_with_context_attribute_0`, `_1`, and `_2` per ticker so that component
estimates are not tied to one evidence item. Clean-margin sign is reported as a
stratum rather than used to select prompts after inference.

> **V1 scope note:** V1 localizes financial-header cells. A future version may
> use factual QA or another entity-sensitive dataset to test whether the same
> cells support world knowledge or transfer across tasks. That extension must
> freeze its dataset, prompt family, outcome, controls, and gates in a new
> version before inference.

### Phase E1: Entity Cell Localization

**Method:** stability-score MLP neuron localization following Barzilay et al.
(2026), adapted to the financial header prompt family.

#### E1 preparation

1. Read the 35 Technology discovery identities and render 12 frozen
   header-prefix variants per identity.
2. Map the last token of the company-name content, excluding the closing `]`,
   with the shared character-to-token span helper. The mapped token must be
   non-empty and must end before evidence begins.
3. Build a versioned baseline prompt file. V1 uses the 399 generic cloze prompts
   released or transcribed from the paper's Appendix A; every prompt receives a
   reproducible ID and SHA-256. Extract baseline activation at the final token
   position of each cloze prompt prefix before generation, matching the paper's
   normalization protocol. If the exact list cannot be recovered, stop V1 before
   inference; a replacement list requires a versioned adaptation.
4. Freeze candidate layers L0–L5, $\varepsilon=10^{-6}$, top-$k=5$, localization
   variants 0–7, and validation variants 8–11 in machine-readable config.

**Hook point:** the post-SwiGLU intermediate vector immediately before
`down_proj`. For each prompt and layer, record only the 9,216-dimensional vector
at the mapped company-name token, transfer it to CPU float32, and release the
GPU tensor. Do not persist per-prompt vectors.

**Generic normalization:** compute per-neuron mean $\mu_{\ell j}$ and standard
deviation $\sigma_{\ell j}$ over the 399 baseline prompts:

$$z_{\ell j}(x) = \frac{a_{\ell j}(x) - \mu_{\ell j}}{\sigma_{\ell j} + \varepsilon}.$$

**Localization score:** for the eight localization prompts of entity $e$:

$$S_{\ell j}(e) = \frac{\left(\mathbb{E}_i[z_{\ell j}(x_i)]\right)^2}{\operatorname{Std}_i[z_{\ell j}(x_i)] + \varepsilon}.$$

Rank all 55,296 L0–L5 neurons. Save the top five candidates, not the full score
array.

#### E1 validation and controls

- **Held-variant stability:** recompute the candidate ranking on the four unseen
  header-prefix variants. Report exact top-1 agreement, top-5 overlap, and
  candidate-rank retention.
- **Same-neuron collision:** report how many other tickers select the same
  $(\ell,j)$ and the candidate's entity-selectivity z-score relative to those
  tickers.
- **Surface-form control:** re-render the frozen `anonymous_ticker`,
  `anonymous_name`, and `name_form_control` variants. A candidate can be called
  form-robust only if it remains top-5 for at least two real forms and is not
  top-5 for `name_form_control`.
- **Targeted amnesia:** on three financial prompts per ticker, scale the
  candidate across all token positions using
  $\alpha\in\{1,0,-1,-2,-3\}$. Compare target prompts with a deterministic
  wrong-entity cell and a random neuron from the same layer matched on generic
  activation mean and variance. For prompt $p$, define signed progress toward
  its frozen anonymous
  margin as

  $$A_p(\alpha)=\frac{(M_p(\alpha)-M_p(1))(M_{p,\mathrm{anon}}-M_p(1))}
  {(M_{p,\mathrm{anon}}-M_p(1))^2+10^{-6}}.$$

  Positive values move toward anonymous; $A=1$ reaches the anonymous margin.
  Exclude the prompt from this gate when
  $|M_{p,\mathrm{anon}}-M_p(1)|<0.1$, while retaining it in descriptive output.
- **Non-collapse checks:** score prompt-format validity and the fixed Buy/Sell
  continuation IDs. Large changes shared by target and controls indicate a
  general-purpose neuron, not an entity-selective cell.

E1 discovery yields **trusted candidate cells**, not confirmed entity cells.
A ticker enters E2/E3 only if held-variant top-5 overlap is non-zero and at
$\alpha=-3$ the top-1 target candidate has $A_p(-3)>0$ and exceeds both
wrong-cell and random-neuron $A_p(-3)$ on at least two eligible prompts. Discovery records
the number passing; it does not set a population success threshold after seeing
the count.

**Compact outputs:** per-ticker top-five candidates, held-variant agreement,
collision/selectivity summaries, form-control results, and amnesia curves. No
raw activation vectors or full neuron-score arrays are persisted.

### Phase E2: Downstream Component Attribution

Qwen3.5-4B uses a hybrid stack of recurrent linear-attention and standard
full-attention layers. Chughtai et al.'s source-token DLA requires an explicit
weighted sum over source positions, so E2 V1 applies only to full-attention
layers L3, L7, L11, L15, L19, L23, L27, and L31. The primary downstream scan
uses L11, L15, and L19 because they bracket the confirmed L16 context-state
site. L3 and L7 form early controls; L23, L27, and L31 form late controls.
Linear-attention state attribution requires a separate derivation and version.

#### E2 preparation

For each of three frozen financial prompts per trusted E1 ticker, resolve four
source groups using shared prompt-span helpers:

- **identity header:** ticker and company-name content tokens;
- **evidence:** qualitative and quantitative evidence tokens;
- **instruction context:** post-evidence instruction and assistant prefix,
  excluding final position;
- **other prefix:** remaining prompt tokens.

The groups must be disjoint and cover every source position before the final
query position. Save only token ranges and hashes.

#### E2 decision-margin attribution

For each full-attention head $h$ and source token $k$, reconstruct its additive
output before the heads are summed, including Qwen3.5's query-dependent output
gate and the head slice of `o_proj`. Contract that vector directly with the
fixed FP32 Buy-minus-Sell unembedding direction. This yields a scalar
approximate direct margin contribution $D_{l,h,k}$ without materializing
full-vocabulary logits. Sum scalars within each source group.

Because final RMSNorm is nonlinear, report two variants:

1. **Frozen-scale DLA:** linearize final RMSNorm with the clean final-residual
   scale held fixed. This is the primary additive decomposition.
2. **Component-alone logit lens:** apply final normalization to the isolated
   component. This is diagnostic and is not expected to sum to the clean
   margin.

Verify the implementation by checking that the sum of reconstructed head
outputs matches the model's attention-block output at the final position within
a frozen numerical tolerance before interpreting attribution. The frozen
tolerance is relative: the reconstruction is an FP32 re-implementation compared
against the model's bf16 forward, so the legitimate gap is bf16 rounding noise
proportional to the attention output scale. Measured on Qwen3.5-4B (5 tickers,
3 prompts each, all 8 full-attention layers, 120 reconstructions): the
max-abs error over max $|\text{clean}|$ ranges from $2.8\times10^{-3}$ (L3)
to $5.7\times10^{-3}$ (L27), i.e. about 0.3-0.6\% relative. The frozen rule is
\[
\max_{i}\,|r_i - c_i| \le \max\left(10^{-3},\; 2\times10^{-2} \cdot \max_j |c_j|\right),
\]
leaving a $\geq 3.5\times$ margin over the measured worst case while any
structural error (missing output gate, wrong RoPE, wrong GQA repeat, wrong
$o_{\mathrm{proj}}$ slice) is orders of magnitude larger. The per-record observed
errors are kept in the compact attribution JSONL so the noise distribution
stays auditable.

#### E2 component screening

The primary quantity for each head is the signed identity-header contribution
to the clean Buy/Sell margin. Rank heads by equal-ticker mean absolute identity
contribution. The 10:1 Subject/Relation ratio from Chughtai et al. is retained
only as a descriptive routing label and uses absolute group contributions with
a frozen denominator floor $\varepsilon_{\mathrm{floor}}=10^{-4}$:

- **identity-dominant:** $|D_{\mathrm{identity}}| / \max(|D_{\mathrm{instruction}}|,\varepsilon_{\mathrm{floor}})>10$;
- **instruction-dominant:** $|D_{\mathrm{instruction}}| / \max(|D_{\mathrm{identity}}|,\varepsilon_{\mathrm{floor}})>10$;
- **mixed:** neither ratio exceeds 10.

Evidence contribution remains a separate axis rather than being merged into the
instruction denominator. A head enters E3 if it is among the frozen top five by
equal-ticker mean absolute identity contribution and its identity contribution
has the same sign in at least two of three prompt columns for at least half of
eligible tickers.

#### E2 readout and controls

- **Jacobian-lens readout:** for selected heads whose source layer is covered by
  the validated canonical lens, transport only the aggregate final-position
  head output, then save top-10 vocabulary tokens and fixed Buy/Sell/sector
  vocabulary scores. Mark unsupported source layers as unavailable rather than
  substituting another layer. Do not treat the readout as source-token-resolved
  unless each source component is transported separately.
- **Anonymous/swap contrasts:** rerun selected heads on original,
  `anonymous_identity`, `same_sector_swap`, and ROT13 name-form prompts. A
  company-identity component should change more under real identity removal or
  replacement than under the surface-form comparator.
- **Head-output patch:** patch a selected head's final-position output between
  matched identity prompts while preserving all other components. This tests
  head-output sufficiency and guards against relying on DLA alone.
- **Additivity audit:** report the fraction of the clean margin reconstructed by
  all attention heads, MLP block outputs, embedding/unembed bias where present,
  and the residual normalization remainder. E2 does not claim a complete
  circuit if a large remainder is unexplained.

**Compact outputs:** per-head group attribution scalars, routing labels,
consistency counts, selected-head readouts, head-output patch effects, and
additivity diagnostics. Never persist attention matrices, per-token head
vectors, full-vocabulary arrays, or recurrent linear-attention states.

### Phase E3: Upstream and Downstream Suppression

E3 separates two interventions that answer different causal questions.

#### E3-A: upstream entity-cell suppression

For each trusted E1 cell $(\ell^*,j^*)$, scale its pre-`down_proj`
activation across **all token positions**, matching Barzilay et al.'s negative
ablation protocol. Use the fixed dose grid
$\alpha\in\{1,0.5,0,-1,-2,-3\}$. Header-only scaling is retained as a
secondary localization control because it is more surgical but no longer a
direct replication.

At each dose, record:

- clean and intervened Buy/Sell margin;
- the selected E2 heads' identity-, evidence-, and instruction-sourced DLA;
- whether the final decision flips;
- distance to the same prompt's frozen `anonymous_identity` margin.

The upstream mediation contrast is

$$\Delta D^{\mathrm{identity}}_{e,h}(\alpha)
= D^{\mathrm{identity}}_{e,h}(\alpha)-D^{\mathrm{identity}}_{e,h}(1).$$

A candidate route receives support when decreasing $\alpha$ reduces the
selected downstream identity contribution and moves the margin toward the
entity-removed condition without erasing the evidence contribution.

#### E3-B: downstream component attenuation

For the E2-selected full-attention heads, multiply only the reconstructed
identity-source contribution at the final query position by
$\beta\in\{1,0.75,0.5,0.25,0\}$ before summing source positions and applying
`o_proj`. Evidence-, instruction-, and other-prefix contributions remain
unchanged. This intervention is the first guidance candidate because it targets
the routed identity update rather than suppressing the entire head or residual.

Run three comparisons. The identity condition uses
$\Delta v_{\mathrm{id}}=-(1-\beta)v_{\mathrm{id}}$. Controls use norm-matched
additive attenuation rather than the same $\beta$:

- attenuate the identity-source contribution by $\Delta v_{\mathrm{id}}$;
- attenuate the evidence-source direction with
  $\Delta v_{\mathrm{evid}}=-\|\Delta v_{\mathrm{id}}\|
  v_{\mathrm{evid}}/(\|v_{\mathrm{evid}}\|+10^{-6})$;
- attenuate a random same-token-count source-subset direction with the same
  perturbation norm.

If a control source vector has norm below $10^{-6}$, mark that control
ineligible rather than substituting another vector after inference.

Also run whole-head attenuation as an upper-bound side-effect control. E3-B
reports a dose-response curve; it does not select the best dose on test data.

#### E3 controls and preservation outcomes

- **Wrong-cell control:** apply a deterministic other-ticker cell from the same
  layer and matched baseline activation magnitude.
- **Random-neuron control:** sample a same-layer neuron matched by generic
  activation mean and variance.
- **Name-form and same-sector controls:** compare movement from original to the
  frozen `name_form_control` and `same_sector_swap` margins.
- **Evidence preservation:** report the selected heads' evidence-sourced DLA and
  its change, together with the intervened clean margins. An intervention that
  removes the evidence-sourced contribution or collapses the clean-margin
  strata is not selective.
- **Format and stability:** verify fixed continuation token IDs, finite logits,
  and no shared collapse across target and controls.
- **Cell-to-component specificity:** upstream suppression should reduce selected
  identity components more than non-selected heads and more than evidence or
  instruction components.

#### Discovery outcomes and confirmation freeze

Discovery reports per-ticker curves and does not issue a confirmatory success
verdict. Before calibration, freeze:

1. the eligible-cell rule from E1;
2. selected E2 layers/heads and whether single-head or grouped attenuation is
   primary;
3. one upstream primary dose and one downstream primary dose, selected from
   discovery only;
4. minimum ticker count;
5. paired contrasts for target versus wrong-cell/random-neuron controls;
6. evidence-preservation and direction-consistency thresholds;
7. bootstrap, sign-flip, Holm family, and test-authorization rule.

Calibration may authorize the 11-ticker held-out test only if all frozen gates
pass. A failed calibration is preserved and blocks the test. Any change to cell
selection, component family, dose, source group, or primary outcome creates V2.

**Compact outputs:** per-ticker dose-response margins, downstream contribution
deltas, anonymous-distance changes, preservation metrics, and control
contrasts. No raw activations, head vectors, attention matrices, gradients, or
KV/recurrent states are persisted.

## Claim boundaries and interpretation limits

- Phase E1 identifies a **candidate entity cell** under financial header
  prompts. It does not prove the cell is necessary for all entity
  knowledge, nor that it is the unique encode point for company identity.
- Phase E2 DLA classification assumes approximate linearity through the
  unembedding. Non-linear LayerNorm interactions mean the decomposition
  is an approximation. Head labels are task-local and prompt-family
  local; they should not be generalized to other tasks without separate
  validation.
- E3-A tests causal necessity under suppression. E3-B tests whether a
  source-resolved downstream update is an actionable control site. Neither
  establishes that the selected route is unique; distributed or redundant
  routes may compensate.
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
| E2 (component attribution) | M3/M4 — controls | Decomposes the entity-to-decision information route for standard full-attention layers; cross-checks selected components with head-output patching and Jacobian-lens transported readout |
| E3 (suppression) | M4 confirmatory + M5 prerequisite | Tests the upstream cell-to-component link and whether source-resolved downstream attenuation can control the margin while preserving evidence effects |

E3 can open M5 only after frozen calibration/test gates support an
entity-specific margin shift, a downstream identity-component decrease, clean
matched controls, and evidence preservation. Discovery alone cannot authorize
guidance. If E3 fails or produces distributed effects across many neurons, M5
design must account for superposition and cannot rely on single-neuron or
single-head suppression.

## Implementation plan

The workflow follows `prepare → forward → analyze → finalize` and lives in the
experiment-owned package `llm_bias/entity_cell/`. The package imports shared core
packages but does not import `jspace_intervention` or `span_sensitivity`. Its public
CLI is `entity-cell`; preparation, E1, E2, E3, discovery summary, confirmation config,
and confirmation evaluation are implemented. No model inference, calibration, or
held-out test has run for V1.

### Shared-core reuse

- model/tokenizer loading: `llm_bias/core/model.py`;
- prompt encoding and character-to-token spans:
  `llm_bias/core/prompt_input/encoding.py`;
- exact FP32 continuation margin: `llm_bias/core/continuation_scoring.py`;
- validated canonical lens: `llm_bias/core/lens_loader.py` and
  `llm_bias/core/analysis/transport.py`;
- run lifecycle and compact serialization: `llm_bias/core/artifacts/`;
- bootstrap, sign-flip, and Holm procedures:
  `llm_bias/core/analysis/statistics.py`.

### New experiment-owned modules

| Module | Responsibility |
|---|---|
| `preparation.py` | Split filtering, header-prefix variants, generic baseline IDs, prompt spans, deterministic E2 donor contracts, hashes, and frozen config validation |
| `mlp_cells.py` | Pre-`down_proj` recorder/scaler, normalization statistics, stability ranking, form controls, and E1 amnesia runs |
| `attention_attribution.py` | Qwen3.5 full-attention head reconstruction, source-group DLA, output-gate handling, additivity checks, and head-output patching |
| `readout.py` | Selected-component Jacobian transport and compact top-k/fixed-vocabulary readout |
| `suppression.py` | E3-A cell scaling, E3-B source-contribution attenuation, dose sweeps, and preservation controls |
| `analysis.py` | Ticker aggregation and discovery summaries |
| `confirmation.py` | Machine-readable frozen confirmation schema, ticker statistics, Holm gates, and calibration authorization |
| `lifecycle.py` | Prepared-run, stage, provenance, and parent-artifact validation |
| `cli.py` | Prepare, run E1/E2/E3, analyze discovery, and analyze confirmation commands |

Do not add Qwen-specific component reconstruction to shared core until a second
experiment needs the same API. Generic hook cleanup or compact component-vector
transport may move to core only when it has model-independent tests.

### Proposed artifact layout

```text
artifacts/qwen3.5-4b/entity-cell-localization/
  configs/
    v1-discovery.json
    v1-confirmation.json              # created only after discovery
  prepared/
    v1-header-variants.jsonl
    v1-generic-baseline.jsonl
    v1-financial-prompts.jsonl
    v1-e2-donor-contracts.jsonl
  runs/<run-id>/
    manifest.json
    prepare/metadata.json
    e1/cells.jsonl
    e1/amnesia.jsonl
    e2/head_attribution.jsonl
    e2/readout.jsonl
    e2/patching.jsonl
    e3/suppression.jsonl
    analyze/summary.json
    analyze/confirmation.json         # calibration/test only
```

Each run records model revision, tokenizer identity, lens SHA-256, split
manifest hash, prepared-input hashes, config hash, dtype, device map, git commit,
and package version. Failed or interrupted runs remain immutable.

### Unit and integration verification

- fake SwiGLU model proves that the recorder reads the pre-`down_proj` vector
  and that scaling one neuron changes only the requested channel;
- fake GQA attention model proves source-group reconstruction sums to the
  attention output, handles repeated KV heads, and includes the output gate;
- span tests cover multi-token ticker/name forms and ensure localization uses
  the final company-name content token rather than the closing bracket;
- artifact tests reject raw tensors and missing provenance;
- deterministic analysis tests cover ranking ties, denominator floors,
  eligibility, bootstrap, sign-flip, Holm, donor matching, lifecycle rejection,
  compact confirmation serialization, and calibration authorization;
- one-model smoke uses one discovery ticker, two header variants, one full
  attention layer, one cell, and two suppression doses before any population
  run.

## Execution order and compute plan

Run all GPU stages sequentially on one device.

1. **Preparation and tests (CPU):** freeze 12 header variants, 399 baseline
   prompts, three financial columns, spans, configs, and hashes.
2. **E1 baseline statistics (GPU):** 399 forwards with L0–L5 hooks. Keep one
   9,216-value vector per layer on CPU during online mean/variance updates.
3. **E1 localization (GPU):** 35 × 12 = 420 forwards. Persist top-five ranks
   only.
4. **E1 amnesia discovery (GPU):** trusted candidates × three prompts × doses
   and controls. Run batch size 1 unless measured memory permits batching.
5. **E2 attribution (GPU):** trusted E1 tickers × three prompts over eight
   full-attention layers; contract directly to scalar Buy/Sell contributions.
   Full-vocabulary
   Jacobian readout runs only for selected heads.
6. **E2 patch controls (GPU):** selected heads and matched identity conditions.
7. **E3 discovery (GPU):** upstream and downstream dose sweeps for eligible
   tickers. Do not run calibration until discovery analysis and confirmation
   config are committed.
8. **Calibration/test:** use the 12/11 Technology ticker splits sequentially.
   For each unseen ticker, rerun E1 with the frozen header variants,
   normalization statistics, ranking rule, form controls, and eligibility gate
   to bind a ticker-specific candidate cell. Keep the E2 layer/head indices and
   all E3 doses/contrasts fixed from discovery. Test remains blocked unless the
   frozen calibration evaluator authorizes it.

Memory rules:

- copy only selected-position MLP vectors to CPU; never cache full-sequence MLP
  activations;
- contract component outputs with the Buy/Sell direction before serialization;
- request or reconstruct attention weights only for one full-attention layer at
  a time;
- never materialize per-head full-vocabulary logits for the population scan;
- run Jacobian-lens top-k readout on selected aggregate component vectors only.

## Statistical reporting

- Tickers are the independent units. Average prompts within ticker before
  population statistics.
- Report equal-ticker means, medians, 10,000-resample ticker bootstrap 95% CIs,
  and paired sign-flip p-values.
- Assess E3 dose monotonicity with Kendall's $\tau$ per ticker and an aggregate
  paired contrast between endpoint and baseline.
- E1 neuron scoring is ranking rather than hypothesis testing. Confirmation
  tests only the frozen candidate-level and route-level contrasts.
- Apply Holm correction across the frozen primary contrast family; keep
  descriptive vocabulary readouts outside the confirmatory family.
- Report eligible and excluded ticker counts at every phase. Do not treat prompt
  rows, doses, heads, or layers as independent samples.

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
