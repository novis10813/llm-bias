# J-space-candidate evaluation for local model comparison

**Status:** proposed, non-runnable optional auxiliary preflight; no implementation
exists in this repository.

This document specifies a future `jspace_eval` design for checking whether
multiple locally hosted decoder models are compatible with Jacobian-lens analysis
and for comparing **synthetic task-local J-space-candidate evidence** across
models. No `jspace_eval` package, CLI, configuration tree, artifact set, or test
suite currently exists in this repository. Every package layout, command, and
artifact described below is conceptual and non-runnable until implemented.

The design may inform the later cross-model phase of the
[entity-bias roadmap](proposal/entity-bias-roadmap.md) as an optional preflight.
It does not establish entity bias, identify a global workspace, replace
domain-specific counterfactual patching, or turn a transported J-lens readout
into standalone causal evidence. It is not a prerequisite, completion gate, or
evidence-quality gate for the entity-bias milestones.

The term *working-space* refers to the concept discussed by Gurnee et al. in
*Verbalizable Representations Form a Global Workspace in Language Models*. This
document does not reproduce or establish that paper's broader scientific
conclusion. Here, a **J-space candidate** is an operational object defined by a
Jacobian-lens readout and a pre-specified synthetic benchmark; a
**workspace-like candidate signal** means only that the signal passed the
benchmark's local readability, intervention, and transfer checks.

## 1. Research objective and result vocabulary

The tool should evaluate which layers show stable, readable, causally useful, and
cross-operation-reusable **candidate evidence under the selected synthetic task**.
The scope is limited to the benchmark family, prompt distribution, calibration
splits, lens targets, model, and tokenizer recorded in the run manifest. It must
not assume that a workspace exists or force every model into a positive result.

The report must distinguish:

```text
stable_jspace_candidate
task_dependent_jspace_candidate
multiple_jspace_candidate_bands
no_detectable_jspace_candidate_evidence
lens_unidentifiable
insufficient_task_competence
incompatible_model
```

The first four are synthetic task-local evidence labels, not global model
properties:

- `stable_jspace_candidate`: candidate evidence is stable across the configured
  task families, lens targets, and calibration seeds;
- `task_dependent_jspace_candidate`: candidate bands differ across configured
  synthetic task families;
- `multiple_jspace_candidate_bands`: one task family produces multiple candidate
  bands; and
- `no_detectable_jspace_candidate_evidence`: lens and competence checks pass, but
  no layer passes the configured candidate-evidence criteria.

The remaining statuses identify technical or interpretive failures:

- `lens_unidentifiable`: the Jacobian estimate is unstable or lacks sufficient
  variance;
- `insufficient_task_competence`: the model cannot solve the benchmark well
  enough for task-local evidence to be interpretable; and
- `incompatible_model`: required residual, unembedding, gradient, or hook
  capabilities are unavailable.

## 2. Compatibility scope

### Required model capabilities

A model adapter must support:

- an autoregressive decoder architecture;
- access to every transformer block's residual stream;
- access to an LM head or unembedding matrix;
- backward gradients;
- custom forward hooks;
- tokenizer token IDs and decoding; and
- local model weights.

### Explicitly unsupported or unguaranteed models

The first version does not guarantee support for API-only models, GGUF runtimes,
engines without gradients, encoder-only or encoder-decoder architectures,
recurrent state-space models, multi-token prediction architectures, or hybrid
architectures with no well-defined residual stream.

An incompatible model must produce a structured result such as:

```json
{
  "status": "incompatible_model",
  "reasons": ["residual_stream_not_accessible"]
}
```

### Model adapter contract

All model-specific behavior is isolated behind a `ModelAdapter` interface:

```python
class ModelAdapter(Protocol):
    @property
    def num_layers(self) -> int: ...

    @property
    def hidden_size(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    def tokenize(self, text: str) -> torch.Tensor: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def get_input_embeddings(self) -> torch.nn.Module: ...
    def get_unembedding_weight(self) -> torch.Tensor: ...
    def get_residual_module(self, layer: int) -> torch.nn.Module: ...
    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False): ...
    def register_residual_hook(self, layer: int, hook_fn): ...
    def validate_capabilities(self) -> list[str]: ...
```

`validate_capabilities()` returns missing capabilities. An empty list is the
only condition under which the benchmark may run.

## 3. Operational screen for a J-space candidate layer

A candidate layer must pass three independent, task-local evidence tests. These
criteria do not define or identify a global workspace.

### 3.1 Representation readability

The layer's transported readout should identify the latent state against matched
distractors within the selected synthetic task. For latent concept `c`, candidate
set `D(c)`, and layer readout probability `p_l`:

```text
R_l = E[ log p_l(c) - log sum(p_l(c') for c' in D(c)) ]
```

The report must preserve episode-level margins, ranks, probabilities, and
baseline-correct flags rather than only an aggregate score.

### 3.2 Causal necessity

Within the selected synthetic task, ablating a concept-related J-space candidate
component should reduce the correct answer margin more than a matched control
ablation:

```text
N_l = delta_margin_concept_ablation - delta_margin_control_ablation
```

A single perturbation strength is insufficient. The tool must report a dose
response and matched controls.

### 3.3 Cross-operator reuse

Within the selected synthetic task, a latent representation captured from a
source episode is patched into a target episode that uses a different operator.
If the target output moves toward the answer associated with the transferred
latent state, the candidate has task-local cross-operation transfer evidence:

```text
U_l = margin_cross_operator_patch - margin_target_baseline
```

The basic J-space-candidate evidence predicate is:

```text
lens_identifiable
and readability_significant
and causal_necessity_significant
and cross_operator_transfer_significant
```

The system must not collapse these criteria into a single weighted JWS score.

## 4. Lens backend and calibration

The lens implementation is wrapped behind a backend so that the pipeline is not
coupled to one checkout:

```python
class LensBackend(Protocol):
    def fit(self, adapter, prompts, source_layers, target_layer, config): ...
    def concept_logits(self, bundle, layer, residual, concept_token_ids): ...
    def concept_direction(self, bundle, layer, token_id): ...
```

A `LensBundle` owns serialization, device movement, and backend-specific shape
handling. The pipeline must not assume a fixed tensor shape.

A calibration run should use multiple disjoint prompt splits and multiple seeds,
with both final and penultimate target layers where supported. A representative
configuration is:

```yaml
calibration:
  prompt_count: 300
  max_sequence_length: 128
  split_count: 3
  seeds: [11, 23, 37]
  target_layers: [final, penultimate]
```

For held-out examples, each layer and calibration split produces a concept-logit
matrix `Z_l^(s)`. Lens stability is evaluated with pairwise linear CKA across
splits and a row-shuffled null distribution. A layer is identifiable only when
its bootstrap lower bound exceeds the 95th percentile of its null distribution
and its concept-logit variance exceeds a configured floor. Constant outputs must
not receive a false stability result.

## 5. Synthetic latent benchmark

The benchmark is designed to reduce trivial or memorised surface-association
solutions, not to establish how the model reasons in natural language. Each
episode samples new numbers, mappings, and output tokens. The minimum viable task
is modular arithmetic:

```text
Compute the remainder of (6 x 7 + 5) divided by 8.
Use the remainder as a zero-based index into this sequence:
[K, R, T, M, Q, B, V, P]
Return the selected item.
```

The latent state and output token are deliberately distinct. Benchmark
constraints include:

- the latent value does not appear as a literal in the target prompt;
- the answer is a single token in the MVP;
- source and target share a template and token length;
- output mappings are resampled per episode;
- latent state and output token are not identical; and
- distractors are other latent values.

After the MVP, the same interface may support finite-state machines, graph
traversal, relational composition, set intersection, and permutation execution.

A task generator should expose:

```python
class LatentTaskGenerator(Protocol):
    def generate_episode(self, seed: int) -> Episode: ...
    def generate_patch_pair(
        self,
        source_latent: int,
        target_latent: int,
        source_operator: str,
        target_operator: str,
        seed: int,
    ) -> PatchPair: ...
```

An episode stores its ID, family, prompt, token IDs, query position, latent and
distractor labels, operator, answer tokens, prompt token inventory, and metadata.
A patch pair stores source/target episode IDs, latent and operator IDs, expected
answers before and after patching, and source/target positions.

## 6. Baseline competence precondition

No synthetic task-local candidate-evidence conclusion is interpretable unless
baseline task competence passes a configured threshold. A representative
precondition is:

```yaml
benchmark:
  min_baseline_accuracy: 0.60
  min_correct_episodes: 200
```

Reports must include both all episodes and baseline-correct episodes. The latter
are used for task-local candidate evidence; the former remain visible so that filtering
does not hide selection effects. A failed gate returns
`insufficient_task_competence` with the measured and required accuracy.

## 7. Readability experiment

For every baseline-correct episode:

1. capture the query-position residual at every candidate layer;
2. apply the calibrated J-lens readout;
3. collect latent and distractor logits;
4. calculate latent margin, rank, and probability; and
5. save a compact episode-level result.

A result record should include:

```text
episode_id, family, layer, lens_seed, lens_target,
latent_margin, latent_rank, latent_probability, baseline_correct
```

Permutation nulls shuffle episode/label assignments while preserving distractor
counts. Layer-wise p-values use Benjamini--Hochberg FDR correction.

## 8. J-space intervention basis

For latent token `c`, obtain a layer-specific J-space direction `v_(l,c)`. For
source and target concepts, construct an orthonormal basis:

```text
B_l = orth([v_(l,source), v_(l,target)])
```

Project the source--target residual difference into that basis:

```text
Delta_h_l = h_l_source - h_l_target
Delta_h_l_J = B_l B_l^T Delta_h_l
```

A target activation can then receive a controlled dose of the projected
contrast:

```text
h_l_patched = h_l_target + alpha * Delta_h_l_J
```

The default strength grid should include negative, zero, partial, and
overshooting values, for example `[-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5]`.
At `alpha = 0`, patched logits must match baseline logits within a configured
tolerance.

## 9. Matched controls

Each intervention run must include controls with independent seeds:

- random orthonormal subspaces with the same rank;
- variance-matched bases from activation covariance directions;
- distractor-concept bases;
- random directions matched to the intervention L2 norm; and
- reversed source/target patching.

A representative control seed list is `[101, 103, 107, 109, 113]`. Control
artifacts must identify the control type, seed, rank, norm, and layer.

## 10. Causal necessity

Concept ablation removes the projection onto the concept direction:

```text
h_l_ablated = h_l - alpha * P_v h_l
P_v = v v^T / (v^T v)
```

Report:

```text
necessity_effect = baseline_margin - concept_ablation_margin
control_effect   = baseline_margin - control_ablation_margin
causal_necessity = necessity_effect - control_effect
```

The result requires a dose-response curve, control comparisons, uncertainty, and
predefined layer-selection rules.

## 11. Cross-operator transfer

A cross-operator pair must have different latent states and different operators.
Patching the source latent into the target episode should be evaluated against
same-operator transfer, random controls, and distractor-concept controls. The
primary evidence is:

```text
cross_operator_transfer > matched_controls
```

A motor-like candidate may be reported separately when same-operator transfer
remains positive, cross-operator transfer decreases, and next-token alignment
increases. This secondary label must not change the J-space-candidate evidence predicate.

## 12. Ambiguous-input ignition

As a boundary corroboration experiment, interpolate between two input embeddings:

```text
 e(alpha) = (1 - alpha) e_A + alpha e_B
```

At each layer, project onto the endpoint difference and fit a logistic transition.
Report transition slope, midpoint, endpoint separation, interpolation residual,
and bootstrap intervals. Ignition is a candidate-localisation signal; it does
not replace causal validation or prove a global workspace.

## 13. Layer evidence and band segmentation

Each layer needs explicit evidence flags and effect/interval fields:

```python
@dataclass
class LayerEvidence:
    layer: int
    lens_identifiable: bool
    readability_significant: bool
    necessity_significant: bool
    cross_operator_transfer_significant: bool
    readability_effect: float
    necessity_effect: float
    cross_operator_effect: float
    readability_ci: tuple[float, float]
    necessity_ci: tuple[float, float]
    cross_operator_ci: tuple[float, float]
    task_family_support: int
    lens_target_support: int
```

A validated layer additionally needs configured support across task families and
lens targets. With one task family, report the result as
`provisional_single_family_result`.

Do not assume fixed sensory, candidate, and motor sections. Merge supported
candidate layers into bands while allowing a configured maximum gap, minimum
width, and membership probability threshold. Bootstrap episodes, recompute
candidate-evidence bands, and report onset/offset intervals and per-layer
membership probabilities. Store normalised layer depth so that models with
different layer counts can be compared.

Representative settings are:

```yaml
segmentation:
  max_gap_layers: 1
  min_band_width: 2
  membership_probability_threshold: 0.70

validation:
  min_task_family_support: 2
  min_lens_target_support: 2
  fdr_alpha: 0.05
  bootstrap_samples: 2000
```

## 14. Final candidate-evidence status decisions

- `stable_jspace_candidate`: a candidate band is stable across the configured
  synthetic task families, lens targets, and calibration seeds;
- `task_dependent_jspace_candidate`: configured task families produce candidate
  bands with low overlap;
- `multiple_jspace_candidate_bands`: one task family produces multiple stable
  candidate bands;
- `no_detectable_jspace_candidate_evidence`: lens and competence checks pass,
  but no layer passes all candidate-evidence criteria;
- `lens_unidentifiable`: lens reliability is insufficient;
- `insufficient_task_competence`: benchmark performance fails the precondition; and
- `incompatible_model`: adapter capability validation fails.

The report must preserve negative outcomes, scope every positive label to the
recorded synthetic task and calibration configuration, and state the reason for
every failed precondition. A negative result concerns only this benchmark and
protocol; it is not evidence against other shared computations or workspace
hypotheses.

## 15. Conceptual future package and CLI (non-runnable)

The future package may be organised as:

```text
jspace_eval/
├── adapters/
├── lens/
├── datasets/
├── activations/
├── interventions/
├── metrics/
├── statistics/
├── pipeline/
├── visualization/
├── cli.py
└── config.py
```

The following commands are a conceptual future interface only. They are not
available in this repository and must not be run as current project commands:

```bash
jspace-eval check-model --model <local-model>
jspace-eval build-benchmark --model <local-model> --config <config>
jspace-eval fit-lens --model <local-model> --config <config>
jspace-eval evaluate-readability --run-dir <run-dir>
jspace-eval evaluate-interventions --run-dir <run-dir>
jspace-eval evaluate-ignition --run-dir <run-dir>
jspace-eval segment --run-dir <run-dir>
jspace-eval report --run-dir <run-dir>
jspace-eval run-all --model <local-model> --config <config>
```

Every command must validate prerequisites, support resume, emit a manifest, avoid
overwriting completed results unless `--force` is used, write failures to
`errors.jsonl`, and return a non-zero exit code on failure.

## 16. Artifact and performance contract

A run should contain resolved configuration, compatibility, benchmark episodes
and patch pairs, lens metadata, compact activation-derived metrics, statistical
results, report JSON/Markdown, plots, and `errors.jsonl`. Full vocabulary logits
and raw model weights must not be copied into the report artifacts. Activations
should use chunked storage and a documented tensor format; tabular outputs may
use Parquet.

Forward computation may use bfloat16, while Jacobian accumulation and metrics
should use float32 or float64. Checkpoints should be updated after each batch;
OOM recovery must record the changed batch setting. Equal configuration and seed
must regenerate the same benchmark. Hooks must be removed when their context
manager exits.

## 17. Tests and invariants

### Unit tests

The future implementation needs tests for atomic-token selection, latent leakage,
source/target length and query alignment, projection idempotence, orthonormal
bases, answer margins, bootstrap intervals, FDR correction, band segmentation,
and configuration serialization.

### Intervention invariants

```text
alpha = 0 -> patched logits approximately equal baseline logits
zero delta -> patched activation equals target activation
projection rank equals requested rank
hook removal -> later forward is unaffected
reversed patch -> effect direction changes
```

### Integration tests

Use a small Hugging Face causal model to test compatibility checks, hidden-state
capture, lens-fit smoke behavior, readout artifacts, intervention artifacts,
resume behavior, and report generation. The small model does not need to produce
positive task-local candidate evidence.

### Mock scientific test

A `MockModelAdapter` should expose artificial regimes such as:

```text
layers 0-2: unreadable
layers 3-6: readable + causal + cross-operator
layers 7-9: readable + same-operator only
```

The expected result is a supported J-space-candidate evidence band at layers
3--6 and a motor candidate at layers 7--9. This validates evidence aggregation
and segmentation independently of a real model; it is not a model-wide workspace
claim.

## 18. Proposed implementation phases (future design)

1. **Repository scaffold:** package, configuration, logging, manifest, and CLI.
2. **Model adapter:** one Hugging Face decoder, capability checks, and residual
   hooks.
3. **Benchmark generator:** modular arithmetic, atomic output inventory,
   episode/patch-pair schemas, and leakage checks.
4. **Lens backend:** fitting wrapper, serialization, concept logits, and concept
   directions.
5. **Lens reliability:** calibration splits, held-out matrices, CKA, permutation
   nulls, and identifiability flags.
6. **Readability:** activations, latent margins, bootstrap, permutation tests,
   and FDR correction.
7. **Intervention engine:** concept bases, projection patching, ablation,
   dose-response, and matched controls.
8. **Cross-operator transfer:** same-operator and cross-operator patch pairs and
   expected-answer margins.
9. **Segmentation:** layer evidence, bootstrap bands, and normalised depth.
10. **Report:** result JSON, summary Markdown, curves, candidate-evidence band
    plots, and explicit failure reasons.

## 19. MVP and non-goals

The MVP requires one decoder adapter, the modular arithmetic benchmark, final and
penultimate lenses, multiple calibration splits, lens stability, readability,
causal necessity, cross-operator transfer, matched controls, dose response,
uncertainty, FDR correction, candidate-evidence band segmentation, reports, resume, unit
tests, and integration tests.

The MVP excludes a dashboard, MoE-specific analysis, encoder-decoder support,
multi-token latent labels, distributed multi-node fitting, automatic architecture
inference, and task-family consensus.

## 20. Scientific limitations

Reports must state that:

1. candidate-evidence decisions depend on the selected synthetic benchmark family;
2. J-space interventions may create off-manifold activations;
3. lens stability does not reconstruct all model computation;
4. baseline-correct filtering changes the sample distribution;
5. synthetic latent tasks do not represent all natural-language reasoning;
6. final and penultimate target disagreement indicates boundary instability;
7. one task family supports only a provisional task-local result; and
8. no detected candidate evidence does not prove that the model lacks other
   shared computation or bear on broader workspace hypotheses.

The core acceptance standard is not that the tool prints a layer range. It must
separate positive synthetic task-local J-space-candidate evidence, absence of
detectable candidate evidence, unreliable lens estimates, insufficient task
competence, and incompatible models.
