# Cone sequence objectives provide a CPU-testable surrogate slice

Status: engineering API for contract item 5 and TODO P1.4. No trained operator,
generated acceptance result, or GPU runner comes with this slice.

## S1. Teacher records bind fit sequences to their sources

`llm_bias/core/stance_cone_objectives.py` defines frozen, slotted
`TeacherResponse` records. Callers supply ticker, exact `fit` role, purpose
(`addition`, `ablation`, or `retain`), immutable token IDs and response mask,
source SHA-256, tokenizer SHA-256, schema SHA-256, token-ID SHA-256,
response source, target decision, and review SHA-256. `response_source` accepts
`construction_generated` or `construction_authored`. Construction means the
fit subset, not validation or calibration. `target_decision` accepts buy/sell
as provenance, not as an answer-token scoring parameter.

The token digest uses `sha256_json(list(token_ids))`. Source and review hashes
bind caller-owned response provenance and evidence/label/schema review records.
This module checks their syntax, not their authenticity or the decoded JSON.
The future compiler must verify original prompt/response bytes, tokenizer,
schema-valid response and review contents before constructing records. A hash
alone does not certify that a teacher is scientifically appropriate.

Every loss call requires an authoritative `roles` mapping from the frozen input
assignments. The module rechecks each record and rejects a ticker that is absent
or not fit, even if the caller has labeled the teacher fit. The future runner
must supply the verified mapping, not a caller-invented assignment. Teachers
within a batch share tokenizer and schema hashes. No evaluation response can
serve as a teacher. Records hold tokens/provenance only, not logits, residuals,
activations, gradients or KV caches.

## S2. The causal shift and reduction order define sequence losses

`sequence_ce(logits, records, *, roles, purpose)` accepts edited teacher-forced
logits `[D,B,T,V]`. `purpose` names addition or ablation. Callers apply the
corresponding intervention before computing logits. The loss function does not
implement hooks, choose target decisions or flip labels.

`sequence_kl(logits, clean_logits, records, *, roles)` accepts edited logits
`[D,B,T,V]` and clean logits `[B,T,V]` for the same retain teacher sequences.
The caller obtains clean logits without the intervention. The module detaches
them and computes `KL(clean || edited)` over the vocabulary at each position.
Both logits must have matching dtype, device, sequence and vocabulary shapes.
Clean distributions remain transient and do not enter teacher records.

For token IDs `x[0:T]`, logit position `t-1` predicts token `x[t]`.
`response_mask[t]` marks target token positions, not predictor positions.
The first mask entry must be false. Callers mark the full response, including
multi-token decision and reason content, and exclude prompt and padding tokens.
The pure API accepts partial masks for explicit tests/adaptations but does not
certify that a supplied mask covers the full response. A future compiler must
check that coverage against source boundaries. This slice has no fixed JSON
prefix margin and no single-step KL approximation.

For each direction and example:

- CE token loss is `-log p_edited(x[t] | x[:t])`.
- Retain token loss is `sum_v p_clean(v) * (log p_clean(v) - log p_edited(v))`.
- The sequence loss is the sum over selected shifted tokens divided by that
  example's selected token count.
- The returned `[D]` loss averages sequence losses over examples with equal
  example weight. Longer responses do not receive extra weight.

Each record needs at least one selected shifted token. Empty panels, all-masked
sequences, negative/out-of-vocabulary token IDs, shape mismatches and nonfinite
inputs/results raise `ValueError`. Finite checks include unselected positions.
Callers use equal padded sequence lengths within each batch. The module does
not infer attention masks, padding semantics or intervention scope.

## S3. Basis and sampled-ray means remain separate

`cone_objective(basis, samples)` consumes two mappings with exactly the keys
`addition`, `ablation`, and `retain`. Each value is a nonempty per-direction
loss vector from the sequence APIs. Direction counts agree across objectives
within each panel but may differ between panels.

For each objective `j`, `L_j = mean(basis[j]) + mean(samples[j])`.
The final loss is `L_addition + L_ablation + L_retain`. `ConeObjective` returns
the three differentiable terms, total and recorded weights `(1., 1., 1.)`.
There is no pooled basis-plus-sample mean and no implicit division by two.
These tensors are transient training values, not serialized artifacts.

The positive/negative finance target choice, source response review policy,
projection-ablation implementation, hook scope, teacher compilation and
training budget still need a centrally approved runner protocol. This module
does not invent those settings. CE/KL are training surrogates. Main must judge
checkpoint acceptance from validation greedy generation and schema/evidence
retention gates, not loss or fixed answer-token margins. A missing or failed
seed is not an accepted operator. This slice has no seed acceptance aggregator;
main must retain failed seeds and report all-seed failure in the future runner.

## S4. Initialization has no DIM dependency

`initialize_basis(d_model, *, dimension, seed, dtype=torch.float32)` returns
independent Gaussian rows `[K,H]`, each normalized to unit L2 norm on CPU.
`K` must be 2 or 4, `H >= K`, and seeds must be 20261003, 20261004 or 20261005.
Supported initialization dtypes are float32 and float64. A local Torch generator
preserves global RNG state. The API takes no DIM, residual data, fitted scale
or anchor. Rows are not forced orthogonal. Callers control gradient tracking,
optimization and device placement. Reproducibility refers to the pinned Torch
version and these explicit parameters.

`positive_rays(basis, coefficients)` accepts `[K,H]` and strictly positive
`[R,K]` tensors of the same dtype/device. It normalizes basis rows, L1-normalizes
coefficients per ray, then L2-normalizes each weighted sum. It returns normalized
coefficients and rays. Both operations retain gradients. Zero, nonfinite,
negative, shape-mismatched or cancelling inputs fail. Strict positivity covers
interior combinations. Individual basis axes form a separate panel. Callers
must freeze ray panel provenance before evaluation; this API does not choose
ray counts or a sampling distribution.

`negative_cone_adaptation(rays)` returns reversed unit rays in `-C`.
This names the bidirectional financial adaptation separately from positive
cone `C`. Positive ray coverage does not establish negative-cone coverage or
validity of the continuous cone. Native dose magnitudes and intervention scope
remain runner responsibilities.

## S5. Verification uses analytic CPU sequences

`tests/test_stance_cone_objectives.py` covers shifted masks, prompt/padding
exclusion, analytic CE/KL and gradients, detached retain teachers, unequal
sequence lengths, equal example weighting, separate basis/sample reductions,
strict authoritative role checks, immutable source-token provenance, declared
initialization seeds/dimensions and differentiable positive/reversed rays.
Tests import no DIM implementation and load no checkpoint.

Run in this worktree with the shared read-only environment:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync pytest -q tests/test_stance_cone_objectives.py
uv lock --check
git diff --check
```

`PYTHONPATH=.` selects this worktree's source instead of the shared environment's
editable main checkout. `--no-sync` avoids changing the shared environment.
