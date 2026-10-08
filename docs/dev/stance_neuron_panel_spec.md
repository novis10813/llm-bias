# Native-neuron panel slice defines candidates, not effects

This API implements item 3 of `stance_parallel_experiments_contract.md` in
`llm_bias/core/stance_neuron_panel.py`. It creates immutable candidate and
role-plan records and applies one native activation edit. It does not execute
discovery, select effective neurons, or authorize a research run.

## D1: Candidate construction covers each dense layer

`build_neuron_panel(model) -> NeuronPanel` inspects `model.layers`, resolves
`dense_down_projection(layer)` (including `_hf_layer` wrappers), and reads
`down_proj.in_features`. This is the intermediate input width, not residual
width or an old model-config coordinate. Each supported layer receives
`min(16, width)` distinct coordinates.

For layer `L`, rank every integer `N` in `[0, width)` by
`(sha256_json({'seed': 20261003, 'layer': L, 'neuron': N}), N)` ascending.
Take the first 16, retaining rank order. Use the repository canonical JSON/SHA
implementation. The seed and count have no override. No localization,
DIM cosine, generated labels, or gradient enters this construction.

`NeuronPanel(status, widths, coordinates, reason='')` validates the entire
ranking, genuine Python integer widths/coordinates (rejecting bool and float),
and complete layer coverage. `coverage` returns tuples of
`(layer, candidate_count, native_width)`, including explicit width<16 caps.

The builder returns `status='unsupported'`, empty widths/coordinates and a
reason if any layer lacks a dense hook or declared input width. It does not
produce a partial panel. GPT-OSS `model_type='gpt_oss'` and MLPs exposing
`experts`, `router`, or `gate` are unsupported routed-MoE cases. The API never
selects an expert or substitutes a residual write vector. Future unusual
projection layouts need a reviewed capability adapter. This inspection does
not certify a real checkpoint's hook semantics.

## D2: Discovery planning keeps roles and scopes separate

`NeuronRole(ticker, issuer_id, role)` records one supplied assignment.
`NeuronDiscoveryPlan(panel, roles, inputs_manifest_sha256, parent_sha256,
model_sha256, protocol_sha256)` requires immutable typed role tuples, four
nonempty roles, unique tickers and issuer-disjoint assignments. It canonicalizes
role order by ticker and requires lowercase SHA-256 provenance strings.

`role_ids(phase)` permits only these phase/role pairs:

| Phase | Role |
|---|---|
| discovery | fit |
| selection | validation |
| calibration | calibration |
| evaluation | evaluation |

These pure validators allow tiny unit-test fixtures. They do not authenticate
supplied hashes or prove full503 membership. A future runner must validate the
approved `BaselineInputs` bundle and exact assignment equality before constructing
the plan. It must not accept a research cohort/count override.

`arms()` yields `NeuronArm(panel_sha256, layer, neuron, delta, scope)` in
layer, hash-rank coordinate, scope, dose order. The fixed native signed scalar
grid is `[-8, -2, -0.5, 0, 0.5, 2, 8]`. Scope order is
`prompt_only`, `prompt_and_decode`. Both scopes edit all original prompt
positions. Only the latter edits decode positions. Zero remains a separate
arm per candidate/scope. No instruction-only arm belongs to this slice.
The budget is `sum(min(16, width)) * 2 * 7` arms before prompt expansion.
An unsupported panel yields zero arms while retaining its reason and plan.

The records use frozen slotted dataclasses and tuple storage. `to_dict()`
returns detached JSON-ready containers. `panel_sha256`, `arm_sha256`, and
`plan_sha256` hash these payloads without their own digest fields. Integer-valued
doses normalize to the fixed grid representation so `0.0` and `0` agree.
The plan hash binds provenance, assignments, panel, doses and scopes. Caller
provenance claims remain unverified until the runner authenticates sources.

## D3: Execution helper edits the actual native input

Use a fresh `GenerationPositionTracker(prompt_length, use_cache)` for each
execution and enter `tracker.track(model)` around model forwards:

```python
with tracker.track(model):
    with native_neuron_edit(model, panel=plan.panel, arm=arm, tracker=tracker):
        output = model(**generation_inputs)
```

`native_neuron_edit` verifies panel equality against current native model widths,
arm digest and membership, then delegates to accepted `scoped_mlp_addition`.
That helper modifies the actual tensor before `down_proj`, selecting positions
through the tracker rather than a sequence-length heuristic. It leaves input
tensors unchanged and removes hooks when the context exits, including exceptions.
The caller owns generation, absolute position/cache metadata and root tracking.
This API does not capture or persist activations, residuals, gradients or caches.

Fake tests observe the actual pre-projection tensor and output for each signed
dose, both scopes, cached single-token decode and uncached full recomputation.
They also check zero edits, unsupported models, caps, deterministic ranking,
role leakage, immutable exports, invalid coordinates and exception cleanup.

## Next runner dependencies

1. **R1:** Bind approved full503 inputs, evidence/schema/model/code identities,
   full role assignments and an authenticated completed baseline. GPT's future
   effective baseline must retain per-row policy and source provenance.
2. **R2:** Freeze prompt expansion, generated-outcome discovery/validation
   selection, multiplicity policy, acceptance thresholds and calibration policy.
   This slice fixes the candidate/dose/scope budget only. Do not invent an
   effect threshold or treat its hash-ranked coordinates as effective neurons.
3. **R3:** Add immutable planned generation keys and resume/completeness checks,
   bounded no-op/model gates, and checkpoint native-hook validation. Preserve
   generated failure/diagnostic rows, unsupported status, both directional
   denominators and schema completion. Use compact artifacts only. Main owns
   reviewer approval and GPU submission.

## Verification

The shared editable environment points imports at main unless the worktree is
placed first on `PYTHONPATH`. Run from this worktree without dependency sync:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync pytest -q tests/test_stance_neuron_panel.py \
  tests/test_mlp_addition.py tests/test_stance_interventions.py
uv lock --check
git diff --check
```

Passing these checks establishes engineering behavior on fake models. It does
not establish native-checkpoint support, decision flips, neuron efficacy or
completion of the neuron experimental milestone.
