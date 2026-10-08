# CPU cone validation compilation fixes the full prospective exploration

This phase implements only `llm_bias/core/stance_cone_validation_plan.py` and its CPU tests. No GPU runner, generated validation results, operator acceptance, selection thresholds, calibration or evaluation are implemented. The declaration is fixed before validation outcomes. It does not confer formal research eligibility or claim that every continuous cone direction is effective.

## D1. All six saved cones are mandatory candidates

Consume the accepted saved-training audit at
`artifacts/glm4-9b-0414/concept-cone-steering/audits/cone-train-stance-rb260930-33.validation.json`
and its sibling `runs/cone-train-stance-rb260930-33/`.
The audit byte SHA is pinned to
`42c5300a224e1b8a4ab2f5508173f3749249d4932679b10c3af5d3dfd362220b`.
K=2,4 and seeds 20261003,20261004,20261005 form the exact mandatory six-run grid. There is no CE ranking, best seed or omitted attempt.

The compiler verifies the full audit artifact inventory and every recorded file digest, including the 1812 compact training step records. Each config, summary, operator and audit run must bind its dimension, seed and canonical config hash. Training must have completed all 302 fit steps and must still declare `accepted_operator=false` and `research_eligible=false`. Bind the full approved input manifest, role hash, reconstructed 2012-key baseline plan/identity, fit ticker membership, audited parent, accepted 906-teacher pack and identical teacher bindings across seeds. Native embedding/head dtype is BF16. Exported bases must be finite K×4096, BF16-representable and within absolute norm tolerance 0.01 of unity. No model weights are loaded or authenticated.

The reviewed audit explicitly records a provenance discrepancy: teacher compiler Git HEAD `5e92b4f` does not match two recorded source hashes (`stance_cone_teachers.py` and `compile_stance_cone_teachers.py`), which match later revision `8e4d9e6`. Preserve that disclosure in the derived panel. This is accepted bounded saved-training authentication, not a hidden discrepancy, a demand for historical reruns or live-weight authentication. Original files and failed-run history are never changed.

## D2. Directions and signed doses are fixed without another method

Normalize each saved basis independently on CPU in float64. Do not rotate, orthogonalize, anchor to DIM or use any oracle or outcome. Candidate families per cone are:

- Every unit basis axis, represented by its one-hot coefficients.
- One unit positive centroid, equal coefficients on the unit basis.
- Four strictly positive random coefficient rays.

One local CPU `torch.Generator` is seeded 20261006. Generate rays in K ascending, training seed ascending, ray index ascending order, with explicit float32 `torch.rand(K) + 0.01`, then L1 coefficients and L2 direction normalization. Export the actual coefficients and derived 4096-element directions. The global RNG remains unchanged. Negative, zero-total or nonfinite coefficients and cancelling/zero directions are rejected, never resampled. Basis axes are boundary directions, whereas centroid and random rays have strictly positive coefficients.

Use native signed unit-L2 doses `[-64,-32,-8,-2,0,2,8,32,64]`, L19, `original_instruction_post`, original-prompt-only residual addition. Negative doses are separately named `negative_cone_financial_adaptation`, the reflected cone −C. Their coverage is not evidence for positive cone C. No arbitrary tickers, layers or doses are accepted and no CLI is added.

## D3. Full validation coverage remains the ITT plan

`compile_cone_validation_plan(inputs, parent_plan, training_directory, audit_path)` reconstructs the approved full 503-ticker/500-issuer/2012-input baseline using the existing strict accepted input boundary. It reads no parent outcomes and selects only the authoritative validation role: all 75 tickers, all four conditions, 300 input keys. Fit, calibration and evaluation keys cannot enter planned validation cells.

Three K2 cones × 7 directions × 9 doses × 300 keys = 56700 cells.
Three K4 cones × 9 directions × 9 doses × 300 keys = 72900 cells.
Total is **129600 logical cells**, 48 directions, 14400 zero-dose cells and 115200 nonzero cells. Mandatory coverage records each cone's count, role membership totals and all four conditions. Cell order is cone, family, dose, parent input order.

Every planned failed generation remains in ITT, with its failure type and unknown baseline accounting. Do not shrink denominators, replace invalid outputs or resample. All logical zero-dose cells remain planned. One legitimate baseline generation per input may be referenced by multiple zero cells only with identical model, prompt, schema, tokenizer, decoding and hook configuration. These references must point to an actual generated record and disclose reuse, never fabricate independent outputs or imply independent replicates. Parent output reuse additionally requires actual exact-config and replay checks. The compiler does not perform these checks or manufacture results.

## D4. Exports are immutable and compact relative to model state

`ConeValidationPlan` stores canonical immutable bytes and returns detached JSON exports. Panel ID is SHA256 of the canonical panel, including coefficients/directions, source/config/operator/teacher hashes, audit disclosure, compiler source hash and torch version. Plan ID is SHA256 of canonical plan bytes without a self-referential ID. Neither is a research eligibility attestation.

`write_cone_validation_plan(fresh_directory, plan)` writes `plan.json`, `panel.json`, mandatory `coverage.json` and `manifest.json` with byte digests and both identifiers. It refuses an existing destination. The plan contains every logical cell plus the derived panel, not raw activations, residuals, gradients, logits, KV cache, checkpoints or optimizer state. Existing training history is read-only. Exported derived operator directions are permitted research artifacts.

## A1. Next phase implements the actual runner, not this compiler

Before full generation, authenticate the saved parent and current checkpoint/tokenizer/schema/prompt/runtime identities, run required gates and exact parent replay once, and record diagnostics without granting formal eligibility on failure. Then execute or explicitly record failures for the exact full cell plan with immutable keyed storage and hash-bound resume. No shortened ticker/ray/dose panel, post-outcome threshold, seed selection, evaluation lookup or raw activation persistence is allowed. Report coverage, schema/reason completion, directional flip denominators, ITT flips and failures separately for C and −C. Establish any operator acceptance protocol prospectively in a later reviewed design, not from this exploration's best observed point.

## Verification

Synthetic saved-training fixtures use the existing accepted-input factory to load the full approved roles and all inputs. Only synthetic audit pins are patched in tests, never production loaders or population validation. Tests exercise exact count/grid, local RNG and generation ordering, normalization and positive-cone geometry, no role leakage, audit and artifact mismatches, token-policy/input/parent/config bindings, missing runs, invalid basis/coefficients, cancellation, immutable IDs and no-overwrite exports.

Run `PYTHONPATH=$PWD uv run --no-sync pytest -q tests/test_stance_cone_validation_plan.py`, `uv lock --check`, `git diff --check`, and verify `uv.lock` has no diff. An additional read-only CPU compilation against the actual accepted six-run grid checks compatibility without publishing a run or generating outcomes.
