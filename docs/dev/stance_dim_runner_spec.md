# DIM runner captures the declared depth panel without selecting an operator

Engineering implementation of the DIM slice in [wave2](stance_parallel_runner_wave2_spec.md).
No checkpoint run or steering-effect claim is included.

## A1. Authenticate inputs and live runtime before capture

`uv run python scripts/fit_stance_dim.py --inputs INPUTS --parent BASELINE
--model ORIGINAL_CHECKPOINT --output-dir FRESH_RUN` consumes the approved full503
input pack and completed2012 baseline. Optional `--recovery RECOVERY` consumes the
accepted merged effective view, retaining each fit row's source and generation
policy. Merged snapshots read original runtime bindings from metadata.original_metadata.bindings, while every generated row uses its own effective source/policy. Verified merged-checkpoint relocation may retain exact metadata hashes at a new physical path, record actual/logical tokenizer names and bind the logical name through the accepted relocation guard before compiling; full tokenizer/grammar hashes remain mandatory. There are no ticker, condition, layer, dose or budget overrides.

Verify original checkpoint path and metadata bytes, backend versions/kernel policy,
actual CUDA placement, embedding/head dtypes, attention policy, rendered wrapper,
body template, schema, tokenizer and grammar. Checkpoint identity is explicitly
metadata-only, not a full-weight hash attestation. Use the actual loaded weights,
not a caller-provided hash or precomputed residual fixture.

## A2. Capture all fit rows at five actual depths

Panel is sorted unique `floor(i*(L-1)/4)` for i=0..4, post-block, four original
spans. GLM L=40 gives 0/9/19/29/39, 1208 fit generations, 20 site families and
80 condition-specific candidates. All full-plan roles remain bound but only fit
rows are generated or supplied as labels. Convert `inputs.members` unchanged and
`inputs.roles['roles']` to the accepted extraction interface.

For each row, install one fresh position tracker and all panel capture contexts
simultaneously around one actual original schema-constrained greedy generation.
Capture only the first complete original prompt, never a generated prefix. Require
parent full token/text/payload/decision/reason/finish/failure equality and matching
configuration provenance before pooling. Arithmetic original-span means are
memory-only. Context cleanup releases raw sources after that row. No tokenwise
residual, per-row pooled array, class mean, gradient or cache is serialized.

## A3. Preserve failure and sufficiency without success claims

On genuine generation failure, parent drift, configuration drift or unsupported
capture, stop immediately without retry. Record the key, named status, compact
match flags and failure type, not arbitrary exception messages that may contain
raw tensors. Discard pooled observations on halt and produce all declared site
extractions with missing-capture coverage and untestable candidates. A completed
capture can still be untestable due to insufficient classes or zero direction.
Only derived directions, scalar norms, counts and provenance may be exported.
No validation selection, teacher construction, calibration, margin or evaluation.

## A4. Write a fresh immutable artifact and verify CPU fakes

Create output directory exclusively. Existing directories, including interrupted
runs, reject before generation. Write-once `config.json`, per-row compact
`records/*.json`, `operators.json`, and `summary.json`. An interrupted fit may
recompute memory-only data only in a new run. Summary reports planned, attempted,
matched and missing rows, elapsed time, fit completion and unsupported status.
`research_eligible` always remains false. Public preflight failures write a compact
failure summary if the fresh directory was created. No resume or overwrite path.

Focused tests exercise actual simultaneous hooks, first full-prompt means,
complete-output and configuration drift, cleanup on exceptions, all four
conditions, fit role conversion, declared panel, immutable outputs, no raw exports,
and a closed CLI. Pure extraction tests independently verify sufficiency math.
Future generated validation selects exactly one sufficient `+-` vector. Main
reviews/integrates this slice and handles GPU execution and full regression.
