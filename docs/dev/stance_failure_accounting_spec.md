# Every generation failure must survive execution accounting

**Goal:** Add the existing structured-generation `unsupported_tokenizer` diagnostic to execution outcomes, ITT failure counts and no-op gates without renaming it, dropping rows, changing denominator rules or rewriting artifacts. This integration slice precedes the baseline runner; it does not change the research design or execute a model.

## 1. Grounded mismatch and narrow scope

Verified `core/inference/structured_output.py:FailureType` has nine names, including unsupported_tokenizer. Both plain and proposed Harmony generation can report it with finish_reason='unsupported' when strict byte/HF decoding equality fails at runtime. Pre-forward compiler/renderer unsupported errors are different: no generation was executed.

Verified `core/experiment_contract.py:_FAILURE_FINISH` accepts unsupported_channel/unsupported but not unsupported_tokenizer; `GenerationOutcome` rejects unknown types and clears primary decision for all failures. Verified `core/decision_metrics.py:_FAILURE_TYPES` and `core/stance_gates.py:_FAILURES` have eight names. Thus direct generated-result conversion currently fails instead of preserving the intended diagnostic. The baseline grounding suggestion to map it to exception is rejected.

In scope: additive accepted failure vocabulary and tests. Non-goals: new generation failure types, changing parser/schema/finish values, marking failed output primary-valid, silently aborting a planned executed row, changing gates/metrics formulas, checkpoint runs, rewriting existing JSON/artifacts, compiler/template/protocol changes. Existing serialized eight-type records must remain readable and byte-identical when roundtripped.

## 2. Numbered contracts (new integration)

**REQ-F1 — Outcome preserves the exact cause.** In experiment_contract, add unsupported_tokenizer -> unsupported to the failure/finish relation. It always requires decision=None; primary_valid is false. Existing diagnostic flag dependencies still apply, but all legal diagnostic completion flags may remain true when a decoder mismatch invalidates an otherwise complete object. Reject this failure with eos/schema_complete/exception/timeouts as finishes and reject any non-null primary decision. Strict JSON imports, copied/forged-record validation, ExecutionRow/Shards, progress and complete merge retain executed failure rows. No new primary success state.

**REQ-F2 — Metrics count failures without shrinking source denominators.** Nine canonical names alphabetically ordered: exception, invalid_json, invalid_reason, invalid_schema, no_legal_token, timeout, truncated, unsupported_channel, unsupported_tokenizer. Every failure-count dictionary includes all nine with zero defaults. An intervention unsupported_tokenizer remains in the baseline-source ITT denominator and contributes no flip/retention; a baseline failure remains unknown, not a source decision. Complete-plan/gate/issuer matching remains strict, and progress still exposes no efficacy. Do not convert source-free NA to0. Bootstrap formulas/RNG ordering unchanged.

**REQ-F3 — Gate consumes failed diagnostic state honestly.** GateGenerationInput admits unsupported_tokenizer as its ninth failure name with the same int64/text/flag validation as all failures. Even identical complete IDs/text/flags cannot pass if any arm has this failure. Full ten checks/twenty diagnostic keys persist and include the exact failure string. Existing invalid caller inputs still raise. No import of the inference module or public common framework is needed; simple documented local tuples are sufficient.

**REQ-F4 — Compatibility is explicit before formal freeze.** This is an additive failure-enum extension, not a changed JSON shape. Existing schema_version=1 envelope/field structures and successful canonical hashes remain unchanged. Previously persisted metrics JSON is not rewritten; new outputs add a zero/count entry, whose provenance is distinguished by the current code hash and future frozen protocol specifying all nine failures. No completed rebuild model runs exist to relabel. Unit fixtures may update their expected canonical failure names. Test old eight-type outcome/row/plan/shard roundtrips unchanged and demonstrate new outcome import/export/merge. Never change Python integer conversion globals or historical sources.

## 3. Slice F1 is one independent integration deliverable

Modify only `llm_bias/core/experiment_contract.py`, `llm_bias/core/decision_metrics.py`, `llm_bias/core/stance_gates.py`, `tests/test_stance_contract.py`, `tests/test_decision_metrics.py`, `tests/test_stance_gates.py`. Leave generation code, compiled inputs, docs and all historical data untouched. This spec governs the deliberate nine-type extension; earlier eight-type spec descriptions are historical contracts superseded only at this integration boundary.

Consumes verified GenerationOutcome/ExecutionRow/ExecutionShard/ExperimentPlan, summarize_decisions/metrics_progress and GateGenerationInput/evaluate_noop_gate. Produces unchanged APIs with nine-type support.

Synthetic red fixtures: unsupported_tokenizer generation-outcome import currently raises, metrics full-plan fixture cannot count it, gate input currently rejects it. Green fixtures: complete source-buy baseline paired to invalid-tokenizer intervention, legal diagnostic flags retained/no primary decision, fixed ITT denominator/counts, gate false, complete shard row counted executed/zero missing. Test both baseline and intervention failure independently. Wrong finish/non-null decision rejected. Parameterized existing all-failure tests include the ninth cause, and old eight causes stay accepted. Independent hand calculation, not merely mirrored constants.

Verification: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q tests/test_stance_contract.py tests/test_decision_metrics.py tests/test_stance_gates.py`; lock/diff; full thread-bounded pytest. Read-only preflight before implementation; independent acceptance and commit before baseline integration.

## 4. Audit skeleton

Core claim: a generation that ran and failed byte integrity must remain a planned executed failure, not disappear or change cause. Supporting lines: typed outcomes, fixed-denominator counts, false no-op gate and backward-readable structure. Deferred: actual baseline execution and frozen model/protocol configuration, not implied by engineering tests.
