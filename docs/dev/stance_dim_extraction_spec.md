# DIM extraction returns fit-only candidates for separate evidence conditions

Implemented pure slice for item 4 of [the parallel contract](stance_parallel_experiments_contract.md).
Code: `llm_bias/core/stance_dim_extraction.py`. Tests: `tests/test_stance_dim_extraction.py`.
Engineering sufficiency does not establish steering efficacy or research eligibility.

## A1. Supply approved baseline bindings and transient captures

Call `extract_dim_candidates` with keyword arguments:

| Argument | Contract |
| --- | --- |
| `plan` | Existing `ExperimentPlan`, baseline stage/arm and dose `0`, covering each supplied population member in all four conditions. Preserve the approved full503 plan in research. |
| `members`, `roles` | Existing `PopulationMember` sequence and four-role ticker lists. `validate_members` and `validate_roles` enforce complete membership and issuer isolation. Tiny fixtures authorize tests only. |
| `rows` | Iterable of existing `ExecutionRow` generated outcomes, exclusively planned fit keys. Reject duplicate, foreign, validation, calibration and evaluation teachers. Bind ticker, issuer, condition and trial to each key. |
| `pooled_residuals` | Memory-only mapping from executed fit baseline `RowKey` to real finite `[d_model]` numeric vectors. Reject bool, complex, object, wrong shape, NaN and infinity. Even unused invalid-label captures must pass numeric checks. |
| `layer`, `span`, `d_model` | Declared nonnegative integer layer, one of `entity`, `evidence1`, `evidence2`, `instruction`, and positive integer dimension. No historical peak default. |
| `parent_sha256` | Immutable generated baseline parent binding. |
| `candidate_panel_sha256` | Future declared layer/span panel binding. This slice does not construct or select that panel. |
| `capture_policy_sha256` | Declared model/hook/original-prompt capture policy binding. |

All three supplied hashes require lowercase SHA-256 format. The output records the full
plan identity, actual membership and role hashes, fit-key hash, generated teacher-row
hash and capture-key hash. Caller-supplied hashes describe provenance claims, not
file authentication. The future runner must use approved `BaselineInputs`, rebuild
the baseline plan and consume an authenticated completed baseline parent. This pure
API cannot prove checkpoint, prompt, hook-site or label authenticity from arrays.

`pool_original_span(residuals, positions, *, original_prompt_length, d_model)` takes
an in-memory real finite `[original_prompt_length,d_model]` array and unique,
nonempty absolute original-prompt indices. It returns the arithmetic span mean as a
transient tuple. Reject generated-prefix length, batch/tokenwise shape, duplicate
or out-of-bounds positions. The runner must establish that the supplied tensor
belongs to the original prompt. Neither API writes files or retains input arrays.

## A2. Compute company-first and issuer-first class means

For each condition independently:

1. Use `GenerationOutcome.primary_valid` buy/sell labels. Retain failed outcomes
   as unknown coverage, even when they completed a diagnostic decision token.
2. For each label and ticker, average its available pooled trial vectors.
3. For each label and issuer, average those ticker means with equal share-class
   weight. Then average issuer means with equal issuer weight within that label.
4. Subtract sell from buy to return one shared `[d_model]` direction for the site.

No class-mean normalization, unit normalization, scale fitting, margin ranking or
cross-condition pooling occurs. Rescaling input vectors rescales the derived
vector. Repeated trials cannot give one ticker extra class weight. An identical
additional share class of a single-ticker issuer cannot increase that issuer's
weight. If trials or share classes give opposite labels, the issuer can occur in
both class means. Report generated and contributing class-issuer overlap rather
than infer an issuer label or drop the issuer.

Use float64 arithmetic, divide before summing means, and reject nonfinite derived
means, differences or norms. Class means remain transient and do not enter exports.

## A3. Report sufficiency and coverage without choosing a winner

Return `DIMExtraction`. `to_dict()` produces independent JSON copies from frozen
canonical bytes. `extraction_sha256` hashes the payload excluding that hash itself.
Only derived directions, scalar norms, compact counts and provenance enter exports.
No raw token arrays, pooled per-row arrays, class-mean arrays, gradients or caches
enter the result. Caller code must discard its transient captures after extraction.

`candidates` contains four records in fixed order `+-`, `-+`, `++`, `--`.
Each record has `condition`, `status`, `reasons`, `direction`, `direction_norm`
and `coverage`. A candidate needs at least **20 distinct contributing issuers
in each generated class**. Otherwise return `status='untestable'`, reasons
`insufficient_buy_issuers` and/or `insufficient_sell_issuers`, and null vector/norm.
A zero difference also returns untestable with `zero_direction` and null vector/norm.
A sufficient candidate only passes this numeric support test.

Coverage includes planned rows/companies/issuers, executed and missing rows,
unknown generated rows and failure-type counts, generated rows/companies/issuers
per class, contributing rows/companies/issuers, missing primary-valid capture rows
and issuer overlap. Missing rows do not become executed failures. Unknown rows do
not acquire guessed labels. `coverage_complete` means no missing generated row or
missing primary-valid capture. Executed failures remain visible even when coverage
is complete. A candidate may have sufficient class support with incomplete coverage,
so the future runner must enforce its separate completion and research gates.

The ordering records provenance priority only. No preferred condition, class,
validation winner or causal effect appears in the output. A zero buy/sell class
cannot borrow labels from another condition or receive a fabricated denominator.
For unsupported capture/model cases, callers may supply no observations and receive
untestable coverage. They must retain the upstream unsupported reason in their
execution artifact. This slice does not infer why an observation is missing.

## A4. Verify math now and defer checkpoint execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync pytest -q tests/test_stance_dim_extraction.py
uv lock --check
git diff --check
```

`PYTHONPATH=.` selects this worktree's sources instead of the shared environment's
editable main checkout. `--no-sync` preserves pinned shared dependencies.
Tests cover analytical means, unequal trial/share-class weighting, scale covariance,
invalid shapes/numerics, class support, absent classes, separate conditions,
missing and executed-failure coverage, role leakage, issuer binding, zero/overflow,
canonical order and immutable provenance.

Next dependencies: approved candidate/capture panel, authenticated baseline adapter,
transient checkpoint capture runner, generated validation selection and completion
gates. GPU fitting, dose calibration, evaluation, neuron/cone conditioning and
research success claims remain outside this slice.
