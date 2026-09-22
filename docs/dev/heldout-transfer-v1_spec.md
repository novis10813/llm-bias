# Held-out Transfer V1 Implementation Spec

**Goal:** Implement the proposed Entity-to-Dial held-out transfer V1 workflow: a deterministic 200-company, all-GICS-sector cohort that evaluates the already frozen L15 \(V_8\) basis through source→target fixed-margin transplants.

**Canonical research contract:** `docs/entity-to-dial/details/proposal-heldout-transfer-v1.md`. This implementation spec does not authorize a formal model run. It only authorizes implementation slices and their fake-model/unit-test verification.

**Architecture:** Keep the research semantics in the existing `llm_bias.entity_to_dial` package. Add a versioned `heldout_transfer.py` module and one explicit script runner. Reuse `llm_bias.core` for artifacts, prompt encoding, model loading, residual recording, continuation scoring and hook lifecycle. Reuse only modules in `llm_bias.entity_to_dial` for the frozen basis and state-transfer transforms. Do not import `selective_intervention`, `evidence_insensitivity`, or any other experiment package.

**Grounding note:** `load_population` establishes the 503-row 2024 S&P 500 contract and deterministic proportionate allocation pattern【已驗證: `llm_bias/evidence_insensitivity/population.py:load_population`, `_stratified_sample`】. `load_pca_basis`, `make_full_transform`, and `make_projected_transplant` already enforce E-01 basis and L15 transplant semantics【已驗證: `llm_bias/entity_to_dial/joint_patch.py:load_pca_basis`, `make_full_transform`, `make_projected_transplant`】. `ArtifactRun` supplies stage recording/finalization but has no experiment semantics【已驗證: `llm_bias/core/artifacts/lifecycle.py:ArtifactRun`】.

## 1. Scope and non-goals

### In scope

- New held-out V1 constants, deterministic cohort manifest, input provenance checks, prompt preparation, two frozen pair graphs, four random 8D controls, compact record analysis, fake-model workflow smoke, and a standalone script.
- The four pipeline stages remain `prepare → forward → analyze → finalize`; the single `forward` stage must first write selection margins and frozen graphs, then run evaluation arms.
- Unit and fake-model tests that prove selection/evaluation separation, transform/no-op behavior, compact serialization, lifecycle behavior and fail-closed cases.

### Non-goals

- Do not modify Phase E/F result files, their existing gates, their PCA basis, M6-V2, or any historical artifact.
- Do not add a `pyproject.toml` top-level command. Entity-to-Dial has phase scripts rather than a package CLI【已驗證: `pyproject.toml:[project.scripts]`, `scripts/entity_to_dial_phase_e.py:main`】.
- Do not run the 200-company formal evaluation, add generation/decision-flip scoring, fit any SVD/direction/cone, add a cross-model/task arm, or persist raw states/deltas/random matrices/model completions.
- Do not change `llm_bias.core`, shared schema behavior, population-loader behavior, or import boundaries.

## 2. Requirements and contracts

### REQ-1 — V1 constants and immutable provenance

Create `llm_bias/entity_to_dial/heldout_transfer.py` with V1-owned constants:

```python
# 【新設計】
HELDOUT_TRANSFER_SCHEMA = "entity-to-dial-heldout-transfer-v1"
HELDOUT_PROTOCOL = "docs/entity-to-dial/details/proposal-heldout-transfer-v1.md"
HELDOUT_MODEL_SLUG = "qwen3.5-4b"
HELDOUT_COHORT_SIZE = 200
HELDOUT_SELECTION_SEED = 20260930
HELDOUT_RANDOM_BASIS_COUNT = 4
HELDOUT_EVAL_VARIANTS = ((True, 0), (False, 1), (True, 1))
HELDOUT_SELECTION_VARIANT = (False, 0)
HELDOUT_LAYER = 15
HELDOUT_BASIS_K = 8
HELDOUT_RATIO_MIN_FULL_DM = 0.2
HELDOUT_RECOVERY_TARGET = 0.8
HELDOUT_BOOTSTRAP_SAMPLES = 10_000
HELDOUT_BOOTSTRAP_SEED = 20260930
HELDOUT_NOOP_TOLERANCE = 1e-12
```

Construction ticker exclusion is the protocol’s exact 16-ticker list. M6 exclusions are read from a supplied M6 V2 manifest rather than copied into code. Validate M6 schema/entries locally: exactly 12 company objects, non-empty `ticker`, `name`, `sector`, ticker unique; the historical M6 population hash is recorded but never compared with this protocol’s population CSV hash. An M6 ticker absent from the 2024 503-row population is a harmless set-intersection no-op.

`verify_heldout_inputs(...)` must fail before model forward if any supplied E-01 manifest is incomplete, E-01 summary is unavailable/invalid under `load_pca_basis`, M6 entry schema is invalid, construction and M6 exclusions overlap, or the population CSV fails the existing 503-row contract. It returns only compact provenance and the CPU `V8: torch.Tensor[d_model, 8]`; it must not serialize the tensor.

### REQ-2 — cohort manifest

`prepare_heldout_cohort(rows, *, m6_manifest, population_path, selection_seed)` must:

1. apply construction and M6 ticker exclusions to the canonical rows;
2. require at least 200 eligible rows;
3. allocate the 200 quota proportionally using the exact floor/fractional-remainder/lexicographic-tie policy of `population._stratified_sample`;
4. derive `local_seed = int.from_bytes(hashlib.sha256(f"{seed}:{sector_index}:{sector}".encode()).digest()[:8], "big")` and use `random.Random(local_seed).shuffle` within each lexicographically sorted sector bucket;
5. select exactly the quota, then ticker-sort the final rows.

It returns only a JSON-serializable manifest. Required shape:

```python
# 【新設計】
{
  "schema_version": "entity-to-dial-heldout-transfer-v1",
  "selection_seed": 20260930,
  "population": {"path": "data/all_constituents_2020_2025.csv", "sha256": "<64 hex>", "n_rows": 503},
  "exclusions": {
    "construction_tickers": ["..."],
    "m6_manifest_sha256": "<64 hex>",
    "m6_tickers": ["..."],
    "eligible_ticker_count": 0
  },
  "sector_counts": {"<sector>": {"eligible": 0, "quota": 0}},
  "companies": [{"ticker": "...", "name": "...", "sector": "..."}],
  "selected_ticker_sha256": "<64 hex>",
  "raw_runtime_payloads": False
}
```

The `companies` list length must be exactly 200, ticker unique, and ticker-sorted. No fallback or candidate replacement is allowed after selection.

### REQ-3 — prompt preparation and selection/evaluation separation

Reuse Entity-to-Dial’s frozen prompt rendering and span resolution, not a second template implementation【已驗證: `llm_bias/entity_to_dial.template`, `llm_bias/entity_to_dial.spans:instruction_char_span`】. Add a helper that creates four records per cohort company and requires exactly one of each `(reverse, order)` pair.

Each compact prompt record must include `prompt_id`, `ticker`, `name`, `sector`, `reverse`, `order`, `role` (`"selection"` or `"evaluation"`), `formatted`, `instruction_span: [start, end]`, `formatted_sha256`, `chat_template_sha256`, `tokenizer_name_or_path`, and `token_count`. It must reject an empty span, a missing formatted body, or non-positive `token_count`. Do not store `input_ids`.

Only `reverse=False, order=0` is `role="selection"`; the other three are evaluation. Analysis functions must reject a selection record in evaluation input and reject any graph constructed from a non-selection margin.

### REQ-4 — deterministic pair graphs

`build_sector_within_graph(selection_records, cohort_manifest)` must sort each sector by `(margin, ticker)` ascending and pair outer values: `(rows[n-1-j], rows[j])`. It emits undirected pair objects with `pair_id`, `high_ticker`, `low_ticker`, margins, sector labels and `stratum="sector_within"`. Odd-sector central entities are returned under `unmatched_tickers`. Pair IDs are deterministic: `within:<sector-slug>:<high>:<low>`.

`build_cross_sector_graph(selection_records, cohort_manifest)` must sort the 200 records by `(margin, ticker)`, take the lower 100 as targets and upper 100 as sources, and find a 100-edge maximum bipartite matching subject to different sectors. Implement deterministic DFS augmenting paths:

- visit targets in `(margin ascending, ticker ascending)` order;
- consider candidate sources in `(margin descending, ticker ascending)` order;
- accept only `source.sector != target.sector`;
- if cardinality is not exactly 100, raise `ValueError`; do not mutate the split or permit same-sector fallback.

Its pair IDs are `cross:<high>:<low>`. Both graph builders must validate all selected tickers occur exactly once within their own graph, except documented sector-within odd leftovers. Graph JSON must include selection margin provenance and no evaluation fields.

### REQ-5 — random controls

`random_orthonormal_bases(width, *, count=4, k=8, seed=20260930)` produces four CPU float32 bases. For index `j`, create an FP64 `torch.randn((width, k), generator=Generator("cpu").manual_seed(seed+j))`, reduce QR, and multiply each Q column by `+1/-1` so its maximum-absolute-entry is positive (tie: first index). Cast to float32 only after this canonical sign correction.

Return `(bases, metadata)`, where metadata has index, seed, shape, row-Gram orthonormality max error, and SHA-256 of tensor bytes. Do not write bases to artifacts. Reject non-positive width/count/k or `k > width`.

### REQ-6 — forward arm contract

`run_heldout_transfer(...)` is the sole workflow entry. It must use `ArtifactRun.create(model=..., dataset="entity-to-dial-heldout-transfer", ...)` and exactly stages `{prepare, forward, analyze}` before `finalize`【已驗證: `llm_bias/core/artifacts/lifecycle.py:ArtifactRun`】.

During `prepare`, write cohort manifest and prompt metadata only. During `forward`:

1. load the model once, score 200 selection prompts clean, write `forward/selection.jsonl`;
2. build and write both frozen graphs to `forward/pair_graphs.json` before any evaluation-arm record;
3. for every graph pair, both directions, and three evaluation variants, capture source/target L15 post states transiently and score target clean, full, V8 and `random_8d_0` through `random_8d_3`;
4. for every company × evaluation variant, run full and V8 self-source no-op and require `abs(patched_margin-clean_margin) <= 1e-12`;
5. write records only after all finite/hook/no-op checks for the associated unit succeed.

Use `make_full_transform` and `make_projected_transplant`; map source and target instruction spans with `nearest_position_mapping`【已驗證: `llm_bias/entity_to_dial.block_patch:nearest_position_mapping`, `llm_bias.entity_to_dial.joint_patch:make_full_transform`, `make_projected_transplant`】. Capture and retain residuals only in memory. Each intervention hook must fire once; a missing or double fire fails closed.

Evaluation record shape:

```python
# 【新設計】
{
  "stratum": "sector_within|cross_sector",
  "pair_id": "...", "direction": "high_to_low|low_to_high",
  "source_ticker": "...", "target_ticker": "...",
  "source_sector": "...", "target_sector": "...",
  "reverse": True, "order": 0,
  "arm": "clean|full|v8|random_8d_0|random_8d_1|random_8d_2|random_8d_3",
  "source_margin": 0.0, "target_margin": 0.0, "patched_margin": 0.0,
  "toward_source_delta_m": 0.0,
  "hook_fires": 0,
  "raw_runtime_payloads": False
}
```

No-op record has `ticker`, `reverse`, `order`, `arm` (`full_self_noop|v8_self_noop`), `clean_margin`, `patched_margin`, `delta_m`, `pass`, and `raw_runtime_payloads=False`.

### REQ-7 — compact analysis

`analyze_heldout_transfer(records, graphs, *, bootstrap_samples, bootstrap_seed)` must require exactly all seven arms per `(stratum, pair_id, direction, reverse, order)` evaluation cell and all three evaluation variants. It computes toward-source shift from raw compact margins, then per edge the median across the three variants.

Eligibility is exactly `abs(median_full_delta_m) >= 0.2`. Excluded edges remain in output with `eligible=False`, `exclusion_reason="full_reference_below_threshold"`, and neither V8 nor random result may influence eligibility.

For eligible edges compute:

```python
# 【新設計】
ratio_v8 = median_v8_delta_m / median_full_delta_m
ratio_random_j = median_random_j_delta_m / median_full_delta_m
paired_advantage = ratio_v8 - median(ratio_random_0, ..., ratio_random_3)
```

For each stratum, separately report all-directions, `high_to_low`, `low_to_high`, V8 median ratio, four random median ratios, median paired advantage, full reference magnitude, eligible/excluded count, sector breakdown, and 10,000-replicate paired-bundle percentile 95% CI. A bootstrap bundle is one undirected pair containing both directed records. Resample eligible bundles with replacement; never combine strata or treat the same entity in different graphs as independent.

Status fields are exactly:

- `"supported"`: V8 median ratio >= 0.8, V8 CI lower >= 0.8, paired-advantage median > 0, and advantage CI lower > 0;
- `"unresolved"`: V8 median ratio >= 0.8 but any of the preceding CI/specificity conditions fails;
- `"not_supported"`: V8 median ratio < 0.8;
- `"not_evaluable"`: zero eligible bundles.

The top-level summary must include `raw_runtime_payloads: false`; it must contain no arrays named `activation`, `residual`, `hidden`, `delta`, `cache`, `basis`, `input_ids`, or `logits`.

### REQ-8 — script and smoke mode

Create `scripts/entity_to_dial_heldout_transfer.py`. It requires `--model`, `--phase-e-run`, `--m6-manifest`, `--population-csv`; it accepts `--artifact-root`, `--run-id`, and `--smoke`.

- Validate paths before creating an artifact run.
- A formal invocation requires `--run-id`; smoke derives `entity-to-dial-heldout-transfer-smoke-<UTC timestamp>`.
- `--smoke` uses only ticker-sorted first two selected companies and `eval_forward_order_reverse`; it does not score all selection rows, does not build formal graphs, and writes `smoke: true` compact metadata.
- Smoke must run source/target clean, full, V8, random-8D-0, both self-source no-ops, artifact lifecycle and finite/single-hook checks. It has no effect-size gate.

## 3. Implementation slices

### Slice 1 — cohort, prompt roles, graphs and random controls

**Requirements:** REQ-1 through REQ-5.

**Files:**
- Create `llm_bias/entity_to_dial/heldout_transfer.py`
- Create `tests/test_entity_to_dial_heldout_transfer.py`

**Consumes:** `load_population` data contract【已驗證: `llm_bias/evidence_insensitivity.population:load_population`】, `load_pca_basis` validation precedent【已驗證: `llm_bias/entity_to_dial.joint_patch:load_pca_basis`】, temporary CSV fixture pattern【已驗證: `tests/test_selective_intervention_m6_manifest.py:_write_csv`】.

**Produces:** pure deterministic helpers: `validate_m6_exclusion_manifest`, `prepare_heldout_cohort`, prompt-role validation, both graph builders, `random_orthonormal_bases`【新設計】.

**Synthetic fixture topology:** a temporary 503-row CSV over at least five sectors; a valid 12-entry M6 object with one absent ticker; synthetic selection margins with ties and one odd sector; tiny width=16 random bases. Use no model.

**Steps:**
1. Add failing tests for 200 deterministic proportional selection, all exclusions, absent M6 ticker no-op, malformed M6 failure, ticker-sorted output, and selected count.
2. Run the new test file and confirm failure before implementation.
3. Implement manifest/cohort validation and test green.
4. Add failing graph tests for tie-break, within leftovers, no same-sector cross edges, perfect matching determinism, impossible matching failure, and selection-only record enforcement.
5. Implement graph helpers and test green.
6. Add/implement random QR reproducibility, sign canonicalization, orthonormality and invalid-shape tests.

**Verification:** `uv run pytest -q tests/test_entity_to_dial_heldout_transfer.py`

**Commit boundary:** `feat: add heldout transfer cohort and pairing helpers`

### Slice 2 — transfer metrics and bundle-bootstrap analysis

**Requirements:** REQ-7.

**Files:**
- Modify `llm_bias/entity_to_dial/heldout_transfer.py`
- Modify `tests/test_entity_to_dial_heldout_transfer.py`

**Consumes:** Slice 1 graph objects【新設計】; `toward_source_delta` semantics【已驗證: `llm_bias/entity_to_dial.analysis:toward_source_delta`】.

**Produces:** `analyze_heldout_transfer` and compact schema validator【新設計】.

**Synthetic fixture topology:** two pair bundles per stratum, two directions per bundle, all three evaluation variants and seven arms; construct ratios to cover supported/unresolved/not_supported/not_evaluable and an excluded low-full-reference edge.

**Steps:**
1. Add failing tests for missing arm/variant rejection, median aggregation, full-only eligibility, preserved exclusions, random advantage, direction separation, and no cross-stratum pooling.
2. Run tests red.
3. Implement per-edge reduction and stratum summaries.
4. Add failing deterministic bootstrap tests showing resampled bundles retain two directed edges and threshold statuses at equality.
5. Implement bootstrap/status handling and run green.
6. Assert serialized summary recursively contains no forbidden raw-payload keys.

**Verification:** `uv run pytest -q tests/test_entity_to_dial_heldout_transfer.py`

**Commit boundary:** `feat: analyze heldout transfer recovery`

### Slice 3 — artifact workflow and fake-model forward

**Requirements:** REQ-1, REQ-3, REQ-6, REQ-7.

**Files:**
- Modify `llm_bias/entity_to_dial/heldout_transfer.py`
- Modify `tests/test_entity_to_dial_heldout_transfer.py`

**Consumes:** core lifecycle【已驗證: `llm_bias.core.artifacts.lifecycle:ArtifactRun`】, final-residual scoring/replay pattern【已驗證: `llm_bias.entity_to_dial.pipeline:_patched_final_margin`, `_live_margin`】, existing fake tokenizer/model/upstream fixture helpers【已驗證: `tests/test_entity_to_dial_pipeline.py:_CharTokenizer`, `_qwen_fake`, `_fake_upstream`】.

**Produces:** `run_heldout_transfer(...) -> Path` with compact prepare/forward/analyze artifacts【新設計】.

**Synthetic fixture topology:** monkeypatch model loading to a deterministic fake with 16 layers / width at least 16; fabricate an E-01 complete manifest plus 16-row PCA summary compatible with `load_pca_basis`; provide a temporary canonical population that the Slice 1 helper can scale via a test-only `cohort_size` argument. The production entry point must retain cohort size 200; only helper-level test injection may use 8.

**Steps:**
1. Add a failing fake pipeline test for completed manifest and all required compact files.
2. Implement prepare provenance/prompt artifacts and test green.
3. Add a failing forward test that asserts selection is written before graphs, graphs precede evaluation records, all arms occur, no-op fields are exact, and graph does not contain evaluation margins.
4. Implement transient capture/intervention and compact writes; run green.
5. Add provenance-tamper, basis-invalid, malformed graph and no-op failure tests; implement fail-closed paths.
6. Verify artifact text has no forbidden raw payload keys.

**Verification:** `uv run pytest -q tests/test_entity_to_dial_heldout_transfer.py tests/test_entity_to_dial_joint.py tests/test_entity_to_dial_pipeline.py`

**Commit boundary:** `feat: add heldout transfer workflow`

### Slice 4 — script and smoke behavior

**Requirements:** REQ-8.

**Files:**
- Create `scripts/entity_to_dial_heldout_transfer.py`
- Modify `tests/test_entity_to_dial_heldout_transfer.py`
- Modify `docs/research-scripts.md`
- Modify `tests/AGENTS.md`

**Consumes:** script argument-validation style【已驗證: `scripts/entity_to_dial_phase_e.py:main`】 and Slice 3 runner【新設計】.

**Produces:** standalone script command and documented research-script ownership entry【新設計】.

**Synthetic fixture topology:** monkeypatch `run_heldout_transfer`; temporary existing/missing argument paths; assert formal run ID requirement and smoke invocation uses `smoke=True` and a generated ID.

**Steps:**
1. Add red CLI tests for required paths, formal `--run-id`, smoke generated ID, and forwarding all paths.
2. Implement parser and run green.
3. Add a fake smoke workflow test verifying it writes only one pair’s compact smoke artifact and does not call graph-building/formal-cohort evaluation helper.
4. Update only the corresponding Entity-to-Dial operator entry in `docs/research-scripts.md` and add one concise workflow entry to `tests/AGENTS.md`; run all Slice verification commands green.

**Verification:** `uv run pytest -q tests/test_entity_to_dial_heldout_transfer.py tests/test_workflow_boundaries.py && uv run python scripts/entity_to_dial_heldout_transfer.py --help`

**Commit boundary:** `feat: expose heldout transfer smoke runner`

## 4. Full verification and acceptance

Before a real-model smoke, all slices must pass:

```bash
uv run pytest -q tests/test_entity_to_dial_heldout_transfer.py \
  tests/test_entity_to_dial_joint.py \
  tests/test_entity_to_dial_pipeline.py \
  tests/test_workflow_boundaries.py
uv run python -m compileall -q llm_bias
uv run python scripts/entity_to_dial_heldout_transfer.py --help
```

The real-model smoke command is allowed only after these checks are green and must not be followed by a formal run in the same authorization:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/entity_to_dial_heldout_transfer.py \
  --model .cache/models/qwen3.5-4b \
  --phase-e-run artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01 \
  --m6-manifest <frozen-m6-v2-manifest> \
  --population-csv data/all_constituents_2020_2025.csv \
  --smoke
```

Acceptance requires: deterministic tests pass; existing Entity-to-Dial tests regress nowhere; workflow boundary test passes; artifacts contain only compact outputs; every no-op is within `1e-12`; all hooks fire exactly once; invalid provenance/graph/arm inputs fail before summary; and no formal 200-company evaluation is executed.
