# A closed CPU panel fixes prospective localization coverage

**Goal:** Bind the historical candidate rule to four canonical models without reading intervention effects or implementing a GPU runner. Research decisions and limitations live in [V2 proposal](../concept-cone-steering/localization-candidates-v2/proposal.md).

## REQ-1: Constructor inputs cannot redefine the panel

【新設計】 `LocalizationCandidatePanel(model_slug: str, actual_layer_count: int)` is a frozen slotted record. Factory `build_localization_candidate_panel(*, model_slug: str, actual_layer_count: int)` has the same validation. Accept exactly qwen3.5-4b/32, glm4-9b-0414/40, gemma4-12b-it/48, gpt-oss-20b/24. Require genuine positive integer count, not bool, and equality to the registered expected count. Reject aliases and unknown slugs. The future runner must supply `len(model.layers)` from the loaded model, never an unchecked CLI value. CPU validation checks a supplied count, not checkpoint authenticity.

No public layers, source, peak, outcome, donor, ticker, span, evaluation or sampling arguments are accepted. Immutable `layers` derives the sorted clipped union of historical peak±1, historical injection, L0, floor((L−1)/8), floor((L−1)/2), L−1. Independent exact test oracles are Qwen `(0,3,14,15,16,31)`, GLM `(0,4,19,20,21,39)`, Gemma `(0,5,23,26,27,28,47)`, GPT `(0,2,7,8,9,11,14,23)`.

## REQ-2: Canonical exports bind provenance without tensors

【已驗證: llm_bias/core/artifact_paths.py:sha256_json】 supplies canonical JSON hashing. 【新設計】 `to_dict()` returns fresh lists and scalar fields: kind, model_slug, actual_layer_count, expected_layer_count, layers, rule, index_base, early_divisor, mid_divisor, historical_peak, historical_injection_layer, historical_source, historical_source_sha256, historical_limitations, site, spans, selector, development_roles, full_pair_count, development_pair_count, development_cell_count, objective. `panel_sha256` hashes that complete record. All stored fields and derived tuples are immutable, and modifying exports cannot mutate identity.

Pin `docs/concept-cone-steering/c2-v3-steering-prompt/status.md` to SHA-256 `8a7762e97b934562167c6450bfdf441dca43c6815dc94bd97c20b2b22669b1b7`, resolving from the module's repository root rather than current directory. Constructor reads only those local document bytes and rejects drift or missing source. Record historical teacher-forced margin, fixed-prefix/off-path and steering-prompt limitations, zero-denominator R7 and GPT band disagreement, and data-informed design. Do not load source raw tensors, historical result artifacts, partial V1 effects or evaluation labels.

## REQ-3: Full development counts are fixed, not a sampled grid

【已驗證: llm_bias/core/stance_localization_pairs.py:build_localization_pairs】 builds the role-preserving global pair table. 【已驗證: llm_bias/core/stance_localization_alignment.py:build_span_alignment】 owns current mapping/selector semantics. 【新設計】 Panel exports post/full, all four spans entity/evidence1/evidence2/instruction, development roles fit/validation, full pair count4024 and development pair count3016. `development_cell_count` is 3016×4×candidate count: Qwen/GLM72384, Gemma84448, GPT96512. Counts are planning constants, not proof a parent/table was authenticated. This module does not accept pair subsets or select evaluation rows.

## Slice 1: Deliver only panel, tests and new protocol documents

Scope is `llm_bias/core/stance_localization_candidate_panel.py`, `tests/test_stance_localization_candidate_panel.py`, this spec and V2 `proposal.md`/`status.md`. No dependencies, GPU calls, runner changes, frozen-document updates, result edits, scheduler operations or stop-manifest fabrication. Tests use only registered scalar cases and local source bytes. Cover exact panels/counts, duplicate collapse, clipped boundary neighbors via the private formula helper (not a public alternate panel), malformed/unknown models, wrong/nonpositive/bool counts, missing count, rejected outcome/sampling overrides, frozen mutation, defensive exports, deterministic hashes and historical drift rejection. No checkpoint is loaded.

Acceptance commands:

```bash
env PYTHONPATH=$PWD uv run --no-sync pytest -q tests/test_stance_localization_candidate_panel.py
uv lock --check
git diff --check
git diff --exit-code -- uv.lock
```

Commit only the five owned files on the task branch. Full regression is a separate integration check, not a reason to widen this bounded implementation.

## A1: Next runner brief preserves all execution gates

Implement a separately reviewed runner that consumes this panel after checking actual model count, authenticates full approved inputs/parent/pair table, binds panel hash into fresh immutable plan/shard/cell identities, and enumerates every fit/validation pair across the full panel and four primary full spans. Preserve current mapping/capture/cache/policy, company-first ITT/failure accounting, exact full-parent replay and layer/span no-op gates. No CLI layer/ticker/span sampling. GPT stays blocked on replay drift, without baseline overwrite or regex/payload-only relaxation. Require verified stop/archive manifest before new GPU dispatch. Do not implement this runner in Slice1 or condition neuron/DIM/cone jobs27/41 on the panel.

## Argument skeleton

Core claim: a fixed historical union reduces the layer budget while retaining honest candidate-only inference. Necessary lines: closed identity and source provenance (REQ-1/2), unsampled four-span development coverage (REQ-3), strict runner boundary (A1). Secondary details: serialization field spellings and private boundary fixtures. This noninteractive task produces the approved bounded slice directly rather than waiting for another design confirmation.
