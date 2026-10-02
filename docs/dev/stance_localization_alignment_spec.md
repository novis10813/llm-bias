# LC1 makes cross-prompt replacement alignment explicit

**Goal:** Produce auditable source-array/target-position maps for all four prompt spans before any real localization generation. This is an independent pure alignment slice, not a donor sampler, layer selector or GPU runner.

## 1. Existing replacement semantics do not require a hook rewrite

【已驗證: llm_bias/core/prompt_input/decision_prompt.py:DecisionPrompt,DecisionSpan】 spans have half-open token ranges indexing actual wrapped inference IDs. 【已驗證: llm_bias/core/inference/stance_transient_capture.py:capture_prompt_residual】 captures selected donor positions into `[1,n,d]` in the supplied order. 【已驗證: llm_bias/core/inference/stance_interventions.py:scoped_residual_intervention】 `source_positions` labels the source tensor rows with **target** absolute positions; it is not required to contain the original donor coordinates. The callback maps those labels to source-array indices. Thus callers can explicitly index captured donor rows into target order, pass the resulting `[1,n_target,d]` tensor and `source_positions=target_positions`. No relaxation of the accepted hook is needed. The earlier exploratory report interpreted these labels as donor coordinates; do not implement that interpretation.

## 2. Two mapping policies have distinct estimands

Create `llm_bias/core/stance_localization_alignment.py` and `tests/test_stance_localization_alignment.py` only. No tensors/GPU/tokenizer downloads; do not modify hooks, renderer, parent loader, baseline artifacts, historical code or dependencies. Inputs are trusted `DecisionPrompt` records produced by the renderer; mapping validation does not authenticate tokenizer execution.

**REQ-LC1-1 【新設計】** API:

```python
build_span_alignment(donor: DecisionPrompt, target: DecisionPrompt, *,
                     span: str, policy: str, selector: str = 'full') -> SpanAlignment
```

Allowed `span`: entity/evidence1/evidence2/instruction. Allowed `policy`: exact_tokens/relative_rank/tail_overlap. Allowed `selector`: full/first/middle/last/tail4. Unknown strings or non-prompt inputs reject. `SpanAlignment` is a frozen tensor-free record with `span`, `policy`, `selector`, `donor_capture_positions: tuple[int,...]`, `target_positions: tuple[int,...]`, `source_indices: tuple[int,...]`, `donor_span_token_count`, `target_span_token_count`, `mapping_sha256`; defensive `to_dict()` exports lists and counts. A public constructor validates genuine nonbool ints, positive lengths, nonempty unique increasing capture/target positions, source_indices length equal target count and values inside capture count, and hash self-consistency. Construction validates structure, not renderer origin. No implicit defaults for mapping policy.

**REQ-LC1-2 【新設計】** For both selected spans, verify genuine integer half-open token bounds `0<=start<end<=len(inference_token_ids)`, stored token tuple equals the actual inference slice, and `token_sha256==sha256_json(list(ids))`. All prompt IDs must be genuine signed-int64 nonnegative integers. Verify the span role is the requested role. Reject malformed ranges, empty spans, wrong role/ID/hash, booleans and nonfinite metadata; never silently skip a target because lengths differ. Char/text checks are renderer responsibility, not new retokenization.

**REQ-LC1-3 【新設計】** Mapping policies:

- `exact_tokens`: require donor and target span ID tuples equal. Map ordinal token indices one-to-one, even when absolute positions differ. This is the primary shared-instruction/shared-evidence mapping, not a cross-tokenizer assertion. Different token IDs reject rather than fall back.
- `relative_rank`: map each target ordinal `j` in a target span of length `T` to donor ordinal `floor((2*j+1)*D/(2*T))`, capped at `D-1`, with integers only. This replaces **every** selected target token; donor rows may repeat. It is relative token-position transport, not semantic matching, and must be named explicitly in outputs. Equal lengths reduce to ordinal mapping.
- `tail_overlap`: let `k=min(D,T)` and pair donor ordinals `D-k..D-1` with target ordinals `T-k..T-1`. Only the overlap tail is replaced. It is an alignment sensitivity arm with different token coverage, never mislabeled as full-span replacement when lengths differ.

Selections operate on the policy's available target ordinal list, after mapping: full takes all; first/middle/last selects one (middle index `len//2`, right center for even lengths); tail4 takes the last `min(4,len)` positions. No zero padding, averaging/interpolation, nearest-ID matching, source answer-prefix capture or token-level semantic claim.

**REQ-LC1-4 【新設計】** After selector application, capture only unique mapped donor absolute positions, sorted increasingly. `source_indices` maps each selected target position to the index of that donor position in `donor_capture_positions`; indices may repeat. Capture once, gather `source[:, source_indices, :]` transiently, and pass target labels to the unchanged residual hook in a later executor slice. Both source-array bounds and target prompt bounds are separately validated; mapping coordinates are never confused.

**REQ-LC1-5 【新設計】** `mapping_sha256` hashes the exact exported record excluding that hash field with `sha256_json`. Counts refer to full original spans, not selected/captured counts. Preserve `donor_span_token_count` and `target_span_token_count`; selected count is derived as `len(target_positions)` (no extra export field), so full-span vs partial/token arms can be distinguished. `to_dict` contains only these explicit fields; no raw prompt text, activation or tensor. Absolute positions and index lists are compact intervention provenance and allowed.

Example 【新設計】: donor entity span absolute positions10–12 (D=3), target entity positions20–24 (T=5), relative_rank/full yields donor_capture_positions=(10,11,12), target_positions=(20,21,22,23,24), source_indices=(0,0,1,2,2). Tail_overlap/full yields donor=(10,11,12), target=(22,23,24), indices=(0,1,2). Shared instruction with identical IDs at donor100–103 vs target105–108 maps ordinally under exact_tokens/full.

## 3. LC1 red-to-green acceptance is independent of the running regression

Synthetic topology: direct tiny `DecisionPrompt`/`DecisionSpan` fixtures with whole-prompt ID tuples and verified span slices; these are unit fixtures, not a smaller research cohort. Red tests absent API and wrong ordinal mapping. Green tests all policies/selectors, singleton/excessive-length ratios, shifted exact tokens, D/T up to bounded exhaustive panel, formula monotonicity/source coverage, donor reuse, constructor malformed indices/hash/types, mutation defense and deterministic hashes. Verify exact_tokens mismatch has no fallback and tail_overlap reports reduced target coverage. No clone of hook formulas as the only evidence: use example maps and an independent test gather into a small tensor passed through the existing fake-model replacement hook; confirm intended target edits only and cleanup. If this integration needs existing fake fixtures, read their concrete signatures first.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q tests/test_stance_localization_alignment.py tests/test_stance_interventions.py tests/test_stance_transient_capture.py
uv lock --check
git diff --check
```

Independent read-only preflight and review precede acceptance. This spec does not freeze donor pairing, selected layers, no-op coverage or effect thresholds. Those choices belong in the localization research proposal and execution contract before GPU dispatch.
