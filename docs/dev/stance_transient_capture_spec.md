# Self-replacement needs a transient source from baseline generation

**Goal:** Add one observational capture context for the unperturbed baseline's first full-prompt forward. Its source stays in memory for immediate prompt-only self-replacement. Initial read-only preflight required explicit observation-state and lifetime amendments. Final read-only feasibility preflight passed with133 existing tracker/cleanup tests and CPU hook-order/failure/replay probes; no further amendments were required. This specification does not implement a runner or certify a model.

## 1. Existing tracking and replacement constrain capture

【已驗證: `llm_bias/core/inference/stance_interventions.py:GenerationPositionTracker.track`】 A fresh tracker installs root pre/post hooks, exposes active absolute positions during root execution and clears them afterward. Trackers cannot be reused across generations. Explicit positions support chunked input; implicit initial cached input requires the full prompt. Later no-cache forwards revisit prompt positions.

【已驗證: `llm_bias/core/inference/stance_interventions.py:scoped_residual_intervention`】 Replacement requires prompt_only, unit dose, finite floating source `[1,n_source,d_model]`, matching target dtype/device and exact source-position mapping. Equal values return the original tensor. It does not replace decode tokens.

【已驗證: `llm_bias/core/inference/interventions.py`】 Pre hooks observe `_block_hidden(args,kwargs)`; mid hooks observe the input to `_post_attention_norm(model,layer)`; post hooks extract a tensor output or `output[0]`. Existing post intervention hooks rebuild the block output, so this capture must use its own observational hooks returning None. `record_block_states` executes a separate forward and is unsuitable for this source.

## 2. Capture has a narrow lifetime and fail-closed state

All contracts below are 【新設計】. Create `llm_bias/core/inference/stance_transient_capture.py` and `tests/test_stance_transient_capture.py` only. Do not edit accepted trackers/hooks/adapters, save tensors, load checkpoints, implement a four-arm loop, or certify eligibility.

**REQ-TC.1 — Explicit scope.** `capture_prompt_residual(model, *, tracker: GenerationPositionTracker, layer: int, hook_site: str, prompt_positions=None)` is a context manager yielding `TransientPromptCapture`. Require active tracker.track on the same resolved root, genuine nonbool in-range layer, site pre/mid/post, and nonempty unique genuine integer prompt positions in range. None selects all prompt positions. Preserve caller position order in the source tensor. Reject unsupported hook targets rather than guessing architecture.

**REQ-TC.2 — Capture the first full root forward only.** Install a root prehook after the tracker's prehook to validate that the first observed forward has active positions exactly `tuple(range(tracker.prompt_length))`. Partial/chunked first prefill is unsupported in this slice, even if the tracker itself supports it; no accumulation or separate forward fallback. Install a root always-call posthook to clear the capture's local forward flag. At the selected layer/site, the first forward must invoke the observation hook exactly once. Require floating finite rank3 batch-one values with sequence length matching tracked positions and positive hidden width. Copy selected values with detach().clone() on the original device/dtype; never cast or change the observed output. A second selected-site invocation in the same root forward invalidates capture and raises ValueError. Count observations explicitly. The state starts unobserved, becomes captured only after successful site extraction, and becomes invalid on an observation error or count mismatch. The root always-call posthook checks the first-forward count is exactly one and the state is captured before clearing the in-forward flag. Missing invocation invalidates capture even when a driver catches the callback exception. The first root attempt must also become invalid if the earlier tracker prehook raises before the capture prehook runs: its always-call root posthook cannot return merely because its local in-forward flag was never set. Also require the first root output to be non-None: PyTorch always-call hooks receive None on a failed root forward, so drop a source captured at pre/mid if the root fails later. A legitimate root returning None is unsupported for this slice, rather than ambiguously accepted.

Later root forwards, including no-cache full-prefix recomputation, must not overwrite the captured source. Ignore their target-site observations while still using tracker-backed sequence-shape validation. The first forward is the authority; do not deduplicate or recapture by seeing prompt positions again.

**REQ-TC.3 — Error state survives driver exception handling.** Invalid root metadata, capture failure, or missing/duplicate observation records an invalid state and drops any source before raising. Because generation drivers may catch callback exceptions and return failed results, consumers must explicitly call `require_source()` before using the source. This method requires state=captured, not in-forward and not released; it raises ValueError for incomplete, invalid, released, or still-in-forward state. A valid source does not make a failed baseline eligible; the later consumer separately validates its generation result. Do not interpret source availability as generation success.

**REQ-TC.4 — Own handles and references.** Holder read-only properties expose layer, hook_site, prompt_length, positions and ready; `require_source() -> torch.Tensor` returns the in-memory source for immediate replay. The holder stores no logits, gradients or KV cache and has no to_dict/to_json/save API; repr must not include tensor values. Context exit removes all its root/site hook handles on normal exit, body/forward errors or partial registration failure, drops its source reference and marks it released. Tracker lifecycle remains the outer caller's responsibility. The holder cannot return a source after context exit. Callers needing a reference beyond exit must extract it inside the valid capture lifetime; caller-retained tensor references remain caller responsibility and must not be serialized. This component cannot prove arbitrary external hooks are absent; unperturbed-baseline orchestration is a later consumer contract.

Example 【新設計】: two ExitStacks allow the baseline tracker to enter before capture yet exit before replay. This is necessary because an ordinary nested `with tracker.track(): with capture():` would release capture before the tracker exits. The source must be consumed while the outer capture stack remains open:

```python
# New consumer example; not a runner implementation.
with ExitStack() as baseline_tracking, ExitStack() as capture_lifetime:
    baseline_tracking.enter_context(baseline_tracker.track(model))
    holder = capture_lifetime.enter_context(capture_prompt_residual(
        model, tracker=baseline_tracker, layer=layer, hook_site=site,
        prompt_positions=positions))
    baseline_result = generate_baseline()  # caller's actual unperturbed driver
    baseline_tracking.close()
    source = holder.require_source()
    # Validate baseline_result separately before accepting any no-op result.
    with replay_tracker.track(model):
        with scoped_residual_intervention(
            model, tracker=replay_tracker, layer=layer, hook_site=site,
            scope='prompt_only', operation='replacement', source=source,
            source_positions=holder.positions, prompt_positions=holder.positions):
            replay_result = generate_baseline()
    del source
# holder.require_source() must now raise: released.
```

`generate_baseline` above denotes the caller's driver call, not a proposed API in this slice. The observation hooks remain installed during replay but must return None immediately when the original tracker is no longer tracking the bound root. Do not validate, overwrite or invalidate on those other invocations. Root-post clearing must use capture-owned flags, because the earlier registered tracker posthook clears active_positions first. Capture entry on an inactive or different-root tracker must reject.

## 3. One slice proves observational capture and cleanup

Consume the verified root/site helpers and tracker fields; reuse their supported tensor structures without introducing model-slug logic. Produce the context and holder only. Fixture: batch-one tiny root with two decoder blocks and hookable post_attention_layernorm, based on `tests/test_stance_interventions.py:Root/Block`; exercise tensor and tuple/list post outputs.

Red: missing module or failing capture assertions. Green tests cover all three sites, ordered subset `[2,0]`, complete source mapping and exact baseline/self outputs with zero changed-token diagnostics; cached decode and no-cache recomputation leave source unchanged. Prove detach/clone has no storage alias and observation preserves original output object identity. Reject chunked first prefill, wrong root/inactive tracker, malformed positions, nonfinite/nonfloating/shape errors, repeated site invocation and skipped selected layer. Verify require_source rejects while active, incomplete/invalid and after release, including exceptions caught by a fake generation loop and root failure after a successful pre/mid capture. A first root output of None must invalidate capture. Test second hook registration failure, unrelated hooks preserved, and no owned hook leaks after errors. Replay with a fresh tracker while capture context remains open must pass without rereading the original source.

Run `uv run pytest -q tests/test_stance_transient_capture.py tests/test_stance_interventions.py tests/test_stance_hook_cleanup.py`, then `uv lock --check`, `git diff --check` and thread-bounded full pytest. Independent preflight precedes dispatch; main verification and independent acceptance precede commit. No real checkpoint or research claim follows from these tests.

## Audit skeleton

Claim: self-replacement can consume the original baseline prompt residual without a separate forward or persistence. Support: exact first-forward/site mapping, observational detached copy, fail-closed state and scoped cleanup. Deferred: concrete model policies, four-arm orchestration, configuration coverage, sharded durability and actual no-op certification.
