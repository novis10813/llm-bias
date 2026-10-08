# Native Harmony grouped execution binds each arm's parent policy

## Scope and API

`llm_bias/core/inference/stance_localization_harmony_grouped.py` exposes
`execute_harmony_grouped_prompt_replacement(model, tokenizer, donor_prompt,
target_prompt, capability, *, donor_policy, target_policy, cells,
expected_donor, expected_target, verified_binding=None)`.

The input is one original donor/target `DecisionPrompt` pair and a nonempty
immutable tuple of existing `ReplacementCell` coordinates. Policies are explicit
`StructuredGenerationPolicy` records, both `harmony_no_tools`. Each parent binds
its own budget, timeout, cache setting and remaining controls. For example, a
1024-token donor and 4096-token target are valid, as is the reverse. The target
policy governs both clean and patched targets. No pair-wide fallback policy or
recovery/resampling occurs here.

Return type is the existing frozen `GroupedReplacementExecution`, containing
shared actual clean results and one frozen `PromptReplacementExecution` per
attempted cell. LC2 record integrity validation permits separate arm policies:
it validates each generation result and full-output match independently, without
requiring donor and target provenance policies to equal. No validator bypass or
new record format is needed.

## Binding and execution

Before any generation, validate cell coordinates, uniqueness, recomputed span
alignments, prompt schema/template/wrapper records, embedding/head token bounds,
exact native prompt suffixes, tokenizer identity and model binding. Require a
genuine factory-owned compiled Harmony capability through
`_check_harmony_capability`. Call actual LC2 `_bind_expected` separately for each
parent and its policy, binding schema bytes, tokenizer, head, stop IDs, policy,
HF controls and policy hash. Additionally require each expected parent's native
channel contract, contract hash and channel-policy hash to match the capability.

Capture the union of required donor positions once per `(layer, hook_site)` during
one original-prompt donor generation. Each arm has a fresh position tracker with
its own policy's `use_cache`. Capture holders remain alive but inert after the
original donor tracker closes. Generate clean target once. Compare both clean
parents' complete generated IDs, text, payload, decision, reason, validity flags,
finish reason and failure/error fields before any patch. LC2 equality deliberately
excludes elapsed time and provenance, which are bound separately in preflight.

For each cell, map its donor capture order into the sorted site union, then expand
its accepted `source_indices` into target order. Generate a fresh target under the
target policy with the unchanged prompt-only, unit-dose scoped replacement hook.
No target cell consumes prior patched activations or KV state. Native analysis,
headers, final JSON and stop remain in the full continuation. Only the accepted
Harmony driver constrains the final body. No forced prefix/EOS, regex parsing,
fabricated control or raw activation serialization is added.

## Outcomes and lifetime

`donor_failed`, `donor_mismatch`, `source_unavailable`, `target_failed`, and
`target_mismatch` halt the entire group with actual available clean results and
an empty execution tuple. A driver-returned intervention failure remains a genuine
`executed` cell with actual failure output and diagnostics. Later cells continue
without retry. Caller/helper errors propagate rather than fabricate outcomes.

Exit stacks remove capture, intervention and tracker hooks on all exits. Capture
holders release source tensors, and executor source/gather references are cleared
in `finally`. No tensor is stored in returned records. Cell diagnostics are the
same compact LC2 reductions, with the same selected-token bound validation.

## Acceptance and next phase

CPU tests use actual factory compilation, native greedy driver calls, torch hooks
and transient capture. Mixed 1024/4096 policies with distinct timeouts and opposing
cache settings match independent three-arm execution, including complete
provenance and diagnostics after excluding only elapsed seconds. Equal-policy
execution also matches the existing LC2 executor. A multi-span/site panel requires
`2 + N` generations instead of `3N`. Tests cover genuine patched decisions,
parent token/text drift, unavailable capture, clean failures, failed intervention
continuation, configuration mismatches, union capture observations and exception
cleanup.

Run:

```bash
PYTHONPATH=$PWD uv run --no-sync pytest -q tests/test_stance_localization_harmony_grouped.py
```

This slice does not implement cohort selection, artifact loading/writing, gates,
GPU submission or a full runner. The GPT effective-2012 cohort (original 1981 plus
31 recovered parents) and per-row policy selection belong to the downstream
runner phase. No cohort counts, checkpoint results or GPU acceptance are claimed
by these CPU tests. Existing GLM execution modules and protocols remain unchanged.
