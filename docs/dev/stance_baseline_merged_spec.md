# GPT effective baseline is complete and intentionally mixed-policy

This slice implements contract item 2. It does not generate outputs, alter source artifacts, certify checkpoint weights, pass no-op gates, or claim research success. The authoritative effective decisions are original1981 primary-valid rows plus exactly31 validated recovery rows, with buy518/sell1494. Source differences never make a row inaccessible and do not require another baseline.

## API

```python
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_merged import (
    MergedBaseline, load_merged_baseline, materialize_merged_baseline,
)
inputs = load_baseline_inputs(inputs_directory)
view = load_merged_baseline(original_directory, recovery_directory, inputs=inputs)
materialize_merged_baseline(view, fresh_output_directory)
```

`load_merged_baseline` returns a frozen `MergedBaseline`, a `CompletedBaseline` subclass. Direct construction, as with LC0, is trusted caller input and is not loader authentication.

- `plan` retains the original complete key/role plan. Its identity describes the original plan, not a homogeneous effective generation policy.
- `rows` is the sorted immutable tuple of all2012 effective `ExecutionRow` records.
- `generation_for(key)` returns a defensive reconstructed full `StructuredGenerationResult`, including the entire Harmony continuation. Unknown or non-`RowKey` keys reject.
- `parent_sha256` and `content_sha256` are the same merged content digest. They differ from the original parent digest and bind both consumed source file inventories, replacement rule, plan, effective records, row sources and policies. Filesystem root names and lock bytes are excluded, so identical copied roots yield identical hashes.
- `metadata` exposes defensive JSON copies with `mixed_generation_policy=True`, `homogeneous_original_policy=False`, original metadata, recovery registration, source hashes and replacement rule. It deliberately has no synthetic homogeneous `bindings` field.
- `summary` preserves the existing baseline row-summary shape, recomputed for the effective rows. Original false gates remain false. This allows LC1's existing structural summary validation without weakening LC0 or inventing passed gates.
- `file_sha256` provides defensive nested original/recovery inventories.
- `row_policy_for(key)` returns a defensive copy of that row's actual recorded policy.
- `row_source_for(key)` returns `{origin, source_sha256, record_sha256}`, where origin is `original` or `recovery`. All sources and policies remain available after source files change.

## Validation and integration

The original directory goes through unmodified `load_completed_baseline` (LC0), including all2012 exact keys, plan, issuer, outcome, strict payload parser, grammar provenance, canonical byte hashes, summary and read-only shared lock. Only the31 registered ticker/condition keys in `RECOVERY_PAIRS` may have `truncated`/`token_budget` originals. Any other original failure rejects. No valid original may be replaced.

Recovery requires the exact complete layout, a nonblocking shared lock, canonical regular nonsymlink files, exact selected keys and record filenames, original parent/plan/input hashes and original continuation hashes. The policy must have a higher token budget while retaining cache, padding and channel settings. Recovery checkpoint metadata bytes must match the original declared model metadata. Grammar/tokenizer/schema/channel bindings must match the original per row, and policy/HF-control hashes must match the recovery policy. Issuer-bound outcomes and strict payload parsing are recomputed. Failed/unresolved recovery rows reject. Recovery summary is recomputed exactly.

For Harmony, the full continuation token IDs replay through `HarmonyBoundaryTracker`. Contract and policy hashes, final boundaries, analysis spans and token counts must match provenance. Successful rows must terminate the complete assistant turn, and saved final text must equal the strict JSON payload followed by native `<|return|>`. No parser repair or partial decision rescue occurs. This validates recorded outcomes without loading a tokenizer or claiming token-byte authenticity beyond LC0's provenance scope.

Baseline row consumers can use this subclass directly, including `build_localization_pairs(inputs, view)`. A consumer needing replay settings must call `row_policy_for(key)`, not treat the original plan policy as effective for all rows. Later runner integration must propagate those per-row settings through clean checks. This slice does not change runner or intervention APIs. Differences in source policy are recorded facts, not an access gate.

## Fresh materialization CLI

```bash
uv run python scripts/merge_stance_baseline_recovery.py \
  --inputs data/concept-cone-steering/rebuild-v1/compiled \
  --original artifacts/gpt-oss-20b/concept-cone-steering/runs/diagnostic-baseline-stance-rb260930-12 \
  --recovery artifacts/gpt-oss-20b/concept-cone-steering/runs/diagnostic-recovery-stance-rb260930-20 \
  --output-dir artifacts/gpt-oss-20b/concept-cone-steering/runs/NEW-MERGED-RUN
```

The output parent must exist. Existing outputs, including empty directories or symlinks, reject. A new directory contains `plan.json`, `metadata.json`, `summary.json`, `row_sources.json`, `row_policies.json`, effective `records/<key-hash>.json`, and `manifest.json` written last with file digests and the content hash. This is explicitly a merged export, not an LC0 homogeneous parent. No materialized-export loader is added in this slice. Consumers load the read-only source view through the API above. Interrupted exports remain visible as incomplete directories and cannot be overwritten.

## Verification scope

Full-cohort synthetic tests validate exact2012 keys,31 replacements,518/1494 counts, failures, policy/grammar/issuer/hash/parser guards, immutable snapshot/defensive lookups, copied-root hash stability and existing-output rejection. A separate actual read-only view check uses the pinned original and recovery bytes. No shared materialization or GPU work is performed.
