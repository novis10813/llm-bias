# Plain cross-model localization changes execution, not research design

This separate runner prospectively executes the accepted grouped primary grid for
Qwen3.5-4B and Gemma4-12B-IT. It does not modify the GLM runners, historical
artifacts, logical protocol or eligibility rules. Acceptance is CPU engineering
verification only. No GPU run, checkpoint gate success or scientific result is
claimed by this document.

## D1. Full inputs and original parent policies remain mandatory

The public entry loads the approved full input bundle and the read-only completed
single-shard baseline through the accepted parent loader. The unchanged pair
builder supplies 4024 pairs, with 3016 fit/validation development pairs executed.
Its seed 20261002, role-preserving issuer ring, reciprocal evidence pairs and
label-independent donor selection are unchanged. Calibration/evaluation remain
in the authenticated full table, not in the primary execution partition.

The parent must retain 512 new tokens, 180 seconds, cache enabled and plain JSON.
Checkpoint resolved path and metadata-file hashes must equal the parent model
binding. Backend Python, torch, transformers, xgrammar, jlens, CUDA, cuDNN,
kernel and deterministic policies must match. The parent requested mode must be
`bfloat16` or `native` and is preserved exactly in loading and runtime bindings.
`bfloat16` loads with `torch.bfloat16`; `native` omits the HF dtype override.
Both require a native BF16 checkpoint declaration, BF16 embedding/head and all
floating model parameters on CUDA in BF16. Grammar,
tokenizer, schema, stop IDs, wrapper, attention policy and genuine full generated
output comparisons use the existing accepted checks. No relocation alias,
recovery-budget substitution, CPU fallback or dtype override is provided.

## D2. Config and native wrapper determine the complete layer grid

`scripts/run_stance_localization_plain_crossmodel.py` reads `config.json` text
configuration, accepting Qwen3.5/Gemma4 model types and native BF16 declarations.
The recorded Gemma unified route requires outer `gemma4_unified` and text
`gemma4_unified_text`, with matching loaded outer/text identities. Its 48-layer
text config can inherit the outer BF16 declaration.
Qwen3.5-4B must declare 32 layers. Gemma depth is read from its actual config,
never a 40-layer default. The config hash must agree with runtime checkpoint
metadata. After native loading, HF text-config family/depth and the wrapper's
complete distinct layer inventory must agree with the checkpoint record.

All actual layers use post hooks and full entity/evidence1/evidence2/instruction
spans. The only partition option is the accepted layer-modulo shard index/count.
There are no cohort, ticker, layer-list, span, selector or generation-policy CLI
options. The unsharded Qwen grid has 386048 planned cells. Gemma has
`3016 * actual_layer_count * 4` cells. Same-label and invalid-parent pairs remain
planned, with original per-role/family/contrast/span/layer denominators.

## D3. Explicit composition preserves grouped execution and immutable storage

The new entry calls the unchanged `grouped.descriptor` and `grouped.execute_run`
with explicit model-bound gate/group callbacks and no prior snapshot. This
reuses the accepted grid, store, import validators, true repeat/zero/self gates,
transient capture executor and halt handling. It does not call the GLM runtime
entry or monkeypatch a global to bypass its 40-layer guard.

Bindings add the checkpoint's model-layer authentication record and exact source
hashes for this runner, grouped/logical runners and recovery store, alongside the
runtime inventory of core/jlens sources and lockfile. Logical-grid identity stays
separate from runtime/model/code identity. Fresh roots are required for a new
model/config; exact-registration same-run resume remains supported. Existing
GLM/prior records cannot be imported, and `--prior-run` is rejected.

Intervention failures remain executed outcomes. Failed no-op gates or clean-arm
drift stop execution without a completion summary. Every planned cell and gate
must be present and revalidated before `complete_shard=true`. Global completion
requires a single shard, with no implicit shard aggregation. All summaries and
failure exports retain `research_eligible=false`. No raw activation, residual,
gradient or KV-cache artifact is written.

## A1. Verify CPU contracts before prospective GPU execution

Focused fake-model tests check config/native-wrapper depth authentication,
closed CLI, all-layer planning, original parent policy, exact runtime bindings,
source inventories, exact requested-mode loader invocation and runtime recording,
actual grouped fake generations,
failed-gate suppression, failure retention and same-run resume. The accepted
grouped tests additionally exercise immutable record corruption, halt handling,
cleanup and grouped equivalence. Harmony fixture cases are intentionally skipped
by the plain-only fake grid. Regression tests read the Qwen16/Gemma17 recorded
parent bindings when installed (both request `bfloat16`) through pure parent
validation and check a synthetic unified Gemma config. Tests do not load weights
or use a GPU.

```bash
PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync pytest -q tests/test_run_stance_localization_plain_crossmodel.py \
  tests/test_run_stance_localization_grouped.py tests/test_stance_localization_grouped.py
uv lock --check
git diff --check
```

## A2. Dispatch only with a matching completed model-specific baseline

A concrete invocation shape is below. Replace the three site-specific paths
with existing approved inputs, native checkpoint and its complete original-policy
baseline. Choose a never-used output root. This command is prospective, not a
record of execution.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD uv run --no-sync python \
  scripts/run_stance_localization_plain_crossmodel.py \
  --model /models/Qwen3.5-4B \
  --inputs /inputs/stance-approved \
  --parent /artifacts/qwen3.5-4b/baseline-original512-bfloat16 \
  --output-dir /artifacts/qwen3.5-4b/localization-plain-crossmodel-new \
  --phase primary --shard-index 0 --num-shards 1
```

For Gemma use its native checkpoint and corresponding parent instead. Single-GPU
VRAM capacity, installed native wrapper compatibility, and actual checkpoint
gates/cells require separate GPU verification. Unsupported config/wrapper or
parent-policy drift fails closed. Position/alignment sensitivity, scientific
aggregation, site selection, eligibility promotion and historical migration are
outside this implementation.
