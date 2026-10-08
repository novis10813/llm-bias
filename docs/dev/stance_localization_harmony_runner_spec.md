# Native Harmony runner replays the effective mixed-policy parent

## Scope

`scripts/run_stance_localization_harmony.py` is a separate primary runner based
on `8e4d9e6`. It does not modify GLM, the accepted grouped executor, LC2 records,
pairing, alignment, generation, or gate contracts. No GPU execution or research
acceptance is claimed by this implementation.

The public entry requires `--parent` (original12) and `--recovery` (recovery20).
It calls strict `load_merged_baseline(original, recovery, inputs=inputs)` directly,
not a materialized export or a homogeneous parent loader. That boundary verifies
all 2012 original records and exactly 31 recovery records, preserves the 1981
valid originals, and supplies authoritative effective decisions of 518 buy and
1494 sell. The original plan is retained, but its original policy is not used
as a common effective policy.

All 4024 role-isolated pairs are built by the accepted pair builder. Only the
3016 fit/validation pairs execute. Calibration/evaluation remain in the full
pair-table identity. The four full spans are entity, evidence1, evidence2, and
instruction. The accepted exact-token/relative-rank alignment rule, prompt-only
unit-dose replacement, and post hook site are unchanged. No public cohort,
layer, span, token budget, timeout, or prior-result override exists.

## Preflight and native loading

Checkpoint metadata-file hashes must equal the original parent's hashes. The
strict merged loader separately verifies recovery checkpoint metadata against
that original. Runtime package/backend controls match the original parent and, where recorded,
the recovery backend. Python alone selects the explicitly approved
`recovery_registration.metadata.backend.python`, not the original Python or an
arbitrary version. The recorded original12/recovery20 pair has Python 3.13.15 /
3.13.14, so the current runner must use 3.13.14. Missing recovery Python or a
current/recovery mismatch rejects before model loading. Torch, Transformers,
xgrammar, jlens, CUDA, cuDNN, kernel policy and determinism remain strict.
Recovery-recorded native dtype, embedding/head dtype, cache and attention fields
must also match the original where available. Actual dtypes and attention remain
authenticated after loading.
Paths may differ only through the accepted `bind_relocated_tokenizer` helper:
the actual loaded checkpoint path stays in runtime metadata while the verified
original logical tokenizer name is restored for tokenizer identity binding.
The identity scope remains metadata-only, not full checkpoint weight hashing.

`config.json` must declare GPT-OSS, a positive text-layer count, and native MXFP4
quantization. Floating dtype declarations are optional. If neither `dtype` nor
`torch_dtype` is present, authentication records `declared_dtype: null` and
`declared_dtype_source: "absent"`. A present declaration must be a nonempty,
non-whitespace string, including declarations shadowed by a higher-priority key.
The selected declaration records its source (`config.dtype`, `config.torch_dtype`,
or the corresponding `text_config` key). No dtype is inferred from the model
family, parent, or Transformers defaults, and no checkpoint metadata is changed.
Layer count is not a model-slug heuristic or a fixed CLI default. The actual HF config must match that depth and family. The wrapper must
expose every distinct layer, in the same order and by object identity as
`hf_model.model.layers`. All configured layers are included. A checkpoint with
24 layers plans 289536 cells globally and 96 layer/span gates. Shards use
`layer % num_shards == shard_index` and form a disjoint complete layer partition.

CUDA is required before loading. The loader receives `dtype='native'` and
`device_map=None`, preserving packed MXFP4 expert storage and checkpoint floating
dtypes. Actual input/output embeddings, returned device and all parameters must
be on CUDA. Input/output dtypes and attention implementation must match the
parent. There is no CPU fallback/offload and no assumption that every parameter
is BF16.

The factory compiles the native Harmony capability from actual tokenizer/head
and parent stops. Every effective row binds its actual source policy, schema,
tokenizer, head, stop IDs, HF controls, policy hash, channel contract, contract
hash, and channel-policy hash before a store opens. Original-source rows use
1024 tokens/300 seconds, recovery-source rows use 4096 tokens/1200 seconds.
All 31 retries remain accessible. Policies are supplied explicitly per key,
never by changing global defaults or rejecting mixed-source pairs.

## Gates and execution

The fixed first fit target key must coincide with the accepted grouped
runner's first development target key. The approved input pack satisfies this
condition. Each owned layer/span receives a genuine four-arm NO1 gate under
that key's effective policy: fresh baseline, repeat, nonzero-direction
addition at dose zero, and captured self-replacement. All gates must pass
full-parent output matching before any replacement group executes.

Runner callbacks supply each pair's effective `donor_policy` and `target_policy`
to `execute_harmony_grouped_prompt_replacement`. This includes both
original-to-recovery and recovery-to-original combinations. The accepted
executor uses fresh trackers, captures only the original donor prompt, verifies
both actual clean parents against full IDs/text and remaining output fields,
and generates each patched target with its target policy. No forced prefix,
removed analysis, parser rescue, reused patched donor state, or policy fallback
is introduced. See [the executor contract](stance_localization_harmony_grouped_spec.md).

The accepted grouped scheduler/store is reused with callbacks and no historical
snapshot. Genuine failed interventions remain executed LC2 cells with their
actual partial output and diagnostics, and later cells continue. A clean-parent
failure, mismatch, or unavailable source writes one closed group-halt diagnostic
with only actual available clean results and planned keys. It never fabricates
aborted cell outputs. Halted runs and failed gates remain incomplete and cannot
silently retry on resume.

## Store identity and resume

Registration binds the unchanged grouped logical grid and execution version,
merged content SHA256, original/recovery metadata and file inventories, complete
original plan, and all 2012 per-row source/policy records. Runtime source hashes
include all current core and jlens Python files, baseline/smoke scripts,
`uv.lock`, this runner, grouped/logical runners and the accepted recovery helper.
Actual checkpoint path, relocation binding, native layer authentication and GPU
metadata remain in the configuration identity. Runtime also binds
`mixed_runtime_python` with `selected_backend_mode: recorded_recovery` and
`current`, `original`, `recovery` versions. Actual backend Python is retained,
not dropped from the configuration hash. Original metadata, recovery registration
and all row source hashes remain unchanged, so this does not relabel the 1981
original rows as recovery-runtime generations.

The immutable grouped store uses persistent exclusive locking, write-once
publication and strict canonical import validation. An identical complete resume
does no generation. Changed configuration/provenance, foreign/staging files,
missing gates/cells in a summarized run and closed halts reject before further
model effects. No prior localization import is supported. Only compact LC2
records, scalar diagnostics, gates, registration, halts and summary are written.
No raw activations, residuals, gradients, KV cache or tensor files are saved.

## Verification and proposed GPU invocation

CPU tests use the actual factory Harmony grammar, native greedy driver, transient
capture and torch intervention hooks. They exercise both mixed-budget directions,
all effective policy/capability bindings, actual gate policy, grouped callback
composition, failed interventions, clean aborts, cleanup, immutable resume,
corruption rejection, native-only loader preflight and the full actual-layer grid.
Full-shaped fake parent tests include 2012 rows, 31 recovery sources and both
policies. The full recorded GPT-OSS-20B config fixture is copied byte-for-byte
from `https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json`. Its SHA256
is `3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce`, matching
original12's `shard-0-of-1.execution_metadata.json` config hash. It has 24 layers,
hidden size 2880, native MXFP4 quantization, and no dtype declaration. CPU-only
public-run tests pass this config through native loading to the actual
embedding/head dtype checks, and reject wrong quantization, depth, or either
actual dtype. These are not checkpoint weight or GPU acceptance tests.

The compact recorded runtime fixture
`tests/fixtures/gpt_oss_20b/merged_runtime_metadata.json` preserves the real
`original_metadata.bindings` / `recovery_registration.metadata` nesting from the
read-only `effective-baseline-original12-recovery20-v1/metadata.json` artifact.
Tests require exact recovery Python, reject missing or mismatched versions and
non-Python backend changes, and capture public-run descriptor binding without
GPU execution or fabricated generation. Existing genuine CPU Harmony callback
tests continue to exercise full-row replay and channel contracts.

This correction does not rewrite failed GPU jobs36/37 or any completed run. It does
not add preflight artifacts or change the cohort, row policies, native Harmony
analysis, gates, or full grid.

```bash
PYTHONPATH=$PWD uv run --no-sync pytest -q tests/test_run_stance_localization_harmony.py
uv lock --check
```

Proposed GPU command, not executed here. Set `GPT_CHECKPOINT` to the actual local
metadata-identical GPT-OSS-20B checkpoint. Use a fresh run ID. Read-only links to
original input artifacts are permitted, but output must not target a historical
run. The following writes only a new shard directory:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD uv run --no-sync python scripts/run_stance_localization_harmony.py \
  --model "$GPT_CHECKPOINT" \
  --inputs data/concept-cone-steering/rebuild-v1/compiled \
  --parent artifacts/gpt-oss-20b/concept-cone-steering/runs/diagnostic-baseline-stance-rb260930-12 \
  --recovery artifacts/gpt-oss-20b/concept-cone-steering/runs/diagnostic-recovery-stance-rb260930-20 \
  --num-shards 4 --shard-index 0 \
  --output-dir artifacts/gpt-oss-20b/concept-cone-steering/runs/NEW-HARMONY-LOCALIZATION/shard-0-of-4
```

Repeat with shard indices 1, 2, 3 and distinct output directories to cover the
remaining layers. There is no cross-shard aggregation or research certification
in this runner slice.
