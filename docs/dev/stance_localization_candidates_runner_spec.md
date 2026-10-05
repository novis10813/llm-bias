# Candidate runner changes only development layer coverage

**D1:** `scripts/run_stance_localization_candidates.py` independently registers
`stance_localization_candidates_v2` with execution version `candidate_grouped_pair_v2`.
The [accepted panel](../../llm_bias/core/stance_localization_candidate_panel.py) and
[binding proposal](../concept-cone-steering/localization-candidates-v2/proposal.md)
own the fixed design. No historical runner or result is rewritten. This is a
production entry point with CPU verification, not a completed GPU experiment.

## D2: Three plain routes authenticate the original parent before execution

Required CLI: `--model-slug`, `--model`, `--inputs`, `--parent`, `--output-dir`.
Canonical slug choices are `glm4-9b-0414`, `qwen3.5-4b`, `gemma4-12b-it`, and
`gpt-oss-20b`. Only `--phase primary` is allowed. Optional `--shard-index` and
`--num-shards` divide **sorted panel order**, not physical layer number.
Every shard retains all fit/validation pairs. There is no layer, span, ticker,
budget, dtype, donor, outcome or migration override.

| Model | Authenticated depth | Fixed layers | Global cells |
|---|---:|---|---:|
| GLM | 40 | 0,4,19,20,21,39 | 72,384 |
| Qwen | 32 | 0,3,14,15,16,31 | 72,384 |
| Gemma | 48 | 0,5,23,26,27,28,47 | 84,448 |
| GPT (unsupported execution) | 24 | 0,2,7,8,9,11,14,23 | 96,512 |

GLM requires native `glm4` config and original parent requested `bfloat16`.
Qwen/Gemma reuse accepted plain-crossmodel config and HF-wrapper checks, including
Gemma unified outer/text family matching. All routes require declared native BF16,
exact configured depth, matching loaded HF text family/depth, and a distinct ordered
wrapper layer inventory. Construct the authenticated panel only after these checks.
CUDA is mandatory, with all parameters CUDA and floating parameters BF16. Preserve
the parent's exact requested dtype mode, head/embedding dtype and attention policy.

Reuse the accepted exact resolved checkpoint path plus complete model metadata and
backend comparison. A name containing GLM is not identity evidence. Keep the full
current core/jlens source inventory and add actual runner hashes, without replacing
old inventories. Parent inputs/pairs remain validated by accepted loaders.
The complete table has fit 2,416, validation 600, calibration 208, evaluation 800.
All original 2,012 baseline keys are compiled/bound, not only selected execution keys.
Plain generation remains 512 tokens, 180 seconds, cache enabled, plain JSON, no margin.

## D3: Gates precede every effect and failures remain immutable

The descriptor binds the full input/table/parent hashes, actual layer inventory,
canonical panel record/hash, full global cell count, selected panel-order shard,
all four post-block/full arms, generation/wrapper/grammar/runtime provenance and
issuer mapping. Cell keys independently name V2 and the panel hash. Registration
and records use the accepted write-once locked store. Resume requires exact current
bindings, revalidates every record and does not retry failed interventions or clean
halts. V1 directories and foreign records are rejected. No historical migration.

Every selected layer × entity/evidence1/evidence2/instruction gets a real NO1 gate
on the first fixed fit target key. Require complete parent token/text equality for
baseline, repeat, dose-zero and self-replacement before any grouped effects.
A failed gate is retained and blocks resume. The accepted pair-grouped executor
captures the donor union transiently, generates one clean target, then executes
all pending coordinates with fresh trackers. Full clean donor/target drift produces
a compact halt record and stops, never fabricated per-cell results. Intervention
runtime failures remain actual executed rows and ITT no-flips. Hooks are scoped,
with no raw activations, residuals, gradients or KV cache output.

The local orchestration uses an explicit panel-aware `_plan`; it does not modify
accepted module globals or construct an all-layer grid and filter afterward.
Alignment and label-independent role-preserving donor pairing remain unchanged.

## D4: Completion describes only the candidate panel

Summary requires every planned record and gate. Missing planned records cannot
produce a completion summary. `candidate_panel_complete` is true only for a complete
unsharded panel, while shards report `complete_shard` and retain global planned counts.
Research eligibility remains false. No global peak/band or outcome selection occurs.
Accepted raw group counts remain visible. Directional groups explicitly include both
source classes, with zero denominators represented by JSON null. Company-first ITT
averages each target company's fixed opposite-clean denominator, including failed
interventions as no flips. Issuer-balanced ITT first averages companies within issuer,
then issuers. Exact pair/generation records remain available for later accepted selection.
No evaluation execution or top3 selection is implemented here.

## R1: GPT is explicitly unsupported, not a fake enabled route

Selecting GPT fails before input/checkpoint loading or store creation. Supporting it
would add merged original/recovery parent loading and row-specific callback plumbing
beyond this bounded first-three-plain dispatch phase. No Harmony flag bypass exists.
A later separately accepted implementation must reuse strict accepted Harmony callbacks
and preflight helpers, original job12 + recovery job20 full 2,012-key/518-recovery/
1,494-original binding, native per-row 1,024/4,096-token budgets, MXFP4/mixed precision
and explicitly recorded Python recovery policy. Known full-native-parent replay drift
must still halt. Payload-only matching, skipped parents, new baselines and relaxed
NO1 gates are prohibited. The CPU GPT panel remains available for planning only.

## A1: Dispatch fresh plain roots only after main's stop/archive handoff

No GPU or scheduler operations are part of this implementation. Main owns verification
of the stopped seven old jobs and SHA-backed archives. There is no dependence on
remote deleted worktrees. Example commands, with actual original checkpoint and parent
paths supplied by the dispatcher:

```bash
PYTHONPATH=$PWD uv run --no-sync python scripts/run_stance_localization_candidates.py \
  --model-slug glm4-9b-0414 --model /original/glm-checkpoint \
  --inputs data/concept-cone-steering/rebuild-v1 \
  --parent /completed/original-glm-baseline \
  --output-dir artifacts/glm4-9b-0414/concept-cone-steering/runs/localization-candidates-v2-new
```

Use the same command for `qwen3.5-4b` or `gemma4-12b-it` with their original checkpoint,
complete parent and fresh model-specific run root. Plain GLM/Qwen/Gemma are dispatch
priority. Optional sharding uses `--shard-index I --num-shards N`, with a distinct root
per shard and `N` no larger than panel size. These paths are placeholders, not verified
installed artifact locations.

## A2: CPU acceptance does not certify parent replay on GPU

```bash
PYTHONPATH=$PWD uv run --no-sync python -m pytest -q \
  tests/test_run_stance_localization_candidates.py tests/test_stance_localization_candidate_panel.py
```

Also run accepted plain/grouped/Harmony execution suites, `uv lock --check`,
`git diff --check`, and verify no `uv.lock` change. Tests use actual fake-model greedy
execution for gates, grouped clean aborts, runtime failures, hook cleanup and exact
resume, plus full-table fixed-cell/shard coverage, source NA/company statistics,
config/inventory/loader mode and explicit GPT rejection. They do not load real weights.
