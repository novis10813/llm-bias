# The GPU runner executes the complete fixed exploration without accepting an operator

This runner implements the next phase of [the accepted CPU plan](stance_cone_validation_plan_spec.md) on base `4a13a19`. The accepted compiler and core execution modules are unchanged. It does not train checkpoint weights, compute validation losses, construct teachers from validation, choose a best seed or point, or confer research eligibility.

## D1. Public execution authenticates the full approved population before allocation

`scripts/run_stance_cone_validation.py` requires `--inputs`, `--parent`, `--model`, `--training-dir`, `--training-audit` and `--output-dir`. Optional `--shard-index` and `--num-shards` assign direction indices modulo shard count. Counts must be between 1 and 48. Each assigned direction retains all nine signed doses and all 300 authoritative validation keys. No ticker, layer, seed, ray, dose, shortened-panel or best-point CLI exists.

The unchanged compiler authenticates the pinned `42c5300a224e1b8a4ab2f5508173f3749249d4932679b10c3af5d3dfd362220b` audit and all six cones. Its teacher compiler provenance discrepancy remains in the panel. The strict parent loader reconstructs the complete 2012-input baseline. Audited and actual parent hashes must match. The compiler fixes 75 validation tickers, 300 inputs, 48 directions and 129600 logical cells, including 115200 nonzero and 14400 zero cells.

Before GPU allocation, the Linux write-once store publishes the full preflight registration and `records/plan.json`, which contains the panel, coverage and full prospective cells. Registration binds current hash-verified source inventory, metadata-only checkpoint identity, original path, parent, approved inputs, roles, plan/panel IDs and shard assignment. Native or BF16 requested mode and parent backend versions/kernel controls must match. Historical parent source hashes are provenance, not a claim of equivalence to current runner code.

Actual loading uses the original GLM4-9B-0414 checkpoint path, native/BF16 parent mode, one CUDA device, 40 layers, H4096 embedding/head, BF16 embedding/head and the original attention implementation. All weights are frozen and the model is in eval mode. `records/runtime.json` binds the actual runtime and all original validation prompt text/ID hashes before any generation. This metadata authentication does not hash all checkpoint weights.

## D2. Genuine NO1 and exact replay precede every nonzero generation

The first fixed validation key runs the accepted four-arm NO1 executor at L19 post-block, original instruction positions, prompt-only scope. Complete generated IDs, text, parse state and gate diagnostics are saved in `gate.json`. Every arm must match the parent full output. Each validation key then has one actual clean replay and an actual dose-zero hook output, both saved. The first key explicitly reuses the genuine gate baseline and zero arm, not another independent generation. Every clean and zero output must be successful and match the parent full output exactly.

Clean failure, zero failure, gate failure or parent drift stops before effects. The incomplete summary retains missing coverage and never repairs or substitutes an output. Generation errors in actual nonzero calls retain their full structured result, token IDs, text, provenance, errors and compact scalar hook diagnostics as executed ITT failures. Unexpected infrastructure exceptions or interruption stop with remaining cells missing. No raw activations, residual arrays, logits, gradients or caches are persisted.

Each addition converts the derived unit direction to embedding dtype/device, applies its signed native dose, and uses a fresh `GenerationPositionTracker` inside `scoped_residual_intervention`. Selection covers only the original instruction token positions at post-block L19. Actual structured generation uses the accepted plain JSON or Harmony driver with the parent grammar, tokenizer, schema, decoding and cache policy.

## D3. Immutable logical zeros reference actual generated sources

Each zero logical cell references its actual clean replay and actual zero-hook record by canonical filename and raw published-byte SHA256. The reference binds runtime/configuration and explicitly declares reuse, source generation origin, gate-arm reuse when applicable and `independent_replicate=false`. It does not copy an invented generation or treat 48 logical references as 48 independent replicates.

The existing Linux hard-link publication and nonblocking store lock enforce single-writer write-once records. Cell filenames hash the complete canonical planned cell, including its baseline row key. Each record has a closed versioned envelope, payload hash and runtime configuration hash. Resume requires the same preflight registration, full plan, actual runtime, valid generation exports, row keys, diagnostic coordinates, source references and passed prerequisites. Foreign, legacy, noncanonical, symlink, staging, hash-mismatched and incomplete prerequisite records are rejected. Existing incomplete summaries are terminal and cannot be repaired. An interruption before summary publication can resume only from valid complete published records. Summaries are also immutable.

The executor constructs its planned cell map once, scans the store once and publishes each new row immediately. It does not rebuild the 129600-cell dictionary or rescan the entire store for each generation.

## D4. Completion is coverage, never efficacy or eligibility

Summaries retain fixed planned/missing counts, schema/reason/decision completion, invalid/failure counts, unknown baseline counts, flips and directional ITT denominators from parent source classes. Positive C uses parent sell as its directional source, while reflected −C uses parent buy. Counts are grouped by cone, direction, signed dose, arm, condition and source class, with issuer-first counts. Missing and invalid outcomes are not flips. There is no efficacy or operator acceptance threshold.

A fully recorded failed nonzero generation counts as executed coverage, not a successful outcome. `complete_shard` requires every assigned cell and passed replay. `complete_global` requires a single unsharded run containing all 129600 cells. Sharded summaries never claim global completion. A hash-authenticating full-shard merge is a later phase, not implemented here. Every summary declares `accepted_operator=false` and `research_eligible=false`. No learned operator is accepted from exploration.

## A1. Verification and GPU execution remain separate

CPU tests use the full approved-role input fixture and unchanged compiler, plus internal microplans with real fake-HF forwards and factory-compiled structured grammars. They exercise NO1, parent drift, signed doses, instruction-only scope, cache modes, genuine failed results, zero source references, hash rejection, write-once registration, full modulo-shard coverage and honest completion. Internal microplans cannot be requested through the CLI.

Proposed GPU invocation, not executed by this implementation:

```bash
uv run python scripts/run_stance_cone_validation.py \
  --inputs data/concept-cone-steering/rebuild-v1/compiled \
  --parent <original-completed-2012-key-parent-directory> \
  --model <original-parent-glm4-9b-0414-checkpoint-path> \
  --training-dir artifacts/glm4-9b-0414/concept-cone-steering/runs/cone-train-stance-rb260930-33 \
  --training-audit artifacts/glm4-9b-0414/concept-cone-steering/audits/cone-train-stance-rb260930-33.validation.json \
  --output-dir artifacts/glm4-9b-0414/concept-cone-steering/runs/<fresh-validation-run-id>
```

An unsharded run attempts 115200 nonzero generations, 300 actual zero generations and 301 clean generations, including the NO1 repeat, plus one self-replacement generation. Each shard repeats its own full 300-input preflight, without subsetting the cohort. The checkpoint and generated-token budgets dominate GPU cost. Wall time and peak memory are not yet measured. Provision one GPU able to hold native BF16 GLM plus generation cache and enough disk for the full plan and structured outputs. No GPU results or resource-fit claim is made.
