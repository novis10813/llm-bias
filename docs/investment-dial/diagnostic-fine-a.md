# A-only fine-grained curve diagnostic preserves the original V1 calibration

Status: auxiliary diagnostic operator implemented; original V1 results remain unchanged.
This is not a new calibration method, a paper reproduction claim, or an independent test.
The question is whether linear interpolation between the original delta=0 and delta=4
samples missed a steep response near zero. This diagnostic does not establish that cause
in advance.

## Fixed input and output contract

`scripts/investment_dial_fine_a.py` owns this diagnostic and reuses the investment-dial
pipeline and shared artifact lifecycle. It takes a complete validated original calibration
run, including manifest and all registered outputs. It verifies model/tokenizer identity,
parent hashes, tokenization, and software/dtype/chat-template compatibility before inference.
Hardware/device differences are allowed and recorded; they may change generated decisions.

The original selected coordinate must be layer index 15, neuron index 8490 (zero-based).
Only split A rows with `positive_count == 2` are generated: 340 prompts from 85 companies.
No B/test generation, candidate selection, or coefficient refitting occurs. The parent
contains other splits, but they are not generated or written into diagnostic trials.

Fixed deltas: 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2 (3,060 generations).
Use bf16, original intervention placement, greedy generation, maximum 256 new tokens,
and unchanged JSON decision/reason prompts. No lens fitting or raw tensor persistence.

Outputs under `artifacts/<model-slug>/investment-dial-fine-a/runs/<run-id>/`:
- `prepare/protocol.json`: parent digest, model/runtime/operator provenance, fixed settings.
- `prepare/trials.json`: selected A-only encoded prompts.
- `forward/delta-0.jsonl` through `forward/delta-8.jsonl`: generated IDs/text, decisions,
  validity, coordinate and delta. Each file is registered after its grid point completes.
- `analyze/result.json`: finite per-delta counts/rates and pi; pi is null if no valid decisions;
  `certified=false`. Pi is (buy-sell)/(buy+sell), excluding invalid decisions from its denominator.
- `manifest.json`: shared prepare → forward → analyze → finalize lifecycle.

There is no automatic resume. Interrupted runs remain incomplete; use a new run ID.
Progress is printed every 20 prompts. Invalid parent/coordinate/count/runtime fails closed;
no fallback to another model, subset or neuron is permitted.

## Run on another server from the repository root

Install the workspace and external dependencies as described in the root README. Transfer
identical checkpoint/tokenizer files and the **complete** source run directory (not just
result.json). Original source:
`artifacts/qwen3.5-4b/investment-dial-calibration/runs/exploratory-v1-20260907-07-gpu0-calibration`.
The source directory is ignored by Git and must be copied separately. Keep its contents
unchanged; `--source-run` may point to its new location. Parent absolute paths are provenance,
not a requirement to recreate the old server's filesystem.

GPU command (physical GPU 0; full model on one device):

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/investment_dial_fine_a.py \
  --model .cache/models/qwen3.5-4b \
  --source-run artifacts/qwen3.5-4b/investment-dial-calibration/runs/exploratory-v1-20260907-07-gpu0-calibration \
  --run-id fine-a-gpu-bf16-01 --device cuda
```

For CPU execution, use `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=32` and `--device cpu`.
The operator rejects CPU requests while CUDA is visible and CUDA requests without CUDA.
Choose a new run ID for each attempt; `--artifact-root` defaults to `artifacts`.

## Interpret the curve without tuning on the test split

First compare delta=0 against original A: buy=125, sell=215, pi=-0.2647058824.
This is a descriptive comparison, not an exact-equality gate across hardware.
Check pi and output-validity rates together across the grid. A steep transition would support
(but not alone prove) the coarse-interpolation explanation. Broad runtime differences at
zero would complicate that attribution. Preserve all points, including invalid generations.
Do not choose a new calibration rule or accuracy gate after seeing test results. Changes to
calibration/gates require a separately versioned protocol and appropriate fresh evaluation;
this diagnostic never overwrites V1 coefficients or claims certified success.

Verification: `uv run pytest -q tests/test_investment_dial_fine_a.py tests/test_investment_dial.py`.
Tests use mocked inference; no full-model run is launched by the test suite.
