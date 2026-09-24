#!/usr/bin/env bash
# v2 development run: 427-company within-company condition-flip on Qwen3.5-4B (GPU0).
# Design: docs/balanced-evidence-gap/details/proposal-phase2-v2.md
#   2A: 427 tickers x {pos,neg} two-sentence conditions x 2 reverses = 1708 forwards
#   2B: per-company pos<->neg direction flips (both orientations), 32 layers x 4 spans
set -u
cd "$(dirname "$0")/../.."
MODEL=".cache/models/qwen3.5-4b"
A_RUN="artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-v2-427-01"
export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== [v2-427] 2A condition probe (gpu 0) ==="
uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \
    --model "$MODEL" --run-id phase2a-v2-427-01 \
    --no-dial --phase1-summary none \
    --family v2 \
    --companies-file data/baseline/investment-dial/exploratory-v1.json
RC=$?
if [ $RC -ne 0 ]; then echo "=== [v2-427] 2A FAILED (rc=${RC}) ==="; exit $RC; fi

echo "=== [v2-427] gate 2A result ==="
uv run --no-sync python - <<'PY'
import json, pathlib
s = json.loads(pathlib.Path("artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-v2-427-01/analyze/summary.json").read_text())
g = s["gate_2a"]
for name, c in g["criteria"].items():
    print(f"  {name}: value={c['value']} pass={c['pass']}")
print("  gate pass:", g["pass"])
PY

echo "=== [v2-427] 2B condition-flip sweep (gpu 0) ==="
uv run --no-sync python scripts/balanced_evidence_gap_phase2_patch.py sweep \
    --model "$MODEL" \
    --phase2a-run "$A_RUN" \
    --run-id phase2b-v2-427-01 --family v2
RC=$?
if [ $RC -ne 0 ]; then echo "=== [v2-427] 2B FAILED (rc=${RC}) ==="; exit $RC; fi
echo "=== [v2-427] COMPLETE ==="
