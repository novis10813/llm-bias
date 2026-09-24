#!/usr/bin/env bash
# v2 427-company condition-flip run for a cross-model member.
# Design: docs/balanced-evidence-gap/details/proposal-phase2-v2.md
#   2A: 427 tickers x {pos,neg} two-sentence conditions x 2 reverses = 1708 forwards
#   2B: per-company pos<->neg direction flips (both orientations), full layer sweep x 4 spans
# usage: run_v2_427_crossmodel.sh <model-dir> <slug> <gpu> [extra load args...]
#   e.g. run_v2_427_crossmodel.sh .cache/models/gpt-oss-20b gpt-oss-20b 1 --dtype native
#   e.g. run_v2_427_crossmodel.sh .cache/models/qwen3.6-27b qwen3.6-27b 0 --device-map qwen27b_two_gpu
set -u
cd "$(dirname "$0")/../.."
MODEL_DIR="$1"; SLUG="$2"; GPU="$3"; shift 3
EXTRA_LOAD_ARGS=("$@")
A_RUN="artifacts/${SLUG}/balanced-evidence-gap-phase2/runs/phase2a-v2-427-01"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
# Long 2B sweeps (854 directions) fragment the allocator; expandable segments
# keep the fp32 lm_head transient (up to 3.75 GiB) satisfiable (Gemma v1/v2 OOMs).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# No HF_HUB_OFFLINE: models load from local paths, but transformers' MXFP4/FP8
# paths resolve runtime triton kernels from the HF hub at load time.
unset HF_HUB_OFFLINE

echo "=== [v2-427 ${SLUG}] 2A condition probe (gpu ${GPU}) ==="
uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \
    --model "$MODEL_DIR" --run-id phase2a-v2-427-01 \
    --no-dial --phase1-summary none --family v2 \
    --companies-file data/baseline/investment-dial/exploratory-v1.json \
    "${EXTRA_LOAD_ARGS[@]}"
RC2A=$?
if [ ! -f "${A_RUN}/analyze/summary.json" ]; then
    echo "=== [v2-427 ${SLUG}] 2A FAILED (rc=${RC2A}, no summary) ==="; exit 1
fi

GATE=$(uv run --no-sync python -c "
import json, pathlib
s = json.loads(pathlib.Path('${A_RUN}/analyze/summary.json').read_text())
print('pass' if s['gate_2a']['pass'] else 'fail')
")
OVERRIDE=()
if [ "$GATE" = "fail" ]; then
    REASON=$(uv run --no-sync python -c "
import json, pathlib
s = json.loads(pathlib.Path('${A_RUN}/analyze/summary.json').read_text())
g = s['gate_2a']
parts = [f\"{n}: value={c['value']} (threshold {c['threshold']})\" for n, c in g['criteria'].items() if c.get('pass') is False]
print('Development override: v2 gate 2A failed [' + '; '.join(parts) + ']; cross-model descriptive comparison')
")
    OVERRIDE=(--gate-override "$REASON")
    echo "=== [v2-427 ${SLUG}] gate 2A FAILED -> 2B with override ==="
fi

echo "=== [v2-427 ${SLUG}] 2B condition-flip sweep (gpu ${GPU}) ==="
uv run --no-sync python scripts/balanced_evidence_gap_phase2_patch.py sweep \
    --model "$MODEL_DIR" \
    --phase2a-run "$A_RUN" \
    --run-id phase2b-v2-427-01 --family v2 \
    "${EXTRA_LOAD_ARGS[@]}" "${OVERRIDE[@]}"
RC=$?
if [ $RC -ne 0 ]; then echo "=== [v2-427 ${SLUG}] 2B FAILED (rc=${RC}) ==="; exit $RC; fi
echo "=== [v2-427 ${SLUG}] COMPLETE ==="
