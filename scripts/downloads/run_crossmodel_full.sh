#!/usr/bin/env bash
# Full cross-model pipeline for the final member: v1 16-company 2A+2B, then
# v2 427-company 2A+2B. Smoke preflight first (aborts the chain on failure).
# Gate failures do not abort: the 2B sweep proceeds with a recorded override
# (development-stage cross-model comparison).
# usage: run_crossmodel_full.sh <model-dir> <slug> <gpu> [extra load args...]
#   e.g. run_crossmodel_full.sh .cache/models/qwen3.6-27b-fp8 qwen3.6-27b-fp8 1 --dtype native
set -u
cd "$(dirname "$0")/../.."
MODEL_DIR="$1"; SLUG="$2"; GPU="$3"; shift 3
EXTRA_LOAD_ARGS=("$@")
RUN_ROOT="artifacts/${SLUG}/balanced-evidence-gap-phase2/runs"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Long 2B sweeps fragment the allocator; expandable segments keep large fp32
# transients satisfiable (Gemma v1/v2 OOMs, 27B lm_head is 3.1 GiB in fp32).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gate_reason() {
    # $1 = phase2a run dir; prints the override reason when gate 2A failed
    local a_run="$1"
    [ -f "${a_run}/analyze/summary.json" ] || return 0
    uv run --no-sync python -c "
import json, pathlib
s = json.loads(pathlib.Path('${a_run}/analyze/summary.json').read_text())
g = s['gate_2a']
if not g['pass']:
    parts = [f\"{n}: value={c['value']} (threshold {c['threshold']})\" for n, c in g['criteria'].items() if c.get('pass') is False]
    print('Development override: gate 2A failed [' + '; '.join(parts) + ']; cross-model descriptive comparison')
"
}

echo "=== [${SLUG}] smoke preflight (gpu ${GPU}) ==="
uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \
    --model "$MODEL_DIR" --smoke --no-dial --phase1-summary none \
    "${EXTRA_LOAD_ARGS[@]}" || { echo "=== [${SLUG}] SMOKE FAILED ==="; exit 1; }

echo "=== [${SLUG}] v1 16-company 2A ==="
uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \
    --model "$MODEL_DIR" --run-id phase2a-crossmodel-01 \
    --no-dial --phase1-summary none "${EXTRA_LOAD_ARGS[@]}"
RC2A=$?
[ -f "${RUN_ROOT}/phase2a-crossmodel-01/analyze/summary.json" ] || { echo "=== [${SLUG}] v1 2A FAILED (rc=${RC2A}) ==="; exit 1; }

echo "=== [${SLUG}] v1 2B sweep ==="
GATE_REASON=$(gate_reason "${RUN_ROOT}/phase2a-crossmodel-01" 2>/dev/null || true)
GATE_OVERRIDE=()
if [ -n "$GATE_REASON" ]; then
    GATE_OVERRIDE=(--gate-override "$GATE_REASON")
    echo "=== [${SLUG}] v1 gate 2A FAILED -> 2B with override ==="
fi
uv run --no-sync python scripts/balanced_evidence_gap_phase2_patch.py sweep \
    --model "$MODEL_DIR" \
    --phase2a-run "${RUN_ROOT}/phase2a-crossmodel-01" \
    --run-id phase2b-crossmodel-01 \
    "${EXTRA_LOAD_ARGS[@]}" ${GATE_OVERRIDE[@]+"${GATE_OVERRIDE[@]}"}
RC=$?
[ $RC -eq 0 ] || { echo "=== [${SLUG}] v1 2B FAILED (rc=${RC}) ==="; exit $RC; }

echo "=== [${SLUG}] v2 427 (runner) ==="
bash scripts/downloads/run_v2_427_crossmodel.sh "$MODEL_DIR" "$SLUG" "$GPU" "${EXTRA_LOAD_ARGS[@]}"
