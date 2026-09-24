#!/usr/bin/env bash
# Cross-model Phase 2 formal run: 2A (cross-entity probe) then 2B (layer sweep).
# Usage: run_crossmodel_phase2.sh <model-slug> <gpu-id> [device-map]
# Artifacts land in artifacts/<slug>/balanced-evidence-gap-phase2/runs/.
set -u
SLUG="$1"
GPU="$2"
DEVICE_MAP="${3:-}"
cd "$(dirname "$0")/../.."
MODEL=".cache/models/$SLUG"
if [ ! -d "$MODEL" ]; then echo "model dir missing: $MODEL"; exit 1; fi

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Long 2B sweeps fragment the allocator; expandable segments keep large fp32
# transmits satisfiable (see run_v2_427_crossmodel.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DM_FLAGS=""
if [ -n "$DEVICE_MAP" ]; then DM_FLAGS="--device-map ${DEVICE_MAP}"; fi

# Optional: proceed with 2B under a recorded development override if gate 2A
# does not pass (e.g. borderline IQR). Unset = protocol behavior (blocked).
GO_FLAGS=""
if [ -n "${GATE_OVERRIDE:-}" ]; then GO_FLAGS="--gate-override ${GATE_OVERRIDE}"; fi

# Pre-warm the page cache with sequential reads: the inference path mmaps the
# safetensors (random page faults), which is painfully slow on the HDD that
# hosts .cache/models. Sequential cat fills the cache in ~1-2 min per 18GB.
echo "=== [${SLUG}] pre-warming page cache ==="
for f in "$MODEL"/*.safetensors; do
  [ -f "$f" ] && cat "$f" > /dev/null
done
echo "=== [${SLUG}] page cache warm ==="

echo "=== [${SLUG}] 2A cross-entity probe (gpu ${GPU}) ==="
uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \
    --model "$MODEL" --run-id phase2a-crossmodel-01 --no-dial --phase1-summary none ${DM_FLAGS}
RC=$?
if [ $RC -ne 0 ]; then echo "=== [${SLUG}] 2A FAILED (rc=${RC}) ==="; exit $RC; fi

echo "=== [${SLUG}] 2B entity-state layer sweep (gpu ${GPU}) ==="
uv run --no-sync python scripts/balanced_evidence_gap_phase2_patch.py sweep \
    --model "$MODEL" \
    --phase2a-run "artifacts/${SLUG}/balanced-evidence-gap-phase2/runs/phase2a-crossmodel-01" \
    --run-id phase2b-crossmodel-01 ${DM_FLAGS} ${GO_FLAGS}
RC=$?
if [ $RC -ne 0 ]; then echo "=== [${SLUG}] 2B FAILED (rc=${RC}) ==="; exit $RC; fi
echo "=== [${SLUG}] COMPLETE ==="
