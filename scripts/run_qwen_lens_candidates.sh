#!/usr/bin/env bash
# Fit English, Simplified-Chinese, and mixed full-layer Qwen lens candidates.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-.cache/models/qwen3.5-4b}"
MODEL_SLUG="${MODEL%/}"
MODEL_SLUG="${MODEL_SLUG##*/}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-data/calibration/qwen3.5-4b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/candidate_lenses/${MODEL_SLUG}}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-128}"
DIM_BATCH="${DIM_BATCH:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
SKIP_FIRST="${SKIP_FIRST:-16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-4}"
GPU="${GPU:-0}"
SESSION="${SESSION:-qwen_lens_candidates}"
RUN_IN_TMUX="${RUN_IN_TMUX:-1}"

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
for file in english chinese_simplified mixed; do
    [[ -f "${CALIBRATION_ROOT}/${file}.jsonl" ]] || {
        echo "Missing calibration corpus: ${CALIBRATION_ROOT}/${file}.jsonl" >&2
        exit 1
    }
done

if [[ "${RUN_IN_TMUX}" == "1" ]]; then
    command -v tmux >/dev/null || {
        echo "tmux is required; set RUN_IN_TMUX=0 to run in foreground" >&2
        exit 1
    }
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "tmux session already exists: ${SESSION}" >&2
        echo "Attach with: tmux attach -t ${SESSION}" >&2
        exit 1
    fi
    mkdir -p "${OUTPUT_ROOT}"
    command=(
        env
        "MODEL=${MODEL}"
        "CALIBRATION_ROOT=${CALIBRATION_ROOT}"
        "OUTPUT_ROOT=${OUTPUT_ROOT}"
        "CALIBRATION_PROMPTS=${CALIBRATION_PROMPTS}"
        "DIM_BATCH=${DIM_BATCH}"
        "MAX_SEQ_LEN=${MAX_SEQ_LEN}"
        "SKIP_FIRST=${SKIP_FIRST}"
        "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
        "GPU=${GPU}"
        "SESSION=${SESSION}"
        "RUN_IN_TMUX=0"
        bash "${SCRIPT_DIR}/run_qwen_lens_candidates.sh"
    )
    command_string=""
    printf -v command_string '%q ' "${command[@]}"
    tmux new-session -d -s "${SESSION}" \
        "cd $(printf '%q' "${REPO_ROOT}") && ${command_string}"
    echo "Started tmux session: ${SESSION}"
    echo "Attach with: tmux attach -t ${SESSION}"
    echo "Logs: ${OUTPUT_ROOT}/{english,chinese_simplified,mixed}/fit.log"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

for condition in english chinese_simplified mixed; do
    condition_dir="${OUTPUT_ROOT}/${condition}"
    lens="${condition_dir}/jacobian_lens.pt"
    log="${condition_dir}/fit.log"
    mkdir -p "${condition_dir}"
    if [[ -f "${lens}" ]]; then
        echo "Skipping completed candidate: ${lens}"
        continue
    fi
    echo "Fitting ${condition} candidate at $(date --iso-8601=seconds)"
    uv run fit-jacobian-lens \
        --model "${MODEL}" \
        --output "${lens}" \
        --calibration-file "${CALIBRATION_ROOT}/${condition}.jsonl" \
        --calibration-field text \
        --calibration-prompts "${CALIBRATION_PROMPTS}" \
        --layer-stride 1 \
        --dim-batch "${DIM_BATCH}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --skip-first "${SKIP_FIRST}" \
        --checkpoint-every "${CHECKPOINT_EVERY}" \
        --chat-template \
        2>&1 | tee -a "${log}"
done

echo "All candidate fits completed at $(date --iso-8601=seconds)"
uv run python scripts/evaluate_qwen_lens_candidates.py \
    --model "${MODEL}" \
    --candidate-root "${OUTPUT_ROOT}" \
    --holdout \
    data/evaluations/qwen3.5-4b/bilingual_intermediate_holdout.jsonl \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --expected-calibration-prompts "${CALIBRATION_PROMPTS}" \
    --output "${OUTPUT_ROOT}/evaluation.json" \
    2>&1 | tee "${OUTPUT_ROOT}/evaluation.log"
uv run python scripts/promote_qwen_lens_candidate.py \
    --model "${MODEL}" \
    --evaluation "${OUTPUT_ROOT}/evaluation.json" \
    2>&1 | tee "${OUTPUT_ROOT}/promotion.log"
echo "Candidate selection and promotion completed at $(date --iso-8601=seconds)"
