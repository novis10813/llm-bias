#!/usr/bin/env bash
# Run fitting, per-date readouts, and sampled generated-token attribution.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-.cache/models/qwen3.5-4b}"
INPUT_CSV="${INPUT_CSV:-sp500_r1k_r2k_entityBiasPrompt.csv}"
RUN_ROOT="${RUN_ROOT:-artifacts/qwen_jspace_experiment}"
LENS="${LENS:-${RUN_ROOT}/jacobian_lens_stride1.pt}"
CALIBRATION_PROMPTS="${CALIBRATION_PROMPTS:-16}"
FIT_LAYER_STRIDE="${FIT_LAYER_STRIDE:-1}"
FIT_DIM_BATCH="${FIT_DIM_BATCH:-16}"
FIT_MAX_SEQ_LEN="${FIT_MAX_SEQ_LEN:-128}"
FIT_SKIP_FIRST="${FIT_SKIP_FIRST:-0}"
READOUT_BATCH_SIZE="${READOUT_BATCH_SIZE:-32}"
READOUT_MAX_SEQ_LEN="${READOUT_MAX_SEQ_LEN:-256}"
TOP_K="${TOP_K:-15}"
ATTR_SAMPLE_PER_CONDITION="${ATTR_SAMPLE_PER_CONDITION:-32}"
ATTR_MAX_NEW_TOKENS="${ATTR_MAX_NEW_TOKENS:-64}"
SESSION="${SESSION:-qwen_jspace_experiment}"
RUN_IN_TMUX="${RUN_IN_TMUX:-1}"

if [[ "${RUN_IN_TMUX}" == "1" && -z "${TMUX:-}" ]]; then
    command -v tmux >/dev/null || {
        echo "tmux is required; set RUN_IN_TMUX=0 to run in foreground" >&2
        exit 1
    }
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "tmux session already exists: ${SESSION}" >&2
        echo "Attach with: tmux attach -t ${SESSION}" >&2
        exit 1
    fi
    mkdir -p "${RUN_ROOT}"
    # Pass the current configuration into the detached child safely.
    env_args=(
        "MODEL=${MODEL}" "INPUT_CSV=${INPUT_CSV}" "RUN_ROOT=${RUN_ROOT}"
        "LENS=${LENS}" "CALIBRATION_PROMPTS=${CALIBRATION_PROMPTS}"
        "FIT_LAYER_STRIDE=${FIT_LAYER_STRIDE}" "FIT_DIM_BATCH=${FIT_DIM_BATCH}"
        "FIT_MAX_SEQ_LEN=${FIT_MAX_SEQ_LEN}" "FIT_SKIP_FIRST=${FIT_SKIP_FIRST}"
        "READOUT_BATCH_SIZE=${READOUT_BATCH_SIZE}"
        "READOUT_MAX_SEQ_LEN=${READOUT_MAX_SEQ_LEN}" "TOP_K=${TOP_K}"
        "ATTR_SAMPLE_PER_CONDITION=${ATTR_SAMPLE_PER_CONDITION}"
        "ATTR_MAX_NEW_TOKENS=${ATTR_MAX_NEW_TOKENS}"
        "SESSION=${SESSION}"
        "RUN_IN_TMUX=0"
    )
    command=(env)
    for value in "${env_args[@]}"; do command+=("${value}"); done
    command+=(bash "${SCRIPT_DIR}/run_qwen_jspace_experiment.sh")
    command_string=""
    printf -v command_string '%q ' "${command[@]}"
    tmux new-session -d -s "${SESSION}" \
        "cd $(printf '%q' "${REPO_ROOT}") && ${command_string} > $(printf '%q' "${RUN_ROOT}/run.log") 2>&1"
    echo "Started tmux session: ${SESSION}"
    echo "Attach with: tmux attach -t ${SESSION}"
    echo "Log: ${RUN_ROOT}/run.log"
    exit 0
fi

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
[[ -f "${INPUT_CSV}" ]] || { echo "Missing input CSV: ${INPUT_CSV}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"

echo "[1/3] Fit stride-1 Jacobian lens: ${LENS}"
uv run python -m llm_bias fit-lens \
    --model "${MODEL}" \
    --output "${LENS}" \
    --calibration-prompts "${CALIBRATION_PROMPTS}" \
    --layer-stride "${FIT_LAYER_STRIDE}" \
    --dim-batch "${FIT_DIM_BATCH}" \
    --max-seq-len "${FIT_MAX_SEQ_LEN}" \
    --skip-first "${FIT_SKIP_FIRST}"

echo "[2/3] Per-date top-k and uncertainty for all prompt columns"
uv run python -m llm_bias analyze-prompt-outputs \
    --model "${MODEL}" \
    --lens "${LENS}" \
    --input "${INPUT_CSV}" \
    --top-k "${TOP_K}" \
    --batch-size "${READOUT_BATCH_SIZE}" \
    --max-seq-len "${READOUT_MAX_SEQ_LEN}" \
    --no-input-attribution \
    --output-dir "${RUN_ROOT}/per_date"

echo "[3/3] Sampled generated-token attribution for all prompt columns"
uv run python -m llm_bias analyze-generated-attribution \
    --model "${MODEL}" \
    --input "${INPUT_CSV}" \
    --sample-per-condition "${ATTR_SAMPLE_PER_CONDITION}" \
    --max-new-tokens "${ATTR_MAX_NEW_TOKENS}" \
    --max-seq-len "${READOUT_MAX_SEQ_LEN}" \
    --output-dir "${RUN_ROOT}/generated_attribution"

echo "Experiment complete: ${RUN_ROOT}"
