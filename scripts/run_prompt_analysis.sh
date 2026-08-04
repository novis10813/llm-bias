#!/usr/bin/env bash
# Run per-date readouts and sampled generated-token attribution with a fitted lens.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-.cache/models/qwen3.5-4b}"
MODEL_SLUG="${MODEL%/}"
MODEL_SLUG="${MODEL_SLUG##*/}"
INPUT_CSV="${INPUT_CSV:-sp500_r1k_r2k_entityBiasPrompt.csv}"
DATASET_FORMAT="${DATASET_FORMAT:-auto}"
MAX_ROWS="${MAX_ROWS:-}"
RUN_ROOT="${RUN_ROOT:-artifacts/prompt_analysis/${MODEL_SLUG}}"
LENS="${LENS:-artifacts/lenses/${MODEL_SLUG}/jacobian_lens.pt}"
READOUT_BATCH_SIZE="${READOUT_BATCH_SIZE:-32}"
if [[ -z "${READOUT_MAX_SEQ_LEN:-}" ]]; then
    if [[ "${DATASET_FORMAT}" == "return-pairs" ]]; then
        READOUT_MAX_SEQ_LEN=512
    else
        READOUT_MAX_SEQ_LEN=256
    fi
fi
TOP_K="${TOP_K:-15}"
ATTR_SAMPLE_PER_CONDITION="${ATTR_SAMPLE_PER_CONDITION:-32}"
ATTR_MAX_NEW_TOKENS="${ATTR_MAX_NEW_TOKENS:-64}"
ATTR_RUNS="${ATTR_RUNS:-1}"
ATTR_TEMPERATURE="${ATTR_TEMPERATURE:-0}"
ATTR_SEED="${ATTR_SEED:-}"
ATTR_TOP_P="${ATTR_TOP_P:-1.0}"
ATTR_TOP_K="${ATTR_TOP_K:-0}"
ATTR_OUTPUT_DIR="${ATTR_OUTPUT_DIR:-${RUN_ROOT}/generated_attribution}"
RUN_READOUT="${RUN_READOUT:-1}"
SESSION="${SESSION:-prompt_analysis}"
RUN_IN_TMUX="${RUN_IN_TMUX:-1}"

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
[[ -f "${INPUT_CSV}" ]] || { echo "Missing input CSV: ${INPUT_CSV}" >&2; exit 1; }
if [[ "${RUN_READOUT}" == "1" ]]; then
    [[ -f "${LENS}" ]] || {
        echo "Missing Jacobian lens: ${LENS}" >&2
        echo "Fit it first with the fit-jacobian-lens CLI or set LENS." >&2
        exit 1
    }
fi

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
        "MODEL=${MODEL}" "INPUT_CSV=${INPUT_CSV}" "DATASET_FORMAT=${DATASET_FORMAT}" "MAX_ROWS=${MAX_ROWS}" "RUN_ROOT=${RUN_ROOT}"
        "LENS=${LENS}" "READOUT_BATCH_SIZE=${READOUT_BATCH_SIZE}"
        "READOUT_MAX_SEQ_LEN=${READOUT_MAX_SEQ_LEN}" "TOP_K=${TOP_K}"
        "ATTR_SAMPLE_PER_CONDITION=${ATTR_SAMPLE_PER_CONDITION}"
        "ATTR_MAX_NEW_TOKENS=${ATTR_MAX_NEW_TOKENS}"
        "ATTR_RUNS=${ATTR_RUNS}" "ATTR_TEMPERATURE=${ATTR_TEMPERATURE}"
        "ATTR_SEED=${ATTR_SEED}" "ATTR_TOP_P=${ATTR_TOP_P}"
        "ATTR_TOP_K=${ATTR_TOP_K}" "ATTR_OUTPUT_DIR=${ATTR_OUTPUT_DIR}"
        "RUN_READOUT=${RUN_READOUT}" "SESSION=${SESSION}"
        "RUN_IN_TMUX=0"
    )
    command=(env)
    for value in "${env_args[@]}"; do command+=("${value}"); done
    command+=(bash "${SCRIPT_DIR}/run_prompt_analysis.sh")
    command_string=""
    printf -v command_string '%q ' "${command[@]}"
    tmux new-session -d -s "${SESSION}" \
        "cd $(printf '%q' "${REPO_ROOT}") && ${command_string} > $(printf '%q' "${RUN_ROOT}/run.log") 2>&1"
    echo "Started tmux session: ${SESSION}"
    echo "Attach with: tmux attach -t ${SESSION}"
    echo "Log: ${RUN_ROOT}/run.log"
    exit 0
fi

mkdir -p "${RUN_ROOT}"

if [[ "${RUN_READOUT}" == "1" ]]; then
    echo "[1/2] Per-date top-k and uncertainty for all prompt columns"
    uv run prompt-analysis readout \
        --model "${MODEL}" \
        --lens "${LENS}" \
        --input "${INPUT_CSV}" \
        --top-k "${TOP_K}" \
        --batch-size "${READOUT_BATCH_SIZE}" \
        --max-seq-len "${READOUT_MAX_SEQ_LEN}" \
        --dataset-format "${DATASET_FORMAT}" \
        --no-input-attribution \
        ${MAX_ROWS:+--max-rows "${MAX_ROWS}"} \
        --output-dir "${RUN_ROOT}/per_date"
else
    echo "[1/2] Skipping readout (RUN_READOUT=${RUN_READOUT})"
fi

echo "[2/2] Sampled generated-token attribution for all prompt columns"
attribute_args=(
    uv run prompt-analysis attribute
    --model "${MODEL}"
    --input "${INPUT_CSV}"
    --sample-per-condition "${ATTR_SAMPLE_PER_CONDITION}"
    --max-new-tokens "${ATTR_MAX_NEW_TOKENS}"
    --max-seq-len "${READOUT_MAX_SEQ_LEN}"
    --dataset-format "${DATASET_FORMAT}"
    --runs "${ATTR_RUNS}"
    --temperature "${ATTR_TEMPERATURE}"
    --top-p "${ATTR_TOP_P}"
    --top-k "${ATTR_TOP_K}"
    --output-dir "${ATTR_OUTPUT_DIR}"
)
if [[ -n "${ATTR_SEED}" ]]; then
    attribute_args+=(--seed "${ATTR_SEED}")
fi
"${attribute_args[@]}"

echo "Experiment complete: ${RUN_ROOT}"
