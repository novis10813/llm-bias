#!/usr/bin/env bash
# Run prompt-analysis readout, generation, and generated-token attribution stages.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-.cache/models/qwen3.5-4b}"
MODEL_SLUG="${MODEL%/}"
MODEL_SLUG="${MODEL_SLUG##*/}"
INPUT_CSV="${INPUT_CSV:-sp500_r1k_r2k_entityBiasPrompt.csv}"
DATASET_FORMAT="${DATASET_FORMAT:-auto}"
DATASET_SLUG="${DATASET_SLUG:-}"
if [[ -z "${DATASET_SLUG}" ]]; then
    DATASET_SLUG="$(basename -- "${INPUT_CSV}")"
    DATASET_SLUG="${DATASET_SLUG%.*}"
    DATASET_SLUG="${DATASET_SLUG,,}"
    DATASET_SLUG="${DATASET_SLUG//[^a-z0-9_.-]/_}"
fi
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-artifacts/${MODEL_SLUG}/${DATASET_SLUG}/runs/${RUN_ID}}"
LENS="${LENS:-artifacts/lenses/${MODEL_SLUG}/jacobian_lens.pt}"
READOUT_BATCH_SIZE="${READOUT_BATCH_SIZE:-32}"
READOUT_MAX_SEQ_LEN="${READOUT_MAX_SEQ_LEN:-}"
if [[ -z "${READOUT_MAX_SEQ_LEN}" ]]; then
    if [[ "${DATASET_FORMAT}" == "return-pairs" ]]; then
        READOUT_MAX_SEQ_LEN=512
    else
        READOUT_MAX_SEQ_LEN=256
    fi
fi
TOP_K="${TOP_K:-15}"
MAX_ROWS="${MAX_ROWS:-}"
GEN_SAMPLE_PER_CONDITION="${GEN_SAMPLE_PER_CONDITION:-32}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-64}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0}"
GEN_SEED="${GEN_SEED:-}"
GEN_TOP_P="${GEN_TOP_P:-1.0}"
GEN_TOP_K="${GEN_TOP_K:-0}"
RUN_READOUT="${RUN_READOUT:-1}"
RUN_GENERATION="${RUN_GENERATION:-0}"
RUN_ATTRIBUTION="${RUN_ATTRIBUTION:-0}"
FORWARD_ARTIFACT="${FORWARD_ARTIFACT:-}"
ATTR_INPUT_TOP_K="${ATTR_INPUT_TOP_K:-}"
SESSION="${SESSION:-prompt_analysis}"
RUN_IN_TMUX="${RUN_IN_TMUX:-1}"

for variable_name in RUN_READOUT RUN_GENERATION RUN_ATTRIBUTION; do
    variable_value="${!variable_name}"
    if [[ "${variable_value}" != "0" && "${variable_value}" != "1" ]]; then
        echo "${variable_name} must be 0 or 1 (got: ${variable_value})" >&2
        exit 1
    fi
done
if [[ "${GEN_SAMPLE_PER_CONDITION}" =~ ^-?[0-9]+$ ]] && (( GEN_SAMPLE_PER_CONDITION < 0 )); then
    echo "GEN_SAMPLE_PER_CONDITION must be zero (all rows) or positive" >&2
    exit 1
fi

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
[[ -f "${INPUT_CSV}" ]] || { echo "Missing input CSV: ${INPUT_CSV}" >&2; exit 1; }
if [[ "${RUN_READOUT}" == "1" ]]; then
    [[ -f "${LENS}" ]] || {
        echo "Missing Jacobian lens: ${LENS}" >&2
        echo "Fit it first with the fit-jacobian-lens CLI or set LENS." >&2
        exit 1
    }
fi
if [[ "${RUN_GENERATION}" == "0" && "${RUN_ATTRIBUTION}" == "1" ]]; then
    [[ -n "${FORWARD_ARTIFACT}" ]] || {
        echo "FORWARD_ARTIFACT is required when RUN_GENERATION=0 and RUN_ATTRIBUTION=1" >&2
        exit 1
    }
    [[ -f "${FORWARD_ARTIFACT}" ]] || {
        echo "Missing forward artifact: ${FORWARD_ARTIFACT}" >&2
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
    [[ ! -e "${RUN_ROOT}" ]] || {
        echo "Run root already exists; refusing to reuse stale output: ${RUN_ROOT}" >&2
        exit 1
    }
    mkdir -p "${RUN_ROOT}"
    env_args=(
        "MODEL=${MODEL}" "INPUT_CSV=${INPUT_CSV}" "DATASET_FORMAT=${DATASET_FORMAT}"
        "DATASET_SLUG=${DATASET_SLUG}" "RUN_ID=${RUN_ID}" "RUN_ROOT=${RUN_ROOT}"
        "RUN_ROOT_READY=1" "LENS=${LENS}" "READOUT_BATCH_SIZE=${READOUT_BATCH_SIZE}"
        "READOUT_MAX_SEQ_LEN=${READOUT_MAX_SEQ_LEN}" "TOP_K=${TOP_K}" "MAX_ROWS=${MAX_ROWS}"
        "GEN_SAMPLE_PER_CONDITION=${GEN_SAMPLE_PER_CONDITION}" "GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS}"
        "GEN_TEMPERATURE=${GEN_TEMPERATURE}" "GEN_SEED=${GEN_SEED}" "GEN_TOP_P=${GEN_TOP_P}"
        "GEN_TOP_K=${GEN_TOP_K}" "RUN_READOUT=${RUN_READOUT}" "RUN_GENERATION=${RUN_GENERATION}"
        "RUN_ATTRIBUTION=${RUN_ATTRIBUTION}" "FORWARD_ARTIFACT=${FORWARD_ARTIFACT}"
        "ATTR_INPUT_TOP_K=${ATTR_INPUT_TOP_K}"
        "SESSION=${SESSION}" "RUN_IN_TMUX=0"
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

if [[ "${RUN_ROOT_READY:-0}" != "1" ]]; then
    [[ ! -e "${RUN_ROOT}" ]] || {
        echo "Run root already exists; refusing to reuse stale output: ${RUN_ROOT}" >&2
        exit 1
    }
    mkdir -p "${RUN_ROOT}"
fi

MANIFEST="${RUN_ROOT}/manifest.json"
READOUT_DIR="${RUN_ROOT}/readout"
FORWARD_OUTPUT="${RUN_ROOT}/forward/generated_outputs.jsonl"
BACKWARD_OUTPUT="${RUN_ROOT}/backward/generated_token_attribution.jsonl"
if [[ "${RUN_GENERATION}" == "1" ]]; then
    FORWARD_ARTIFACT="${FORWARD_OUTPUT}"
fi

write_manifest() {
    local status="${1}"
    local readout_status="${2}"
    local generation_status="${3}"
    local attribution_status="${4}"
    MANIFEST_STATUS="${status}" \
    MANIFEST_READOUT_STATUS="${readout_status}" \
    MANIFEST_GENERATION_STATUS="${generation_status}" \
    MANIFEST_ATTRIBUTION_STATUS="${attribution_status}" \
    uv run python - "${MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = {
    "schema": "prompt-analysis-run-manifest-v1",
    "status": os.environ["MANIFEST_STATUS"],
    "model": os.environ["MODEL"],
    "model_slug": os.environ["MODEL_SLUG"],
    "dataset": os.environ["DATASET_SLUG"],
    "dataset_format": os.environ["DATASET_FORMAT"],
    "run_id": os.environ["RUN_ID"],
    "input": os.environ["INPUT_CSV"],
    "stages": {
        "readout": {"enabled": os.environ["RUN_READOUT"] == "1", "status": os.environ["MANIFEST_READOUT_STATUS"], "path": os.environ["READOUT_DIR"]},
        "generation": {"enabled": os.environ["RUN_GENERATION"] == "1", "status": os.environ["MANIFEST_GENERATION_STATUS"], "path": os.environ["FORWARD_OUTPUT"]},
        "attribution": {"enabled": os.environ["RUN_ATTRIBUTION"] == "1", "status": os.environ["MANIFEST_ATTRIBUTION_STATUS"], "path": os.environ["BACKWARD_OUTPUT"]},
    },
    "forward_artifact": os.environ["FORWARD_ARTIFACT"] or None,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

export MODEL MODEL_SLUG DATASET_SLUG DATASET_FORMAT RUN_ID INPUT_CSV RUN_READOUT RUN_GENERATION RUN_ATTRIBUTION
export READOUT_DIR FORWARD_OUTPUT BACKWARD_OUTPUT FORWARD_ARTIFACT
run_status="running"
readout_status="disabled"
generation_status="disabled"
attribution_status="disabled"
[[ "${RUN_READOUT}" == "1" ]] && readout_status="pending"
[[ "${RUN_GENERATION}" == "1" ]] && generation_status="pending"
[[ "${RUN_ATTRIBUTION}" == "1" ]] && attribution_status="pending"
write_manifest "${run_status}" "${readout_status}" "${generation_status}" "${attribution_status}"
trap 'rc=$?; if [[ "${run_status}" != "completed" ]]; then write_manifest failed "${readout_status}" "${generation_status}" "${attribution_status}" || true; fi; exit "${rc}"' EXIT

if [[ "${RUN_READOUT}" == "1" ]]; then
    readout_status="running"
    write_manifest "${run_status}" "${readout_status}" "${generation_status}" "${attribution_status}"
    readout_args=(
        uv run prompt-analysis readout --model "${MODEL}" --lens "${LENS}" --input "${INPUT_CSV}"
        --top-k "${TOP_K}" --batch-size "${READOUT_BATCH_SIZE}" --max-seq-len "${READOUT_MAX_SEQ_LEN}"
        --dataset-format "${DATASET_FORMAT}" --output-dir "${READOUT_DIR}"
    )
    if [[ -n "${MAX_ROWS}" ]]; then readout_args+=(--max-rows "${MAX_ROWS}"); fi
    "${readout_args[@]}"
    readout_status="completed"
    write_manifest "${run_status}" "${readout_status}" "${generation_status}" "${attribution_status}"
fi

if [[ "${RUN_GENERATION}" == "1" ]]; then
    generation_status="running"
    write_manifest "${run_status}" "${readout_status}" "${generation_status}" "${attribution_status}"
    mkdir -p "${RUN_ROOT}/forward"
    generation_args=(
        uv run prompt-analysis generate --model "${MODEL}" --input "${INPUT_CSV}" --output "${FORWARD_OUTPUT}"
        --sample-per-condition "${GEN_SAMPLE_PER_CONDITION}" --max-new-tokens "${GEN_MAX_NEW_TOKENS}"
        --max-seq-len "${READOUT_MAX_SEQ_LEN}" --temperature "${GEN_TEMPERATURE}" --top-p "${GEN_TOP_P}"
        --top-k "${GEN_TOP_K}" --dataset-format "${DATASET_FORMAT}"
    )
    if [[ -n "${GEN_SEED}" ]]; then generation_args+=(--seed "${GEN_SEED}"); fi
    "${generation_args[@]}"
    [[ -s "${FORWARD_OUTPUT}" ]] || { echo "Generation did not produce a non-empty forward artifact" >&2; exit 1; }
    generation_status="completed"
    write_manifest "${run_status}" "${readout_status}" "${generation_status}" "${attribution_status}"
fi

if [[ "${RUN_ATTRIBUTION}" == "1" ]]; then
    attribution_status="running"
    write_manifest "${run_status}" "${readout_status}" "${generation_status}" "${attribution_status}"
    mkdir -p "${RUN_ROOT}/backward"
    attribution_args=(
        uv run prompt-analysis attribute-generated --model "${MODEL}"
        --forward-artifact "${FORWARD_ARTIFACT}" --output "${BACKWARD_OUTPUT}"
        --max-seq-len "${READOUT_MAX_SEQ_LEN}"
    )
    if [[ -n "${ATTR_INPUT_TOP_K}" ]]; then attribution_args+=(--input-top-k "${ATTR_INPUT_TOP_K}"); fi
    "${attribution_args[@]}"
    [[ -s "${BACKWARD_OUTPUT}" ]] || { echo "Attribution did not produce a non-empty backward artifact" >&2; exit 1; }
    attribution_status="completed"
    write_manifest "${run_status}" "${readout_status}" "${generation_status}" "${attribution_status}"
fi

run_status="completed"
write_manifest "${run_status}" "${readout_status}" "${generation_status}" "${attribution_status}"
trap - EXIT
echo "Experiment complete: ${RUN_ROOT}"
