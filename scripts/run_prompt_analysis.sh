#!/usr/bin/env bash
# Run prompt-analysis readout, generation, and generated-token attribution stages.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-.cache/models/qwen3.5-4b}"
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
MODEL_SLUG="$(uv run python -c \
    'from llm_bias.core.lens_artifacts import model_slug; import sys; print(model_slug(sys.argv[1]))' \
    "${MODEL}")"
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
LENS="${LENS:-artifacts/${MODEL_SLUG}/jacobian-lens/jacobian_lens.pt}"
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
BACKWARD_INPUT_TOP_K="${BACKWARD_INPUT_TOP_K:-}"
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
        "BACKWARD_INPUT_TOP_K=${BACKWARD_INPUT_TOP_K}"
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

manifest_update() {
    local action="${1}"
    local stage="${2:-}"
    local error="${3:-}"
    MANIFEST_ACTION="${action}" \
    MANIFEST_STAGE="${stage}" \
    MANIFEST_ERROR="${error}" \
    MANIFEST_FILES="${MANIFEST_FILES:-}" \
    uv run python - "${MANIFEST}" <<'PY'
import os
import sys
from pathlib import Path

from llm_bias.core.artifact_manifest import RunManifest

manifest_path = Path(sys.argv[1])
action = os.environ["MANIFEST_ACTION"]
stage = os.environ.get("MANIFEST_STAGE", "")
manifest_files = os.environ.get("MANIFEST_FILES", "")

if action == "init":
    manifest = RunManifest(
        model=os.environ["MODEL"],
        dataset=os.environ["DATASET_SLUG"],
        run_id=os.environ["RUN_ID"],
        run_directory=Path(os.environ["RUN_ROOT"]),
    )
    manifest.register_artifact(
        os.environ["INPUT_CSV"], artifact_type="prompt_input", stage="prepare", role="input",
        metadata={"dataset_format": os.environ["DATASET_FORMAT"], "provenance": "runner input CSV"},
    )
    lens_path = Path(os.environ["LENS"])
    if lens_path.is_file():
        manifest.register_artifact(
            lens_path, artifact_type="jacobian_lens", stage="prepare", role="lens",
            metadata={"provenance": "configured Jacobian lens"},
        )
    for name, enabled in (("readout", "RUN_READOUT"), ("generation", "RUN_GENERATION"), ("attribution", "RUN_ATTRIBUTION")):
        if os.environ[enabled] == "1":
            manifest.stages[name] = {"status": "created"}
    manifest.save()
else:
    manifest = RunManifest.load(manifest_path)
    if action == "start":
        manifest.start().save()
    elif action == "stage_running":
        manifest.start_stage(stage).save()
    elif action == "stage_complete":
        for entry in filter(None, manifest_files.split("\n")):
            path, artifact_type, role = entry.split("|", 2)
            manifest.register_artifact(
                path, artifact_type=artifact_type, stage=stage, role=role,
                metadata={"provenance": f"prompt-analysis {stage} stage output"},
            )
        manifest.finish_stage(stage).save()
    elif action == "fail":
        if stage:
            manifest.finish_stage(stage, status="failed").save()
        manifest.fail(os.environ.get("MANIFEST_ERROR") or "prompt-analysis stage failed").save()
    elif action == "complete":
        manifest.complete().save()
    else:
        raise ValueError(f"unknown manifest action: {action}")
PY
}

export MODEL MODEL_SLUG DATASET_SLUG DATASET_FORMAT RUN_ID INPUT_CSV RUN_READOUT RUN_GENERATION RUN_ATTRIBUTION LENS
export READOUT_DIR FORWARD_OUTPUT BACKWARD_OUTPUT FORWARD_ARTIFACT
manifest_update init
manifest_update start
trap 'rc=$?; if [[ "${run_status}" != "completed" ]]; then manifest_update fail "${active_stage:-}" "runner exited with status ${rc}" || true; fi; exit "${rc}"' EXIT
run_status="running"
active_stage=""

if [[ "${RUN_READOUT}" == "1" ]]; then
    active_stage="readout"
    manifest_update stage_running "${active_stage}"
    readout_args=(
        uv run prompt-analysis readout --model "${MODEL}" --lens "${LENS}" --input "${INPUT_CSV}"
        --top-k "${TOP_K}" --batch-size "${READOUT_BATCH_SIZE}" --max-seq-len "${READOUT_MAX_SEQ_LEN}"
        --dataset-format "${DATASET_FORMAT}" --output-dir "${READOUT_DIR}"
    )
    if [[ -n "${MAX_ROWS}" ]]; then readout_args+=(--max-rows "${MAX_ROWS}"); fi
    "${readout_args[@]}"
    MANIFEST_FILES="${READOUT_DIR}/prompt_layer_topk.jsonl|prompt_layer_topk|output
${READOUT_DIR}/prompt_layer_uncertainty.jsonl|prompt_layer_uncertainty|output
${READOUT_DIR}/average_layer_topk.jsonl|average_layer_topk|output
${READOUT_DIR}/metadata.json|readout_metadata|output" \
        manifest_update stage_complete "${active_stage}"
    active_stage=""
fi

if [[ "${RUN_GENERATION}" == "1" ]]; then
    active_stage="generation"
    manifest_update stage_running "${active_stage}"
    mkdir -p "${RUN_ROOT}/forward"
    generation_args=(
        uv run prompt-analysis generate --model "${MODEL}" --input "${INPUT_CSV}" --output "${FORWARD_OUTPUT}"
        --sample-per-condition "${GEN_SAMPLE_PER_CONDITION}" --max-new-tokens "${GEN_MAX_NEW_TOKENS}"
        --max-seq-len "${READOUT_MAX_SEQ_LEN}" --temperature "${GEN_TEMPERATURE}" --top-p "${GEN_TOP_P}"
        --top-k "${GEN_TOP_K}" --dataset-format "${DATASET_FORMAT}"
    )
    if [[ "${GEN_SAMPLE_PER_CONDITION}" == "0" ]]; then generation_args+=(--full-generation); fi
    if [[ -n "${GEN_SEED}" ]]; then generation_args+=(--seed "${GEN_SEED}"); fi
    "${generation_args[@]}"
    MANIFEST_FILES="${FORWARD_OUTPUT}|generated_outputs|output
${RUN_ROOT}/forward/metadata.json|generation_metadata|output" \
        manifest_update stage_complete "${active_stage}"
    active_stage=""
fi

if [[ "${RUN_ATTRIBUTION}" == "1" ]]; then
    active_stage="attribution"
    manifest_update stage_running "${active_stage}"
    mkdir -p "${RUN_ROOT}/backward"
    attribution_args=(
        uv run prompt-analysis attribute-generated --model "${MODEL}"
        --forward-artifact "${FORWARD_ARTIFACT}" --output "${BACKWARD_OUTPUT}"
        --max-seq-len "${READOUT_MAX_SEQ_LEN}"
    )
    if [[ -n "${BACKWARD_INPUT_TOP_K}" ]]; then attribution_args+=(--input-top-k "${BACKWARD_INPUT_TOP_K}"); fi
    "${attribution_args[@]}"
    MANIFEST_FILES="${BACKWARD_OUTPUT}|generated_token_attribution|output
${RUN_ROOT}/backward/metadata.json|attribution_metadata|output" \
        manifest_update stage_complete "${active_stage}"
    active_stage=""
fi

run_status="completed"
manifest_update complete
trap - EXIT
echo "Experiment complete: ${RUN_ROOT}"
