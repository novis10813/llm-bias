#!/usr/bin/env bash
# Start the MAG7 8-K return-pair prompt-analysis workflow.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-.cache/models/qwen3.5-4b}"
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
cd "${REPO_ROOT}"
MODEL_SLUG="$(uv run python -c \
    'from llm_bias.core.lens_artifacts import model_slug; import sys; print(model_slug(sys.argv[1]))' \
    "${MODEL}")"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
LENS="${LENS:-${ARTIFACT_ROOT}/${MODEL_SLUG}/jacobian-lens/jacobian_lens.pt}"
INPUT_CSV="${INPUT_CSV:-mag7_8k_return_prompts.csv}"
DATASET_FORMAT="${DATASET_FORMAT:-return-pairs}"
DATASET="${DATASET:-mag7_8k_return_pairs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
READOUT_MAX_SEQ_LEN="${READOUT_MAX_SEQ_LEN:-512}"
# Zero means all pairs. Return-pairs expands each pair into original and counterfactual records.
GEN_SAMPLE_PER_CONDITION="${GEN_SAMPLE_PER_CONDITION:-0}"
RUN_READOUT="${RUN_READOUT:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_ATTRIBUTION="${RUN_ATTRIBUTION:-0}"
RUN_IN_TMUX="${RUN_IN_TMUX:-1}"
SESSION="${SESSION:-prompt_analysis}"

exec env \
    MODEL="${MODEL}" \
    MODEL_SLUG="${MODEL_SLUG}" \
    LENS="${LENS}" \
    INPUT_CSV="${INPUT_CSV}" \
    DATASET_FORMAT="${DATASET_FORMAT}" \
    ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    DATASET="${DATASET}" \
    RUN_ID="${RUN_ID}" \
    READOUT_MAX_SEQ_LEN="${READOUT_MAX_SEQ_LEN}" \
    GEN_SAMPLE_PER_CONDITION="${GEN_SAMPLE_PER_CONDITION}" \
    RUN_READOUT="${RUN_READOUT}" \
    RUN_GENERATION="${RUN_GENERATION}" \
    RUN_ATTRIBUTION="${RUN_ATTRIBUTION}" \
    RUN_IN_TMUX="${RUN_IN_TMUX}" \
    SESSION="${SESSION}" \
    bash "${SCRIPT_DIR}/run_prompt_analysis.sh"
