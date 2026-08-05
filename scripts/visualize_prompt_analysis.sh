#!/usr/bin/env bash
# Build prompt uncertainty plots and a Semantic Scope attribution dashboard.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# By default this consumes the artifacts produced by the documented prompt-analysis
# workflow. Set RUN_ROOT to use a separate runner output directory.
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to artifacts/<model-slug>/<dataset-slug>/runs/<run-id>}"
UNCERTAINTY_ROOT="${UNCERTAINTY_ROOT:-${RUN_ROOT}}"
MODEL="${MODEL:-.cache/models/qwen3.5-4b}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
INPUT_CSV="${INPUT_CSV:-sp500_r1k_r2k_entityBiasPrompt.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/visualization}"
INPUT_TOP_K="${INPUT_TOP_K:-15}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"

command -v uv >/dev/null || {
    echo "uv is required" >&2
    exit 1
}

for required_file in "${INPUT_CSV}"; do
    [[ -f "${required_file}" ]] || {
        echo "Missing required file: ${required_file}" >&2
        echo "Set INPUT_CSV if the experiment used a custom input file." >&2
        exit 1
    }
done

[[ -d "${TOKENIZER}" ]] || {
    echo "Missing tokenizer/model directory: ${TOKENIZER}" >&2
    echo "Set TOKENIZER=/path/to/the/model if the model is stored elsewhere." >&2
    exit 1
}

resolve_optional_file() {
    local variable_name="$1"
    shift
    if [[ -n "${!variable_name:-}" ]]; then
        return 0
    fi
    local candidate
    for candidate in "$@"; do
        if [[ -f "${candidate}" ]]; then
            printf -v "${variable_name}" '%s' "${candidate}"
            return 0
        fi
    done
    printf -v "${variable_name}" '%s' ""
}

# The dashboard always reads forward generated outputs. A backward artifact is
# optional and only enables the attribution panel after parent-hash validation.
resolve_optional_file FORWARD \
    "${RUN_ROOT}/forward/generated_outputs.jsonl"

resolve_optional_file BACKWARD \
    "${RUN_ROOT}/backward/generated_token_attribution.jsonl"

resolve_optional_file VALIDATION \
    "${RUN_ROOT}/attribution_validation/semantic_scope_aopc.jsonl"

combined_uncertainty=""
for candidate in \
    "${UNCERTAINTY_ROOT}/readout/prompt_layer_uncertainty.jsonl" \
    "${UNCERTAINTY_ROOT}/prompt_layer_uncertainty.jsonl"; do
    if [[ -f "${candidate}" ]]; then
        combined_uncertainty="${candidate}"
        break
    fi
done
if [[ -z "${combined_uncertainty}" ]]; then
    echo "No combined uncertainty file found below: ${UNCERTAINTY_ROOT}" >&2
    echo "Expected readout/prompt_layer_uncertainty.jsonl or a prompt_layer_uncertainty.jsonl at the root." >&2
    exit 1
fi

if [[ -z "${FORWARD}" || ! -f "${FORWARD}" ]]; then
    echo "Missing forward generated-output JSONL." >&2
    echo "Set FORWARD=/path/to/forward/generated_outputs.jsonl" >&2
    exit 1
fi

if [[ -n "${BACKWARD}" && ! -f "${BACKWARD}" ]]; then
    echo "Backward artifact does not exist; attribution panel disabled: ${BACKWARD}" >&2
    BACKWARD=""
fi

echo "Building prompt-analysis visualizations..."
echo "  uncertainty root: ${UNCERTAINTY_ROOT}"
echo "  forward outputs: ${FORWARD}"
if [[ -n "${BACKWARD}" ]]; then
    echo "  backward attribution: ${BACKWARD}"
    if [[ -n "${VALIDATION}" ]]; then
        echo "  attribution validation: ${VALIDATION}"
    fi
else
    echo "  attribution panel: disabled"
fi
echo "  output: ${OUTPUT_DIR}"

visualize_args=(
    uv run prompt-analysis visualize
    --uncertainty-root "${UNCERTAINTY_ROOT}"
    --forward "${FORWARD}"
    --prices "${INPUT_CSV}"
    --tokenizer "${TOKENIZER}"
    --input-top-k "${INPUT_TOP_K}"
    --max-seq-len "${MAX_SEQ_LEN}"
    --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${BACKWARD}" ]]; then
    visualize_args+=(--backward "${BACKWARD}")
    if [[ -n "${VALIDATION}" ]]; then
        visualize_args+=(--validation "${VALIDATION}")
    fi
fi

"${visualize_args[@]}"

echo
echo "Visualization complete: ${OUTPUT_DIR}"
echo "Open: ${OUTPUT_DIR}/attribution_dashboard.html"
