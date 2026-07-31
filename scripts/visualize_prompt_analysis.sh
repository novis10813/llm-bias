#!/usr/bin/env bash
# Build prompt uncertainty plots and a Semantic Scope attribution dashboard.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# By default this consumes the artifacts produced by the documented prompt-analysis
# workflow. Set RUN_ROOT to use a separate runner output directory.
RUN_ROOT="${RUN_ROOT:-artifacts/prompt_analysis/qwen3.5-4b}"
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

# Prefer the Semantic Scope artifact used by the dashboard. Fall back to the
# portable runner's generated-attribution output when necessary.
resolve_optional_file ATTRIBUTION \
    "${RUN_ROOT}/generated_attribution/generated_token_attribution.jsonl"

resolve_optional_file VALIDATION \
    "${RUN_ROOT}/attribution_validation/semantic_scope_aopc.jsonl"

combined_uncertainty=""
for candidate in \
    "${UNCERTAINTY_ROOT}/per_date/prompt_layer_uncertainty.jsonl" \
    "${UNCERTAINTY_ROOT}/prompt_layer_uncertainty.jsonl"; do
    if [[ -f "${candidate}" ]]; then
        combined_uncertainty="${candidate}"
        break
    fi
done
if [[ -z "${combined_uncertainty}" ]]; then
    echo "No combined uncertainty file found below: ${UNCERTAINTY_ROOT}" >&2
    echo "Expected per_date/ or a prompt_layer_uncertainty.jsonl at the root." >&2
    exit 1
fi

if [[ -z "${ATTRIBUTION}" || ! -f "${ATTRIBUTION}" ]]; then
    echo "Missing generated-token attribution JSONL." >&2
    echo "Set ATTRIBUTION=/path/to/generated_token_attribution.jsonl" >&2
    exit 1
fi

if [[ -n "${VALIDATION}" && ! -f "${VALIDATION}" ]]; then
    echo "Validation file does not exist; continuing without validation: ${VALIDATION}" >&2
    VALIDATION=""
fi

echo "Building prompt-analysis visualizations..."
echo "  uncertainty root: ${UNCERTAINTY_ROOT}"
echo "  Semantic Scope: ${ATTRIBUTION}"
if [[ -n "${VALIDATION}" ]]; then
    echo "  validation: ${VALIDATION}"
else
    echo "  validation: disabled"
fi
echo "  output: ${OUTPUT_DIR}"

visualize_args=(
    uv run prompt-analysis visualize
    --uncertainty-root "${UNCERTAINTY_ROOT}"
    --attribution "${ATTRIBUTION}"
    --prices "${INPUT_CSV}"
    --tokenizer "${TOKENIZER}"
    --input-top-k "${INPUT_TOP_K}"
    --max-seq-len "${MAX_SEQ_LEN}"
    --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${VALIDATION}" ]]; then
    visualize_args+=(--validation "${VALIDATION}")
fi

"${visualize_args[@]}"

echo
echo "Visualization complete: ${OUTPUT_DIR}"
echo "Open: ${OUTPUT_DIR}/attribution_dashboard.html"
