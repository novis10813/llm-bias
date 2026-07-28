#!/usr/bin/env bash
# Build the Qwen Semantic Scope dashboard and uncertainty plots.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-.cache/models/qwen3.5-4b}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
INPUT_CSV="${INPUT_CSV:-sp500_r1k_r2k_entityBiasPrompt.csv}"
RUN_ROOT="${RUN_ROOT:-artifacts/qwen_jspace_experiment}"
ATTRIBUTION="${ATTRIBUTION:-${RUN_ROOT}/generated_attribution/generated_token_attribution.jsonl}"
UNCERTAINTY="${UNCERTAINTY:-${RUN_ROOT}/per_date/prompt_layer_uncertainty.jsonl}"
VALIDATION="${VALIDATION:-${RUN_ROOT}/no_validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/result_visualization}"
INPUT_TOP_K="${INPUT_TOP_K:-15}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"

command -v uv >/dev/null || {
    echo "uv is required" >&2
    exit 1
}

for required_file in "${INPUT_CSV}" "${ATTRIBUTION}" "${UNCERTAINTY}"; do
    [[ -f "${required_file}" ]] || {
        echo "Missing required file: ${required_file}" >&2
        echo "Set RUN_ROOT, ATTRIBUTION, or UNCERTAINTY if the experiment used custom paths." >&2
        exit 1
    }
done

[[ -d "${TOKENIZER}" ]] || {
    echo "Missing tokenizer/model directory: ${TOKENIZER}" >&2
    echo "Set TOKENIZER=/path/to/the/model if the model is stored elsewhere." >&2
    exit 1
}

# The visualizer currently discovers one combined uncertainty file through this
# compatibility directory name. The runner writes the same file under per_date/.
UNCERTAINTY_ALIAS_DIR="${RUN_ROOT}/qwen3.5_temperature_scope_per_date"
UNCERTAINTY_ALIAS="${UNCERTAINTY_ALIAS_DIR}/prompt_layer_uncertainty.jsonl"
UNCERTAINTY_ALIAS_TARGET="$(realpath --relative-to="${UNCERTAINTY_ALIAS_DIR}" "${UNCERTAINTY}")"
mkdir -p "${UNCERTAINTY_ALIAS_DIR}"

if [[ -e "${UNCERTAINTY_ALIAS}" || -L "${UNCERTAINTY_ALIAS}" ]]; then
    if [[ ! -L "${UNCERTAINTY_ALIAS}" ]] || \
        [[ "$(readlink "${UNCERTAINTY_ALIAS}")" != "${UNCERTAINTY_ALIAS_TARGET}" ]]; then
        echo "Refusing to replace existing file: ${UNCERTAINTY_ALIAS}" >&2
        exit 1
    fi
else
    ln -s "${UNCERTAINTY_ALIAS_TARGET}" "${UNCERTAINTY_ALIAS}"
fi

echo "Building Qwen result visualizations..."
echo "  uncertainty: ${UNCERTAINTY}"
echo "  Semantic Scope: ${ATTRIBUTION}"
echo "  output: ${OUTPUT_DIR}"

uv run python -m llm_bias visualize-qwen-results \
    --uncertainty-root "${RUN_ROOT}" \
    --attribution "${ATTRIBUTION}" \
    --validation "${VALIDATION}" \
    --prices "${INPUT_CSV}" \
    --tokenizer "${TOKENIZER}" \
    --input-top-k "${INPUT_TOP_K}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --output-dir "${OUTPUT_DIR}"

echo
echo "Visualization complete: ${OUTPUT_DIR}"
echo "Open: ${OUTPUT_DIR}/attribution_dashboard.html"
