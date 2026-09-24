#!/usr/bin/env bash
# Cross-model Phase 2 (2A/2B) model downloads for the layer-localization sweep.
# Target slugs match .cache/models/<slug> convention used by load_model().
set -u
cd "$(dirname "$0")/../.."
CACHE_DIR=".cache/models"

# slug:repo_id
JOBS=(
  "glm4-9b-0414:zai-org/GLM-4-9B-0414"
  "hunyuan-7b-instruct:tencent/Hunyuan-7B-Instruct"
  "mimo-7b-rl:XiaomiMiMo/MiMo-7B-RL"
  "gemma4-12b-it:google/gemma-4-12B-it"
  "phi-4:microsoft/phi-4"
  "gpt-oss-20b:openai/gpt-oss-20b"
)

download_one() {
  local slug="$1" repo="$2"
  local out="$CACHE_DIR/$slug"
  if [ -f "$out/config.json" ]; then
    echo "[$slug] already present, skipping"
    return 0
  fi
  echo "[$slug] downloading $repo -> $out"
  if hf download "$repo" --local-dir "$out" 2>&1; then
    echo "[$slug] DONE"
  else
    echo "[$slug] FAILED"
    return 1
  fi
}

for job in "${JOBS[@]}"; do
  slug="${job%%:*}"
  repo="${job#*:}"
  download_one "$slug" "$repo" || echo "[$slug] will retry on rerun"
done
echo "ALL DOWNLOADS COMPLETE"
