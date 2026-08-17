#!/usr/bin/env bash
set -euo pipefail

cd /home/sam/Projects/llm-bias

mkdir -p artifacts/logs

echo "========================================================"
echo "=== [$(date)] Starting Synthetic Entity Bias: Qwen 3.6-27B (two-GPU sharded) ==="
echo "========================================================"

# GPU-only sharded loading: layers 0-31 -> GPU 0, layers 32-63 + norm + lm_head -> GPU 1.
# Budgets must fit within the free memory of each GPU at launch time.
uv run synthetic-entity-bias run \
  --constituents data/sp500_constituents_2020_2025.csv \
  --constituents data/russell1000_constituents_2020_2025.csv \
  --constituents data/russell2000_constituents_2020_2025.csv \
  --model .cache/models/qwen3.6-27b \
  --lens artifacts/qwen3.6-27b/jacobian-lens/jacobian_lens.pt \
  --artifact-root artifacts \
  --dataset synthetic-entity-bias-2020-2025 \
  --run-id expanded-12templates-27b \
  --batch-size 1 \
  --device-map qwen27b_two_gpu \
  --max-memory-json '{"0": "30GiB", "1": "30GiB"}' 2>&1 | tee artifacts/logs/synthetic-entity-bias-expanded-27b.log

echo "=== [$(date)] Running Statistical Analysis for Qwen 3.6-27B ==="
uv run synthetic-entity-bias analyze \
  --run-root artifacts/qwen3.6-27b/synthetic-entity-bias-2020-2025/runs/expanded-12templates-27b \
  --replace-existing

echo "=== [$(date)] Generating Visualizations for Qwen 3.6-27B ==="
uv run synthetic-entity-bias visualize \
  --run-root artifacts/qwen3.6-27b/synthetic-entity-bias-2020-2025/runs/expanded-12templates-27b \
  --replace-existing \
  --with-dashboard

echo "========================================================"
echo "=== [$(date)] Qwen 3.6-27B sharded run finished successfully! ==="
echo "========================================================"
