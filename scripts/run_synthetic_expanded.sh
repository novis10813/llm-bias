#!/usr/bin/env bash
set -euo pipefail

cd /home/sam/Projects/llm-bias

mkdir -p artifacts/logs

echo "========================================================"
echo "=== [$(date)] Starting Synthetic Entity Bias: Qwen 3.5-4B ==="
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 uv run synthetic-entity-bias run \
  --constituents data/sp500_constituents_2020_2025.csv \
  --constituents data/russell1000_constituents_2020_2025.csv \
  --constituents data/russell2000_constituents_2020_2025.csv \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt \
  --artifact-root artifacts \
  --dataset synthetic-entity-bias-2020-2025 \
  --run-id expanded-12templates-4b \
  --batch-size 16 2>&1 | tee artifacts/logs/synthetic-entity-bias-expanded-4b.log

echo "=== [$(date)] Running Statistical Analysis for Qwen 3.5-4B ==="
uv run synthetic-entity-bias analyze \
  --run-root artifacts/qwen3.5-4b/synthetic-entity-bias-2020-2025/runs/expanded-12templates-4b \
  --replace-existing

echo "=== [$(date)] Generating Visualizations for Qwen 3.5-4B ==="
uv run synthetic-entity-bias visualize \
  --run-root artifacts/qwen3.5-4b/synthetic-entity-bias-2020-2025/runs/expanded-12templates-4b \
  --replace-existing \
  --with-dashboard

echo "========================================================"
echo "=== [$(date)] Starting Synthetic Entity Bias: Qwen 3.6-27B ==="
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 uv run synthetic-entity-bias run \
  --constituents data/sp500_constituents_2020_2025.csv \
  --constituents data/russell1000_constituents_2020_2025.csv \
  --constituents data/russell2000_constituents_2020_2025.csv \
  --model .cache/models/qwen3.6-27b \
  --lens artifacts/qwen3.6-27b/jacobian-lens/jacobian_lens.pt \
  --artifact-root artifacts \
  --dataset synthetic-entity-bias-2020-2025 \
  --run-id expanded-12templates-27b \
  --batch-size 1 2>&1 | tee artifacts/logs/synthetic-entity-bias-expanded-27b.log

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
echo "=== [$(date)] All Qwen expanded runs and analyses finished successfully! ==="
echo "========================================================"
