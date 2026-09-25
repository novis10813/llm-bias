#!/usr/bin/env bash
# Sequential confirmation-v1 jobs on one GPU (one model load per job).
# usage: scripts/run_confirmation_wave.sh GPU PHASE RUN_ID SLUG:ARM[,ARM...] [SLUG:ARM[,ARM...] ...]
#   e.g. scripts/run_confirmation_wave.sh 1 full confirmation-v1-20260926-full-01 gemma4-12b-it:tier1 gpt-oss-20b:tier1
# While one job runs, the next job's checkpoint is read into the page cache at idle I/O priority
# (the HDD is shared and slow; RAM is 60 GB, so only the next model is pre-warmed).
set -u
cd "$(dirname "$0")/.."
GPU=$1 PHASE=$2 RUN_ID=$3
shift 3
jobs=("$@")
kernel_cache=~/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels
for i in "${!jobs[@]}"; do
  IFS=: read -r SLUG ARMS <<< "${jobs[$i]}"
  next=${jobs[$((i + 1))]:-}
  if [ -n "$next" ]; then
    (ionice -c3 nice -n 19 cat .cache/models/"${next%%:*}"/*.safetensors > /dev/null) &
  fi
  offline=1
  if [ "$SLUG" = gpt-oss-20b ] && [ ! -d "$kernel_cache" ]; then
    offline=0   # first GPT-OSS run must fetch the MXFP4 triton kernels
  fi
  echo "==> START $SLUG $PHASE [$ARMS] $(date -Is)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=$offline uv run --no-sync python scripts/probe_steering_confirmation.py \
    --model .cache/models/"$SLUG" --phase "$PHASE" --run-id "$RUN_ID" --arms ${ARMS//,/ }
  echo "==> END $SLUG rc=$? $(date -Is)"
done
wait
