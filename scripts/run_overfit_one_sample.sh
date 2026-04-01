#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set."
  echo "Export it first, then rerun:"
  echo "  export WANDB_API_KEY='<your_key>'"
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-tinyaya-tts-lab}"
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=1 -m torchtitan.train \
  --job.config_file config/tinyaya_q1_fleurs_overfit_1sample_strict.toml
