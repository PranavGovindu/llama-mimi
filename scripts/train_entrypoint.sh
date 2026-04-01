#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_PROJECT="${WANDB_PROJECT:-tinyaya-tts-lab}"
export WANDB_DIR="${WANDB_DIR:-/cache/wandb}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${WANDB_DIR}/artifacts}"
export TORCH_HOME="${TORCH_HOME:-/cache/torch}"
export TMPDIR="${TMPDIR:-/cache/tmp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${WANDB_DIR}" "${WANDB_ARTIFACT_DIR}" "${TORCH_HOME}" "${TMPDIR}"

mode="${1:-train}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${mode}" in
  probe-ngc-stack)
    exec python /workspace/scripts/portable/probe_ngc_stack.py "$@"
    ;;
  probe-fa3)
    exec python /workspace/scripts/portable/probe_fa3_runtime.py "$@"
    ;;
  sync-dataset)
    exec python /workspace/scripts/sync_hf_dataset.py "$@"
    ;;
  train)
    nproc_per_node="${NPROC_PER_NODE:-}"
    if [[ -z "${nproc_per_node}" ]]; then
      nproc_per_node="$(python - <<'PY'
import torch
count = torch.cuda.device_count()
print(count if count > 0 else 1)
PY
)"
    fi
    exec torchrun \
      --nproc_per_node="${nproc_per_node}" \
      --nnodes="${NNODES:-1}" \
      --node_rank="${NODE_RANK:-0}" \
      --master_addr="${MASTER_ADDR:-127.0.0.1}" \
      --master_port="${MASTER_PORT:-29500}" \
      -m torchtitan.train "$@"
    ;;
  bash|sh|zsh)
    exec "${mode}" "$@"
    ;;
  *)
    exec "${mode}" "$@"
    ;;
esac
