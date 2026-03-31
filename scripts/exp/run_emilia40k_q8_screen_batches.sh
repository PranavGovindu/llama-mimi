#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="codecs/mimi/configs/tinyaya_mimi_q8_s4096_emilia40k_en.toml"
MODAL_PATH="mimi/ablation_emilia40k_q8_s4096_en"
CAMPAIGN_ID="${CAMPAIGN_ID:-emilia40k-wave1}"
PHASE="mimi_q8_s4096_emilia40k_en_ablation"
TRACK="english_only"
FAMILY="mimi_q8_s4096"
OWNER="${OWNER:-codex}"
CODEC="mimi"
QUESTION="Which English-only Q8/4096 settings improve held-out TTS quality on the frozen Emilia-40k subset?"
DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d-%H%M%S)}"

usage() {
  cat <<'EOF'
Usage:
  scripts/exp/run_emilia40k_q8_screen_batches.sh batch1
  scripts/exp/run_emilia40k_q8_screen_batches.sh batch2
  scripts/exp/run_emilia40k_q8_screen_batches.sh batch3
  scripts/exp/run_emilia40k_q8_screen_batches.sh all
  scripts/exp/run_emilia40k_q8_screen_batches.sh dry-run batch1

Notes:
  - Every command launches one detached Modal app.
  - Each Modal app uses one GPU only (`nproc_per_node=1` in modal/app.py::train).
  - No DDP is used here.
EOF
}

run_launch() {
  local experiment_id="$1"
  local variant="$2"
  local axis="$3"
  local hypothesis="$4"
  shift 4

  python scripts/exp/launch.py \
    --mode modal \
    --config "${CONFIG}" \
    --modal-path "${MODAL_PATH}" \
    --experiment-id "${experiment_id}" \
    --campaign-id "${CAMPAIGN_ID}" \
    --phase "${PHASE}" \
    --stage screen5k \
    --variant "${variant}" \
    --track "${TRACK}" \
    --axis "${axis}" \
    --family "${FAMILY}" \
    --question "${QUESTION}" \
    --hypothesis "${hypothesis}" \
    --owner "${OWNER}" \
    --codec "${CODEC}" \
    --tags "tinyaya,tts,mimi,q8,4096,emilia40k,english,screen5k,${axis},${variant}" \
    "$@"
}

batch1() {
  run_launch \
    "exp-${DATE_TAG}-screen5k-anchor-s0" \
    "anchor" \
    "anchor" \
    "Baseline config for all English-only Q8/4096 comparisons." \
    --override training.seed=0 \
    --override training.deterministic=false

  run_launch \
    "exp-${DATE_TAG}-screen5k-anchor-s1" \
    "anchor" \
    "anchor" \
    "Baseline config for all English-only Q8/4096 comparisons." \
    --override training.seed=1 \
    --override training.deterministic=false

  run_launch \
    "exp-${DATE_TAG}-screen5k-lr-2e-5-s0" \
    "lr_2e-5" \
    "lr" \
    "Lower learning-rate challenger." \
    --override training.seed=0 \
    --override training.deterministic=false \
    --override optimizer.lr=2e-5

  run_launch \
    "exp-${DATE_TAG}-screen5k-lr-1e-4-s0" \
    "lr_1e-4" \
    "lr" \
    "Higher learning-rate challenger." \
    --override training.seed=0 \
    --override training.deterministic=false \
    --override optimizer.lr=1e-4
}

batch2() {
  run_launch \
    "exp-${DATE_TAG}-screen5k-gbs-2-s0" \
    "gbs_2" \
    "global_batch_size" \
    "Smaller effective update size." \
    --override training.seed=0 \
    --override training.deterministic=false \
    --override training.global_batch_size=2

  run_launch \
    "exp-${DATE_TAG}-screen5k-gbs-8-s0" \
    "gbs_8" \
    "global_batch_size" \
    "Larger effective update size." \
    --override training.seed=0 \
    --override training.deterministic=false \
    --override training.global_batch_size=8

  run_launch \
    "exp-${DATE_TAG}-screen5k-maxsec-12-s0" \
    "maxsec_12" \
    "max_audio_seconds" \
    "Shorter audio crop challenger." \
    --override training.seed=0 \
    --override training.deterministic=false \
    --override training.max_audio_seconds=12

  run_launch \
    "exp-${DATE_TAG}-screen5k-maxsec-20-s0" \
    "maxsec_20" \
    "max_audio_seconds" \
    "Longer audio crop challenger." \
    --override training.seed=0 \
    --override training.deterministic=false \
    --override training.max_audio_seconds=20
}

batch3() {
  run_launch \
    "exp-${DATE_TAG}-screen5k-warmup-2pct-s0" \
    "warmup_2pct" \
    "warmup_ratio" \
    "Short warmup challenger." \
    --override training.seed=0 \
    --override training.deterministic=false \
    --override lr_scheduler.warmup_steps=100

  run_launch \
    "exp-${DATE_TAG}-screen5k-warmup-8pct-s0" \
    "warmup_8pct" \
    "warmup_ratio" \
    "Long warmup challenger." \
    --override training.seed=0 \
    --override training.deterministic=false \
    --override lr_scheduler.warmup_steps=400
}

DRY_RUN=0
if [[ "${1:-}" == "dry-run" ]]; then
  DRY_RUN=1
  shift
fi

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  usage
  exit 1
fi

cd "${REPO_ROOT}"
source .venv/bin/activate

if [[ "${DRY_RUN}" == "1" ]]; then
  run_launch() {
    local experiment_id="$1"
    local variant="$2"
    local axis="$3"
    local hypothesis="$4"
    shift 4
    printf 'python scripts/exp/launch.py --mode modal --config %q --modal-path %q --experiment-id %q --campaign-id %q --phase %q --stage screen5k --variant %q --track %q --axis %q --family %q --question %q --hypothesis %q --owner %q --codec %q --tags %q' \
      "${CONFIG}" "${MODAL_PATH}" "${experiment_id}" "${CAMPAIGN_ID}" "${PHASE}" "${variant}" "${TRACK}" "${axis}" "${FAMILY}" "${QUESTION}" "${hypothesis}" "${OWNER}" "${CODEC}" "tinyaya,tts,mimi,q8,4096,emilia40k,english,screen5k,${axis},${variant}"
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
  }
fi

case "${MODE}" in
  batch1)
    batch1
    ;;
  batch2)
    batch2
    ;;
  batch3)
    batch3
    ;;
  all)
    batch1
    batch2
    batch3
    ;;
  *)
    usage
    exit 1
    ;;
esac
