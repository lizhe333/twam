#!/usr/bin/env bash
set -euo pipefail
HDMI_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export CBWAM_ROOT="$HDMI_PROJECT"
export TWAM_ROOT="$(cd -- "$HDMI_PROJECT/.." && pwd -P)"
export CBW_PYTHON="${CBW_PYTHON:-$HDMI_PROJECT/../contact-belief-wam/.venv/bin/python}"
export PYTHONPATH="$HDMI_PROJECT/src:$HDMI_PROJECT/vendor/wave0/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
HDMI_COMMAND="${1:-train}"
if [ "$#" -gt 0 ]; then shift; fi
cd "$HDMI_PROJECT"
case "$HDMI_COMMAND" in
  train)
    exec bash scripts/train.sh train --config configs/finetune_insert_HDMI_4xa800.yaml --resume auto "$@"
    ;;
  plan)
    exec bash scripts/train.sh train --config configs/finetune_insert_HDMI_4xa800.yaml --plan "$@"
    ;;
  prepare)
    exec bash scripts/train.sh prepare --config configs/finetune_insert_HDMI_4xa800.yaml "$@"
    ;;
  check)
    exec "$CBW_PYTHON" -m contact_belief_wam.finetune_data --config configs/finetune_insert_HDMI_4xa800.yaml --check-loaders "$@"
    ;;
  *)
    echo 'Usage: bash scripts/finetune_hdmi.sh {plan|prepare|check|train}' >&2
    exit 2
    ;;
esac
