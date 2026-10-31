#!/usr/bin/env bash
set -euo pipefail
mixed_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mixed_workspace="$(dirname -- "$mixed_project")"
export PYTHONPATH="$mixed_project/src:$mixed_workspace/pi05-univtac/src:$mixed_workspace/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$mixed_workspace/contact-belief-wam/.venv/bin/python" -m contact_belief_wam.mixed_recovery_campaign "$@"
