#!/usr/bin/env bash
set -euo pipefail
CBW_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CBW_PYTHON="${CBW_PYTHON:-$CBW_PROJECT/.venv/bin/python}"
export PYTHONPATH="$CBW_PROJECT/src:$CBW_PROJECT/vendor/wave0/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$CBW_PYTHON" -m contact_belief_wam.aligned_pipeline "$@"
