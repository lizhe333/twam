#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(dirname -- "$project_dir")"
# The model environment has the compatible AnyTouch/Transformers stack. Native
# workers are launched separately through sim.sh by the campaign supervisor.
export PYTHONPATH="$project_dir/src:$workspace_dir/pi05-univtac/src:$workspace_dir/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec "$workspace_dir/contact-belief-wam/.venv/bin/python" -m contact_belief_wam.oracle_batch_campaign "$@"
