#!/usr/bin/env bash
set -euo pipefail
preview_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
preview_workspace="$(dirname -- "$preview_project")"
preview_output="$preview_project/data/insert_hole_perturb_preview_v1"
preview_gpu="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
exec 9>"$preview_workspace/outputs/eval_suites/.$preview_gpu.lock"
flock -n 9
export PYTHONPATH="$preview_project/src:$preview_workspace/pi05-univtac/src:$preview_workspace/openpi/packages/openpi-client/src"
export PI05_SIM_GRAPHICS_LIBRARY_PATH="$preview_workspace/.cache/sim-graphics-runtime/lib"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONUTF8=1
if [[ "${1:-}" == "retry" ]]; then
  bash "$preview_project/scripts/sim.sh" -m contact_belief_wam.sim_hole_preview_retry \
    --config "$preview_output/config.json" \
    --trial-result "$preview_output/run-01/trials/02_y300/result.json" \
    --output "$preview_output/oracle_retry-01" --headless --device cuda:0
elif [[ $# == 0 ]]; then
  bash "$preview_project/scripts/sim.sh" -m contact_belief_wam.sim_hole_preview \
    --config "$preview_output/config.json" --output "$preview_output/run-01" \
    --headless --device cuda:0
else
  exit 2
fi
