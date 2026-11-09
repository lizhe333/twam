#!/usr/bin/env bash
#SBATCH --job-name=wave0-shared-8tasks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=wave0-%j.log
set -euo pipefail

# Submit from WAVE-0/; pass site-specific --partition/--account to sbatch.
# Example: sbatch scripts/train/univtac/train_all.sbatch --output /scratch/wave0/run1
# CUDA_VISIBLE_DEVICES is assigned by Slurm and must not be overwritten here.
WAVE0_PROJECT="${WAVE0_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the wave0-univtac directory}}"
WAVE0_WORKSPACE="$WAVE0_PROJECT"
while [[ ! -d "$WAVE0_WORKSPACE/UniVTAC" ]]; do
    if [[ "$WAVE0_WORKSPACE" == / ]]; then
        echo "Cannot locate tactile-wam workspace; set TWAM_ROOT." >&2
        exit 2
    fi
    WAVE0_WORKSPACE="$(dirname "$WAVE0_WORKSPACE")"
done
export TWAM_ROOT="${TWAM_ROOT:-$WAVE0_WORKSPACE}"
cd "$WAVE0_PROJECT"
exec bash scripts/train/univtac/train_all.sh --config configs/univtac/shared_wm_8xa800.yaml "$@"
