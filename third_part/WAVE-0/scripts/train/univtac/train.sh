#!/usr/bin/env bash
set -euo pipefail
WAVE0_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WAVE0_WORKSPACE="${TWAM_ROOT:-$WAVE0_PROJECT}"
while [[ ! -d "$WAVE0_WORKSPACE/UniVTAC" ]]; do
    if [[ "$WAVE0_WORKSPACE" == / ]]; then
        echo "Cannot locate tactile-wam workspace; set TWAM_ROOT." >&2
        exit 2
    fi
    WAVE0_WORKSPACE="$(dirname "$WAVE0_WORKSPACE")"
done
export TWAM_ROOT="${TWAM_ROOT:-$WAVE0_WORKSPACE}"
export WAVE0_ROOT="$WAVE0_PROJECT"
export PYTHONPATH="$WAVE0_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
WAVE0_PYTHON="${WAVE0_PYTHON:-$TWAM_ROOT/contact-belief-wam/.venv/bin/python}"
WAVE0_CONFIG=""
for ((i=1;i<=$#;i++)); do
    if [[ "${!i}" == "--config" ]]; then j=$((i+1)); WAVE0_CONFIG="${!j}"; fi
done
if [[ -z "$WAVE0_CONFIG" ]]; then
    echo 'Usage: scripts/train/univtac/train.sh --config configs/univtac/insert_hole.yaml --output PATH [--workspace PATH] [--resume]' >&2
    exit 2
fi
WAVE0_TASK="$("$WAVE0_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["task"])' "$WAVE0_CONFIG")"
if [[ "$WAVE0_TASK" == "shared_wm" ]]; then
    exec bash "$WAVE0_PROJECT/scripts/train/univtac/train_all.sh" "$@"
fi
WAVE0_GPUS="$("$WAVE0_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["num_gpus"])' "$WAVE0_CONFIG")"
"$WAVE0_PYTHON" "$WAVE0_PROJECT/scripts/debug/univtac/verify_assets.py" "$@"
"$WAVE0_PYTHON" -m lerobot.scripts.lerobot_train --univtac prepare --verify-hashes "$@"
for WAVE0_STAGE in stage1 stage2; do
    "$WAVE0_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$WAVE0_GPUS" \
        -m lerobot.scripts.lerobot_train --univtac train --stage "$WAVE0_STAGE" --device cuda "$@"
done
