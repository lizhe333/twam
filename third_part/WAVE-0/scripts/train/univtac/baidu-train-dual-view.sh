#!/usr/bin/env bash
# Same eight-A800 recipe as gpu-003, with the EMA shadow disabled.
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
export WAVE0_PYTHON="${WAVE0_PYTHON:-$TWAM_ROOT/contact-belief-wam/.venv/bin/python}"
WAVE0_OUTPUT="$TWAM_ROOT/outputs/wave0-shared-8task-head-wrist-noema-001"
WAVE0_CONFIG="$WAVE0_PROJECT/configs/univtac/shared_wm_head_wrist_8xa800_noema.yaml"
WAVE0_PLAN=false
WAVE0_ARGS=()
for WAVE0_ARG in "$@"; do
    case "$WAVE0_ARG" in
        --plan) WAVE0_PLAN=true; WAVE0_ARGS+=(--plan) ;;
        --resume) WAVE0_ARGS+=(--resume) ;;
        *) echo "Usage: bash $0 [--plan] [--resume]" >&2; exit 2 ;;
    esac
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}"
export WAVE0_ROOT="$WAVE0_PROJECT"
export PYTHONPATH="$WAVE0_PROJECT/src"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
unset PYTHONHOME
if [[ ! -x "$WAVE0_PYTHON" ]]; then
    echo "WAVE-0 Python unavailable: $WAVE0_PYTHON. Use the existing training image, or set WAVE0_PYTHON." >&2
    exit 2
fi
WAVE0_TEACHER="$("$WAVE0_PYTHON" "$WAVE0_PROJECT/scripts/debug/univtac/selected-dual-view-teacher.py" "$TWAM_ROOT")"
WAVE0_ARGS+=(--stage1-checkpoint "$WAVE0_TEACHER")
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for WAVE0_PATH in "$WAVE0_CONFIG" "$WAVE0_PROJECT/assets/univtac/baseline" "$WAVE0_PROJECT/src"; do
    if [[ ! -e "$WAVE0_PATH" ]]; then
        echo "Missing shared workspace path: $WAVE0_PATH. Mount the complete tactile-wam workspace, including WAVE-0." >&2
        exit 2
    fi
done
cd "$WAVE0_PROJECT"
# Always reuse the completed dual-view teacher and initialize a new Stage 2 run.
"$WAVE0_PYTHON" -c 'import sys; from lerobot.configs.univtac import load_config; r=load_config(sys.argv[1], workspace=sys.argv[2], output=sys.argv[3]); assert r["camera_mode"] == "head_wrist" and r["task"] == "shared_wm" and r["ema_enabled"] is False, "Expected the dual-view recipe with EMA disabled"' \
    "$WAVE0_CONFIG" "$TWAM_ROOT" "$WAVE0_OUTPUT"
if ! "$WAVE0_PLAN"; then
    IFS=',' read -r -a WAVE0_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    if [[ ${#WAVE0_GPUS[@]} != 8 ]]; then
        echo "This WAVE-0 job requires eight visible A800 80GB GPUs; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        exit 2
    fi
    mkdir -p "$TWAM_ROOT/outputs/cloud-training-logs"
    exec > >(tee -a "$TWAM_ROOT/outputs/cloud-training-logs/${WAVE0_OUTPUT##*/}.log") 2>&1
    echo "Reusing the completed head+wrist Stage 1: $WAVE0_TEACHER"
    echo "EMA is disabled. AdamW and its two-rank GPU state sharding remain unchanged."
    echo "Stage 2: micro batch 8, accumulation 4. New run noema-001."
    echo "Stage 2: two GPUs/task, four tasks concurrently, two waves; batch 64, 10000 updates/task."
    echo "Progress: $WAVE0_OUTPUT/pipeline_status.json"
    echo "Loss logs: $WAVE0_OUTPUT/logs/"
fi
exec bash "$WAVE0_PROJECT/scripts/train/univtac/train_all.sh" \
    --config "$WAVE0_CONFIG" --workspace "$TWAM_ROOT" --output "$WAVE0_OUTPUT" "${WAVE0_ARGS[@]}"
