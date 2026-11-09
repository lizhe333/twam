#!/usr/bin/env bash
# One PRO 6000 Blackwell GPU per tactile task; 50 episodes per task.
set -euo pipefail
export TWAM_ROOT="${TWAM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
bootstrap_python="$TWAM_ROOT/contact-belief-wam/.venv/bin/python"
eval_args=(--root "$TWAM_ROOT" --gpus 0,1,2,3,4,5,6,7 "$@" --per-gpu 1 --dedicated-task-per-gpu)
"$bootstrap_python" -u "$TWAM_ROOT/twam/scripts/prepare_pro6k.py" --root "$TWAM_ROOT" --check-only
"$bootstrap_python" -u "$TWAM_ROOT/twam/scripts/reeval_pro6k.py" --check-hardware "${eval_args[@]}"
exec "$bootstrap_python" -u "$TWAM_ROOT/twam/scripts/reeval_pro6k.py" "${eval_args[@]}"
