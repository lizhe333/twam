#!/usr/bin/env bash
# Single-GPU formal head-only tactile evaluation: 50 fixed-seed episodes per task.
set -euo pipefail

export TWAM_ROOT="${TWAM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export PATH="$TWAM_ROOT/.conda/univtac-sim/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PI05_SIM_GRAPHICS_LIBRARY_PATH="$TWAM_ROOT/.cache/sim-graphics-runtime/lib"

model_python="$TWAM_ROOT/contact-belief-wam/.venv/bin/python"
"$model_python" "$TWAM_ROOT/pi05-univtac/scripts/prepare-sim-graphics.py"

exec "$model_python" -u "$TWAM_ROOT/twam/scripts/watch_eval_10k.py" \
  --root "$TWAM_ROOT" \
  --run-root "$TWAM_ROOT/twam/runs/head-only-expert-only-8task-all-train-fast-mb16-003/full" \
  --suite "$TWAM_ROOT/outputs/eval_suites/baidu-tactile-head-only-mb16-003-eval50" \
  --gpu 0 \
  --episodes 50 \
  --skip-driver-version-check \
  --skip-sim-check \
  "$@"
