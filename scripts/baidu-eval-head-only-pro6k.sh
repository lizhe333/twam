#!/usr/bin/env bash
# Two PRO 6000 Blackwell GPUs; eight tactile-only tasks, 50 episodes each.
set -euo pipefail
export TWAM_ROOT="${TWAM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
bootstrap_python="$TWAM_ROOT/contact-belief-wam/.venv/bin/python"
"$bootstrap_python" -u "$TWAM_ROOT/twam/scripts/prepare_pro6k.py" --root "$TWAM_ROOT" --check-only
"$bootstrap_python" -u "$TWAM_ROOT/twam/scripts/reeval_pro6k.py" --check-hardware --gpus 0,1
exec "$bootstrap_python" -u "$TWAM_ROOT/twam/scripts/reeval_pro6k.py" \
  --root "$TWAM_ROOT" --gpus 0,1 --per-gpu 4 "$@"
