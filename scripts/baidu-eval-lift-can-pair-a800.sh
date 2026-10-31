#!/usr/bin/env bash
# Formal matched comparison: base 50 episodes, then aligned tactile 50 episodes.
set -euo pipefail
export TWAM_ROOT="${TWAM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PI05_SIM_GRAPHICS_LIBRARY_PATH="$TWAM_ROOT/.cache/sim-graphics-runtime/lib"
model_python="$TWAM_ROOT/contact-belief-wam/.venv/bin/python"
"$model_python" "$TWAM_ROOT/pi05-univtac/scripts/prepare-sim-graphics.py"
(
  source "$TWAM_ROOT/pi05-univtac/scripts/sim-env.sh"
  "$model_python" -u "$TWAM_ROOT/pi05-univtac/scripts/prepare-a800-native.py" --jobs "${A800_BUILD_JOBS:-2}"
)
export PI05_SIM_NATIVE_ROOT="$TWAM_ROOT/.cache/sim-a800-sm80"
export PATH="$TWAM_ROOT/.conda/univtac-sim/bin:$PATH"
exec "$model_python" -u "$TWAM_ROOT/twam/scripts/reeval_lift_can_a800.py" --root "$TWAM_ROOT" --gpu 0 "$@"
