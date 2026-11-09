#!/usr/bin/env bash
# Build/reuse A800 native libraries, then run the formal 50-episode evaluation.
set -euo pipefail
export TWAM_ROOT="${TWAM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export PYTHONDONTWRITEBYTECODE=1
(
  source "$TWAM_ROOT/pi05-univtac/scripts/sim-env.sh"
  "$TWAM_ROOT/contact-belief-wam/.venv/bin/python" -u \
    "$TWAM_ROOT/pi05-univtac/scripts/prepare-a800-native.py" --jobs "${A800_BUILD_JOBS:-2}"
)
export PI05_SIM_NATIVE_ROOT="$TWAM_ROOT/.cache/sim-a800-sm80"
exec bash "$TWAM_ROOT/twam/scripts/baidu-eval-head-only.sh" \
  --suite "$TWAM_ROOT/outputs/eval_suites/baidu-tactile-head-only-mb16-003-eval50-sm80" "$@"
