#!/usr/bin/env bash
set -euo pipefail
CBW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export TWAM_ROOT="${TWAM_ROOT:-$(dirname "$CBW_ROOT")}"
source "$CBW_ROOT/scripts/sim-env.sh"
export PYTHONPATH="$CBW_ROOT/src:$PYTHONPATH"
exec "$SIM_PREFIX/bin/python" "$@"
