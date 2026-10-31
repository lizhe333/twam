#!/usr/bin/env bash
set -euo pipefail
OPENPI_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export OPENPI_ROOT="$OPENPI_PROJECT"
export TWAM_ROOT="${TWAM_ROOT:-$(dirname "$OPENPI_PROJECT")}"
export PYTHONPATH="$OPENPI_PROJECT/src:$OPENPI_PROJECT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
PI05_PYTHON="${PI05_PYTHON:-$TWAM_ROOT/.venvs/openpi/bin/python}"
exec "$PI05_PYTHON" "$OPENPI_PROJECT/scripts/train.py" --univtac "$@"
