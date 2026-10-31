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

exec "$WAVE0_PYTHON" -u -m lerobot.scripts.univtac_pipeline "$@"
