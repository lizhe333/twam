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
WAVE0_BASE_PYTHON="${WAVE0_BASE_PYTHON:-python3.10}"
WAVE0_ENV="${WAVE0_ENV:-$WAVE0_PROJECT/.venv}"
"$WAVE0_BASE_PYTHON" -m venv "$WAVE0_ENV"
WAVE0_PYTHON="$WAVE0_ENV/bin/python"
# Same patched Transformers revision as the original WAVE-0 environment.
# Existing training environments need no reinstall; this script is opt-in setup.
if [[ -n "${WAVE0_WHEELHOUSE:-}" ]]; then
    "$WAVE0_PYTHON" -m pip install --no-index --find-links "$WAVE0_WHEELHOUSE" torch==2.7.1 torchvision==0.22.1
else
    "$WAVE0_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
fi
"$WAVE0_PYTHON" -m pip install -r "$WAVE0_PROJECT/requirements-wave0.txt"
"$WAVE0_PYTHON" -m pip install --no-deps --no-build-isolation -e "$WAVE0_PROJECT"
"$WAVE0_PYTHON" -m pip install h5py sentencepiece PyYAML safetensors
"$WAVE0_PYTHON" -m pip check
