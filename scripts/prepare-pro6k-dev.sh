#!/usr/bin/env bash
# Prepare shared runtime on the development machine; never launch evaluation.
set -euo pipefail
export TWAM_ROOT="${TWAM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
bootstrap_python="$TWAM_ROOT/contact-belief-wam/.venv/bin/python"
runtime_home="$TWAM_ROOT/.cache/pro6k-isaac51-transfer-v1"
mkdir -p "$runtime_home"
"$bootstrap_python" -u "$TWAM_ROOT/twam/scripts/prepare_pro6k.py" --root "$TWAM_ROOT" \
  --jobs "${PRO6K_BUILD_JOBS:-4}" 2>&1 | tee -a "$runtime_home/dev-install.log"
"$bootstrap_python" -u "$TWAM_ROOT/twam/scripts/prepare_pro6k.py" --root "$TWAM_ROOT" --check-only
