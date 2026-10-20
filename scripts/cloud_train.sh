#!/usr/bin/env bash
set -euo pipefail
CBW_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CBW_PYTHON=python3.10
CBW_BOOTSTRAP=true
CBW_WHEELS="$CBW_PROJECT/wheelhouse"
CBW_CONFIG=""
CBW_ARGUMENTS=("$@")
while (($#)); do
  case "$1" in
    --python) CBW_PYTHON="$2"; shift 2 ;;
    --bootstrap-env) CBW_BOOTSTRAP="$2"; shift 2 ;;
    --wheelhouse) CBW_WHEELS="$2"; shift 2 ;;
    --run-config) CBW_CONFIG="$2"; shift 2 ;;
    *) shift ;;
  esac
done
# A resume configuration was created by this project environment, so prefer it
# when available. The complete YAML is resolved by cloud.py after bootstrapping.
if [[ -n "$CBW_CONFIG" && -x "$CBW_PROJECT/.venv/bin/python" ]]; then
  CBW_PYTHON="$CBW_PROJECT/.venv/bin/python"
  CBW_BOOTSTRAP=false
fi
if [[ "$CBW_BOOTSTRAP" == true ]]; then
  CBW_INSTALL_OPTIONS=()
  if [[ -f "$CBW_WHEELS/complete.json" ]]; then
    CBW_INSTALL_OPTIONS+=(--offline)
  fi
  bash "$CBW_PROJECT/scripts/bootstrap_env.sh" --python "$CBW_PYTHON" \
    --venv "$CBW_PROJECT/.venv" --wheelhouse "$CBW_WHEELS" "${CBW_INSTALL_OPTIONS[@]}"
  CBW_PYTHON="$CBW_PROJECT/.venv/bin/python"
elif [[ -x "$CBW_PROJECT/.venv/bin/python" ]]; then
  CBW_PYTHON="$CBW_PROJECT/.venv/bin/python"
fi
export PYTHONPATH="$CBW_PROJECT/src:$CBW_PROJECT/vendor/wave0/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "$CBW_PYTHON" -m contact_belief_wam.cloud "${CBW_ARGUMENTS[@]}"
