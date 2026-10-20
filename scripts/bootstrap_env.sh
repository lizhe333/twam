#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_PYTHON="python3.10"
VENV_PATH="$PROJECT_ROOT/.venv"
WHEELHOUSE="$PROJECT_ROOT/wheelhouse"
OFFLINE=0
while (($#)); do
    case "$1" in
        --python)
            BASE_PYTHON="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --wheelhouse)
            WHEELHOUSE="$2"
            shift 2
            ;;
        --offline)
            OFFLINE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--python PYTHON] [--venv PATH] [--wheelhouse PATH] [--offline]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

mkdir -p reports "$WHEELHOUSE"
exec > >(tee -a reports/environment-bootstrap.log) 2>&1

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-12}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}"
export PIP_RETRIES="${PIP_RETRIES:-12}"
export UV_INDEX_URL="https://pypi.org/simple"
export PIP_INDEX_URL="https://pypi.org/simple"
PYTHON="$VENV_PATH/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$BASE_PYTHON" "$VENV_PATH"
    else
        "$BASE_PYTHON" -m venv "$VENV_PATH"
    fi
fi

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "$VENV_PATH/bin/uv" ]]; then
    UV_BIN="$VENV_PATH/bin/uv"
elif ((OFFLINE)); then
    "$PYTHON" -m pip install --no-index --find-links "$WHEELHOUSE" uv
    UV_BIN="$VENV_PATH/bin/uv"
else
    "$PYTHON" -m pip install --index-url https://pypi.org/simple uv
    UV_BIN="$VENV_PATH/bin/uv"
fi

"$PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Expected Python 3.10, found {sys.version.split()[0]}")
PY

# Use the official PyTorch CUDA 12.6 index first. PyPI is only the fallback for
# CUDA component wheels and non-PyTorch dependencies; no machine-default mirror.
if [[ ! -f vendor/transformers/setup.py ]]; then
    if ((OFFLINE)); then
        echo "Prepare vendor/transformers first: python scripts/prepare_transformers.py --archive PATH" >&2
        exit 1
    fi
    "$PYTHON" scripts/prepare_transformers.py
fi
if ((OFFLINE)); then
    "$UV_BIN" pip install --python "$PYTHON" --offline --find-links "$WHEELHOUSE" \
        torch==2.7.1 torchvision==0.22.1
else
    "$UV_BIN" pip install \
        --python "$PYTHON" \
        --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple \
        torch==2.7.1 torchvision==0.22.1
fi

# Keep all remaining dependencies on official PyPI and avoid the repository's
# unrelated robot/policy extras. The patched Transformers and WAVE-0 source
# checkouts are built locally below, then installed from project wheelhouse.
if ((OFFLINE)); then
    if [[ -s "$WHEELHOUSE/requirements-resolved.txt" ]]; then
        RUNTIME_REQUIREMENTS="$WHEELHOUSE/requirements-resolved.txt"
    else
        RUNTIME_REQUIREMENTS="requirements-wave0-runtime.txt"
    fi
    "$UV_BIN" pip install --python "$PYTHON" --offline --find-links "$WHEELHOUSE" -r "$RUNTIME_REQUIREMENTS"
else
    "$UV_BIN" pip install --python "$PYTHON" --index-url https://pypi.org/simple -r requirements-wave0-runtime.txt
fi

LOCAL_BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wave0-local-wheels.XXXXXX")"
trap 'rm -rf "$LOCAL_BUILD_DIR"' EXIT
cp -a vendor/transformers "$LOCAL_BUILD_DIR/transformers"
cp -a vendor/wave0 "$LOCAL_BUILD_DIR/wave0"
mkdir -p "$LOCAL_BUILD_DIR/contact-belief-wam"
cp pyproject.toml "$LOCAL_BUILD_DIR/contact-belief-wam/pyproject.toml"
cp -a src "$LOCAL_BUILD_DIR/contact-belief-wam/src"
"$PYTHON" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$WHEELHOUSE" \
    "$LOCAL_BUILD_DIR/transformers" "$LOCAL_BUILD_DIR/wave0" "$LOCAL_BUILD_DIR/contact-belief-wam"
shopt -s nullglob
LOCAL_WHEELS=(
    "$WHEELHOUSE"/transformers-4.53.3*.whl
    "$WHEELHOUSE"/lerobot-*.whl
    "$WHEELHOUSE"/contact_belief_wam-*.whl
)
if ((${#LOCAL_WHEELS[@]} != 3)); then
    echo "Expected local wheels for patched Transformers, WAVE-0, and contact-belief-wam" >&2
    exit 1
fi
"$UV_BIN" pip install --python "$PYTHON" --no-deps "${LOCAL_WHEELS[@]}"

# Save exact resolved runtime pins and every runtime wheel for a Python 3.10
# Linux offline install. Local source projects are built into separate wheels.
"$PYTHON" -m pip freeze --all | sort > reports/environment-pip-freeze.txt
if ((!OFFLINE)); then
    UV_VERSION="$("$UV_BIN" --version | awk '{print $2}')"
    "$PYTHON" - "$UV_VERSION" <<'PY'
from pathlib import Path
import sys

version = sys.argv[1]
freeze = Path("reports/environment-pip-freeze.txt").read_text().splitlines()
resolved = [
    line.strip()
    for line in freeze
    if line.strip() and not line.lstrip().startswith("-e ") and " @ file:" not in line
    and line.partition("==")[0].lower().replace("_", "-") not in {
        "transformers", "lerobot", "contact-belief-wam"
    }
]
if not any(line.lower().startswith("uv==") for line in resolved):
    resolved.append(f"uv=={version}")
Path("reports/environment-resolved.txt").write_text("\n".join(sorted(set(resolved))) + "\n")
PY
    cp reports/environment-resolved.txt "$WHEELHOUSE/requirements-resolved.txt"
    # evdev has no PyPI wheel for Linux/Python 3.10, so build the small local
    # wheel first and let the binary-only download step consume it from here.
    "$PYTHON" -m pip wheel \
        --disable-pip-version-check \
        --no-deps \
        --no-build-isolation \
        --wheel-dir "$WHEELHOUSE" \
        --index-url https://pypi.org/simple \
        evdev==1.9.3
    "$PYTHON" -m pip download \
        --disable-pip-version-check \
        --only-binary=:all: \
        --no-deps \
        --dest "$WHEELHOUSE" \
        --find-links "$WHEELHOUSE" \
        --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple \
        -r reports/environment-resolved.txt
fi
if [[ -d "$WHEELHOUSE" ]]; then
    find "$WHEELHOUSE" -maxdepth 1 -type f -print0 | sort -z | xargs -0 -r sha256sum > reports/environment-wheelhouse.sha256
fi

# Import smoke check only; this does not allocate or execute on a GPU.
"$PYTHON" - <<'PY'
import torch
import transformers
from lerobot.policies.pi05 import modeling_pi05

assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
assert transformers.__version__ == "4.53.3", transformers.__version__
assert modeling_pi05.PaliGemmaForConditionalGeneration is not None
print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
print(f"transformers={transformers.__version__} source=vendor/transformers")
print("lerobot PI0.5 patched Gemma/SigLIP imports: OK")
PY
