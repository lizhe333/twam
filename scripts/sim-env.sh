#!/usr/bin/env bash
# Source this before launching UniVTAC / Isaac Sim. All compiler and CUDA
# paths stay inside the workspace-local simulator environment.

_PI05_SIM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export TWAM_ROOT="${TWAM_ROOT:-$(cd -- "${_PI05_SIM_SCRIPT_DIR}/../.." && pwd -P)}"
export SIM_PREFIX="${SIM_PREFIX:-${TWAM_ROOT}/.conda/univtac-sim}"
export TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-${TWAM_ROOT}/.toolchains}"
export TOOLCHAIN_BIN="${TOOLCHAIN_ROOT}/bin"
# Keep the model subprocess independent of Isaac Sim's CUDA/Python libraries.
export PI05_MODEL_LD_LIBRARY_PATH="${PI05_MODEL_LD_LIBRARY_PATH-${LD_LIBRARY_PATH:-}}"

# Preserve the existing workspace data/cache setup used by the policy runner.
if [[ -f "${TWAM_ROOT}/pi05-univtac/scripts/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${TWAM_ROOT}/pi05-univtac/scripts/env.sh"
fi

# Route Kit's shared data/cache/log roots into this workspace. Kit reads
# omniverse.toml from OMNI_CONFIG_PATH, whose [paths] entries override defaults
# under the user's home directory. Keep this setup in the runtime environment
# as well as the installer because Kit may be started by a separate process.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_WORKSPACE_ROOT="${OMNI_WORKSPACE_ROOT:-${TWAM_ROOT}/.cache/omniverse}"
export OMNI_CONFIG_PATH="${OMNI_CONFIG_PATH:-${OMNI_WORKSPACE_ROOT}/config}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${TWAM_ROOT}/.cache/xdg-data}"
# NVIDIA supports EGL as the Vulkan ICD for containers without X11/GLX.
# Scope this to the simulator; leave the host driver and system manifest intact.
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES="${_PI05_SIM_SCRIPT_DIR}/nvidia-egl-icd.json"
fi
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${TWAM_ROOT}/.cache/xdg-runtime}"
mkdir -p -- "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
mkdir -p -- "${OMNI_CONFIG_PATH}" \
  "${OMNI_WORKSPACE_ROOT}/data" \
  "${OMNI_WORKSPACE_ROOT}/cache" \
  "${OMNI_WORKSPACE_ROOT}/logs" \
  "${XDG_DATA_HOME}"
cat > "${OMNI_CONFIG_PATH}/omniverse.toml" <<EOF
[paths]
data_root = "${OMNI_WORKSPACE_ROOT}/data"
cache_root = "${OMNI_WORKSPACE_ROOT}/cache"
logs_root = "${OMNI_WORKSPACE_ROOT}/logs"
EOF

_pi05_prepend_path() {
  local _path="$1"
  [[ -d "${_path}" ]] || return 0
  case ":${PATH:-}:" in
    *":${_path}:"*) ;;
    *) export PATH="${_path}${PATH:+:${PATH}}" ;;
  esac
}

_pi05_prepend_library_path() {
  local _path="$1"
  [[ -d "${_path}" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${_path}:"*) ;;
    *) export LD_LIBRARY_PATH="${_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

_pi05_prepend_path "${TOOLCHAIN_BIN}"
_pi05_prepend_path "${SIM_PREFIX}/bin"

export CUDA_HOME="${SIM_PREFIX}"
export CUDA_PATH="${SIM_PREFIX}"
export CUDACXX="${SIM_PREFIX}/bin/nvcc"
export CMAKE_PREFIX_PATH="${SIM_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
if [[ -z "${CC:-}" ]]; then
  if [[ -x "${SIM_PREFIX}/bin/gcc" ]]; then
    export CC="${SIM_PREFIX}/bin/gcc"
  else
    export CC="$(command -v gcc || true)"
  fi
fi
if [[ -z "${CXX:-}" ]]; then
  if [[ -x "${SIM_PREFIX}/bin/g++" ]]; then
    export CXX="${SIM_PREFIX}/bin/g++"
  else
    export CXX="$(command -v g++ || true)"
  fi
fi

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  _pi05_cuda_caps=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    _pi05_cuda_caps="$(
      nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | sed -nE 's/^[[:space:]]*([0-9]+\.[0-9]+)[[:space:]]*$/\1/p' \
        | awk '!seen[$0]++' \
        | paste -sd ';' -
    )"
  fi
  export TORCH_CUDA_ARCH_LIST="${_pi05_cuda_caps:-8.0;8.9}"
fi
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-${TORCH_CUDA_ARCH_LIST//./}}"

_pi05_prepend_library_path "${SIM_PREFIX}/lib"
_pi05_prepend_library_path "${SIM_PREFIX}/lib64"
_pi05_prepend_library_path "${SIM_PREFIX}/lib/python3.10/site-packages/torch/lib"
_pi05_prepend_library_path "${SIM_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
_pi05_prepend_library_path "${SIM_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib"
_pi05_prepend_library_path "${SIM_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib"

# Opt-in for thin cloud containers. Capture PI05_MODEL_LD_LIBRARY_PATH above
# before adding these simulator-only X11/GLU dependencies.
if [[ -n "${PI05_SIM_GRAPHICS_LIBRARY_PATH:-}" ]]; then
  _pi05_prepend_library_path "${PI05_SIM_GRAPHICS_LIBRARY_PATH}"
fi

unset _PI05_SIM_SCRIPT_DIR
unset _pi05_cuda_caps
unset -f _pi05_prepend_path _pi05_prepend_library_path
