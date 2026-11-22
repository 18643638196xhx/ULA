#!/usr/bin/env bash
set -euo pipefail
ULA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROBOTWIN_ROOT=${ULA_ROOT}/tra_pipe/RoboTwin
PI_ROOT=${ULA_ROOT}/external/pi05
CACHE_ROOT=${ULA_ROOT}/.runtime_cache
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH=${CUDA_HOME}
export CUDACXX=${CUDA_HOME}/bin/nvcc
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/home/xuhuixin/miniconda3/envs/RoboTwin/lib:/usr/lib/x86_64-linux-gnu:/usr/lib
export PYTHONPATH=${ROBOTWIN_ROOT}/envs/curobo/src:${ULA_ROOT}:${ROBOTWIN_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export OPENPI_RUNTIME_CACHE_DIR=${CACHE_ROOT}
export OPENPI_DATA_HOME=${CACHE_ROOT}/openpi
export HF_HOME=${CACHE_ROOT}/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export JAX_COMPILATION_CACHE_DIR=${CACHE_ROOT}/jax
export TORCH_EXTENSIONS_DIR=${CACHE_ROOT}/torch_extensions
export MAX_JOBS=${MAX_JOBS:-4}
export WARP_CACHE_PATH=${CACHE_ROOT}/warp
export NUMBA_CACHE_DIR=${CACHE_ROOT}/numba
export MPLCONFIGDIR=${CACHE_ROOT}/matplotlib
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export TMPDIR=${CACHE_ROOT}/tmp
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export WORLDLA_SKIP_ACTION_RENDER=1
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${JAX_COMPILATION_CACHE_DIR}" "${OPENPI_DATA_HOME}" "${XDG_CACHE_HOME}/torch/kernels"
unset LD_PRELOAD
exec "${PI_PY:-${PI_ROOT}/.venv/bin/python}" "${ULA_ROOT}/scripts/run_adjust_bottle_ula.py" "$@"
