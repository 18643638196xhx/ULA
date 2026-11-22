#!/usr/bin/env bash
set -euo pipefail
ULA_CUP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ULA_CUP_CACHE=${ULA_CUP_ROOT}/.runtime_cache
: "${GPU:?Select one shared GPU}"
export CUDA_VISIBLE_DEVICES=${GPU}
export CUDA_HOME=/usr/local/cuda-12.4 CUDA_PATH=/usr/local/cuda-12.4
export CUDACXX=${CUDA_HOME}/bin/nvcc PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/home/xuhuixin/miniconda3/envs/RoboTwin/lib:/usr/lib/x86_64-linux-gnu:/usr/lib
export PYTHONPATH=${ULA_CUP_ROOT}/tra_pipe/RoboTwin/envs/curobo/src:${ULA_CUP_ROOT}:${ULA_CUP_ROOT}/tra_pipe/RoboTwin
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_CLIENT_MEM_FRACTION=0.40
export OPENPI_RUNTIME_CACHE_DIR=${ULA_CUP_CACHE} OPENPI_DATA_HOME=${ULA_CUP_CACHE}/openpi
export HF_HOME=${ULA_CUP_CACHE}/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JAX_COMPILATION_CACHE_DIR=${ULA_CUP_CACHE}/jax TORCH_EXTENSIONS_DIR=${ULA_CUP_CACHE}/torch_extensions
export MAX_JOBS=${MAX_JOBS:-4} WARP_CACHE_PATH=${ULA_CUP_CACHE}/warp NUMBA_CACHE_DIR=${ULA_CUP_CACHE}/numba
export MPLCONFIGDIR=${ULA_CUP_CACHE}/matplotlib XDG_CACHE_HOME=${ULA_CUP_CACHE}/xdg TMPDIR=${ULA_CUP_CACHE}/tmp
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json WORLDLA_SKIP_ACTION_RENDER=1
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${JAX_COMPILATION_CACHE_DIR}" "${OPENPI_DATA_HOME}" "${XDG_CACHE_HOME}/torch/kernels"
unset LD_PRELOAD XLA_PYTHON_CLIENT_MEM_FRACTION WORLDLA_RENDER_GPU
exec "${PI_PY:-${ULA_CUP_ROOT}/external/pi05/.venv/bin/python}" "${ULA_CUP_RUNNER:-${ULA_CUP_ROOT}/scripts/run_place_empty_cup_randomized.py}" "$@"
