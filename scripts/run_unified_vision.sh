#!/usr/bin/env bash
set -euo pipefail
ULA_UNIFIED_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Select a GPU}"
ULA_UNIFIED_CACHE=${ULA_UNIFIED_ROOT}/.runtime_cache
ULA_UNIFIED_ENV=${ULA_UNIFIED_ROOT}/tra_pipe/RoboTwin/.conda_envs/RoboTwin_yzy
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=${ULA_UNIFIED_ROOT}:${ULA_UNIFIED_ROOT}/tra_pipe/RoboTwin
export LD_LIBRARY_PATH=${ULA_UNIFIED_ENV}/lib:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME=${ULA_UNIFIED_CACHE}/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=${ULA_UNIFIED_CACHE}/tmp MPLCONFIGDIR=${ULA_UNIFIED_CACHE}/matplotlib
export XDG_CACHE_HOME=${ULA_UNIFIED_CACHE}/xdg TORCH_EXTENSIONS_DIR=${ULA_UNIFIED_CACHE}/torch_extensions
export CUDA_CACHE_PATH=${ULA_UNIFIED_ROOT}/.runtime_cache/cuda
export WARP_CACHE_PATH=${ULA_UNIFIED_ROOT}/.runtime_cache/warp_vision
mkdir -p "${CUDA_CACHE_PATH}" "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
exec "${ULA_UNIFIED_ENV}/bin/python" "${ULA_UNIFIED_ROOT}/scripts/run_unified_vision.py" "$@"
