#!/usr/bin/env bash
set -euo pipefail
ULA_NATIVE_PICKUP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU_LIST:?Select available training GPUs}"
export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENPI_RUNTIME_CACHE_DIR=${ULA_NATIVE_PICKUP_ROOT}/.runtime_cache
export HF_HOME=${ULA_NATIVE_PICKUP_ROOT}/.runtime_cache/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME=${ULA_NATIVE_PICKUP_ROOT}/.runtime_cache/openpi
export XDG_CACHE_HOME=${ULA_NATIVE_PICKUP_ROOT}/.runtime_cache/xdg
export TMPDIR=${ULA_NATIVE_PICKUP_ROOT}/.runtime_cache/tmp
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
exec "${ULA_NATIVE_PICKUP_ROOT}/external/pi05/.venv/bin/python" "${ULA_NATIVE_PICKUP_ROOT}/scripts/train_pi05_handover_pickup.py" "$@"
