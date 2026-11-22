#!/usr/bin/env bash
set -euo pipefail
ULA_SAFE_YAW_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_SAFE_YAW_ROOT}/scripts/eval_pi05_handover_safe_yaw.py
export HANDOVER_VISION_GPU=${HANDOVER_VISION_GPU:-${GPU:?Select a GPU}}
export CUDA_CACHE_PATH=${ULA_SAFE_YAW_ROOT}/.runtime_cache/cuda
mkdir -p "${CUDA_CACHE_PATH}"
exec bash "${ULA_SAFE_YAW_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
