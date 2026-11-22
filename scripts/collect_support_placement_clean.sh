#!/usr/bin/env bash
set -euo pipefail
ULA_CLEAN_COLLECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${GPU:?Select one physical collection GPU}"
export ULA_CUP_RUNNER=${ULA_CLEAN_COLLECT_ROOT}/scripts/collect_support_placement_clean.py
export ULA_TASK_CONFIG=${ULA_TASK_CONFIG:-demo_clean}
# The worker is a separate subprocess and sets its own CUDA_VISIBLE_DEVICES,
# so it needs the selected physical id rather than the parent's logical id.
export ULA_GPU=${GPU}
exec bash "${ULA_CLEAN_COLLECT_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
