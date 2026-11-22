#!/usr/bin/env bash
set -euo pipefail
ULA_KNOWN_CUP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_TARGET_FOCUS=1 ULA_CUP_VLA_FOCUS=0
# Reuse the established shared-GPU environment, replacing only the executable.
exec env ULA_CUP_RUNNER="${ULA_KNOWN_CUP_ROOT}/scripts/run_place_empty_cup_known_randomized.py" \
  bash "${ULA_KNOWN_CUP_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
