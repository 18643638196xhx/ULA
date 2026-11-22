#!/usr/bin/env bash
set -euo pipefail
ULA_FOCUS_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_TARGET_FOCUS=1
export ULA_CUP_VLA_FOCUS=${CUP_VLA_FOCUS:-0}
exec bash "${ULA_FOCUS_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
