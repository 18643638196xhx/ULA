#!/usr/bin/env bash
set -euo pipefail
ULA_PI05_FULL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_PI05_FULL_ROOT}/scripts/eval_pi05_handover_full.py
export HANDOVER_VISION_GPU=${HANDOVER_VISION_GPU:-${GPU:?Select one shared GPU}}
exec bash "${ULA_PI05_FULL_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
