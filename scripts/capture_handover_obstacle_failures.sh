#!/usr/bin/env bash
set -euo pipefail
ULA_CAPTURE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_CAPTURE_ROOT}/scripts/capture_handover_obstacle_failures.py
export HANDOVER_VISION_GPU=${HANDOVER_VISION_GPU:-${GPU:?Select one shared GPU}}
exec bash "${ULA_CAPTURE_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
