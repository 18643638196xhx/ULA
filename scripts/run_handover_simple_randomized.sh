#!/usr/bin/env bash
set -euo pipefail
ULA_HANDOVER_SIMPLE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_HANDOVER_SIMPLE_ROOT}/scripts/run_handover_simple_randomized.py
export HANDOVER_VISION_GPU=${HANDOVER_VISION_GPU:-${GPU:?Select one shared GPU}}
exec bash "${ULA_HANDOVER_SIMPLE_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
