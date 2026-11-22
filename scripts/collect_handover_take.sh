#!/usr/bin/env bash
set -euo pipefail
ULA_TAKE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_TAKE_ROOT}/scripts/collect_handover_take.py
export HANDOVER_VISION_GPU=${HANDOVER_VISION_GPU:-${GPU:?Select one shared GPU}}
exec bash "${ULA_TAKE_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
