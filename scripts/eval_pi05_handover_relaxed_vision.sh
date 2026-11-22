#!/usr/bin/env bash
set -euo pipefail
ULA_RELAXED_VISION_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_RELAXED_VISION_ROOT}/scripts/eval_pi05_handover_relaxed_vision.py
export HANDOVER_VISION_GPU=${HANDOVER_VISION_GPU:-${GPU:?Select one shared GPU}}
exec bash "${ULA_RELAXED_VISION_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
