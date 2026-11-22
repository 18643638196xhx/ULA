#!/usr/bin/env bash
set -euo pipefail
ULA_BOTTLE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_BOTTLE_ROOT}/scripts/run_move_pillbottle_pad_cup_method.py
export ULA_NO_TRIAL_VISION_GPU=${ULA_NO_TRIAL_VISION_GPU:-${PILLBOTTLE_RANDOM_VISION_GPU:-${GPU:?Select one shared GPU}}}
export ULA_CUP_TARGET_FOCUS=1 ULA_CUP_VLA_FOCUS=0
exec bash "${ULA_BOTTLE_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
