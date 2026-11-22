#!/usr/bin/env bash
set -euo pipefail
ULA_PILL_INITIAL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_PILL_INITIAL_ROOT}/scripts/run_pillbottle_initial_randomized.py
export ULA_NO_TRIAL_VISION_GPU=${ULA_NO_TRIAL_VISION_GPU:-${PILLBOTTLE_RANDOM_VISION_GPU:-${GPU:?Select one shared GPU}}}
exec bash "${ULA_PILL_INITIAL_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
