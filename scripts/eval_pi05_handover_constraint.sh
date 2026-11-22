#!/usr/bin/env bash
set -euo pipefail
ULA_CONSTRAINT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_CONSTRAINT_ROOT}/scripts/eval_pi05_handover_constraint.py
export HANDOVER_VISION_GPU=${HANDOVER_VISION_GPU:-${GPU:?Select one shared GPU}}
exec bash "${ULA_CONSTRAINT_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
