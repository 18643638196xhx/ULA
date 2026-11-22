#!/usr/bin/env bash
set -euo pipefail
ULA_PICKUP_COLLECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_PICKUP_COLLECT_ROOT}/scripts/collect_handover_pickup.py
export HANDOVER_VISION_GPU=${HANDOVER_VISION_GPU:-${GPU:?Select one shared GPU}}
exec bash "${ULA_PICKUP_COLLECT_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
