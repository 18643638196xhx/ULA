#!/usr/bin/env bash
set -euo pipefail
ULA_AUDIT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_CUP_RUNNER=${ULA_AUDIT_ROOT}/scripts/audit_handover_obstacle_snapshots.py
exec bash "${ULA_AUDIT_ROOT}/scripts/run_place_empty_cup_randomized.sh" "$@"
