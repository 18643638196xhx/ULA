#!/usr/bin/env bash
set -euo pipefail
ULA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EXPECTED=${ULA_ROOT}/scripts/run_open_microwave_panel_randomized.py
if [[ ${1:-} != "$EXPECTED" ]]; then
  echo "Unexpected panel entry point" >&2
  exit 2
fi
shift
exec "${ULA_MICROWAVE_DIRECT_PY:?}" "${ULA_ROOT}/scripts/run_open_microwave_direct_randomized.py" "$@"
