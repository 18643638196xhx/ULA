#!/usr/bin/env bash
set -euo pipefail
ULA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} != "${ULA_ROOT}/scripts/run_open_microwave_panel_randomized.py" ]]; then
  echo "Unexpected panel entry point" >&2
  exit 2
fi
shift
exec "${ULA_MICROWAVE_NO_EXTENT_PY:?}" "${ULA_ROOT}/scripts/run_open_microwave_no_extent_randomized.py" "$@"
