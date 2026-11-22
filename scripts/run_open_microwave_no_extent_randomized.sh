#!/usr/bin/env bash
set -euo pipefail
ULA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ULA_MICROWAVE_NO_EXTENT_PY=${PI_PY:-${ULA_ROOT}/external/pi05/.venv/bin/python}
export PI_PY=${ULA_ROOT}/scripts/microwave_no_extent_python.sh
exec bash "${ULA_ROOT}/scripts/run_open_microwave_panel_randomized.sh" "$@"
