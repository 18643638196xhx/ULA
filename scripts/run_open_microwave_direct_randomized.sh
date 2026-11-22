#!/usr/bin/env bash
set -euo pipefail
ULA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Reuse the frozen panel launch environment, replacing only its Python entry.
export ULA_MICROWAVE_DIRECT_PY=${PI_PY:-${ULA_ROOT}/external/pi05/.venv/bin/python}
export PI_PY=${ULA_ROOT}/scripts/microwave_direct_python.sh
exec bash "${ULA_ROOT}/scripts/run_open_microwave_panel_randomized.sh" "$@"
