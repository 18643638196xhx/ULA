#!/usr/bin/env bash
set -euo pipefail
ULA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} != "${ULA_ROOT}/scripts/run_move_pillbottle_pad_known_model.py" ]]; then
  echo "Unexpected known-bottle entry point" >&2
  exit 2
fi
shift
exec "${ULA_PILLBOTTLE_RANDOM_PY:?}" "${ULA_ROOT}/scripts/run_move_pillbottle_pad_known_randomized.py" "$@"
