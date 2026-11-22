#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/xuhuixin/ula
cd "$ROOT"

if [[ -z "${GPU:-}" ]]; then
  echo "Set GPU=<physical GPU index>" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export ULA_GPU=0
exec "$ROOT/external/pi05/.venv/bin/python" \
  "$ROOT/perception_cad_search/run_move_pillbottle_cad_search.py" "$@"
