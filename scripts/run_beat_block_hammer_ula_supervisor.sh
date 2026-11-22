#!/usr/bin/env bash
# Supervisor for the beat_block_hammer batch (resumable runner + crash restart).
set -uo pipefail
ULA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:?set OUT to the output directory}
SEED_START=${SEED_START:-0}
EPISODES=${EPISODES:-100}
MAX_RETRIES=${MAX_RETRIES:-40}
SLEEP_S=${SLEEP_S:-15}

count_terminal() {
  /usr/bin/python3 - "$OUT" "$SEED_START" "$EPISODES" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); start = int(sys.argv[2]); n = int(sys.argv[3])
done = 0
for i in range(start, start + n):
    p = out / f"seed_{i:05d}" / "report.json"
    if not p.exists():
        continue
    try:
        r = json.loads(p.read_text())
    except Exception:
        continue
    if r.get("status") in ("finished", "error"):
        done += 1
print(done)
PY
}

attempt=0
while :; do
  done=$(count_terminal)
  if [ "$done" -ge "$EPISODES" ]; then
    echo "[supervisor] complete: $done/$EPISODES"
    break
  fi
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "[supervisor] giving up after $MAX_RETRIES attempts (done=$done/$EPISODES)"
    exit 1
  fi
  attempt=$((attempt+1))
  echo "[supervisor] attempt $attempt done=$done/$EPISODES $(date '+%F %T')"
  GPU=${GPU:-0} VISION_PRESET=${VISION_PRESET:-full} bash "$ULA_ROOT/scripts/run_beat_block_hammer_ula.sh" \
    --seed "$SEED_START" --episodes "$EPISODES" --gpu "${GPU:-0}" --output "$OUT"
  rc=$?
  echo "[supervisor] runner exit=$rc done=$(count_terminal)/$EPISODES $(date '+%F %T')"
  if [ "$rc" -eq 0 ] && [ "$(count_terminal)" -ge "$EPISODES" ]; then break; fi
  sleep "$SLEEP_S"
done
