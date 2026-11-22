#!/usr/bin/env bash
# Parallel multi-GPU runner + merge for the beat_block_hammer batch.
set -uo pipefail
ULA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
SEED_START=${SEED_START:-0}
EPISODES=${EPISODES:-100}
N=${N:-4}
GPU_LIST=${GPU_LIST:-0,1,2,3}
COMBINED="$OUT_ROOT/combined"
mkdir -p "$OUT_ROOT" "$COMBINED"
IFS=',' read -ra GPUS <<< "$GPU_LIST"
n=${#GPUS[@]}
[ "$N" -gt "$n" ] && N=$n
echo "[parallel] beat_block_hammer seeds $SEED_START..$((SEED_START+EPISODES-1)) across $N gpu(s): ${GPUS[*]:0:$N}"

pids=()
for ((i=0; i<N; i++)); do
  s=$((SEED_START + i*EPISODES/N))
  e=$((SEED_START + (i+1)*EPISODES/N))
  cnt=$((e-s))
  if [ "$cnt" -le 0 ]; then continue; fi
  g=${GPUS[$i]}
  part="$OUT_ROOT/part_$i"
  mkdir -p "$part"
  echo "[parallel] part $i: seed $s..$((e-1)) ($cnt) gpu $g"
  OUT="$part" SEED_START="$s" EPISODES="$cnt" GPU="$g" VISION_PRESET="${VISION_PRESET:-full}" \
    nohup bash "$ULA_ROOT/scripts/run_beat_block_hammer_ula_supervisor.sh" \
      > "$OUT_ROOT/part_$i.log" 2>&1 &
  pids+=($!)
done

for p in "${pids[@]}"; do wait "$p"; done
echo "[parallel] all parts done"

for ((i=0; i<N; i++)); do
  s=$((SEED_START + i*EPISODES/N))
  e=$((SEED_START + (i+1)*EPISODES/N))
  for ((k=s; k<e; k++)); do
    name=$(printf 'seed_%05d' "$k")
    d="$OUT_ROOT/part_$i/$name"
    [ -d "$d" ] && ln -sfn "$d" "$COMBINED/$name"
  done
done

/usr/bin/python3 - "$COMBINED" "$SEED_START" "$EPISODES" <<'PY'
import json, sys, pathlib
D = pathlib.Path(sys.argv[1]); start = int(sys.argv[2]); n = int(sys.argv[3])
eps = []
for i in range(start, start + n):
    p = D / ("seed_%05d" % i) / "report.json"
    r = json.loads(p.read_text()) if p.exists() else {}
    eps.append(dict(seed=i, status=r.get("status", "missing_report"),
        official_success=r.get("evaluation_only", {}).get("official_success"),
        error=r.get("error"), elapsed_s=r.get("elapsed_s")))
succ = sum(e["official_success"] is True for e in eps)
el = [e["elapsed_s"] for e in eps if e["official_success"] is True and e["elapsed_s"] is not None]
summary = dict(task="beat_block_hammer", seed_start=start, requested_episodes=n,
    completed_episodes=len(eps), official_successes=succ,
    official_success_rate=(succ/len(eps)) if eps else None,
    successful_mean_elapsed_s=(sum(el)/len(el)) if el else None,
    model_reuse=True, episodes=eps)
(D / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
from collections import Counter
print("=== COMBINED beat_block_hammer ===")
print("overall: %d/%d = %.1f%%" % (succ, len(eps), (succ/len(eps)*100) if eps else 0))
print("errors:", dict(Counter(e.get("status") for e in eps)))
PY
