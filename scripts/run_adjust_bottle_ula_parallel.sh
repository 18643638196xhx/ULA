#!/usr/bin/env bash
# Parallel multi-GPU runner + merge for the adjust_bottle batch.
# Splits [SEED_START, SEED_START+EPISODES) across N GPUs, runs one resumable
# supervisor per GPU, then merges per-seed reports into one combined summary.
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
echo "[parallel] seeds $SEED_START..$((SEED_START+EPISODES-1)) across $N gpu(s): ${GPUS[*]:0:$N}"

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
  OUT="$part" SEED_START="$s" EPISODES="$cnt" GPU="$g" \
    nohup bash "$ULA_ROOT/scripts/run_adjust_bottle_ula_supervisor.sh" \
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
        error=r.get("error"), elapsed_s=r.get("elapsed_s"),
        active_arm=r.get("active_arm"), known_model=r.get("known_model")))
succ = sum(e["official_success"] is True for e in eps)
el = [e["elapsed_s"] for e in eps if e["official_success"] is True and e["elapsed_s"] is not None]
summary = dict(task="adjust_bottle", seed_start=start, requested_episodes=n,
    completed_episodes=len(eps), official_successes=succ,
    official_success_rate=(succ/len(eps)) if eps else None,
    successful_mean_elapsed_s=(sum(el)/len(el)) if el else None,
    model_reuse=True, episodes=eps)
(D / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
unstable = [e for e in eps if (e.get("error") or "").startswith("UnStableError")]
valid = [e for e in eps if not (e.get("error") or "").startswith("UnStableError")]
sv = [e for e in valid if e["official_success"] is True]
print("=== COMBINED 100-round ===")
print("overall: %d/%d = %.1f%%" % (succ, len(eps), (succ/len(eps)*100) if eps else 0))
print("init-fail(excluded): %d" % len(unstable))
print("effective: %d/%d = %.1f%%" % (len(sv), len(valid), (len(sv)/len(valid)*100) if valid else 0))
PY
