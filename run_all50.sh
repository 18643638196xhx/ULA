#!/bin/bash
# 10 tasks x 50 episodes, 4 parallel workers.
cd /data1/xuhuixin/ula || exit 1
TASKS=(place_empty_cup move_pillbottle_pad place_fan place_mouse_pad
       move_stapler_pad place_shoe place_object_stand place_object_scale
       place_container_plate put_object_cabinet)
BASE=results/all50
mkdir -p "$BASE"
run_one() {
  local gpu=$1 task=$2
  local out="$BASE/$task"
  rm -rf "$out"
  ULA_PIPELINE=known_cad_unified ULA_SKIP_TRIAL_LIFT=1 GPU=$gpu \
  ULA_TASK_CONFIG=demo_randomized \
    bash scripts/run_ula_task.sh --task "$task" --mode hybrid \
      --episodes 50 --seed 0 --instruction-from-demo --output "$out" \
      > "$BASE/$task.log" 2>&1 < /dev/null
  echo "done $task" >> "$BASE/progress.txt"
}
for gpu in 0 1 2 3; do
  (
    for i in "${!TASKS[@]}"; do
      t=${TASKS[$i]}
      if (( i % 4 == gpu )); then run_one "$gpu" "$t"; fi
    done
  ) &
done
wait
echo "ALL DONE" > "$BASE/DONE"
