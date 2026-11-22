#!/usr/bin/env bash
set -euo pipefail

ULA_CLEAN9_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/xuhuixin/ula_clean9_pi05_20261113_v2}
mkdir -p "${OUTPUT_ROOT}/collections" "${OUTPUT_ROOT}/datasets" \
         "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/logs"

TASKS=(
  place_fan place_mouse_pad move_stapler_pad place_object_scale place_object_stand
  place_shoe place_container_plate place_empty_cup move_pillbottle_pad
)
SEEDS=(21000 22000 23000 24000 25000 26000 27000 28000 29000)

collect_one() {
  local task=$1 gpu=$2 seed=$3
  local output=${OUTPUT_ROOT}/collections/${task}_clean50
  local resume=()
  if [[ -f "${output}/report.json" ]]; then
    if "${ULA_CLEAN9_ROOT}/external/pi05/.venv/bin/python" -c \
      "import json; r=json.load(open('${output}/report.json')); raise SystemExit(0 if r.get('successful_episodes')==50 and r.get('run_status')=='finished' else 1)"; then
      return 0
    fi
    resume=(--resume --resume-after-compatible-source-change)
  fi
  GPU=${gpu} bash "${ULA_CLEAN9_ROOT}/scripts/collect_support_placement_clean.sh" \
    --task "${task}" --episodes 50 --seed "${seed}" --max-attempts 1000 \
    --stride 15 --output "${output}" "${resume[@]}" \
    > "${OUTPUT_ROOT}/logs/collect_${task}.log" 2>&1
}

# At most one simulator/vision process per requested GPU; only GPUs 0-3.
pids=()
for index in 0 1 2 3; do
  collect_one "${TASKS[$index]}" "${index}" "${SEEDS[$index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

pids=()
for index in 4 5 6 7; do
  gpu=$((index-4))
  collect_one "${TASKS[$index]}" "${gpu}" "${SEEDS[$index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

collect_one "${TASKS[8]}" 0 "${SEEDS[8]}"

for task in "${TASKS[@]}"; do
  dataset=${OUTPUT_ROOT}/datasets/${task}_clean50
  if [[ ! -f "${dataset}/dataset.json" ]]; then
    "${ULA_CLEAN9_ROOT}/external/pi05/.venv/bin/python" \
      "${ULA_CLEAN9_ROOT}/scripts/build_support_placement_dataset.py" \
      --collection "${OUTPUT_ROOT}/collections/${task}_clean50" \
      --output "${dataset}" --repo-id "ula_${task}_clean50_v1" \
      > "${OUTPUT_ROOT}/logs/build_${task}.log" 2>&1
  fi
done

# Train one independent model at a time on GPUs 0-3.
for task in "${TASKS[@]}"; do
  checkpoint=${OUTPUT_ROOT}/checkpoints/${task}_pi05_10000
  resume=()
  if [[ -f "${checkpoint}/protocol.json" ]]; then resume=(--resume); fi
  GPU_LIST=0,1,2,3 bash "${ULA_CLEAN9_ROOT}/scripts/train_pi05_handover_pickup.sh" \
    --dataset "${OUTPUT_ROOT}/datasets/${task}_clean50" \
    --output "${checkpoint}" --steps 10000 --batch-size 8 --fsdp-devices 4 \
    --validation-interval 1000 --save-interval 1000 "${resume[@]}" \
    > "${OUTPUT_ROOT}/logs/train_${task}.log" 2>&1
done

touch "${OUTPUT_ROOT}/PIPELINE_FINISHED"
