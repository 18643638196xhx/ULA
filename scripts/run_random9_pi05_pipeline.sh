#!/usr/bin/env bash
set -euo pipefail

# End-to-end randomized collection -> conversion -> independent pi0.5 training.
# The pipeline deliberately waits for the three requested orig50 evaluations and
# refuses to consume partial collections/datasets.  Existing clean9 artifacts are
# never reused or modified.

ULA_RANDOM9_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROBOTWIN_RESULTS=${ULA_RANDOM9_ROOT}/tra_pipe/RoboTwin/results
OUTPUT_ROOT=${OUTPUT_ROOT:-/data2/xuhuixin/ula_random9_pi05_20261114_v1}
TASK_CONFIG=demo_randomized
PYTHON=${ULA_RANDOM9_ROOT}/external/pi05/.venv/bin/python
GPU_IDS=${GPU_IDS:-0,1}
WAIT_FOR_EVALUATIONS=${WAIT_FOR_EVALUATIONS:-1}
IFS=',' read -r -a PIPELINE_GPUS <<< "${GPU_IDS}"
if [[ ${#PIPELINE_GPUS[@]} -ne 2 ]] || [[ ${PIPELINE_GPUS[0]} == "${PIPELINE_GPUS[1]}" ]] \
    || [[ ! ${PIPELINE_GPUS[0]} =~ ^[0-9]+$ ]] || [[ ! ${PIPELINE_GPUS[1]} =~ ^[0-9]+$ ]]; then
  echo "GPU_IDS must contain two distinct physical GPU ids, for example GPU_IDS=1,3" >&2
  exit 2
fi

TASKS=(
  place_fan place_mouse_pad move_stapler_pad place_object_scale place_object_stand
  place_shoe place_container_plate place_empty_cup move_pillbottle_pad
)
SEEDS=(31000 32000 33000 34000 35000 36000 37000 38000 39000)
EVALUATIONS=(orig50_place_empty_cup orig50_place_shoe orig50_place_container_plate)

mkdir -p "${OUTPUT_ROOT}/collections" "${OUTPUT_ROOT}/datasets" \
         "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/logs"

log() {
  printf '[random9] %s %s\n' "$(date '+%F %T')" "$*"
}

result_count() {
  local directory=$1
  if [[ ! -d ${directory} ]]; then
    echo 0
    return 0
  fi
  find "${directory}" -type f -path '*/episode_*/result.json' 2>/dev/null | wc -l
}

wait_for_evaluations() {
  local active name directory count
  while true; do
    active=0
    for name in "${EVALUATIONS[@]}"; do
      if pgrep -f "run_ula_task.py .*--output results/${name}([[:space:]]|$)" >/dev/null; then
        active=1
      fi
    done
    if [[ ${active} -eq 0 ]]; then
      break
    fi
    log "waiting for orig50 evaluations to exit"
    sleep 30
  done

  for name in "${EVALUATIONS[@]}"; do
    directory=${ROBOTWIN_RESULTS}/${name}
    count=$(result_count "${directory}")
    if [[ ${count} -ne 50 ]]; then
      log "refusing to collect: ${name} has ${count}/50 result.json files"
      return 1
    fi
  done
  log "all three orig50 evaluations finished with 50 result files"
}

collection_complete() {
  local report=$1
  [[ -f ${report} ]] || return 1
  "${PYTHON}" -c \
    'import json,sys; r=json.load(open(sys.argv[1])); ok=(r.get("run_status")=="finished" and r.get("task_config")=="demo_randomized" and r.get("requested_successful_episodes")==100 and r.get("successful_episodes")==100 and len(r.get("successful_trajectories",[]))==100 and r.get("sources_unchanged") is True); raise SystemExit(0 if ok else 1)' \
    "${report}"
}

collect_one() {
  local task=$1 gpu=$2 seed=$3
  local output=${OUTPUT_ROOT}/collections/${task}_randomized100
  local args=()
  if collection_complete "${output}/report.json"; then
    log "collection already complete: ${task}"
    return 0
  fi
  if [[ -f ${output}/report.json ]]; then
    args=(--resume)
  elif [[ -d ${output} ]] && [[ -n $(find "${output}" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    log "refusing nonempty collection without report: ${output}"
    return 1
  fi
  log "collect start task=${task} gpu=${gpu} seed=${seed}"
  GPU=${gpu} ULA_TASK_CONFIG=${TASK_CONFIG} \
    bash "${ULA_RANDOM9_ROOT}/scripts/collect_support_placement_clean.sh" \
      --task "${task}" --task-config "${TASK_CONFIG}" \
      --episodes 100 --seed "${seed}" --max-attempts 10000 --stride 15 \
      --output "${output}" "${args[@]}" \
      > "${OUTPUT_ROOT}/logs/collect_${task}.log" 2>&1
  collection_complete "${output}/report.json" || {
    log "collection completeness check failed: ${task}"
    return 1
  }
  log "collect finished task=${task}"
}

collect_batch() {
  local begin=$1 end=$2 index gpu
  local pids=() labels=()
  for ((index=begin; index<end; index++)); do
    gpu=${PIPELINE_GPUS[$((index-begin))]}
    collect_one "${TASKS[$index]}" "${gpu}" "${SEEDS[$index]}" &
    pids+=("$!")
    labels+=("${TASKS[$index]}")
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      log "collection worker failed: ${labels[$index]}"
      return 1
    fi
  done
}

dataset_complete() {
  local metadata=$1
  [[ -f ${metadata} ]] || return 1
  "${PYTHON}" -c \
    'import json,sys,pathlib; d=json.load(open(sys.argv[1])); ok=(d.get("task_config")=="demo_randomized" and len(d.get("episodes",[]))==100 and d.get("splits",{}).get("train")==95 and d.get("splits",{}).get("val")==5 and pathlib.Path(d["dataset_root"]).is_dir()); raise SystemExit(0 if ok else 1)' \
    "${metadata}"
}

build_one() {
  local task=$1
  local collection=${OUTPUT_ROOT}/collections/${task}_randomized100
  local output=${OUTPUT_ROOT}/datasets/${task}_randomized100
  local staging=${output}.building.$$
  if dataset_complete "${output}/dataset.json"; then
    log "dataset already complete: ${task}"
    return 0
  fi
  collection_complete "${collection}/report.json" || {
    log "refusing conversion of incomplete collection: ${task}"
    return 1
  }
  if [[ -d ${output} ]] && [[ -n $(find "${output}" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    log "refusing incomplete nonempty dataset directory: ${output}"
    return 1
  fi
  log "conversion start task=${task}"
  "${PYTHON}" "${ULA_RANDOM9_ROOT}/scripts/build_support_placement_dataset.py" \
    --collection "${collection}" --output "${staging}" \
    --published-output "${output}" \
    --repo-id "ula_${task}_randomized100_v1" \
    > "${OUTPUT_ROOT}/logs/build_${task}.log" 2>&1
  if [[ ! -f ${staging}/dataset.json ]] || [[ ! -d ${staging}/lerobot ]]; then
    log "staged dataset completeness check failed: ${task}"
    return 1
  fi
  if [[ -d ${output} ]]; then
    rmdir "${output}"
  fi
  mv "${staging}" "${output}"
  dataset_complete "${output}/dataset.json" || {
    log "published dataset completeness check failed: ${task}"
    return 1
  }
  log "conversion finished task=${task}"
}

training_complete() {
  local protocol=$1
  [[ -f ${protocol} ]] || return 1
  "${PYTHON}" -c \
    'import json,sys,pathlib; p=json.load(open(sys.argv[1])); latest=p.get("latest_checkpoint"); ok=(p.get("status")=="finished" and p.get("completed_steps")==10000 and p.get("sources_unchanged") is True and latest and pathlib.Path(latest).is_dir()); raise SystemExit(0 if ok else 1)' \
    "${protocol}"
}

train_one() {
  local task=$1
  local dataset=${OUTPUT_ROOT}/datasets/${task}_randomized100
  local output=${OUTPUT_ROOT}/checkpoints/${task}_pi05_randomized_10000
  local args=()
  dataset_complete "${dataset}/dataset.json" || {
    log "refusing training with incomplete dataset: ${task}"
    return 1
  }
  if training_complete "${output}/protocol.json"; then
    log "training already complete: ${task}"
    return 0
  fi
  if [[ -f ${output}/protocol.json ]]; then
    args=(--resume)
  elif [[ -d ${output} ]] && [[ -n $(find "${output}" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    log "refusing nonempty checkpoint directory without protocol: ${output}"
    return 1
  fi
  log "training start task=${task} GPUs=${GPU_IDS}"
  GPU_LIST=${GPU_IDS} bash "${ULA_RANDOM9_ROOT}/scripts/train_pi05_handover_pickup.sh" \
    --dataset "${dataset}" --output "${output}" \
    --steps 10000 --batch-size 8 --fsdp-devices 2 \
    --validation-interval 1000 --save-interval 1000 "${args[@]}" \
    > "${OUTPUT_ROOT}/logs/train_${task}.log" 2>&1
  training_complete "${output}/protocol.json" || {
    log "training completeness check failed: ${task}"
    return 1
  }
  log "training finished task=${task}"
}

if [[ ${WAIT_FOR_EVALUATIONS} == 1 ]]; then
  wait_for_evaluations
else
  log "evaluation wait disabled by request"
fi

# Two independent collectors at a time on the selected physical GPUs. Training starts
# only after all nine collections and all nine conversions have passed checks.
collect_batch 0 2
collect_batch 2 4
collect_batch 4 6
collect_batch 6 8
collect_one "${TASKS[8]}" "${PIPELINE_GPUS[0]}" "${SEEDS[8]}"

for task in "${TASKS[@]}"; do
  build_one "${task}"
done

for task in "${TASKS[@]}"; do
  train_one "${task}"
done

touch "${OUTPUT_ROOT}/PIPELINE_FINISHED"
log "all randomized collections, conversions and trainings finished"
