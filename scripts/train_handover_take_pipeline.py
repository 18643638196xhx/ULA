#!/usr/bin/env python3
"""50 actual-prefix takeover demos -> disjoint LeRobot splits -> native pi0.5."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT/"external/pi05/.venv/bin/python"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--collection",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=50)
    parser.add_argument("--steps",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=31000)
    parser.add_argument("--simulation-gpu",default="7")
    parser.add_argument("--vision-gpu",default="6")
    parser.add_argument("--training-gpus",default="0,1,2,3")
    args=parser.parse_args()
    training_gpus=args.training_gpus.split(",")
    if len(training_gpus)!=4 or len(set(training_gpus))!=4 or args.episodes<2 or args.steps<1:
        parser.error("Need four distinct training GPUs and positive training budget")
    for path in (args.run_dir,args.collection,args.dataset,args.model):
        if path.exists() and any(path.iterdir()):
            parser.error(f"Use fresh output paths: {path}")
        if not path.resolve().is_relative_to(ROOT):
            parser.error("All outputs must remain under ULA")
    args.run_dir.mkdir(parents=True,exist_ok=True)
    protocol_path=args.run_dir/"protocol.json"
    protocol=dict(status="running",stage="starting",supervisor_pid=os.getpid(),
        task="handover_block",trained_stage="right_take_left_release_and_retreat",
        episodes=args.episodes,steps=args.steps,training_gpus=training_gpus,
        simulation_gpu=args.simulation_gpu,vision_gpu=args.vision_gpu,
        collection=str(args.collection),dataset=str(args.dataset),model=str(args.model),
        previous_left_model_modified=False,started_unix=time.time())
    write_json(protocol_path,protocol)
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",HF_HOME=str(ROOT/".runtime_cache/hf_home"),
        XDG_CACHE_HOME=str(ROOT/".runtime_cache/xdg"),
        OPENPI_DATA_HOME=str(ROOT/".runtime_cache/openpi"),
        TMPDIR=str(ROOT/".runtime_cache/tmp"))

    def run(stage, command, additions=None):
        protocol.update(stage=stage,child_pid=None,stage_started_unix=time.time())
        write_json(protocol_path,protocol)
        print(f"[take-pipeline] starting {stage}",flush=True)
        with (args.run_dir/f"{stage}.log").open("x") as log:
            process=subprocess.Popen([str(v) for v in command],cwd=ROOT,
                env=dict(env,**(additions or {})),stdout=log,stderr=subprocess.STDOUT)
            protocol["child_pid"]=process.pid
            write_json(protocol_path,protocol)
            try:
                while process.poll() is None:
                    protocol["heartbeat_unix"]=time.time()
                    write_json(protocol_path,protocol)
                    time.sleep(5)
            except BaseException:
                process.terminate()
                process.wait(timeout=30)
                raise
        protocol["child_returncode"]=process.returncode
        if process.returncode:
            raise RuntimeError(f"{stage} failed with exit {process.returncode}; inspect {args.run_dir/stage}.log")

    try:
        run("collection",["bash",ROOT/"scripts/collect_handover_take.sh",
            "--episodes",args.episodes,"--seed",args.seed,"--max-attempts",500,
            "--output",args.collection],
            dict(GPU=args.simulation_gpu,HANDOVER_VISION_GPU=args.vision_gpu))
        report=json.loads((args.collection/"report.json").read_text())
        if report["run_status"]!="finished" or not report["sources_unchanged"] or report["successful_episodes"]!=args.episodes:
            raise RuntimeError("Collection integrity/size check failed; refusing to train")
        run("conversion",[PYTHON,ROOT/"scripts/build_handover_take_dataset.py",
            "--collection",args.collection,"--output",args.dataset,
            "--repo-id","ula_handover_take50_v1"])
        # Recheck reservations when training actually starts, not only now.
        gpu_rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits"],text=True)
        usage={k.strip():int(v.strip()) for k,v in (line.split(",") for line in gpu_rows.splitlines())}
        if any(usage[gpu]>2048 for gpu in training_gpus):
            raise RuntimeError("A selected training GPU is occupied (>2 GiB); not starting a conflicting job")
        run("training",["bash",ROOT/"scripts/train_pi05_handover_pickup.sh",
            "--dataset",args.dataset,"--output",args.model,"--steps",args.steps,
            "--batch-size",8,"--fsdp-devices",4],dict(GPU_LIST=args.training_gpus))
        trained=json.loads((args.model/"protocol.json").read_text())
        if trained["status"]!="finished" or trained["completed_steps"]!=args.steps:
            raise RuntimeError("Training did not reach the requested optimizer step")
        protocol.update(status="finished",stage="finished",completed_steps=args.steps,
                        latest_checkpoint=trained["latest_checkpoint"])
    except BaseException as error:
        protocol.update(status="failed_or_interrupted",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        protocol["last_update_unix"]=time.time()
        write_json(protocol_path,protocol)


if __name__=="__main__":
    main()
