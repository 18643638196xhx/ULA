#!/usr/bin/env python3
"""Detached, logged native pi0.5 takeover training using the existing dataset.

Does not modify the previous run or the left-pickup trainer/model. A fresh
session with closed terminal descriptors avoids dependence on the launch shell.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def training_command(dataset, output, steps=10000):
    return ["bash",str(ROOT/"scripts/train_pi05_handover_pickup.sh"),
        "--dataset",str(dataset),"--output",str(output),"--steps",str(steps),
        "--batch-size","8","--fsdp-devices","4",
        "--save-interval","250","--validation-interval","1000"]


def validate_dataset(dataset):
    metadata=json.loads((dataset/"dataset.json").read_text())
    if len(metadata["episodes"])!=50 or metadata["splits"]!={"train":45,"val":5}:
        raise ValueError("Expected the existing 50 demonstrations with 45/5 episode split")
    if "right arm" not in metadata["instruction"] or "left gripper" not in metadata["instruction"]:
        raise ValueError("Wrong training stage instruction")
    train={row["seed"] for row in metadata["episodes"] if row["split"]=="train"}
    validation={row["seed"] for row in metadata["episodes"] if row["split"]=="val"}
    if train & validation:
        raise ValueError("Train/validation scene overlap")
    for row in metadata["episodes"]:
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=ROOT/"datasets/handover_take50_20261107_v1")
    parser.add_argument("--output",type=Path,default=ROOT/"checkpoints/pi05_handover_take50_10000_4gpu_20261107_v2")
    parser.add_argument("--run-dir",type=Path,default=ROOT/"results/pi05_handover_take_restart_20261107_v2")
    parser.add_argument("--gpus",default="0,1,2,3")
    parser.add_argument("--steps",type=int,default=10000)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.dataset,args.output,args.run_dir=(p.resolve() for p in (args.dataset,args.output,args.run_dir))
    gpus=args.gpus.split(",")
    if len(gpus)!=4 or len(set(gpus))!=4 or args.steps<1:
        parser.error("Select four distinct GPUs and a positive step budget")
    for path in (args.output,args.run_dir):
        if not path.is_relative_to(ROOT):
            parser.error("Outputs must remain under ULA")
    if not args.worker:
        validate_dataset(args.dataset)
        for path in (args.output,args.run_dir):
            if path.exists():
                parser.error(f"Preserve existing files; select a fresh directory: {path}")
        rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits"],text=True)
        usage={k.strip():int(v.strip()) for k,v in (line.split(",") for line in rows.splitlines())}
        if any(gpu not in usage or usage[gpu]>2048 for gpu in gpus):
            parser.error("A selected GPU is unavailable or occupied (>2 GiB)")
        args.run_dir.mkdir(parents=True)
        with (args.run_dir/"supervisor.log").open("x") as log:
            worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1"))
        print(json.dumps(dict(worker_pid=worker.pid,run_dir=str(args.run_dir),
            model_output=str(args.output),detached_session=True)),flush=True)
        return

    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    metadata=validate_dataset(args.dataset)
    status=dict(status="starting",supervisor_pid=os.getpid(),session_id=os.getsid(0),
        target_steps=args.steps,completed_steps=0,dataset=str(args.dataset),output=str(args.output),
        dataset_sha256=hashlib.sha256((args.dataset/"dataset.json").read_bytes()).hexdigest(),
        instruction=metadata["instruction"],training_gpus=gpus,train_episodes=45,validation_episodes=5,
        checkpoint_interval=250,validation_interval=1000,previous_run_preserved=True,
        initialization="pi05_base_new_optimizer_step_zero_not_resume_from_875",
        started_unix=time.time())
    path=args.run_dir/"status.json"
    write_json(path,status)
    process=None
    try:
        with (args.run_dir/"training.log").open("x") as log:
            process=subprocess.Popen(training_command(args.dataset,args.output,args.steps),
                cwd=ROOT,env=dict(os.environ,GPU_LIST=args.gpus),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            status.update(status="running",training_pid=process.pid)
            while process.poll() is None:
                protocol=args.output/"protocol.json"
                if protocol.exists():
                    current=json.loads(protocol.read_text())
                    for key in ("completed_steps","last_training_loss","latest_checkpoint","last_validation"):
                        if key in current:
                            status[key]=current[key]
                status["heartbeat_unix"]=time.time()
                write_json(path,status)
                time.sleep(5)
            status["returncode"]=process.returncode
            if process.returncode:
                raise RuntimeError(f"Trainer exited with code {process.returncode}")
        protocol=json.loads((args.output/"protocol.json").read_text())
        if protocol["status"]!="finished" or protocol["completed_steps"]!=args.steps:
            raise RuntimeError("Trainer did not finish the requested steps")
        status.update(status="finished",completed_steps=protocol["completed_steps"],
            latest_checkpoint=protocol["latest_checkpoint"])
    except BaseException as error:
        status.update(status="failed_or_interrupted",error=f"{type(error).__name__}: {error}")
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        raise
    finally:
        status["last_update_unix"]=time.time()
        write_json(path,status)


if __name__=="__main__":
    main()
