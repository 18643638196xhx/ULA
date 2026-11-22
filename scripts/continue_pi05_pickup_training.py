#!/usr/bin/env python3
"""Finish the current run, preserve it, then resume parameters AND optimizer.

The source trainer is never edited or stopped. Checkpoint files are immutable;
hard-linking them into a new run preserves the original 10k baseline even when
the new run's rolling checkpoint retention unlinks its own 10k copy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid,0)
        return Path(f"/proc/{pid}/stat").read_text().split(")",1)[1].strip().split()[0]!="Z"
    except (ProcessLookupError,FileNotFoundError):
        return False


def validate_source(protocol, expected_step, target_steps):
    if protocol.get("architecture")!="native_pi0.5_LoRA_only" or protocol.get("WAM") or protocol.get("WorldLA"):
        raise ValueError("Only this native pi0.5 LoRA experiment may be continued")
    if protocol.get("status")!="finished" or protocol.get("completed_steps")!=expected_step:
        raise ValueError("Source run has not completed the expected optimizer step")
    if protocol.get("target_steps")!=expected_step or target_steps<=expected_step:
        raise ValueError("Invalid continuation budget")
    if not protocol.get("sources_unchanged"):
        raise ValueError("Source run changed during training; inspect it before continuation")
    checkpoint=Path(protocol["latest_checkpoint"])
    if checkpoint.name!=str(expected_step) or not all((checkpoint/key).is_dir() for key in ("params","train_state","assets")):
        raise ValueError("Missing complete native parameter/optimizer/assets checkpoint")
    return checkpoint


def prepare_checkpoint(source_protocol,output,expected_step,target_steps):
    checkpoint=validate_source(source_protocol,expected_step,target_steps)
    if (output/"checkpoints").exists() or (output/"protocol.json").exists():
        raise FileExistsError("Refusing to replace an existing continuation checkpoint")
    target=output/"checkpoints"/str(expected_step)
    target.parent.mkdir(parents=True)
    shutil.copytree(checkpoint,target,copy_function=os.link)
    protocol=dict(source_protocol,latest_checkpoint=str(target))
    write_json(output/"protocol.json",protocol)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-pid",type=int)
    parser.add_argument("--expected-source-step",type=int,default=10000)
    parser.add_argument("--target-steps",type=int,default=100000)
    parser.add_argument("--gpu-list",default="0,1,2,3")
    parser.add_argument("--timeout-seconds",type=float,default=7200.)
    parser.add_argument("--max-validation-batches",type=int,help="Smoke test only")
    args=parser.parse_args()
    args.source_run=args.source_run.resolve()
    args.output=args.output.resolve()
    if args.output==args.source_run or args.target_steps<=args.expected_source_step:
        parser.error("Need a new output and a larger TOTAL training budget")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty continuation output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    plan_path=args.output/"continuation.json"
    plan=dict(status="waiting_for_source_completion",pid=os.getpid(),source_run=str(args.source_run),
        source_pid=args.source_pid,source_step=args.expected_source_step,total_target_steps=args.target_steps,
        additional_optimizer_steps=args.target_steps-args.expected_source_step,gpu_list=args.gpu_list,
        preserve_source_checkpoint=True,resume_weights_and_optimizer=True,source_trainer_edited=False,
        learning_rate_policy="preserve_original_decay_horizon_then_hold_original_final_learning_rate",
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            ROOT/"scripts/train_pi05_handover_pickup_extended.py",
            ROOT/"scripts/train_pi05_handover_pickup_extended.sh"]})
    write_json(plan_path,plan)
    print(f"[pi05-continuation] waiting for completed step {args.expected_source_step}; target TOTAL={args.target_steps}",flush=True)
    started=time.monotonic()
    try:
        while True:
            try:
                source=json.loads((args.source_run/"protocol.json").read_text())
            except (FileNotFoundError,json.JSONDecodeError):
                source={}
            if source.get("status")=="finished":
                validate_source(source,args.expected_source_step,args.target_steps)
                if not process_alive(args.source_pid):
                    break
            elif source.get("status")=="failed_or_interrupted":
                raise RuntimeError(f"Source training stopped: {source.get('error')}")
            if args.source_pid is not None and not process_alive(args.source_pid) and source.get("status")!="finished":
                raise RuntimeError("Source process exited before completing its checkpoint")
            if time.monotonic()-started>args.timeout_seconds:
                raise TimeoutError("Source run did not finish within continuation wait budget")
            time.sleep(15)
        checkpoint=prepare_checkpoint(source,args.output,args.expected_source_step,args.target_steps)
        decay=int(source.get("lr_decay_steps",max(3,args.expected_source_step)))
        command=["bash",str(ROOT/"scripts/train_pi05_handover_pickup_extended.sh"),
            "--dataset",source["dataset"],"--output",str(args.output),"--resume",
            "--init-params",source["init_params"],"--steps",str(args.target_steps),
            "--lr-decay-steps",str(decay),"--batch-size",str(source["batch_size"]),
            "--fsdp-devices",str(source["fsdp_devices"]),"--learning-rate",str(source["learning_rate"]),
            "--validation-interval","1000","--save-interval","1000"]
        if args.max_validation_batches is not None:
            command += ["--max-validation-batches",str(args.max_validation_batches)]
        environment=os.environ.copy()
        environment["GPU_LIST"]=args.gpu_list
        plan.update(status="resuming",resume_checkpoint=str(checkpoint),command=command,
                    lr_decay_steps=decay,batch_size=source["batch_size"],fsdp_devices=source["fsdp_devices"])
        write_json(plan_path,plan)
        print(f"[pi05-continuation] resuming optimizer step {args.expected_source_step}->{args.target_steps}; GPUs={args.gpu_list}",flush=True)
        subprocess.run(command,cwd=ROOT,env=environment,check=True)
        result=json.loads((args.output/"protocol.json").read_text())
        if result["status"]!="finished" or result["completed_steps"]!=args.target_steps:
            raise RuntimeError("Continuation did not reach the requested total optimizer steps")
        plan.update(status="finished",completed_steps=result["completed_steps"],latest_checkpoint=result["latest_checkpoint"])
    except BaseException as error:
        plan.update(status="failed_or_interrupted",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(plan_path,plan)


if __name__=="__main__":
    main()
