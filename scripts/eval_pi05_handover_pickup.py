#!/usr/bin/env python3
"""Evaluate native pi0.5 10k checkpoint on ULA left pickup, unseen scenes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PI_ROOT=ROOT/"external/pi05"
sys.path[:0]=[str(ROOT),str(ROOT/"tra_pipe/RoboTwin"),str(PI_ROOT),str(PI_ROOT/"src")]
SPEC=importlib.util.spec_from_file_location("pi05_pickup_eval_harness",ROOT/"scripts/run_handover_pickup_randomized.py")
legacy=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)
from adapters.handover.native_pi05_pickup import NativePi05Pickup,rollout_fixed_budget
from adapters.handover.pickup_data import INSTRUCTION,assert_disjoint_seeds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",type=int,default=50)
    parser.add_argument("--seed",type=int,default=20000)
    parser.add_argument("--max-vla-steps",type=int,default=200)
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/pi05_handover_pickup50_10000_4gpu_20261107_v1/checkpoints/10000")
    parser.add_argument("--collection",type=Path,default=ROOT/"results/handover_pickup_collect50_20261107_v1")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.episodes<1 or args.seed<0 or args.max_vla_steps<5:
        parser.error("Invalid evaluation size or budget")
    args.checkpoint=args.checkpoint.resolve()
    if not all((args.checkpoint/name).is_dir() for name in ("params","assets","train_state")):
        parser.error("Missing complete native pi0.5 checkpoint")
    protocol=json.loads((args.checkpoint.parent.parent/"protocol.json").read_text())
    if protocol.get("architecture")!="native_pi0.5_LoRA_only" or protocol.get("WAM"):
        parser.error("Not a pure pi0.5 run")
    if protocol["status"]!="finished" or protocol["completed_steps"]!=10000 or args.checkpoint.name!="10000":
        parser.error("This evaluation requires the completed 10000-step checkpoint")
    collection=json.loads((args.collection.resolve()/"report.json").read_text())
    seen_seeds=[row["seed"] for row in collection["attempts"]]
    assert_disjoint_seeds(seen_seeds,range(args.seed,args.seed+max(200,5*args.episodes)))
    args.output=args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    args.instruction=INSTRUCTION
    args.robot_rise_m=None
    # Reuse the established approach/artifacts/passive audit harness, but not
    # its old 8cm hand-rise endpoint or its WorldLA model constructor.
    legacy.rollout_pickup=rollout_fixed_budget
    paths=[Path(__file__),ROOT/"scripts/eval_pi05_handover_pickup.sh",
           ROOT/"scripts/run_handover_pickup_randomized.py",ROOT/"scripts/run_handover_simple_randomized.py",
           PI_ROOT/"src/openpi/policies/policy_config.py",PI_ROOT/"src/openpi/policies/aloha_policy.py"]
    def fingerprints():
        return dict(legacy.shared.initial.sources(),**{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    before=fingerprints()
    resource=legacy.shared.initial.shared.method.cup.randomized.configure_runtime()
    config=legacy.shared.initial.task_config()
    worker=legacy.shared.initial.HandoverInitialWorker(ROOT,args.output/"vision_worker",backend="pointcloud")
    policy=NativePi05Pickup(args.checkpoint)
    report=dict(run_status="running",task="handover_block",task_config="demo_randomized",
        architecture="native_pi0.5_LoRA",checkpoint_step=10000,checkpoint=str(args.checkpoint),
        WAM=False,WorldLA=False,stage="left_pickup_only",requested_valid_scenes=args.episodes,
        seed_start=args.seed,collection_seed_overlap=False,instruction=INSTRUCTION,resources=resource,
        max_vla_steps=args.max_vla_steps,controller_stop="fixed_VLA_action_budget_only",
        hand_rise_early_stop=False,trial_lift=False,platform_localization_required=False,
        replan_mode="native_fresh_chunks_no_tail_blend",chunk_steps=10,
        inference_seed=20261107,inference_seed_reset_each_scene=True,
        perception_and_traditional_approach_changed=False,
        truth_usage="passive_pickup_audit_only_never_returned_to_control",
        success_definition="same_PickupAudit_3cm_5_samples_relative_drift_and_end_raised",
        denominator="all_initialized_scenes_including_localization_and_approach_failures",
        not_strict_paired_ablation_vs_previous_1_of_50="new_seeds_training_instruction_native_policy_and_no_hand_rise_stop",
        effective_domain_randomization=config["domain_randomization"],sources_before=before,episodes=[])
    previous=Path.cwd()
    try:
        os.chdir(ROOT/"tra_pipe/RoboTwin")
        from envs.handover_block import handover_block
        task=handover_block()
        legacy.shared.dump(args.output/"report.json",report)
        while sum(row["initialized"] for row in report["episodes"])<args.episodes:
            index=len(report["episodes"])
            if index>=max(200,5*args.episodes):
                raise RuntimeError("Too many scene initialization failures")
            print(f"[pi05-10k-eval] start seed={args.seed+index}",flush=True)
            row=legacy.run_episode(task,policy,worker,config,args,index)
            report["episodes"].append(row)
            report.update(summary=legacy.summarize(report["episodes"]),worker=worker.summary())
            legacy.shared.dump(args.output/"report.json",report)
            print(f"[pi05-10k-eval] valid={report['summary']['valid_scenes']}/{args.episodes} seed={row['seed']} pickup={row['success']} failure={row['failure_stage']}",flush=True)
            if any(word in (row.get("error") or "").lower() for word in
                   ("out of memory","resource_exhausted","cuda error","worker died","worker exited","worker is closed")):
                raise RuntimeError("Infrastructure failure; incomplete evaluation")
        report["run_status"]="finished"
    finally:
        worker.close()
        os.chdir(previous)
        report.update(summary=legacy.summarize(report["episodes"]),worker=worker.summary(),sources_after=fingerprints())
        report["sources_unchanged"]=before==report["sources_after"]
        if report["run_status"]!="finished":
            report["run_status"]="interrupted_or_error"
        legacy.shared.dump(args.output/"report.json",report)


if __name__=="__main__":
    main()
