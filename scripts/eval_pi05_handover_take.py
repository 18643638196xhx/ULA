#!/usr/bin/env python3
"""Unseen randomized scenes: left pi0.5, middle controller, right pi0.5 takeover."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"tra_pipe/RoboTwin"),str(ROOT/"external/pi05"),str(ROOT/"external/pi05/src")]
SPEC = importlib.util.spec_from_file_location("take_eval_middle",ROOT/"scripts/eval_pi05_handover_middle.py")
middle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(middle)
shared, pickup = middle.shared, middle.pickup
from adapters.handover.pickup_audit import PickupAudit
from adapters.handover.pickup_data import assert_disjoint_seeds
from adapters.handover.middle_control import closure_attachment, carry_to_middle, hold_interval
from adapters.handover.middle_audit import evaluate_middle
from adapters.handover.take_data import TakeAudit, INSTRUCTION
from adapters.handover.take_rollout import rollout_take


def summarize(rows):
    valid = [r for r in rows if r["initialized"]]
    prefix = [r for r in valid if r.get("prefix_success")]
    successes = [r for r in valid if r["success"]]
    return dict(valid_scenes=len(valid),raw_attempts=len(rows),
        initialization_rejection_seeds=[r["seed"] for r in rows if not r["initialized"]],
        localized=sum(bool(r.get("block_localized")) for r in valid),
        left_policy_entered=sum(bool(r.get("left_entered")) for r in valid),
        pickup_successes=sum(bool(r.get("pickup_success")) for r in valid),
        verified_middle_arrivals=len(prefix),
        right_policy_entered=sum(bool(r.get("take_entered")) for r in valid),
        successes=len(successes),
        overall_success_rate_through_handover=len(successes)/len(valid) if valid else None,
        right_take_success_rate_given_verified_middle=len(successes)/len(prefix) if prefix else None,
        successful_seeds=[r["seed"] for r in successes],
        right_take_failed_after_valid_prefix_seeds=[r["seed"] for r in prefix if not r["success"]],
        failure_stages=dict(Counter(r.get("first_failed_stage") or "unknown" for r in valid if not r["success"])))


def take_failure(evaluation):
    if not evaluation.get("samples"):
        return "no_right_take_evaluation"
    last = evaluation["samples"][-1]
    if last["clearance_m"] < .03:
        return "block_not_held_above_table"
    if last["left_command"] <= .8:
        return "left_gripper_not_released"
    if not 0 <= last["right_command"] < .05:
        return "right_gripper_not_closed"
    if last["left_retreat_m"] < .06 or last["left_to_block_m"] < .17:
        return "left_hand_not_clear_of_block"
    if last["right_to_block_m"] > .20:
        return "block_not_near_right_hand"
    return "right_held_geometry_not_sustained"


def run_episode(task,left_policy,right_policy,worker,config,args,index):
    seed = args.seed+index
    directory = args.output/f"episode_{index:03d}_seed{seed}"
    directory.mkdir()
    row = dict(seed=seed,initialized=False,success=False,pickup_success=False,
        prefix_success=False,left_entered=False,take_entered=False,failure_stage=None,
        control_uses_object_truth=False,truth_authorizes_stage_switch=False,trial_lift=False)
    stage, started = "setup",time.perf_counter()
    take_audit = None
    try:
        task.setup_demo(now_ep_num=index,seed=seed,**config)
        row["initialized"] = True
        task.step_lim = args.pickup_steps
        task.set_instruction(pickup.INSTRUCTION)
        audit = PickupAudit(task.box.get_pose().to_transformation_matrix())
        with shared.ObjectTruthTerminationGuard(task):
            shared.snapshot(task,directory,"initial")
            stage = "initial_localization"
            perception = shared.ExternalPerception(ROOT,directory/"perception",worker=worker)
            camera = task.get_camera_observation("head_camera")
            visual = perception.estimate(camera["rgb"],camera["depth"],camera["intrinsic_cv"],camera["extrinsic_cv"])
            row["initial_observation"] = visual.observation_audit
            row["block_localized"] = bool(visual.observation_audit["block_output"])
            if not row["block_localized"]:
                raise RuntimeError("No usable visual red-block localization")
            stage = "traditional_approach"
            row["approach"] = {}
            shared.approach(task,visual,row["approach"])
            shared.snapshot(task,directory,"pregrasp")
            stage = "left_model_load"
            left_policy.load()
            stage = "left_pi05_pickup"
            row["left_entered"] = True
            original = task.take_action
            def observed_pickup(values,*a,**kw):
                original(values,*a,**kw)
                audit.observe(task.box.get_pose().to_transformation_matrix(),
                    shared.pose_wxyz_to_matrix(task.get_arm_pose("left")),
                    float(task.robot.get_left_gripper_val()),task.take_action_cnt)
            task.take_action = observed_pickup
            row["left_pickup"] = {}
            try:
                with shared.policy_without_truth(left_policy):
                    pickup.rollout_fixed_budget(task,left_policy,row["left_pickup"])
            finally:
                task.take_action = original
            row["pickup_evaluation"] = audit.report()
            row["pickup_success"] = bool(row["pickup_evaluation"]["pickup_success"])
            shared.snapshot(task,directory,"pickup")
            stage = "traditional_middle"
            attachment,row["attachment"] = closure_attachment(visual.cup_world_pose,row["left_pickup"]["commands"])
            row["middle_control"] = {}
            carry_to_middle(task,attachment,row["middle_control"])
            samples = []
            for _ in range(5):
                hold_interval(task,15)
                samples.append(dict(block_world=task.box.get_pose().to_transformation_matrix().tolist(),
                    left_ee_world=shared.pose_wxyz_to_matrix(task.get_arm_pose("left")).tolist(),
                    left_command=float(task.robot.get_left_gripper_val())))
            row["middle_evaluation"] = evaluate_middle(samples,audit.initial_lowest_z)
            row["prefix_success"] = bool(row["pickup_success"] and row["middle_evaluation"]["success"])
            # Deliberately NO branch on prefix_success or any object-truth score.
            # Even an empty left grasp proceeds if robot-only transport succeeds.
            shared.snapshot(task,directory,"handover_start")
            task.set_instruction(INSTRUCTION)
            stage = "right_model_load"
            right_policy.load()
            take_audit = TakeAudit(audit.initial_lowest_z,
                shared.pose_wxyz_to_matrix(task.get_arm_pose("left")))
            def observe_take():
                take_audit.observe(task.box.get_pose().to_transformation_matrix(),
                    shared.pose_wxyz_to_matrix(task.get_arm_pose("left")),
                    shared.pose_wxyz_to_matrix(task.get_arm_pose("right")),
                    task.robot.get_left_gripper_val(),task.robot.get_right_gripper_val(),
                    len(take_audit.samples)+1)
            stage = "right_pi05_take"
            row["take_entered"] = True
            row["right_take"] = {}
            original = task.take_action
            saved = set()
            def observed_take(values,*a,**kw):
                original(values,*a,**kw)
                observe_take()
                for name,condition in (("first_right_closed",task.robot.get_right_gripper_val()<.05),
                                       ("first_left_release",task.robot.get_left_gripper_val()>.8)):
                    if condition and name not in saved:
                        shared.snapshot(task,directory,name)
                        saved.add(name)
            task.take_action = observed_take
            take_started = time.perf_counter()
            try:
                with shared.policy_without_truth(right_policy):
                    rollout_take(task,right_policy,row["right_take"],args.take_steps)
            finally:
                task.take_action = original
            row["right_take_wall_s"] = time.perf_counter()-take_started
            # Fixed 150 physical ticks for EVERY completed rollout, no lift,
            # no retries and no early termination based on the passive audit.
            stage = "fixed_final_hold"
            for _ in range(10):
                hold_interval(task,15)
                observe_take()
            row["take_evaluation"] = take_audit.report()
            row["success"] = bool(row["prefix_success"] and row["right_take"]["complete"]
                and row["take_evaluation"]["success"])
    except Exception as error:
        row.update(failure_stage=stage,error=f"{type(error).__name__}: {error}")
        print(f"[pi05-take-eval] seed={seed} stage={stage} error={row['error']}",flush=True)
    finally:
        if take_audit is not None and "take_evaluation" not in row:
            row["take_evaluation"] = take_audit.report()
        if not row["success"]:
            if row["left_entered"] and not row["pickup_success"]:
                row["first_failed_stage"] = "pickup_not_verified"
            elif row["pickup_success"] and not row["prefix_success"]:
                row["first_failed_stage"] = row["failure_stage"] or "middle_not_verified"
            elif row["prefix_success"]:
                row["first_failed_stage"] = row["failure_stage"] or take_failure(row.get("take_evaluation",{}))
            else:
                row["first_failed_stage"] = row["failure_stage"]
        if row["initialized"]:
            try:
                shared.snapshot(task,directory,"final")
            except Exception as error:
                row["snapshot_error"] = str(error)
        row["elapsed_s"] = time.perf_counter()-started
        shared.dump(directory/"result.json",row)
        try:
            task.close_env()
        except Exception as error:
            row["close_error"] = str(error)
    return row


def check_checkpoint(path,instruction):
    protocol = json.loads((path.parent.parent/"protocol.json").read_text())
    if protocol.get("architecture")!="native_pi0.5_LoRA_only" or protocol.get("WAM"):
        raise ValueError("Requires a native pi0.5 model, not WorldLA")
    if protocol.get("status")!="finished" or protocol.get("completed_steps")!=10000 or path.name!="10000":
        raise ValueError("Requires a finished 10000-step checkpoint")
    if protocol["instruction"]!=instruction:
        raise ValueError("Checkpoint instruction does not match its assigned stage")
    if not all((path/name).is_dir() for name in ("params","assets","train_state")):
        raise ValueError("Incomplete native checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",type=int,default=50)
    parser.add_argument("--seed",type=int,default=40000)
    parser.add_argument("--pickup-steps",type=int,default=200)
    parser.add_argument("--take-steps",type=int,default=200)
    parser.add_argument("--left-checkpoint",type=Path,default=ROOT/"checkpoints/pi05_handover_pickup50_10000_4gpu_20261107_v1/checkpoints/10000")
    parser.add_argument("--right-checkpoint",type=Path,default=ROOT/"checkpoints/pi05_handover_take50_10000_4gpu_20261107_v2/checkpoints/10000")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--detach",action="store_true")
    args = parser.parse_args()
    if min(args.episodes,args.pickup_steps,args.take_steps)<1 or args.seed<0:
        parser.error("Invalid evaluation budgets")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        parser.error("Evaluation outputs must stay under ULA")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh output directory")
    for checkpoint,instruction in ((args.left_checkpoint,pickup.INSTRUCTION),(args.right_checkpoint,INSTRUCTION)):
        check_checkpoint(checkpoint,instruction)
    seen = []
    for name in ("handover_pickup_collect50_20261107_v1","handover_take_collect50_20261107_v1"):
        seen.extend(row["seed"] for row in json.loads((ROOT/"results"/name/"report.json").read_text())["attempts"])
    assert_disjoint_seeds(seen,range(args.seed,args.seed+max(200,5*args.episodes)))
    if args.detach:
        log_path = args.output.with_suffix(".log")
        with log_path.open("x") as log:
            process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                *[arg for arg in sys.argv[1:] if arg!="--detach"]],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                close_fds=True,start_new_session=True)
        print(json.dumps(dict(pid=process.pid,output=str(args.output),log=str(log_path),detached=True)),flush=True)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    paths = [Path(__file__),ROOT/"scripts/eval_pi05_handover_take.sh",ROOT/"scripts/eval_pi05_handover_middle.py"]
    def fingerprints():
        return dict(shared.initial.sources(),**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    before = fingerprints()
    resource = shared.initial.shared.method.cup.randomized.configure_runtime()
    config = shared.initial.task_config()
    worker = shared.initial.HandoverInitialWorker(ROOT,args.output/"vision_worker",backend="pointcloud")
    left_policy,right_policy = (pickup.NativePi05Pickup(p) for p in (args.left_checkpoint,args.right_checkpoint))
    report = dict(run_status="running",pid=os.getpid(),session_id=os.getsid(0),
        task="handover_block",task_config="demo_randomized",stage="through_right_take_no_platform_placement",
        requested_valid_scenes=args.episodes,seed_start=args.seed,resources=resource,
        left_checkpoint=str(args.left_checkpoint),right_checkpoint=str(args.right_checkpoint),
        pickup_steps=args.pickup_steps,take_steps=args.take_steps,
        pickup_instruction=pickup.INSTRUCTION,take_instruction=INSTRUCTION,
        controller_stop="fixed_separate_policy_action_budgets_no_closed_command_early_stop",
        ground_truth_in_model_inputs=False,truth_authorizes_stage_switch=False,
        truth_usage="passive_same_collection_geometric_scores_only",trial_lift=False,
        placement_enabled=False,official_full_task_score=False,
        overall_denominator="all_initialized_scenes_including_perception_and_approach_failures",
        conditional_denominator="independently_verified_left_pickup_and_middle_arrivals_before_take",
        final_hold_physics_steps=150,conditional_metrics_do_not_filter_execution=True,
        effective_domain_randomization=config["domain_randomization"],sources_before=before,episodes=[])
    previous = Path.cwd()
    try:
        os.chdir(ROOT/"tra_pipe/RoboTwin")
        from envs.handover_block import handover_block
        task = handover_block()
        shared.dump(args.output/"report.json",report)
        while sum(row["initialized"] for row in report["episodes"])<args.episodes:
            index = len(report["episodes"])
            if index>=max(200,5*args.episodes):
                raise RuntimeError("Too many scene initialization rejections")
            row = run_episode(task,left_policy,right_policy,worker,config,args,index)
            report["episodes"].append(row)
            report.update(summary=summarize(report["episodes"]),worker=worker.summary())
            shared.dump(args.output/"report.json",report)
            print(f"[pi05-take-eval] valid={report['summary']['valid_scenes']}/{args.episodes} seed={row['seed']} middle={row['prefix_success']} take={row['success']} failure={row.get('first_failed_stage')}",flush=True)
            if any(word in row.get("error","").lower() for word in
                ("out of memory","resource_exhausted","cuda error","worker died","worker exited","worker is closed")):
                raise RuntimeError("Infrastructure failure; evaluation incomplete")
        report["run_status"] = "finished"
    finally:
        worker.close()
        os.chdir(previous)
        report.update(summary=summarize(report["episodes"]),worker=worker.summary(),sources_after=fingerprints())
        report["sources_unchanged"] = before==report["sources_after"]
        if report["run_status"]!="finished":
            report["run_status"] = "interrupted_or_error"
        shared.dump(args.output/"report.json",report)


if __name__=="__main__":
    main()
