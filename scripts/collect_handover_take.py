#!/usr/bin/env python3
"""Collect the RIGHT takeover stage from the deployed left-pickup/middle prefix."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"tra_pipe/RoboTwin"),str(ROOT/"external/pi05"),str(ROOT/"external/pi05/src")]
SPEC = importlib.util.spec_from_file_location("take_middle_shared",ROOT/"scripts/eval_pi05_handover_middle.py")
middle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(middle)
shared, pickup = middle.shared, middle.pickup
from adapters.handover.pickup_audit import PickupAudit
from adapters.handover.pickup_data import choose_splits, assert_disjoint_seeds
from adapters.handover.middle_control import closure_attachment, carry_to_middle, hold_interval
from adapters.handover.middle_audit import evaluate_middle
from adapters.handover.take_data import INSTRUCTION, TakeAudit, TakeRecorder, expert


def run_episode(task, policy, worker, config, args, index):
    seed = args.seed+index
    directory = args.output/f"attempt_{index:04d}_seed{seed}"
    directory.mkdir()
    row = dict(seed=seed,initialized=False,success=False,failure_stage=None,
        control_uses_object_truth=False,success_filter_uses_object_truth=True)
    stage, started = "setup", time.perf_counter()
    try:
        task.setup_demo(now_ep_num=index,seed=seed,**config)
        row["initialized"] = True
        task.step_lim = args.max_vla_steps
        task.set_instruction(pickup.INSTRUCTION)
        audit = PickupAudit(task.box.get_pose().to_transformation_matrix())
        with shared.ObjectTruthTerminationGuard(task):
            stage = "initial_localization"
            perception = shared.ExternalPerception(ROOT,directory/"perception",worker=worker)
            camera = task.get_camera_observation("head_camera")
            visual = perception.estimate(camera["rgb"],camera["depth"],camera["intrinsic_cv"],camera["extrinsic_cv"])
            row["initial_observation"] = visual.observation_audit
            if not visual.observation_audit["block_output"]:
                raise RuntimeError("No usable visual red-block localization")
            stage = "traditional_approach"
            row["approach"] = {}
            shared.approach(task,visual,row["approach"])
            stage = "left_pi05_pickup"
            original = task.take_action
            def observed_action(values,*a,**kw):
                original(values,*a,**kw)
                audit.observe(task.box.get_pose().to_transformation_matrix(),
                    shared.pose_wxyz_to_matrix(task.get_arm_pose("left")),
                    float(task.robot.get_left_gripper_val()),task.take_action_cnt)
            task.take_action = observed_action
            row["left_pickup"] = {}
            try:
                with shared.policy_without_truth(policy):
                    pickup.rollout_fixed_budget(task,policy,row["left_pickup"])
            finally:
                task.take_action = original
            row["pickup_evaluation"] = audit.report()
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
            # Evaluation fields above NEVER authorize the teacher: even a failed
            # prefix proceeds if the robot-only controller has completed.
            shared.snapshot(task,directory,"handover_start")
            task.set_instruction(INSTRUCTION)
            take_audit = TakeAudit(audit.initial_lowest_z,
                shared.pose_wxyz_to_matrix(task.get_arm_pose("left")))
            class Observer:
                def observe(self,block,unused_left,unused_command,step):
                    take_audit.observe(block,
                        shared.pose_wxyz_to_matrix(task.get_arm_pose("left")),
                        shared.pose_wxyz_to_matrix(task.get_arm_pose("right")),
                        task.robot.get_left_gripper_val(),task.robot.get_right_gripper_val(),step)
            recorder = TakeRecorder(task,Observer(),shared.pose_wxyz_to_matrix,args.stride)
            row["expert"] = {}
            stage = "right_take_teacher"
            expert(task,attachment,recorder,row["expert"])
            row["take_evaluation"] = take_audit.report()
            row["frames"] = len(recorder.frames)
            row["success"] = bool(row["pickup_evaluation"]["pickup_success"]
                and row["middle_evaluation"]["success"] and row["take_evaluation"]["success"])
            if row["success"]:
                recorder.save(directory/"trajectory.hdf5",row)
                row["trajectory"] = str(directory/"trajectory.hdf5")
            else:
                row["failure_stage"] = "independent_success_filter"
            shared.snapshot(task,directory,"final")
    except Exception as error:
        row.update(failure_stage=stage,error=f"{type(error).__name__}: {error}")
        print(f"[take-collect] seed={seed} stage={stage} error={row['error']}",flush=True)
        if row["initialized"]:
            try:
                shared.snapshot(task,directory,"final")
            except Exception:
                pass
    finally:
        row["elapsed_s"] = time.perf_counter()-started
        shared.dump(directory/"result.json",row)
        try:
            task.close_env()
        except Exception as error:
            row["close_error"] = str(error)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",type=int,default=50)
    parser.add_argument("--seed",type=int,default=30000)
    parser.add_argument("--stride",type=int,default=15)
    parser.add_argument("--max-attempts",type=int,default=500)
    parser.add_argument("--max-vla-steps",type=int,default=200)
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/pi05_handover_pickup50_10000_4gpu_20261107_v1/checkpoints/10000")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.episodes<2 or args.max_attempts<args.episodes or min(args.stride,args.max_vla_steps)<1:
        parser.error("Invalid collection budgets")
    protocol = json.loads((args.checkpoint.parent.parent/"protocol.json").read_text())
    if protocol.get("architecture")!="native_pi0.5_LoRA_only" or protocol.get("status")!="finished":
        parser.error("Prefix requires a finished native pi0.5 checkpoint")
    # Keep new training seeds disjoint from previous left training/evaluation.
    previous_seeds = []
    for path,key in [(ROOT/"results/handover_pickup_collect50_20261107_v1/report.json","attempts"),
                     (ROOT/"results/pi05_middle10k_randomized_50valid_seed20000_20261107_v1/report.json","episodes")]:
        previous_seeds.extend(row["seed"] for row in json.loads(path.read_text())[key])
    assert_disjoint_seeds(previous_seeds,range(args.seed,args.seed+args.max_attempts))
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    paths = [Path(__file__),ROOT/"scripts/collect_handover_take.sh",ROOT/"scripts/eval_pi05_handover_middle.py"]
    def fingerprints():
        return dict(shared.initial.sources(),**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    before = fingerprints()
    resource = shared.initial.shared.method.cup.randomized.configure_runtime()
    config = shared.initial.task_config()
    worker = shared.initial.HandoverInitialWorker(ROOT,args.output/"vision_worker",backend="pointcloud")
    policy = pickup.NativePi05Pickup(args.checkpoint)
    report = dict(run_status="running",requested_successful_episodes=args.episodes,
        instruction=INSTRUCTION,collection_seed_start=args.seed,resources=resource,
        prefix_checkpoint=str(args.checkpoint),prefix_vla_steps=args.max_vla_steps,
        expert="robot_propagated_initial_RGBD_known_primitive_lower_grasp_release_retreat",
        recorded_stage="right_approach_insert_close_left_release_left_retreat_right_hold",
        training_includes_prefix=False,trial_lift=False,
        object_truth_used_for_success_filter_only=True,object_truth_in_model_inputs=False,
        state_semantics="commanded_drive_targets_matching_native_pi05_runtime",
        action_semantics="next_sample_commanded_targets",stride_physics_steps=args.stride,
        sources_before=before,attempts=[],successful_trajectories=[])
    previous = Path.cwd()
    try:
        os.chdir(ROOT/"tra_pipe/RoboTwin")
        from envs.handover_block import handover_block
        task = handover_block()
        shared.dump(args.output/"report.json",report)
        while len(report["successful_trajectories"])<args.episodes:
            index = len(report["attempts"])
            if index>=args.max_attempts:
                raise RuntimeError("Not enough verified right-take demonstrations within attempt budget")
            row = run_episode(task,policy,worker,config,args,index)
            report["attempts"].append(row)
            if row["success"]:
                report["successful_trajectories"].append(dict(seed=row["seed"],path=row["trajectory"],frames=row["frames"]))
            report["successful_episodes"] = len(report["successful_trajectories"])
            shared.dump(args.output/"report.json",report)
            print(f"[take-collect] success={report['successful_episodes']}/{args.episodes} attempts={len(report['attempts'])} seed={row['seed']} accepted={row['success']}",flush=True)
            if any(word in row.get("error","").lower() for word in
                   ("out of memory","cuda error","worker died","worker exited","worker is closed")):
                raise RuntimeError("Infrastructure error during collection")
        splits = choose_splits(args.episodes,validation=max(1,min(5,args.episodes//5)))
        for row,split in zip(report["successful_trajectories"],splits):
            row["split"] = split
        report.update(run_status="finished",train_episodes=splits.count("train"),val_episodes=splits.count("val"))
    finally:
        worker.close()
        os.chdir(previous)
        report.update(worker=worker.summary(),sources_after=fingerprints())
        report["sources_unchanged"] = before==report["sources_after"]
        if report["run_status"]!="finished":
            report["run_status"]="interrupted_or_error"
        shared.dump(args.output/"report.json",report)


if __name__=="__main__":
    main()
