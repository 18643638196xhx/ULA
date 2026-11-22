"""Independent comparison: visual traditional approach, then continuous VLA.

No traditional transport, descent, release, retreat, or trial lift. Completion
is a visual heuristic; official simulator scoring is only read after control.
"""
from contextlib import nullcontext
import time
import traceback

import numpy as np

from adapters.pillbottle_close import bounds_distance
from adapters.pillbottle_control import measured_joint_input, robot_state, audit_topp
from adapters.pillbottle_known_model import support_point


class VisualFinish(Exception):
    pass


def visual_finish_decision(value, tcp, metadata, *, xy_tolerance=.03,
                           height_tolerance=.015, retreat_distance=.03):
    """Known CAD bottom + current visual pad; never a hidden bottom measurement."""
    pose = np.asarray(value.cup_world_pose, dtype=float)
    target = np.asarray(value.coaster_world_center, dtype=float)
    tcp = np.asarray(tcp, dtype=float)
    bounds = np.asarray(value.cup_bounds_world, dtype=float)
    if (pose.shape != (4, 4) or target.shape != (3,) or tcp.shape != (3,)
            or bounds.shape != (2, 3)
            or not all(np.isfinite(v).all() for v in (pose, target, tcp, bounds))
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=.01)
            or np.linalg.det(pose[:3, :3]) < .99):
        raise ValueError("Invalid visual finish geometry")
    bottom = support_point(pose, metadata)
    # The CAD's upright direction is its local Y axis (model metadata frame).
    up_cosine = float(pose[2, 1])
    delta = bottom-target
    clearance = bounds_distance(bounds, tcp)
    ready = (np.all(np.abs(delta[:2]) <= xy_tolerance)
             and abs(delta[2]) <= height_tolerance
             and up_cosine >= np.cos(np.radians(15.)) and clearance >= retreat_distance)
    return dict(ready=bool(ready), source="current_rgbd_cad_and_visual_pad_with_robot_fk",
        estimated_support_world=bottom.tolist(), visual_pad_world=target.tolist(),
        support_error_m=delta.tolist(), upright_cosine=up_cosine,
        tcp_distance_to_visible_bounds_m=clearance, xy_tolerance_m=xy_tolerance,
        height_tolerance_m=height_tolerance, minimum_retreat_distance_m=retreat_distance,
        maximum_tilt_deg=15., uses_object_truth=False,
        limitation="visual_heuristic_not_physical_stability_or_official_success")


def rollout_finish(task, policy, perception, arm, record, metadata, *, synchronize=True,
                   measured_observation=False, check_every=10, required_checks=2):
    if check_every < 1 or required_checks < 1:
        raise ValueError("Invalid visual finish budget")
    audit = record["vla_finish"] = dict(complete=False, actions=[], visual_checks=[],
        uses_object_truth=False, trial_lift_enabled=False, policy_resets=0,
        continuous_grasp_transport_place=True, required_visual_checks=required_checks,
        check_every_actions=check_every, completion_source="visual_release_and_retreat")
    original, owned = task.take_action, "take_action" in vars(task)
    closed_count = open_count = good_checks = 0
    saw_closed, last_check = False, -check_every
    if synchronize:
        record["handoff_state_sync"] = dict(before=robot_state(task.robot), uses_object_truth=False)
        for name in ("left", "right"):
            actual = np.asarray(getattr(task.robot, f"get_{name}_arm_real_jointState")(), dtype=float)[:6]
            task.robot.set_arm_joints(actual, np.zeros(6), name)
        record["handoff_state_sync"]["after"] = robot_state(task.robot)

    def action(value, *a, **kw):
        nonlocal closed_count, open_count, saw_closed, last_check, good_checks
        values = np.asarray(value)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("WorldLA must return a finite 14-D joint action")
        item = dict(action=values.tolist(), before=robot_state(task.robot), topp=[])
        audit["actions"].append(item)
        # Preserve training command-state observations, but start interpolation
        # at encoders. Do not nest an already active measured-input context.
        context = measured_joint_input(task.robot) if synchronize and not measured_observation else nullcontext()
        with audit_topp(task.robot, item["topp"]), context:
            returned = original(value, *a, **kw)
        step = int(task.take_action_cnt)
        command = float(getattr(task.robot, f"get_{arm}_gripper_val")())
        item.update(step=step, after=robot_state(task.robot), gripper_command=command)
        closed_count = closed_count+1 if 0 <= command < .05 else 0
        saw_closed = saw_closed or closed_count >= 3
        open_count = open_count+1 if saw_closed and .95 <= command <= 1. else 0
        audit.update(saw_closed_command=saw_closed, open_command_samples=open_count)
        if open_count < 3:
            good_checks = 0
            return returned
        if step-last_check < check_every:
            return returned
        last_check = step
        check = dict(step=step, ready=False, uses_object_truth=False)
        audit["visual_checks"].append(check)
        try:
            estimate = perception.localize_held(task, arm, check)
            tcp = np.asarray(getattr(task.robot, f"get_{arm}_tcp_pose")(), dtype=float)[:3]
            check.update(visual_finish_decision(estimate, tcp, metadata))
        except (ValueError, RuntimeError) as error:
            check["error"] = str(error)
        good_checks = good_checks+1 if check["ready"] else 0
        check["consecutive"] = good_checks
        if good_checks >= required_checks:
            audit.update(complete=True, finish_step=step)
            raise VisualFinish()
        return returned

    task.take_action = action
    try:
        policy.reset()
        audit["policy_resets"] += 1
        while task.take_action_cnt < task.step_lim:
            before = int(task.take_action_cnt)
            policy.step(task, task.get_obs())
            print(f"[ula-pillbottle-vla-finish] step={task.take_action_cnt}/{task.step_lim}", flush=True)
            if before == task.take_action_cnt:
                raise RuntimeError("WorldLA executed no actions")
    except VisualFinish:
        print(f"[ula-pillbottle-vla-finish] visual finish step={task.take_action_cnt}", flush=True)
    finally:
        if owned:
            task.take_action = original
        else:
            del task.take_action
        audit["executed_actions"] = int(task.take_action_cnt)
    if not audit["complete"]:
        raise RuntimeError("No visual release/place/retreat confirmation within VLA budget")


def run_episode(base, task, policy, worker, args, config, index):
    import json
    from pathlib import Path
    seed = args.seed+index
    directory = args.output/f"episode_{index:03d}_seed{seed}"
    directory.mkdir(parents=True)
    timer = base.EpisodeTiming()
    record = dict(seed=seed, instruction=args.instruction, success=False, task_success=None,
        controller_complete=False, error=None, failure_stage=None, handoff=None,
        traditional_placement=None, evaluation_only={}, control_uses_object_truth=False,
        sequence="traditional_approach -> continuous_vla_grasp_transport_place_retreat",
        trial_lift_enabled=False)
    started, stage, setup = time.perf_counter(), "setup", False
    try:
        with timer.phase("setup"):
            task.setup_demo(now_ep_num=index, seed=seed, **config)
            setup = True
            task.step_lim = args.max_vla_steps
            task.set_instruction(args.instruction)
        with base.ObjectTruthTerminationGuard(task):
            with timer.phase("artifacts"):
                base.snapshot(task, directory, "initial")
            perception = base.PillbottlePerception(base.ROOT, directory/"perception", worker=worker)
            stage = "traditional_approach"
            with timer.phase(stage), base.traditional_timing(timer, task, perception):
                record["handoff"] = base.approach_bottle(task, perception, args)
            with timer.phase("artifacts"):
                base.dump(directory/"handoff.json", record["handoff"])
                base.snapshot(task, directory, "handoff")
            if not record["handoff"]["ready"]:
                raise RuntimeError(f"Approach handoff rejected: {record['handoff']['reason']}")
            if args.mode == "hybrid":
                stage = "vla_load"
                with timer.phase(stage):
                    policy.load()
                arm = record["handoff"]["arm"]
                vision = base.PillbottlePerception(base.ROOT, directory/"finish_perception", worker=worker)
                metadata = json.loads(Path(args.known_metadata).read_text())
                stage = "vla_grasp_transport_place"
                with base.policy_without_truth(policy), measured_joint_input(task.robot,
                        enabled=args.joint_state_source == "measured"), timer.phase(stage), base.vla_timing(timer, task, policy.model):
                    rollout_finish(task, policy, vision, arm, record, metadata,
                        synchronize=args.joint_state_source == "synced-commanded",
                        measured_observation=args.joint_state_source == "measured")
            record["controller_complete"] = True
    except Exception as error:
        record.update(error=f"{type(error).__name__}: {error}", failure_stage=stage)
        traceback.print_exc()
    finally:
        # No official scores or live actors are read until all control stops.
        if setup:
            try:
                with timer.phase("evaluation"):
                    record["task_success"] = bool(task.check_success())
                    bottle = np.asarray(task.pillbottle.get_pose().p)
                    pad = np.asarray(task.pad.get_pose().p)
                    record["evaluation_only"] = dict(source="simulation_after_control_only",
                        bottle_model_id=int(task.pillbottle_id), final_bottle_xyz=bottle.tolist(),
                        final_pad_xyz=pad.tolist(), final_xy_error_m=float(np.linalg.norm(bottle[:2]-pad[:2])),
                        left_gripper_open=bool(task.robot.is_left_gripper_open()),
                        right_gripper_open=bool(task.robot.is_right_gripper_open()))
                record["success"] = bool(record["controller_complete"] and (
                    record["task_success"] if args.mode == "hybrid" else record["handoff"]["ready"]))
                with timer.phase("artifacts"):
                    base.snapshot(task, directory, "final")
            except Exception as error:
                record["evaluation_error"] = f"{type(error).__name__}: {error}"
                record["success"] = False
            record["vla_steps"] = int(task.take_action_cnt)
        record["elapsed_s"] = time.perf_counter()-started
        record["timing_profile"] = timer.snapshot(record["elapsed_s"])
        base.dump(directory/"result.json", record)
        try:
            task.close_env()
        except Exception as error:
            record["close_env_error"] = str(error)
            base.dump(directory/"result.json", record)
    return record
