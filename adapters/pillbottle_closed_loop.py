"""Bottle-only visual goal correction during real transport, never a trial lift.

The pad is static: its initial RGB-D center is retained when it is occluded.
No object actor, physics contact, identity mask, or official score is read.
"""
import json
from pathlib import Path

import numpy as np

from adapters.handoff import handoff_ready
from adapters.pillbottle_close import bounds_distance
from adapters.pillbottle_known_placement import placement_candidates, reverse_trajectory, model_vertices
from adapters.pillbottle_scene_geometry import gravity_upright


def hold_robot(task, arm, steps, *, gripper=None):
    """Bounded encoder-target hold; not an object contact or stability test."""
    if not 0 <= steps <= 500:
        raise ValueError("Invalid bounded settling duration")
    q = np.asarray(getattr(task.robot, f"get_{arm}_arm_real_jointState")(), dtype=float)[:6]
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Invalid measured settling joints")
    for _ in range(steps):
        task.robot.set_arm_joints(q, np.zeros(6), arm)
        if gripper is not None:
            task.robot.set_gripper(gripper, arm)
        task.scene.step()


def capture_stage(task, perception, arm, label):
    """Save camera observations for diagnosis, never simulator actor state."""
    if not hasattr(task, "get_camera_observation") or not hasattr(perception, "output"):
        return
    from PIL import Image
    perception.output.mkdir(parents=True, exist_ok=True)
    for camera_name in ("head_camera", f"{arm}_camera"):
        frame = task.get_camera_observation(camera_name)
        Image.fromarray(np.asarray(frame["rgb"][..., :3], dtype=np.uint8)).save(
            perception.output/f"{label}_{camera_name}.png")


def execute_with_local_retry(task, arm, target, plan, audit):
    from adapters.pillbottle_known_model import execute_checked, joint_limit_audit
    try:
        execute_checked(task, arm, target, plan, audit)
        return plan
    except RuntimeError as error:
        audit["first_execution_error"] = str(error)
        # Do not push again after a large Cartesian miss. A nearby final pose
        # may have an unreachable joint branch; replan from ACTUAL encoders.
        if not handoff_ready(task.get_arm_pose(arm), target, task.plan_success).ready:
            raise
        retry = getattr(task.robot, f"{arm}_plan_path")(target)
        audit["local_retry"] = dict(status=retry["status"], source="same_goal_from_actual_joints")
        if retry["status"] != "Success" or not joint_limit_audit(task.robot, arm, retry)["accepted"]:
            raise RuntimeError("Local arrival retry planning rejected; do not release") from error
        execute_checked(task, arm, target, retry, audit["local_retry"])
        return retry


def finish_visual_placement(task, perception, arm, metadata_path, result, *,
                            upright_reference, release_gap_m=.003):
    from adapters.pillbottle_known_model import joint_limit_audit, support_point
    from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz, pose_wxyz_to_matrix, placement_ee_pose

    metadata = json.loads(Path(metadata_path).read_text())
    vertices = model_vertices(str(metadata_path))
    functional = np.asarray(metadata["functional_matrix"][0], dtype=float).copy()
    functional[:3, 3] *= np.asarray(metadata["scale"])
    result.update(placement_policy="visual", actions=[], planning_attempts=[],
        arrival_checks=[], observations=[], complete=False, trial_lift_enabled=False,
        lifted_grasp_verified=False, uses_prepared_object_model=True,
        uses_object_truth_for_planning=False, geometry_source="fresh_rgbd_goal_correction",
        support_rise_required=False, release_gap_m=release_gap_m,
        maximum_goal_corrections=2, post_release_hold_steps=250,
        pad_source="initial_rgbd_static_pad_not_hidden_current_center")
    cached_pad = getattr(perception.worker, "reference_pad", None)
    result["pad_target_validation"] = getattr(perception.worker, "pad_target_validation", None)
    destination = None

    def held_proximity(value):
        tcp = np.asarray(getattr(task.robot, f"get_{arm}_tcp_pose")(), dtype=float)[:3]
        visible_distance = bounds_distance(value.cup_bounds_world, tcp)
        world = vertices@value.cup_world_pose[:3, :3].T+value.cup_world_pose[:3, 3]
        distance = bounds_distance(np.array([world.min(0), world.max(0)]), tcp)
        return dict(accepted=bool(np.isfinite(distance) and distance <= .03),
            reason="within_hand_proximity" if np.isfinite(distance) and distance <= .03
                else "Visual bottle no longer near hand; do not place or release",
            tcp_to_visible_object_bounds_m=visible_distance,
            tcp_to_observed_cad_bounds_m=distance, maximum_tcp_to_cad_distance_m=.03,
            proximity_source="known_cad_extent_at_current_depth_validated_pose")

    def observe(label, hint=None):
        nonlocal destination
        audit = dict(stage=label)
        result["observations"].append(audit)
        value = perception.localize_held(task, arm, audit, pose_hint_world=hint,
            candidate_validator=held_proximity)
        if destination is None:
            destination = np.asarray(value.coaster_world_center if cached_pad is None else cached_pad, dtype=float).copy()
            if destination.shape != (3,) or not np.isfinite(destination).all():
                raise RuntimeError("No finite initial visual pad center")
        audit.update(cup_world_pose=value.cup_world_pose.tolist(),
            current_visible_pad_center=value.coaster_world_center.tolist(),
            retained_static_pad_center=destination.tolist())
        proximity = held_proximity(value)
        audit.update(proximity)
        # Defense in depth: even a replacement perception implementation must
        # not bypass the unchanged proximity gate and authorize a release.
        if not proximity["accepted"]:
            raise RuntimeError(proximity["reason"])
        return value

    value = observe("before_transport")
    initial_ee = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    ee_to_bottle = np.linalg.solve(initial_ee, value.cup_world_pose)
    upright = gravity_upright(upright_reference)
    result.update(cup_world_pose=value.cup_world_pose.tolist(),
        coaster_world_center=destination.tolist(), ee_to_cup=ee_to_bottle.tolist(),
        initial_ee_pose=matrix_to_pose_wxyz(initial_ee).tolist(), timing_s=value.timings_s)
    planner = getattr(task.robot, f"{arm}_plan_path")
    entity = getattr(task.robot, f"{arm}_entity")
    names = [j.get_name() for j in entity.get_active_joints()]
    indices = [names.index(n) for n in getattr(task.robot, f"{arm}_arm_joints_name")]
    chosen = None
    for tilt, yaw, candidate, compensation in placement_candidates(upright, metadata_path):
        support_target = destination.copy()
        support_target[2] += release_gap_m+compensation
        desired = placement_ee_pose(candidate, ee_to_bottle, functional, support_target)
        for clearance in (.05, .03, .02, .08):
            above = desired.copy()
            above[2, 3] += clearance
            pre = planner(matrix_to_pose_wxyz(above))
            attempt = dict(tilt_deg=tilt, yaw_rad=yaw, clearance_m=clearance, pre_status=pre["status"])
            result["planning_attempts"].append(attempt)
            if pre["status"] != "Success":
                if "INVALID_START" in str(pre.get("planner_status", "")):
                    raise RuntimeError("Traditional planner rejected measured robot start")
                continue
            if not joint_limit_audit(task.robot, arm, pre)["accepted"]:
                continue
            future = np.asarray(entity.get_qpos()).copy()
            future[indices] = pre["position"][-1]
            descent = planner(matrix_to_pose_wxyz(desired), last_qpos=future)
            attempt["place_status"] = descent["status"]
            if descent["status"] != "Success" or not joint_limit_audit(task.robot, arm, descent)["accepted"]:
                continue
            chosen = candidate, support_target, above, pre
            result.update(chosen_release_tilt_deg=tilt, chosen_yaw_rad=yaw,
                chosen_clearance_m=clearance, support_height_compensation_m=compensation,
                target_ee_pose=matrix_to_pose_wxyz(desired).tolist())
            break
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError("No feasible visual bottle transport/descent pair")
    candidate, support_target, above, pre = chosen
    route = []

    def move(label, goal, plan):
        start = np.asarray(task.get_arm_pose(arm), dtype=float)
        audit = dict(stage=label)
        result["arrival_checks"].append(audit)
        executed = execute_with_local_retry(task, arm, matrix_to_pose_wxyz(goal), plan, audit)
        route.append((start, executed))
        result["actions"].append(dict(action=label, success=True))

    move("move_pre_place", above, pre)
    # This is the already requested transport, not an extra grasp-test motion.
    # A rigid-hold prediction is only a pose initializer; fresh RGB-D must fit.
    predicted = pose_wxyz_to_matrix(task.get_arm_pose(arm))@ee_to_bottle
    current = observe("after_transport", predicted)
    for correction in range(3):
        current_ee = pose_wxyz_to_matrix(task.get_arm_pose(arm))
        relation = np.linalg.solve(current_ee, current.cup_world_pose)
        goal = placement_ee_pose(candidate, relation, functional, support_target)
        translation = float(np.linalg.norm(goal[:3, 3]-current_ee[:3, 3]))
        angle = handoff_ready(matrix_to_pose_wxyz(current_ee), matrix_to_pose_wxyz(goal), True).orientation_error_deg
        if translation > .10 or angle > 45.:
            raise RuntimeError("Visual descent correction exceeds local budget; do not release")
        plan = planner(matrix_to_pose_wxyz(goal))
        if plan["status"] != "Success" or not joint_limit_audit(task.robot, arm, plan)["accepted"]:
            raise RuntimeError("Updated visual descent planning failed; do not release")
        result["target_ee_pose"] = matrix_to_pose_wxyz(goal).tolist()
        move("move_place" if correction == 0 else f"visual_correction_{correction}", goal, plan)
        predicted = pose_wxyz_to_matrix(task.get_arm_pose(arm))@relation
        current = observe(f"before_release_{correction}", predicted)
        support = support_point(current.cup_world_pose, metadata)
        delta = support-support_target
        ready = (np.linalg.norm(delta[:2]) <= .012 and abs(delta[2]) <= .008
                 and current.cup_world_pose[2, 1] >= np.cos(np.radians(12.)))
        result["observations"][-1].update(support_error_m=delta.tolist(), release_goal_ready=bool(ready))
        if ready:
            break
    else:
        raise RuntimeError("Visual bottle goal not confirmed after bounded corrections; do not release")
    # Choose a local exit BEFORE release. Withdraw away from the pinch point
    # along -tool X, rather than rotating back through the entire carry route.
    end = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    retreat = None
    result["retreat_planning"] = []
    for distance in (.06, .04):
        target = end.copy()
        target[:3, 3] -= end[:3, 0]*distance
        plan = planner(matrix_to_pose_wxyz(target))
        result["retreat_planning"].append(dict(distance_m=distance, status=plan["status"]))
        if plan["status"] == "Success" and joint_limit_audit(task.robot, arm, plan)["accepted"]:
            retreat = [(matrix_to_pose_wxyz(target), plan)]
            result["retreat_source"] = "preplanned_local_tool_axis_withdrawal"
            break
    if retreat is None:
        # Retrace only verified descent/corrections; returning to the original
        # pickup location is unnecessary and can sweep open fingers into bottle.
        retreat = [(target, reverse_trajectory(plan)) for target, plan in reversed(route[1:])]
        result["retreat_source"] = "verified_descent_only_not_transport_return"
    capture_stage(task, perception, arm, "before_release")
    if not task.move(task.open_gripper(arm, pos=1.)) or not task.plan_success:
        raise RuntimeError("Bottle release failed")
    result["actions"].append(dict(action="release", success=True))
    hold_robot(task, arm, 250, gripper=1.)
    capture_stage(task, perception, arm, "after_release_before_retreat")
    result["retreat_execution"] = []
    for target, plan in retreat:
        audit = {}
        result["retreat_execution"].append(audit)
        execute_with_local_retry(task, arm, target, plan, audit)
    hold_robot(task, arm, 150, gripper=1.)
    capture_stage(task, perception, arm, "after_retreat")
    result["actions"].append(dict(action="retreat", success=True))
    result["complete"] = True
