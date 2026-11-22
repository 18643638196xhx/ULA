"""Dual-arm hold-and-place: main arm places the object, opposite arm lifts container."""

import numpy as np

from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz, pose_wxyz_to_matrix


def _gripper_open(task, arm):
    return bool(task.is_left_gripper_open() if arm == "left" else task.is_right_gripper_open())


def _move(task, arm, matrix, stage, result):
    from envs.utils import Action, ArmTag

    target = matrix_to_pose_wxyz(matrix)
    moved = bool(task.move((ArmTag(arm), [Action(ArmTag(arm), "move", target_pose=target)])))
    success = bool(moved and task.plan_success)
    result.setdefault("actions", []).append(dict(
        action=stage, arm=arm, target_pose=target.tolist(), success=success))
    if not success:
        raise RuntimeError(f"Dual-arm failed at {stage}: motion plan failed")
    return pose_wxyz_to_matrix(task.get_arm_pose(arm))


def _measure_center(perception, task, which):
    camera = task.get_camera_observation("head_camera")
    estimate = perception.estimate(camera["rgb"], camera["depth"],
                                   camera["intrinsic_cv"], camera["extrinsic_cv"])
    points = np.asarray(estimate.cup_points_world if which == "object"
                        else estimate.coaster_points_world, float)
    if len(points) < 6:
        raise RuntimeError(f"Trial lift: {which} has too few measured points")
    return points.mean(axis=0), estimate


def _trial_lift(task, perception, arm, which, result, *, lift_m=0.03, min_follow_m=0.02):
    """Raise the wrist and confirm the measured target really follows."""
    before, _ = _measure_center(perception, task, which)
    ee = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    up = ee.copy()
    up[:3, 3] = ee[:3, 3] + np.array([0.0, 0.0, lift_m])
    _move(task, arm, up, f"trial_lift_{which}", result)
    after, _ = _measure_center(perception, task, which)
    follow = float(after[2] - before[2])
    accepted = bool(follow >= min_follow_m)
    result[f"trial_lift_{which}"] = dict(
        before_z_m=float(before[2]), after_z_m=float(after[2]), follow_m=follow,
        min_follow_m=float(min_follow_m), accepted=accepted,
        uses_simulator_truth=False)
    if not accepted:
        raise RuntimeError(
            f"Trial lift failed for {which}: it did not follow the gripper "
            f"({follow:.3f} m < {min_follow_m:.3f} m)")
    return accepted


def _container_threshold(rule, default=0.02):
    for item in (getattr(rule, "success", {}) or {}).get("criteria") or []:
        if isinstance(item, dict) and "container_lifted" in item:
            return float(item["container_lifted"].get("min_dz_m", default))
    return default


def dual_arm_hold_place(task, perception, main_arm, rule, result, *,
                        approach_margin_m=0.06, clearance_m=0.05,
                        position_tolerance_m=0.02, orientation_tolerance_deg=15.0):
    """Main arm drops the object; opposite arm then grasps and lifts the container."""
    from adapters.handoff import handoff_ready
    from envs.utils import Action, ArmTag

    opposite = "right" if main_arm == "left" else "left"
    result.setdefault("actions", [])
    camera = task.get_camera_observation("head_camera")
    estimate = perception.estimate(camera["rgb"], camera["depth"],
                                   camera["intrinsic_cv"], camera["extrinsic_cv"])
    container = np.asarray(estimate.coaster_points_world, float)
    if len(container) < 6:
        raise RuntimeError("Dual-arm: container has too few measured points")
    container_center = np.asarray(estimate.coaster_world_center, float).copy()
    container_center[:2] = np.median(container[:, :2], axis=0)
    container_top = float(np.percentile(container[:, 2], 90))
    object_points = np.asarray(estimate.cup_points_world, float)
    if len(object_points) < 6:
        raise RuntimeError("Dual-arm: held object has too few measured points")
    object_center = object_points.mean(axis=0)
    object_bottom = float(np.min(object_points[:, 2]))

    # --- main arm: verify the object really follows, then place -------------
    _trial_lift(task, perception, main_arm, "object", result)
    ee = pose_wxyz_to_matrix(task.get_arm_pose(main_arm))
    offset = object_center - ee[:3, 3]
    drop_height = container_top + object_center[2] - object_bottom + 0.05
    above = ee.copy()
    above[:3, 3] = np.array([container_center[0], container_center[1], drop_height]) - offset
    _move(task, main_arm, above, "dual_place_above", result)
    opened = bool(task.move(task.open_gripper(main_arm, pos=1.0)))
    result["actions"].append(dict(action="dual_release", arm=main_arm, success=opened))
    if not opened or not _gripper_open(task, main_arm):
        raise RuntimeError("Dual-arm: main gripper did not open (handoff D rejected)")

    # --- main arm: clear upward and return to origin ------------------------
    current = pose_wxyz_to_matrix(task.get_arm_pose(main_arm))
    clear = current.copy()
    clear[:3, 3] = current[:3, 3] + np.array([0.0, 0.0, clearance_m])
    _move(task, main_arm, clear, "dual_main_clear", result)
    moved = bool(task.move(task.back_to_origin(main_arm)))
    ee_main = pose_wxyz_to_matrix(task.get_arm_pose(main_arm))
    away = float(np.linalg.norm(ee_main[:3, 3] - container_center)) > 0.10
    result["handoff_d"] = dict(gripper_open=_gripper_open(task, main_arm),
                               ee_distance_m=float(np.linalg.norm(ee_main[:3, 3] - container_center)),
                               moved=moved, accepted=bool(_gripper_open(task, main_arm) and away))
    if not (moved and away):
        raise RuntimeError("Dual-arm: main arm did not clear the container")

    # --- opposite arm: approach the container and close ---------------------
    pre = np.array([container_center[0], container_center[1],
                    container_top + approach_margin_m, 0.5, 0.5, 0.5, -0.5])
    moved = bool(task.move((ArmTag(opposite), [Action(ArmTag(opposite), "move", target_pose=pre)])))
    decision = handoff_ready(task.get_arm_pose(opposite), pre, moved and task.plan_success,
                             position_tolerance_m, orientation_tolerance_deg)
    result["handoff_e"] = decision.to_dict()
    if not decision.ready:
        raise RuntimeError(f"Dual-arm: container approach rejected ({decision.reason})")
    closed = bool(task.move(task.close_gripper(opposite, pos=0.0)))
    result["handoff_f"] = dict(gripper_open=_gripper_open(task, opposite), closed=closed,
                               accepted=bool(closed and not _gripper_open(task, opposite)))
    if not closed or _gripper_open(task, opposite):
        raise RuntimeError("Dual-arm: opposite gripper did not close on the container")
    _trial_lift(task, perception, opposite, "support", result)

    # --- opposite arm: lift by the declared success threshold + margin ------
    threshold = _container_threshold(rule)
    lift_m = threshold + 0.02
    current = pose_wxyz_to_matrix(task.get_arm_pose(opposite))
    lifted = current.copy()
    lifted[:3, 3] = current[:3, 3] + np.array([0.0, 0.0, lift_m])
    _move(task, opposite, lifted, "dual_lift_container", result)
    result["dual_arm"] = dict(
        main_arm=main_arm, opposite_arm=opposite,
        container_center_world=container_center.tolist(),
        container_top_m=container_top,
        lift_threshold_m=threshold, lift_applied_m=lift_m,
        uses_simulator_truth=False,
    )
    result["complete"] = True
    return result
