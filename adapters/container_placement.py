"""Container placement primitive: carry a held object above a measured container
interior and release it, then retreat.  No simulator truth is read."""

from pathlib import Path

import numpy as np

from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz, pose_wxyz_to_matrix


def _move(task, arm, matrix, stage, result):
    from envs.utils import Action, ArmTag

    target = matrix_to_pose_wxyz(matrix)
    moved = bool(task.move((ArmTag(arm), [Action(ArmTag(arm), "move", target_pose=target)])))
    success = bool(moved and task.plan_success)
    result.setdefault("actions", []).append(dict(
        action=stage, target_pose=target.tolist(), success=success))
    if not success:
        raise RuntimeError(f"Container placement failed at {stage}: motion plan failed")
    return pose_wxyz_to_matrix(task.get_arm_pose(arm))


def finish_container_placement(task, perception, arm, reference_path, result, *,
                               rule=None, registration_config=None, release_gap_m=None,
                               reobserve=False, candidate_budget=6):
    """Place the held object into the measured container and release it."""
    camera = task.get_camera_observation("head_camera")
    estimate = perception.estimate(camera["rgb"], camera["depth"],
                                   camera["intrinsic_cv"], camera["extrinsic_cv"])
    container = np.asarray(estimate.coaster_points_world, float)
    if len(container) < 6:
        raise RuntimeError("Container has too few measured points")
    container_center = np.asarray(estimate.coaster_world_center, float).copy()
    container_center[:2] = np.median(container[:, :2], axis=0)
    container_top = float(np.percentile(container[:, 2], 90))
    object_points = np.asarray(estimate.cup_points_world, float)
    if len(object_points) < 6:
        raise RuntimeError("Held object has too few measured points")
    object_center = object_points.mean(axis=0)
    object_bottom = float(np.min(object_points[:, 2]))

    ee = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    offset = object_center - ee[:3, 3]
    clearance = 0.05
    drop_height = container_top + clearance + (object_center[2] - object_bottom)
    target = ee.copy()
    target[:3, 3] = np.array([container_center[0], container_center[1], drop_height]) - offset

    result.update(
        complete=False,
        geometry_source="measured_container_interior",
        uses_simulator_truth=False,
        container_center_world=container_center.tolist(),
        container_top_m=container_top,
        object_center_world=object_center.tolist(),
        drop_height_m=drop_height,
        release_mode=("drop_into" if rule is None else rule.release_mode),
        candidate_budget=int(candidate_budget),
    )
    stage = "container_move_above"
    current = _move(task, arm, target, stage, result)
    stage = "container_pre_release"
    above = target.copy()
    above[:3, 3] = current[:3, 3]
    _move(task, arm, above, stage, result)
    opened = bool(task.move(task.open_gripper(arm, pos=1.0)))
    result.setdefault("actions", []).append(dict(action="container_release", success=opened))
    if not opened or not task.plan_success:
        raise RuntimeError("Container placement failed to release")
    retreat = above.copy()
    retreat[:3, 3] = above[:3, 3] - np.array([0.0, 0.0, -0.10])
    _move(task, arm, retreat, "container_retreat", result)
    result["complete"] = True
    return result
