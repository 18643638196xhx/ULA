"""Bottle-only paired transport/place search, including low clearances.

This task has a horizontal table and a millimetre-thick target pad. A 2 cm
support-point clearance is an explicit scene assumption, not a trial lift.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=8)
def model_vertices(metadata_path):
    """Explicit known CAD, transformed exactly like the visual pose estimator."""
    import trimesh
    path = Path(metadata_path)
    metadata = json.loads(path.read_text())
    suffix = path.stem.removeprefix("model_data")
    mesh = trimesh.load(path.parent/"visual"/f"base{suffix}.glb", force="scene")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    mesh.apply_scale(np.asarray(metadata["scale"]))
    mesh.apply_transform(np.asarray(metadata.get("transform_matrix", np.eye(4))))
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 8 or not np.isfinite(vertices).all():
        raise ValueError("Invalid known CAD support geometry")
    return vertices


def placement_candidates(upright_reference, metadata_path):
    """Prefer upright. Only failed upright searches reach a 5/10-degree tilt.

    Keep the CAD's LOWEST vertex at least as high as for its upright pose.
    Tilting does not authorize model penetration into the pad/table.
    """
    for tilt in (0., 5., 10., -5., -10.):
        vertices = None if tilt == 0 else model_vertices(str(metadata_path))
        if vertices is not None:
            metadata = json.loads(Path(metadata_path).read_text())
            anchor = np.asarray(metadata["functional_matrix"][0], dtype=float)[:3, 3] * metadata["scale"]
            vertices = vertices-anchor
        angle = np.radians(tilt)
        lean = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                         [0., np.sin(angle), np.cos(angle)]])
        for yaw in (0., -.15, .15, -.3, .3, -.6, .6, -1.2, 1.2, -np.pi/2, np.pi/2, np.pi):
            candidate = np.asarray(upright_reference, dtype=float).copy()
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ candidate[:3, :3]
            candidate[:3, :3] = lean @ rotation
            shift = 0. if vertices is None else max(0., float(
                np.min(vertices @ rotation[2])-np.min(vertices @ candidate[2, :3])))
            yield tilt, yaw, candidate, shift


def reverse_trajectory(plan):
    """Retrace a checked approach: reverse waypoints AND velocity sign."""
    position, velocity = np.asarray(plan["position"]), np.asarray(plan["velocity"])
    if (position.ndim != 2 or len(position) < 2 or position.shape != velocity.shape
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
        raise ValueError("Invalid trajectory for retracing")
    return dict(plan, position=position[::-1].copy(), velocity=-velocity[::-1].copy(),
                reverse_source="executed_transport_or_descent")


def finish_placement(task, perception, arm, metadata_path, result, *, upright_reference):
    from traditional_place_empty_cup.geometry import (
        matrix_to_pose_wxyz, placement_ee_pose, pose_wxyz_to_matrix,
    )
    from adapters.pillbottle_known_model import joint_limit_audit, execute_checked

    camera = task.get_camera_observation("head_camera")
    value = perception.estimate(camera["rgb"], camera["depth"],
                                camera["intrinsic_cv"], camera["extrinsic_cv"])
    metadata = json.loads(Path(metadata_path).read_text())
    functional = np.asarray(metadata["functional_matrix"][0], dtype=float).copy()
    functional[:3, 3] *= np.asarray(metadata["scale"])
    ee_to_bottle = np.linalg.solve(pose_wxyz_to_matrix(task.get_arm_pose(arm)), value.cup_world_pose)
    result.update(cup_world_pose=value.cup_world_pose.tolist(),
        coaster_world_center=value.coaster_world_center.tolist(), ee_to_cup=ee_to_bottle.tolist(),
        initial_ee_pose=list(task.get_arm_pose(arm)), timing_s=value.timings_s,
        actions=[], planning_attempts=[], complete=False, uses_object_truth_for_planning=False,
        final_goal_probes=[], release_tilt_candidates_deg=[0., 5., 10., -5., -10.],
        orientation_source="pregrasp_visual_upright_reference",
        transport_clearance_candidates_m=[.05, .03, .02, .08],
        placement_yaw_is_free=True,
        minimum_clearance_assumption="flat_table_with_thin_pad_no_tall_transport_obstacles")
    plan_path = getattr(task.robot, f"{arm}_plan_path")
    entity = getattr(task.robot, f"{arm}_entity")
    names = [joint.get_name() for joint in entity.get_active_joints()]
    indices = [names.index(name) for name in getattr(task.robot, f"{arm}_arm_joints_name")]
    chosen, chosen_retreat = None, None
    for tilt, yaw, candidate, compensation in placement_candidates(upright_reference, metadata_path):
        support_target = np.asarray(value.coaster_world_center).copy()
        support_target[2] += compensation
        desired = placement_ee_pose(candidate, ee_to_bottle, functional, support_target)
        probe = plan_path(matrix_to_pose_wxyz(desired))
        result["final_goal_probes"].append(dict(tilt_deg=tilt, yaw_rad=yaw,
            target_ee_pose=matrix_to_pose_wxyz(desired).tolist(), status=probe["status"],
            planner_status=probe.get("planner_status")))
        for clearance in result["transport_clearance_candidates_m"]:
            above = desired.copy()
            above[2, 3] += clearance
            pre_plan = plan_path(matrix_to_pose_wxyz(above))
            attempt = dict(yaw_rad=yaw, tilt_deg=tilt, support_compensation_m=compensation,
                clearance_m=clearance, pre_status=pre_plan["status"],
                planner_status=pre_plan.get("planner_status"))
            result["planning_attempts"].append(attempt)
            if pre_plan["status"] != "Success":
                result["planner_failure"] = pre_plan
                if "INVALID_START" in str(pre_plan.get("planner_status", "")):
                    raise RuntimeError("Traditional planner rejected the VLA end state")
                continue
            attempt["pre_joint_limits"] = joint_limit_audit(task.robot, arm, pre_plan)
            if not attempt["pre_joint_limits"]["accepted"]:
                continue
            future = np.asarray(entity.get_qpos()).copy()
            future[indices] = pre_plan["position"][-1]
            place_plan = plan_path(matrix_to_pose_wxyz(desired), last_qpos=future)
            attempt["place_status"] = place_plan["status"]
            if place_plan["status"] != "Success":
                result["planner_failure"] = place_plan
                continue
            attempt["place_joint_limits"] = joint_limit_audit(task.robot, arm, place_plan)
            if not attempt["place_joint_limits"]["accepted"]:
                continue
            retreat = None
            if tilt != 0.:
                # Avoid a fresh IK branch beyond the workspace boundary. These
                # exact routes must pass measured arrival checks on the forward
                # execution before release permits retracing them with open jaws.
                retreat = [(matrix_to_pose_wxyz(above), reverse_trajectory(place_plan)),
                           (np.asarray(result["initial_ee_pose"]), reverse_trajectory(pre_plan))]
                attempt["retreat_source"] = "reverse_forward_verified_descent_and_transport"
            chosen = pre_plan, place_plan
            chosen_retreat = retreat
            result.update(target_ee_pose=matrix_to_pose_wxyz(desired).tolist(),
                chosen_yaw_rad=yaw, chosen_clearance_m=clearance, chosen_release_tilt_deg=tilt,
                support_height_compensation_m=compensation)
            break
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError("No feasible bottle transport/place pair including low-clearance candidates")
    # Both trajectories are accepted BEFORE any motion or release.
    for name, trajectory in zip(("move_pre_place", "move_place"), chosen):
        controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
        controls[f"{arm}_arm"] = trajectory
        moved = bool(task.take_dense_action(controls))
        result["actions"].append(dict(action=name, success=moved))
        if not moved:
            raise RuntimeError(f"Traditional placement did not reach {name}; do not release")
    for name, action in (("release", lambda: task.open_gripper(arm, pos=1.)),):
        moved = bool(task.move(action()))
        result["actions"].append(dict(action=name, success=moved))
        if not moved or not task.plan_success:
            raise RuntimeError(f"Traditional placement failed at {name}")
    if chosen_retreat is None:
        moved = bool(task.move(task.move_by_displacement(arm, z=.05, move_axis="world")))
        if not moved or not task.plan_success:
            raise RuntimeError("Traditional placement failed at retreat")
    else:
        result["retreat_execution"] = []
        for target, trajectory in chosen_retreat:
            audit = {}
            result["retreat_execution"].append(audit)
            execute_checked(task, arm, target, trajectory, audit)
    result["actions"].append(dict(action="retreat", success=True))
    result["complete"] = True
