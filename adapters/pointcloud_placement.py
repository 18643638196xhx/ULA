"""Model-free placement using an online RGB-D reference and table-contact prior.

Restricted pilot: a rigid cup initially upright and resting on a horizontal
table, with sufficiently overlapping visible surfaces before/after grasp.
The support anchor is approximate, not a recovered hidden CAD functional point.
"""

import json
from pathlib import Path

import numpy as np

from adapters.handoff import handoff_ready
from adapters import pointcloud_registration


def placement_candidates(reference_bounds, table_z, motion, ee_world, target, release_gap_m):
    """Carry an approximate initial support anchor through the observed motion."""
    bounds = np.asarray(reference_bounds,dtype=float)
    motion = np.asarray(motion,dtype=float)
    ee_world = np.asarray(ee_world,dtype=float)
    target = np.asarray(target,dtype=float)
    if (bounds.shape != (2,3) or not np.isfinite(bounds).all()
            or np.any(bounds[1,:2] <= bounds[0,:2]) or bounds[1,2] <= table_z+0.01):
        raise ValueError("Invalid pregrasp visible bounds/table contact reference")
    if (motion.shape != (4,4) or ee_world.shape != (4,4) or target.shape != (3,)
            or not all(np.isfinite(v).all() for v in (motion,ee_world,target))
            or not np.isfinite(table_z) or not np.isfinite(release_gap_m)
            or not 0 <= release_gap_m <= 0.02):
        raise ValueError("Invalid pointcloud placement geometry or release gap")
    initial_anchor = bounds.mean(axis=0)
    initial_anchor[2] = table_z
    held_anchor = motion[:3,:3]@initial_anchor+motion[:3,3]
    target_anchor = target.copy()
    target_anchor[2] += release_gap_m
    goals = []
    for yaw in (0.,-0.15,0.15,-0.30,0.30,-0.60,0.60):
        c, s = np.cos(yaw),np.sin(yaw)
        restore = np.array([[c,-s,0],[s,c,0],[0,0,1]])@motion[:3,:3].T
        desired = np.eye(4)
        desired[:3,:3] = restore@ee_world[:3,:3]
        desired[:3,3] = target_anchor+restore@(ee_world[:3,3]-held_anchor)
        goals.append((yaw,desired))
    audit = dict(
        support_anchor_source="initial_visible_xy_bounds_center_at_visual_table_height",
        initial_support_anchor_world=initial_anchor.tolist(),
        held_support_anchor_world=held_anchor.tolist(), target_support_anchor_world=target_anchor.tolist(),
        initial_observed_height_m=float(bounds[1,2]-table_z), release_gap_m=release_gap_m,
        orientation_source="restore_pregrasp_orientation_from_observed_rigid_motion_plus_yaw",
        initial_upright_assumed=True, initial_upright_verified=False,
        hidden_bottom_reconstructed=False, support_xy_is_approximate=True,
    )
    return goals, audit


def finish_pointcloud_placement(task, perception, arm, reference_path, result, *, release_gap_m=0.003,
                               registration_config=None, reobserve=False):
    from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz, pose_wxyz_to_matrix

    result.update(geometry_source="online_reference_rgbd_registration", complete=False,
                  uses_prepared_object_model=False, uses_foundationpose=False,
                  uses_object_truth_for_planning=False, uses_functional_point_annotations=False,
                  reference_cloud_npz=str(Path(reference_path).resolve()), actions=[], planning_attempts=[])
    # Only this episode's observed surface is loaded; never a mesh/model metadata file.
    with np.load(reference_path,allow_pickle=False) as reference:
        points = reference["cup_points_world"].copy()
        bounds = reference["cup_bounds_world"].copy()
        observation = json.loads(str(reference["observation_audit_json"]))
    config = registration_config or pointcloud_registration.RegistrationConfig()
    if reobserve:
        from adapters.reobservation import observe_for_placement

        result["reobservation"] = {}
        estimate, motion, registration = observe_for_placement(
            task, perception, arm, reference_path, config, result["reobservation"])
    else:
        camera = task.get_camera_observation("head_camera")
        estimate = perception.estimate(camera["rgb"],camera["depth"],
                                       camera["intrinsic_cv"],camera["extrinsic_cv"])
        motion, registration = pointcloud_registration.estimate_pointcloud_motion(
            points,estimate.cup_points_world,config=config)
    result["timing_s"] = estimate.timings_s
    result["registration"] = registration
    if motion is None:
        raise RuntimeError("Pointcloud placement rejected: unreliable surface registration")
    ee_world = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    candidates, geometry = placement_candidates(bounds,observation["table_height_m"],motion,ee_world,
                                                estimate.coaster_world_center,release_gap_m)
    result.update(geometry, coaster_world_center=estimate.coaster_world_center.tolist(),
                  initial_ee_pose=task.get_arm_pose(arm))
    plan_path = getattr(task.robot,f"{arm}_plan_path")
    entity = getattr(task.robot,f"{arm}_entity")
    all_names = [joint.get_name() for joint in entity.get_active_joints()]
    indices = [all_names.index(name) for name in getattr(task.robot,f"{arm}_arm_joints_name")]
    chosen = None
    for yaw, desired in candidates:
        for clearance in (0.05,0.08,0.10):
            above = desired.copy()
            above[2,3] += clearance
            pre_plan = plan_path(matrix_to_pose_wxyz(above))
            attempt = dict(yaw_rad=yaw,clearance_m=clearance,pre_status=pre_plan["status"],
                           planner_status=pre_plan.get("planner_status"))
            result["planning_attempts"].append(attempt)
            if pre_plan["status"] != "Success":
                if "INVALID_START" in str(pre_plan.get("planner_status","")):
                    raise RuntimeError("Pointcloud placement planner rejected VLA end state")
                continue
            future = np.asarray(entity.get_qpos()).copy()
            future[indices] = pre_plan["position"][-1]
            place_plan = plan_path(matrix_to_pose_wxyz(desired),last_qpos=future)
            attempt["place_status"] = place_plan["status"]
            if place_plan["status"] == "Success":
                chosen = (pre_plan,above),(place_plan,desired)
                result.update(chosen_yaw_rad=yaw,chosen_clearance_m=clearance,
                              target_ee_pose=matrix_to_pose_wxyz(desired).tolist())
                break
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError("Pointcloud placement: no feasible transport/place pair")
    for name, (trajectory, target) in zip(("move_pre_place","move_place"),chosen):
        controls = dict(left_arm=None,right_arm=None,left_gripper=None,right_gripper=None)
        controls[f"{arm}_arm"] = trajectory
        moved = bool(task.take_dense_action(controls))
        arrival = handoff_ready(task.get_arm_pose(arm),matrix_to_pose_wxyz(target),
                                moved and task.plan_success,0.02,15.)
        result["actions"].append(dict(action=name,success=arrival.ready,arrival=arrival.to_dict()))
        if not arrival.ready:
            raise RuntimeError(f"Pointcloud placement failed at {name}: {arrival.reason}")
    for name, build in (("release",lambda:task.open_gripper(arm,pos=1.)),
                        ("retreat",lambda:task.move_by_displacement(arm,z=0.05,move_axis="world"))):
        moved = bool(task.move(build()))
        result["actions"].append(dict(action=name,success=moved))
        if not moved or not task.plan_success:
            raise RuntimeError(f"Pointcloud placement failed at {name}")
    result["complete"] = True
