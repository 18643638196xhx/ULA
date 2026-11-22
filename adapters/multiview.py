"""Bounded supplementary wrist observations; no robot-mask or ICP changes."""

import time
from types import SimpleNamespace

import numpy as np

from adapters.handoff import handoff_ready


def clipped(estimate):
    return bool((estimate.observation_audit or {}).get("cup_mask_touches_border",False))


def move_ee_checked(task,arm,target):
    plan = getattr(task.robot,f"{arm}_plan_path")(target)
    audit = dict(status=plan["status"],planner_status=plan.get("planner_status"),target_ee_pose=list(target))
    if plan["status"] != "Success":
        return False,audit
    controls = dict(left_arm=None,right_arm=None,left_gripper=None,right_gripper=None)
    controls[f"{arm}_arm"] = plan
    moved = bool(task.take_dense_action(controls))
    decision = handoff_ready(task.get_arm_pose(arm),target,moved and task.plan_success)
    audit["arrival"] = decision.to_dict()
    return decision.ready,audit


def centred_wrist_target(ee_pose,camera,observed_bounds,max_shift_m=0.10):
    """Translate the wrist camera in its image plane; keep EE orientation."""
    from traditional_place_empty_cup.geometry import homogeneous_transform

    world_to_camera = homogeneous_transform(camera["extrinsic_cv"])
    centre = np.asarray(observed_bounds,dtype=float).mean(axis=0)
    point = world_to_camera@np.r_[centre,1.]
    if point[2] <= 0:
        raise ValueError("Cup centre lies behind the wrist camera")
    shift = world_to_camera[:3,:3].T@np.array([point[0],point[1],0.])
    if not np.isfinite(shift).all() or np.linalg.norm(shift) > max_shift_m:
        raise ValueError("Required wrist centring shift exceeds bounded observation motion")
    target = np.asarray(ee_pose,dtype=float).copy()
    target[:3] += shift
    return target


def supplement_after_approach(task,perception,head,arm,*,validate_geometry=False):
    """Called only before VLA, while the cup is assumed supported on the table."""
    start = time.perf_counter()
    audit = dict(trigger="head_cup_mask_touches_image_border",camera=f"{arm}_camera",
                 views=[],motions=[],selected="head_camera",resolved=False,
                 robot_region_filtering=False)
    camera_name = f"{arm}_camera"
    selected = None
    for attempt in range(2):
        try:
            camera = task.get_camera_observation(camera_name)
            options = dict(refine_cup=True) if validate_geometry else {}
            estimate = perception.supplementary(camera,f"wrist_view_{attempt}", **options)
            view = dict(attempt=attempt,clipped=clipped(estimate),
                        bounds_world=estimate.cup_bounds_world.tolist(),observation=estimate.observation_audit)
            audit["views"].append(view)
            if validate_geometry:
                from adapters.reobservation import observation_quality

                view["quality"] = observation_quality(estimate)
                if not view["quality"]["geometry_ok"]:
                    view["rejected"] = "unreliable_wrist_geometry"
                    break  # Do not centre a camera using untrusted bounds.
            table_difference = abs(estimate.observation_audit["table_height_m"]
                                   -head.observation_audit["table_height_m"])
            view["table_height_difference_m"] = table_difference
            if table_difference > 0.02:
                view["rejected"] = "inconsistent_observed_table_height"
                break
            if not clipped(estimate):
                selected = estimate
                audit.update(selected=camera_name,resolved=True,selected_attempt=attempt)
                break
            if attempt == 0:
                target = centred_wrist_target(task.get_arm_pose(arm),camera,estimate.cup_bounds_world)
                moved, motion = move_ee_checked(task,arm,target)
                audit["motions"].append(motion)
                if not moved:
                    break
        except (RuntimeError,ValueError,KeyError) as error:
            audit["error"] = f"{type(error).__name__}: {error}"
            break
    audit["wall_s"] = time.perf_counter()-start
    if selected is None:
        # No invented completed bounds. Caller must reject unresolved partial observation.
        return head,audit
    combined = SimpleNamespace(**vars(selected))
    # Wrist improves spatial bounds; retain the head surface for same-view
    # before/after registration. Do not mix opposite partial surfaces in ICP.
    combined.cup_points_world = head.cup_points_world.copy()
    combined.coaster_world_center = head.coaster_world_center.copy()
    combined.observation_audit = dict(selected.observation_audit,
        selected_cup_camera=camera_name,coaster_camera="head_camera",multiview=audit,
        registration_reference_camera="head_camera",bounds_camera=camera_name,
        table_height_m=head.observation_audit["table_height_m"])
    combined.timings_s = dict(head.timings_s,supplementary_observation_wall=audit["wall_s"])
    perception.select_reference(combined)
    return combined,audit
