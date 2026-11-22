"""Finite current-frame visual close checks using this episode's RGB-D reference."""
from itertools import product

import numpy as np

from adapters.pointcloud_registration import RegistrationConfig, estimate_pointcloud_motion


def bounds_distance(bounds, point):
    bounds, point = np.asarray(bounds, dtype=float), np.asarray(point, dtype=float)
    if (bounds.shape != (2, 3) or point.shape != (3,) or not np.isfinite(bounds).all()
            or not np.isfinite(point).all() or np.any(bounds[1] < bounds[0])):
        raise ValueError("Invalid bounds/point")
    outside = np.maximum(bounds[0]-point, 0.)+np.maximum(point-bounds[1], 0.)
    return float(np.linalg.norm(outside))


def registered_proximity(reference, estimate, tcp, *, threshold=.015):
    """Observed reference bounds can cover a flank occluded in the current view.

    This is a close-candidate heuristic only. Neither an accepted ICP nor an
    interior TCP is a grasp certificate. Visual trial lift remains mandatory.
    """
    current_bounds = np.asarray(estimate.cup_bounds_world, dtype=float)
    result = dict(distance_to_visible_bounds_m=bounds_distance(current_bounds, tcp),
        visible_bounds_world=current_bounds.tolist(), threshold_m=threshold, accepted=False,
        uses_object_truth=False)
    if result["distance_to_visible_bounds_m"] <= threshold:
        result.update(accepted=True, proximity_geometry="current_observed_bounds",
            registration_required_for_candidate=False)
        return result
    motion, registration = estimate_pointcloud_motion(reference.cup_points_world,
        estimate.cup_points_world, RegistrationConfig())
    result["registration"] = registration
    if motion is None:
        return result
    bounds = np.asarray(reference.cup_bounds_world)
    corners = np.array(list(product(*zip(bounds[0], bounds[1]))))
    moved = corners@motion[:3, :3].T+motion[:3, 3]
    propagated = np.array([moved.min(0), moved.max(0)])
    distance = bounds_distance(propagated, tcp)
    result.update(registered_reference_bounds_world=propagated.tolist(),
        distance_to_registered_reference_bounds_m=distance,
        proximity_geometry="same_episode_observed_bounds_propagated_by_measured_icp",
        accepted=distance <= threshold)
    return result


def observe_close_candidate(task, perception, arm, step, reference, checks, *, threshold=.015):
    from adapters.pillbottle import observation_quality

    candidates = [("head_camera", False), ("head_camera", True), (f"{arm}_camera", True)]
    for camera_name, prompted in candidates:
        if prompted and not hasattr(perception, "hinted"):
            continue
        audit = dict(vla_step=int(step), camera=camera_name, prompted=prompted,
                     threshold_m=threshold, uses_object_truth=False)
        checks.append(audit)
        try:
            frame = task.get_camera_observation(camera_name)
            label = f"close_{step}_{camera_name}_{int(prompted)}"
            estimate = (perception.hinted(frame, label, reference.cup_points_world) if prompted else
                        perception.supplementary(frame, label, cup_only=True, refine_cup=False))
            audit["quality"] = observation_quality(estimate, reference.cup_bounds_world)
            if not audit["quality"]["geometry_ok"]:
                continue
            tcp = np.asarray(getattr(task.robot, f"get_{arm}_tcp_pose")(), dtype=float)[:3]
            audit.update(tcp_world=tcp.tolist(), **registered_proximity(reference, estimate, tcp, threshold=threshold))
            if audit["accepted"]:
                perception.select_reference(estimate)
                return dict(audit, observation_reference=str(perception.reference_path))
        except (RuntimeError, ValueError) as error:
            audit["error"] = str(error)
    return None
