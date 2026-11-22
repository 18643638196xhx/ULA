"""Contact-annotation-free, top-down approach goals for the upright-cup pilot.

This is a geometric heuristic, not a learned grasp detector or a safety proof.
Targets can use either model geometry plus pose, or measured RGB-D bounds.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class GeometryApproachConfig:
    top_clearance_m: float = 0.10
    edge_fraction: float = 1.0
    yaw_offsets_rad: tuple = (0.0, -0.15, 0.15)

    def __post_init__(self):
        if not np.isfinite(self.top_clearance_m) or self.top_clearance_m <= 0:
            raise ValueError("Geometry top clearance must be finite and positive")
        if not np.isfinite(self.edge_fraction) or not 0 <= self.edge_fraction <= 1:
            raise ValueError("Geometry edge fraction must be finite and in [0,1]")
        if not self.yaw_offsets_rad or not np.isfinite(self.yaw_offsets_rad).all():
            raise ValueError("Geometry yaw offsets must be a nonempty finite sequence")

    def to_dict(self):
        return dict(top_clearance_m=float(self.top_clearance_m),
                    edge_fraction=float(self.edge_fraction),
                    yaw_offsets_rad=[float(v) for v in self.yaw_offsets_rad])


def _transform(value, name):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-4)):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return value


def generate_geometry_candidates(cup_world, metadata, arm, ee_to_tcp, config):
    """Return world-frame TCP/EE targets and auditable model bounding geometry.

    World +Z is table-up; left/right use the -X/+X side of the world AABB.
    The canonical EE/TCP local +X points down, following RoboTwin's convention.
    EE-to-TCP comes from robot kinematics, never from a guessed centre distance.
    """
    if arm not in ("left", "right"):
        raise ValueError("Geometry approach arm must be left or right")
    cup_world = _transform(cup_world, "cup_world")
    ee_to_tcp = _transform(ee_to_tcp, "ee_to_tcp")
    model_transform = _transform(metadata.get("transform_matrix", np.eye(4)), "model transform")
    vectors = {}
    for name in ("center", "extents", "scale"):
        value = np.asarray(metadata[name], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Model {name} must be a finite 3-vector")
        if name != "center" and np.any(value <= 0):
            raise ValueError(f"Model {name} must be positive")
        vectors[name] = value
    corners = (np.asarray(list(product((-1, 1), repeat=3)))
               * vectors["extents"] / 2 + vectors["center"]) * vectors["scale"]
    object_world = cup_world @ model_transform
    corners_world = corners @ object_world[:3, :3].T + object_world[:3, 3]
    lower, upper = corners_world.min(axis=0), corners_world.max(axis=0)
    candidates, audit = generate_bounds_candidates(np.stack((lower, upper)), arm, ee_to_tcp, config)
    audit.update(source="estimated_pose_and_model_bounding_box", requires_known_model_geometry=True,
                 model_geometry_keys=["center", "extents", "scale", "transform_matrix"])
    return candidates, audit


def generate_bounds_candidates(bounds_world, arm, ee_to_tcp, config, *, reference_tcp_world=None):
    """Build approach templates directly from measured bounds, without model input.

    The arm-side point remains first for backwards compatibility.  When the
    current TCP pose is available, two additional *position* hypotheses are
    generated on the object side facing the arm: one on the observed bounding
    surface and one bounding-sphere radius farther away.  This makes the
    fallback search positions as well as orientations, and derives its scale
    from the current observation instead of a task-specific distance.
    """
    if arm not in ("left", "right"):
        raise ValueError("Geometry approach arm must be left or right")
    ee_to_tcp = _transform(ee_to_tcp, "ee_to_tcp")
    bounds = np.asarray(bounds_world, dtype=float)
    if (bounds.shape != (2,3) or not np.isfinite(bounds).all()
            or np.any(bounds[1] < bounds[0]) or np.any(bounds[1,:2] <= bounds[0,:2])):
        raise ValueError("Expected finite ordered world bounds with positive horizontal extent")
    lower, upper = bounds
    centre = (lower + upper) / 2
    tcp_position = centre.copy()
    side = -1 if arm == "left" else 1
    tcp_position[0] += side * config.edge_fraction * (upper[0] - lower[0]) / 2
    tcp_position[2] = upper[2] + config.top_clearance_m
    canonical = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
    if arm == "right":
        canonical = np.diag([-1., -1., 1.]) @ canonical
    positions = [("arm_side_bounds", tcp_position)]
    if reference_tcp_world is not None:
        reference_tcp_world = _transform(reference_tcp_world, "reference_tcp_world")
        direction = reference_tcp_world[:2, 3] - centre[:2]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction /= norm
            half = (upper[:2] - lower[:2]) / 2.
            active = np.abs(direction) > 1e-9
            ray_distance = float(np.min(half[active] / np.abs(direction[active])))
            surface = centre.copy()
            surface[:2] += direction * ray_distance
            surface[2] = upper[2] + config.top_clearance_m
            radius = float(np.linalg.norm(upper-lower) / 2.)
            standoff = surface.copy()
            standoff[:2] += direction * radius
            for source, position in (("reach_facing_surface", surface),
                                     ("reach_facing_observed_radius", standoff)):
                if not any(np.allclose(position, old, atol=1e-9) for _, old in positions):
                    positions.append((source, position))
    candidates = []
    tcp_to_ee = np.linalg.inv(ee_to_tcp)
    for position_source, position in positions:
        for yaw in config.yaw_offsets_rad:
            cosine, sine = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
            tcp = np.eye(4)
            tcp[:3, :3] = rotation @ canonical
            tcp[:3, 3] = position
            candidates.append(dict(yaw_offset_rad=float(yaw), tcp_world=tcp,
                                   ee_world=tcp @ tcp_to_ee,
                                   position_source=position_source))
    audit = dict(
        source="observed_world_bounds",
        uses_contact_annotations=False, uses_simulator_truth_for_target=False,
        requires_known_model_geometry=False, config=config.to_dict(),
        orientation_source=f"fixed_top_down_{arm}_template_with_optional_yaw",
        world_up_axis="+Z", side_axis="-X" if arm == "left" else "+X",
        model_geometry_keys=[],
        bbox_world_min=lower.tolist(), bbox_world_max=upper.tolist(),
        bbox_world_center=centre.tolist(), ee_to_tcp=ee_to_tcp.tolist(),
        tcp_reference_source="robot_current_EE_and_TCP_kinematics",
        position_hypotheses=[dict(source=source, tcp_position_world=position.tolist())
                             for source, position in positions],
        position_search_source=("observed_bounds_and_current_tcp_reach_direction"
                                if reference_tcp_world is not None
                                else "arm_side_bounds_only"),
        planning_attempts=[], selected_candidate=None,
    )
    return candidates, audit


def execute_geometry_approach(task, cup_world, metadata, arm, config):
    """Plan without moving, then execute the first feasible candidate exactly once."""
    from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix

    ee = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    tcp = pose_wxyz_to_matrix(getattr(task.robot, f"get_{arm}_tcp_pose")())
    candidates, audit = generate_geometry_candidates(
        cup_world, metadata, str(arm), np.linalg.solve(ee, tcp), config)
    return _execute_candidates(task, arm, candidates, audit)


def execute_pointcloud_approach(task, bounds_world, arm, config, observation_audit,
                                *, measured_points=None, object_pose_world=None,
                                demo_prior_path=None):
    from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix

    ee = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    tcp = pose_wxyz_to_matrix(getattr(task.robot, f"get_{arm}_tcp_pose")())
    candidates, audit = generate_bounds_candidates(
        bounds_world, str(arm), np.linalg.solve(ee, tcp), config,
        reference_tcp_world=tcp)
    audit.update(source="segmented_rgbd_visible_surface_bounds", uses_prepared_object_model=False,
                 uses_foundationpose=False, observation=observation_audit)
    if demo_prior_path is not None:
        from adapters.adaptive_interaction import measured_resolution
        from adapters.demo_relative_approach import (
            load_demo_approach_prior, prepend_demo_candidates,
        )
        if measured_points is None:
            raise ValueError("Demo-relative approach requires measured object points")
        prior = load_demo_approach_prior(demo_prior_path)
        candidates, audit = prepend_demo_candidates(
            candidates, audit, prior, bounds_world, str(arm), np.linalg.solve(ee, tcp),
            object_pose_world=object_pose_world,
            uncertainty_m=measured_resolution(measured_points),
        )
    return _execute_candidates(task, arm, candidates, audit)


def _execute_candidates(task, arm, candidates, audit):
    from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz

    plan_path = getattr(task.robot, f"{arm}_plan_path")
    target = matrix_to_pose_wxyz(candidates[0]["ee_world"])
    for index, candidate in enumerate(candidates):
        target = matrix_to_pose_wxyz(candidate["ee_world"])
        plan = plan_path(target)
        attempt = dict(candidate_index=index, yaw_offset_rad=candidate["yaw_offset_rad"],
                       candidate_source=candidate.get("source"),
                       position_source=candidate.get("position_source"),
                       target_ee_pose=target.tolist(),
                       target_tcp_pose=matrix_to_pose_wxyz(candidate["tcp_world"]).tolist(),
                       status=plan["status"], planner_status=plan.get("planner_status"))
        audit["planning_attempts"].append(attempt)
        if plan["status"] != "Success":
            if "INVALID_START" in str(plan.get("planner_status", "")):
                audit["failure_reason"] = "invalid_start_state"
                break
            continue
        audit["selected_candidate"] = index
        attempt["trajectory_waypoints"] = len(plan["position"])
        controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
        controls[f"{arm}_arm"] = plan
        # Never try a new candidate after physical execution has begun.
        moved = bool(task.take_dense_action(controls))
        attempt["execution_returned_success"] = moved
        return target, moved, audit
    audit.setdefault("failure_reason", "no_feasible_candidate")
    return target, False, audit
