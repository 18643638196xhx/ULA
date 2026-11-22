"""Task-independent geometry for escape, pre-contact and contact motion.

All distances are inferred from the current RGB-D samples, robot arrival
residuals and object geometry.  The module contains no task/object names and
does not read simulator actors, contacts, labels or success functions.
"""

from dataclasses import dataclass

import numpy as np


def _vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite {size}-vector")
    return result


def _rigid(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1])
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1.0, atol=1e-5)):
        raise ValueError("Expected a finite rigid transform")
    return result


def _cloud(value):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 6 or not np.isfinite(result).all():
        raise ValueError("Expected at least six finite 3-D points")
    return result


def measured_resolution(points):
    """Robust native spacing of a measured point cloud, in its own units."""
    points = _cloud(points)
    try:
        from scipy.spatial import cKDTree
        distance = cKDTree(points).query(points, k=2)[0][:, 1]
    except ImportError:
        # Exact bounded-memory fallback for lightweight test environments.
        squared_norm = np.sum(points * points, axis=1)
        nearest = np.empty(len(points), dtype=float)
        chunk = 512
        for start in range(0, len(points), chunk):
            stop = min(start + chunk, len(points))
            squared = (squared_norm[start:stop, None] + squared_norm[None, :]
                       - 2.0 * points[start:stop] @ points.T)
            squared = np.maximum(squared, 0.0)
            rows = np.arange(stop - start)
            squared[rows, np.arange(start, stop)] = np.inf
            nearest[start:stop] = np.sqrt(np.min(squared, axis=1))
        distance = nearest
    distance = distance[np.isfinite(distance) & (distance > 0)]
    if not len(distance):
        raise ValueError("Point cloud has no measurable spatial resolution")
    return float(np.median(distance))


def combined_uncertainty(source_points, target_points, arrival_residuals=()):
    """Root-sum-square of independently measured spatial resolutions."""
    terms = [measured_resolution(source_points), measured_resolution(target_points)]
    residuals = np.asarray(tuple(arrival_residuals), dtype=float)
    if residuals.size:
        if residuals.ndim != 1 or not np.isfinite(residuals).all() or np.any(residuals < 0):
            raise ValueError("Arrival residuals must be finite nonnegative distances")
        terms.append(float(np.sqrt(np.mean(residuals * residuals))))
    return float(np.linalg.norm(terms))


@dataclass(frozen=True)
class ManipulationGeometryState:
    tcp_world: np.ndarray
    hand_to_object: np.ndarray
    functional_point_object: np.ndarray
    target_point_world: np.ndarray
    target_normal_world: np.ndarray
    up_world: np.ndarray
    object_points_world: np.ndarray
    target_points_world: np.ndarray
    scene_points_world: np.ndarray
    uncertainty_m: float

    def __post_init__(self):
        object.__setattr__(self, "tcp_world", _rigid(self.tcp_world))
        object.__setattr__(self, "hand_to_object", _rigid(self.hand_to_object))
        object.__setattr__(self, "functional_point_object", _vector(self.functional_point_object, 3))
        object.__setattr__(self, "target_point_world", _vector(self.target_point_world, 3))
        normal = _vector(self.target_normal_world, 3)
        up = _vector(self.up_world, 3)
        if np.linalg.norm(normal) == 0 or np.linalg.norm(up) == 0:
            raise ValueError("Directions must be nonzero")
        object.__setattr__(self, "target_normal_world", normal / np.linalg.norm(normal))
        object.__setattr__(self, "up_world", up / np.linalg.norm(up))
        for name in ("object_points_world", "target_points_world", "scene_points_world"):
            object.__setattr__(self, name, _cloud(getattr(self, name)))
        if not np.isfinite(self.uncertainty_m) or self.uncertainty_m <= 0:
            raise ValueError("Geometry uncertainty must be positive")

    @property
    def object_world(self):
        return self.tcp_world @ self.hand_to_object

    @property
    def functional_world(self):
        return (self.object_world[:3, :3] @ self.functional_point_object
                + self.object_world[:3, 3])


def _mode_plane_coordinate(points, direction):
    values = np.asarray(points) @ np.asarray(direction)
    edges = np.histogram_bin_edges(values, bins="fd")
    if len(edges) < 2:
        return float(np.median(values))
    counts, _ = np.histogram(values, bins=edges)
    index = int(np.argmax(counts))
    selected = values[(values >= edges[index]) & (values <= edges[index + 1])]
    return float(np.median(selected)) if len(selected) else float((edges[index] + edges[index + 1]) / 2)


def minimum_safe_escape(state):
    """Return the shortest world-up escape clearing the measured corridor."""
    current = state.functional_world
    target = state.target_point_world
    delta = target[:2] - current[:2]
    length2 = float(delta @ delta)
    scene = state.scene_points_world
    if length2 > 0:
        t = np.clip(((scene[:, :2] - current[:2]) @ delta) / length2, 0.0, 1.0)
        nearest = current[:2] + t[:, None] * delta
        radius = float(np.max(np.linalg.norm(
            state.object_points_world[:, :2] - state.object_world[:2, 3], axis=1)))
        corridor = np.linalg.norm(scene[:, :2] - nearest, axis=1) <= radius + state.uncertainty_m
    else:
        corridor = np.ones(len(scene), dtype=bool)
    # Remove the measured source and target neighbourhoods from obstacle evidence.
    source_radius = float(np.max(np.linalg.norm(
        state.object_points_world[:, :2] - state.object_world[:2, 3], axis=1)))
    target_radius = float(np.max(np.linalg.norm(
        state.target_points_world[:, :2] - state.target_point_world[:2], axis=1)))
    corridor &= np.linalg.norm(scene[:, :2] - current[:2], axis=1) > source_radius
    corridor &= np.linalg.norm(scene[:, :2] - target[:2], axis=1) > target_radius
    # Points above the held object's top are robot links / gripper, not
    # obstacles for its horizontal path; ignoring them stops the escape from
    # trying to lift the object over the robot itself.
    object_top = float(np.max(state.object_points_world @ state.up_world))
    corridor &= (scene @ state.up_world) <= object_top + state.uncertainty_m
    support = _mode_plane_coordinate(scene, state.up_world)
    obstacle_top = support
    if np.any(corridor):
        obstacle_top = max(obstacle_top, float(np.max(scene[corridor] @ state.up_world)))
    object_bottom = float(np.min(state.object_points_world @ state.up_world))
    distance = max(0.0, obstacle_top + state.uncertainty_m - object_bottom)
    target_tcp = state.tcp_world.copy()
    target_tcp[:3, 3] += state.up_world * distance
    return target_tcp, dict(
        escape_distance_m=distance,
        measured_support_coordinate_m=support,
        measured_corridor_obstacle_top_m=obstacle_top,
        measured_object_bottom_m=object_bottom,
        uncertainty_m=state.uncertainty_m,
        task_specific_clearance_constant=False,
    )


def interaction_targets(state, *, desired_object_rotation_world=None, candidate_budget=6):
    """Compute surface, robust pre-contact and uncertainty-bounded contact poses."""
    if not isinstance(candidate_budget, int) or candidate_budget < 1:
        raise ValueError("Candidate budget must be a positive integer")
    rotation = state.object_world[:3, :3]
    if desired_object_rotation_world is not None:
        candidate_rotation = np.asarray(desired_object_rotation_world, dtype=float)
        candidate = np.eye(4)
        candidate[:3, :3] = candidate_rotation
        rotation = _rigid(candidate)[:3, :3]
    surface_object = state.object_world.copy()
    surface_object[:3, :3] = rotation
    surface_object[:3, 3] = (state.target_point_world
                              - rotation @ state.functional_point_object)
    limit_object = surface_object.copy()
    limit_object[:3, 3] -= state.target_normal_world * state.uncertainty_m
    object_to_hand = np.linalg.inv(state.hand_to_object)
    object_span = float(np.ptp(state.object_points_world @ state.target_normal_world))
    target_span = float(np.ptp(state.target_points_world @ state.target_normal_world))
    # The closest candidate is the measured uncertainty bound.  Extra
    # candidates exist only for planner fallback and are derived from current
    # measured geometry rather than a task-specific distance ladder.
    upper_gap = state.uncertainty_m + object_span + target_span
    precontact_gaps = np.linspace(
        state.uncertainty_m, upper_gap, candidate_budget
    ).tolist()
    precontact_tcp_candidates = []
    for gap in precontact_gaps:
        candidate = surface_object.copy()
        candidate[:3, 3] += state.target_normal_world * gap
        precontact_tcp_candidates.append(candidate @ object_to_hand)
    return dict(
        precontact_tcp=precontact_tcp_candidates[0],
        precontact_tcp_candidates=precontact_tcp_candidates,
        surface_tcp=surface_object @ object_to_hand,
        contact_limit_tcp=limit_object @ object_to_hand,
        audit=dict(
            approach_normal_world=state.target_normal_world.tolist(),
            inferred_precontact_gap_m=state.uncertainty_m,
            planner_fallback_gaps_m=precontact_gaps,
            planner_candidate_budget=candidate_budget,
            desired_object_rotation_world=rotation.tolist(),
            inferred_contact_travel_bound_m=state.uncertainty_m,
            task_specific_distance_constant=False,
        ),
    )


def contact_feedback(state, actual_tcp, commanded_tcp, baseline_tracking_m):
    """Assess contact progress from geometry and measured TCP tracking only."""
    actual_tcp, commanded_tcp = _rigid(actual_tcp), _rigid(commanded_tcp)
    actual_object = actual_tcp @ state.hand_to_object
    point = actual_object[:3, :3] @ state.functional_point_object + actual_object[:3, 3]
    signed_gap = float((point - state.target_point_world) @ state.target_normal_world)
    residual = float(np.linalg.norm(actual_tcp[:3, 3] - commanded_tcp[:3, 3]))
    if not np.isfinite(baseline_tracking_m) or baseline_tracking_m < 0:
        raise ValueError("Baseline tracking residual must be finite and nonnegative")
    numerical_slack = 8.0 * np.spacing(max(1.0, abs(signed_gap), state.uncertainty_m))
    within_uncertainty = signed_gap <= state.uncertainty_m + numerical_slack
    surface_crossed = signed_gap <= numerical_slack
    tracking_changed = residual > baseline_tracking_m + state.uncertainty_m
    return dict(
        relation_observed=bool(surface_crossed or tracking_changed),
        signed_surface_gap_m=signed_gap,
        tcp_tracking_residual_m=residual,
        baseline_tracking_residual_m=float(baseline_tracking_m),
        geometry_within_uncertainty=bool(within_uncertainty),
        estimated_surface_crossed=bool(surface_crossed),
        tracking_change_detected=bool(tracking_changed),
        uses_simulator_contact_truth=False,
    )
