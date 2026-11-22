"""Local multi-start trimmed ICP against this episode's measured surface, not CAD.

This estimates observed rigid motion, not a semantic object coordinate frame.
Symmetries, limited overlap and occlusion can make the motion ambiguous.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RegistrationConfig:
    voxel_m: float = 0.003
    correspondence_m: float = 0.03
    inlier_m: float = 0.012
    max_rmse_m: float = 0.008
    min_observation_overlap: float = 0.55
    min_reference_overlap: float = 0.40
    max_tilt_deg: float = 25.0
    iterations: int = 45
    min_points: int = 40
    yaw_starts_rad: tuple = (0., -0.3, 0.3, -0.6, 0.6)

    def __post_init__(self):
        distances = (self.voxel_m,self.correspondence_m,self.inlier_m,self.max_rmse_m)
        if (not np.isfinite(distances).all() or min(distances) <= 0
                or not self.voxel_m < self.inlier_m <= self.correspondence_m
                or not 0 < self.min_observation_overlap <= 1
                or not 0 < self.min_reference_overlap <= 1
                or not 0 <= self.max_tilt_deg < 90 or self.iterations < 1 or self.min_points < 3
                or not self.yaw_starts_rad or not np.isfinite(self.yaw_starts_rad).all()):
            raise ValueError("Invalid pointcloud registration configuration")


def _cloud(value, config):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 measured point cloud")
    if len(points) < config.min_points:
        raise RuntimeError("Too few measured surface points for registration")
    _, selected = np.unique(np.floor(points/config.voxel_m).astype(np.int64), axis=0,
                            return_index=True)
    points = points[np.sort(selected)]
    if len(points) < config.min_points:
        raise RuntimeError("Too few spatially distinct surface points for registration")
    return points


def _rigid_fit(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, singular, vt = np.linalg.svd((source-a).T @ (target-b))
    if singular[1] < 1e-10:
        raise RuntimeError("Degenerate point correspondences")
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, b-rotation@a


def estimate_pointcloud_motion(reference_points, observed_points, config=RegistrationConfig()):
    """Return (reference-world -> current-world motion, quality audit).

    Rejected estimates return None plus diagnostics, so callers retain failure
    evidence and must not silently execute a CAD/ground-truth fallback.
    """
    from scipy.spatial import cKDTree

    reference = _cloud(reference_points, config)
    observed = _cloud(observed_points, config)
    reference_tree, observed_tree = cKDTree(reference), cKDTree(observed)
    attempts = []
    solutions = []
    for yaw in config.yaw_starts_rad:
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c,-s,0], [s,c,0], [0,0,1.]])
        translation = np.median(observed,axis=0)-rotation@np.median(reference,axis=0)
        previous_error = float("inf")
        failed = None
        for iteration in range(config.iterations):
            distances, nearest = reference_tree.query((observed-translation)@rotation)
            threshold = min(config.correspondence_m, float(np.quantile(distances,0.8)))
            keep = distances <= threshold + 1e-12
            if keep.sum() < config.min_points:
                failed = "too_few_correspondences"
                break
            try:
                new_rotation, new_translation = _rigid_fit(reference[nearest[keep]],observed[keep])
            except RuntimeError as error:
                failed = str(error)
                break
            error = float(np.sqrt(np.mean(distances[keep]**2)))
            rotation, translation = new_rotation, new_translation
            if abs(previous_error-error) < 1e-7:
                break
            previous_error = error
        distances, _ = reference_tree.query((observed-translation)@rotation)
        reverse, _ = observed_tree.query(reference@rotation.T+translation)
        closest = distances <= np.quantile(distances,0.8)+1e-12
        rmse = float(np.sqrt(np.mean(distances[closest]**2)))
        overlap = float(np.mean(distances <= config.inlier_m))
        reference_overlap = float(np.mean(reverse <= config.inlier_m))
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2,2],-1,1))))
        accepted = (failed is None and rmse <= config.max_rmse_m
                    and overlap >= config.min_observation_overlap
                    and reference_overlap >= config.min_reference_overlap
                    and tilt <= config.max_tilt_deg)
        attempt = dict(yaw_start_rad=yaw, iterations=iteration+1, trimmed_rmse_m=rmse,
                       observation_overlap=overlap, reference_overlap=reference_overlap,
                       inferred_tilt_deg=tilt, accepted=accepted, failure=failed)
        attempts.append(attempt)
        if accepted:
            motion = np.eye(4)
            motion[:3,:3], motion[:3,3] = rotation, translation
            # Geometric fit first, small overlap penalty; no simulator feedback.
            solutions.append((rmse+0.005*(1-overlap), len(attempts)-1, motion))
    audit = dict(method="multistart_trimmed_point_to_point_icp", config=asdict(config),
                 reference_points=len(reference), observed_points=len(observed),
                 uses_prepared_object_model=False, uses_simulator_truth=False,
                 reference_source="same_episode_pregrasp_segmented_rgbd",
                 semantic_6d_pose=False, symmetry_ambiguity_resolved=False,
                 attempts=attempts, accepted=bool(solutions), selected_attempt=None)
    if not solutions:
        audit["failure_reason"] = "no_registration_passed_quality_gate"
        return None, audit
    _, index, motion = min(solutions,key=lambda item:item[0])
    audit.update(selected_attempt=index, observed_rigid_motion=motion.tolist())
    return motion, audit
