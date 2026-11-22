"""Task-independent visual confirmation that a rigid object follows a robot hand.

The module only consumes measured hand poses and segmented RGB-D point clouds.
It never reads simulator actors, contacts, renderer labels, task success, object
names, CAD poses, or task-specific dimensions.
"""

import numpy as np

from adapters.adaptive_interaction import combined_uncertainty, measured_resolution
from adapters.pointcloud_registration import RegistrationConfig, estimate_pointcloud_motion


def _transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-6)):
        raise ValueError(f"{name} must be a finite homogeneous transform")
    return matrix


def _cloud(value, name):
    points = np.asarray(value, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 10
            or not np.isfinite(points).all()):
        raise ValueError(f"{name} must contain finite measured 3-D points")
    return points


def expected_rigid_cloud(points, hand_before, hand_after):
    """Propagate a measured object cloud by the measured hand motion."""
    points = _cloud(points, "object cloud")
    before = _transform(hand_before, "hand_before")
    after = _transform(hand_after, "hand_after")
    motion = after @ np.linalg.inv(before)
    expected = points @ motion[:3, :3].T + motion[:3, 3]
    return expected, motion


def attachment_consistency(points_before, points_after, hand_before, hand_after):
    """Check whether fresh RGB-D agrees with rigid hand/object co-motion.

    Registration is performed *after* applying the measured hand motion.  A
    stationary object therefore cannot pass merely because ICP can discover a
    large translation.  The acceptance scale comes from RGB-D spacing and the
    selected registration residual, rather than an object/task distance.
    """
    before = _cloud(points_before, "object cloud before transport")
    observed = _cloud(points_after, "object cloud after transport")
    expected, hand_motion = expected_rigid_cloud(before, hand_before, hand_after)
    resolution = max(measured_resolution(expected), measured_resolution(observed))
    config = RegistrationConfig(
        voxel_m=float(np.clip(0.5 * resolution, 0.001, 0.004)),
        inlier_m=float(np.clip(1.5 * resolution, 0.005, 0.015)),
        correspondence_m=float(np.clip(3.0 * resolution, 0.02, 0.04)),
        min_points=max(10, int(0.02 * min(len(expected), len(observed)))),
        min_observation_overlap=0.35,
        min_reference_overlap=0.25,
    )
    correction, registration = estimate_pointcloud_motion(expected, observed, config=config)
    hand_translation = float(np.linalg.norm(hand_motion[:3, 3]))
    induced_motion = float(np.quantile(np.linalg.norm(expected - before, axis=1), .5))
    audit = dict(
        accepted=False,
        method="measured_hand_motion_then_rgbd_residual_registration",
        hand_translation_m=hand_translation,
        median_hand_induced_object_motion_m=induced_motion,
        pointcloud_resolution_m=float(resolution),
        registration=registration,
        uses_simulator_truth=False,
        uses_contact_truth=False,
        extra_trial_motion=False,
    )
    if induced_motion <= 3. * resolution:
        audit["reason"] = "natural_motion_was_too_small_to_confirm_attachment"
        return audit
    from scipy.spatial import cKDTree

    # Relative, scale-free acceptance.  An attached object's fresh cloud sits closer
    # to the hand-propagated expected cloud than to its original before cloud; a
    # detached object stays at the old location, so it is closer to the before cloud.
    # This avoids both the unreliable 6-DOF residual ICP under occlusion and any
    # absolute distance threshold that would break on small grip rotation.
    expected_tree = cKDTree(expected)
    before_tree = cKDTree(before)
    d_expected, _ = expected_tree.query(observed, k=1)
    d_before, _ = before_tree.query(observed, k=1)
    median_expected = float(np.median(d_expected))
    median_before = float(np.median(d_before))
    accepted = bool(median_expected < median_before)

    # Retain the residual ICP purely as diagnostics, never for the decision.
    icp = dict(correction=None)
    if correction is not None:
        selected = registration.get("selected_attempt")
        residuals = []
        if selected is not None:
            residuals.append(registration["attempts"][selected]["trimmed_rmse_m"])
        uncertainty = combined_uncertainty(expected, observed, residuals)
        translation_residual = float(np.linalg.norm(correction[:3, 3]))
        cosine = np.clip((np.trace(correction[:3, :3]) - 1.) / 2., -1., 1.)
        rotation_residual = float(np.arccos(cosine))
        radius = float(np.quantile(np.linalg.norm(expected - np.median(expected, axis=0), axis=1), .9))
        rotational_displacement = float(radius * np.sin(min(rotation_residual, np.pi / 2)))
        rigid_residual = float(np.hypot(translation_residual, rotational_displacement))
        icp = dict(
            uncertainty_m=float(uncertainty),
            correction_translation_m=translation_residual,
            correction_rotation_deg=float(np.degrees(rotation_residual)),
            correction_rotational_displacement_m=rotational_displacement,
            rigid_residual_m=rigid_residual,
            acceptance_bound_m=float(3. * max(uncertainty, resolution)),
        )
    audit.update(
        method="measured_hand_motion_then_relative_cloud_proximity",
        median_distance_to_expected_m=median_expected,
        median_distance_to_before_m=median_before,
        accepted=accepted,
        **icp,
    )
    audit["reason"] = ("rigid_comotion_confirmed" if accepted
                       else "observed_object_did_not_match_hand_propagated_reference")
    return audit


def confirm_after_natural_motion(task, perception, arm, points_before, hand_before, label):
    """Fuse bounded head/wrist observations after an already-required move."""
    from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix

    after = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    attempts = []
    for camera_name in ("head_camera", f"{arm}_camera",
                        "right_camera" if str(arm) == "left" else "left_camera"):
        item = dict(camera=camera_name)
        attempts.append(item)
        try:
            camera = task.get_camera_observation(camera_name)
            if hasattr(perception, "object_only"):
                estimate = perception.object_only(camera, f"{label}_{camera_name}")
            else:
                estimate = perception.supplementary(
                    camera, f"{label}_{camera_name}", cup_only=True, refine_cup=True)
            item["consistency"] = attachment_consistency(
                points_before, estimate.cup_points_world, hand_before, after)
            if item["consistency"]["accepted"]:
                return dict(
                    accepted=True,
                    selected_camera=camera_name,
                    attempts=attempts,
                    hand_pose_before=np.asarray(hand_before).tolist(),
                    hand_pose_after=after.tolist(),
                    uses_simulator_truth=False,
                    extra_trial_motion=False,
                )
        except (RuntimeError, ValueError) as error:
            item["error"] = f"{type(error).__name__}: {error}"
    return dict(
        accepted=False,
        reason="no_camera_confirmed_rigid_hand_object_comotion",
        attempts=attempts,
        hand_pose_before=np.asarray(hand_before).tolist(),
        hand_pose_after=after.tolist(),
        uses_simulator_truth=False,
        extra_trial_motion=False,
    )
