"""Task-independent placement on a visually measured support surface.

The caller supplies segmented RGB-D surfaces.  No object/task names, CAD
models, simulator poses, renderer labels or contact truth are read here.
"""

import json
from pathlib import Path

import numpy as np

from adapters.adaptive_interaction import (
    ManipulationGeometryState,
    combined_uncertainty,
    contact_feedback,
    interaction_targets,
    measured_resolution,
    minimum_safe_escape,
)
from adapters.handoff import handoff_ready
from adapters import pointcloud_registration
from adapters.attachment_confirmation import confirm_after_natural_motion


def _finite_cloud(value, name):
    points = np.asarray(value, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 6
            or not np.isfinite(points).all()):
        raise RuntimeError(f"{name} does not contain enough finite RGB-D points")
    return points


def scene_points_from_rgbd(path):
    """Back-project a saved RGB-D frame without semantic or simulator labels."""
    from traditional_place_empty_cup.geometry import (
        backproject_mask_to_world, depth_to_metres,
    )

    with np.load(path, allow_pickle=False) as frame:
        depth = depth_to_metres(frame["depth"])
        valid = np.isfinite(depth) & (depth > 0)
        return _finite_cloud(backproject_mask_to_world(
            valid, depth, frame["intrinsic"], frame["world_to_camera"]
        ), "scene")


def fuse_support_clouds(clouds):
    """Fuse independently verified support surfaces in a common world frame.

    Each input is already a semantic RGB-D result.  Robust centre agreement
    prevents an unrelated surface from enlarging the target, while the union
    recovers support edges missed by a single occluded view.
    """
    valid = []
    for index, value in enumerate(clouds):
        points = np.asarray(value, dtype=float)
        if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 6
                or not np.isfinite(points).all()):
            continue
        z = points[:, 2]
        lo, hi = np.quantile(z, (0.02, 0.98))
        points = points[(z >= lo) & (z <= hi)]
        if len(points) < 6:
            continue
        center = np.median(points[:, :2], axis=0)
        radius = float(np.quantile(np.linalg.norm(points[:, :2] - center, axis=1), .9))
        valid.append(dict(index=index, points=points, center=center, radius=radius,
                          resolution=measured_resolution(points)))
    if not valid:
        raise RuntimeError("No finite support surface was observed")
    reference = np.median(np.stack([item["center"] for item in valid]), axis=0)
    reference_radius = float(np.median([item["radius"] for item in valid]))
    selected = []
    rejected = []
    for item in valid:
        bound = item["radius"] + reference_radius + 3. * item["resolution"]
        distance = float(np.linalg.norm(item["center"] - reference))
        audit = dict(index=item["index"], center_xy=item["center"].tolist(),
                     radius_m=item["radius"], center_consensus_error_m=distance,
                     center_consensus_bound_m=bound)
        if len(valid) == 1 or distance <= bound:
            selected.append(item["points"])
            audit["accepted"] = True
        else:
            audit["accepted"] = False
            rejected.append(audit)
            continue
        rejected.append(audit)
    merged = np.concatenate(selected, axis=0)
    # The same static support is often observed twice from an unchanged head
    # view.  Exact duplicate samples would collapse nearest-neighbour spacing
    # to zero, so deduplicate at half the measured sensor spacing.
    voxel = float(max(np.median([item["resolution"] for item in valid]) * .5,
                      np.finfo(float).eps))
    keys = np.rint(merged / voxel).astype(np.int64)
    _, unique = np.unique(keys, axis=0, return_index=True)
    raw_points = len(merged)
    merged = merged[np.sort(unique)]
    return merged, dict(
        method="world_frame_union_of_semantically_verified_rgbd_support_surfaces",
        input_views=len(valid), accepted_views=len(selected), raw_points=int(raw_points),
        points=int(len(merged)), deduplication_voxel_m=voxel,
        views=rejected, uses_simulator_truth=False,
    )


def support_fit(reference_points, initial_anchor, target_points, uncertainty_m):
    """Measured footprint containment in the target surface's PCA frame."""
    reference = _finite_cloud(reference_points, "object reference")
    target = _finite_cloud(target_points, "support target")
    anchor = np.asarray(initial_anchor, dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("Initial support anchor must be a finite 3-vector")
    centered = target[:, :2] - np.median(target[:, :2], axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt.T
    target_local = centered @ basis
    vertical_resolution = measured_resolution(reference)
    # Minimum layer support is derived from the measured object point count, not
    # a task constant: a sparse object still needs a contiguous bottom layer.
    min_layer_points = max(6, int(0.02 * len(reference)))
    layer_index = np.floor(
        (reference[:, 2] - np.min(reference[:, 2])) / vertical_resolution
    ).astype(int)
    support_points = np.empty((0, 3))
    selected_layer = None
    for selected_layer in np.unique(layer_index):
        support_points = reference[layer_index <= selected_layer]
        if len(support_points) >= min_layer_points:
            break
    object_local = (support_points[:, :2] - anchor[:2]) @ basis
    target_bounds = np.quantile(target_local, (0.02, 0.98), axis=0)
    object_bounds = np.quantile(object_local, (0.02, 0.98), axis=0)
    target_half = (target_bounds[1] - target_bounds[0]) / 2
    object_half = np.maximum(np.abs(object_bounds[0]), np.abs(object_bounds[1]))
    margin = target_half - object_half - uncertainty_m
    return dict(
        basis_xy=basis,
        target_center_xy=np.median(target[:, :2], axis=0),
        target_half_extent_m=target_half,
        object_half_extent_m=object_half,
        containment_margin_m=margin,
        geometry_can_contain_object=bool(np.all(margin >= 0)),
        object_support_layer_points=int(len(support_points)),
        object_support_layer_height_m=float(np.max(support_points[:, 2])
                                            - np.min(reference[:, 2])),
        pointcloud_resolution_m=vertical_resolution,
    )


def release_decision(state, actual_tcp, fit, contact, *, require_containment=True):
    """Authorize release after contact, and containment when the rule requires it."""
    actual = np.asarray(actual_tcp, dtype=float)
    actual_object = actual @ state.hand_to_object
    anchor = (actual_object[:3, :3] @ state.functional_point_object
              + actual_object[:3, 3])
    basis = np.asarray(fit["basis_xy"], dtype=float)
    error = (anchor[:2] - np.asarray(fit["target_center_xy"])) @ basis
    margin = np.asarray(fit["containment_margin_m"], dtype=float)
    contained = bool(np.all(margin >= 0) and np.all(np.abs(error) <= margin))
    ready = bool(contact.get("relation_observed") and (contained or not require_containment))
    return dict(
        ready=ready,
        support_contact_observed=bool(contact.get("relation_observed")),
        footprint_contained=contained,
        tangential_error_m=error.tolist(),
        containment_margin_m=margin.tolist(),
        uses_simulator_truth=False,
    )


def _move(task, arm, matrix, stage, result, tolerance, *, required=True):
    from envs.utils import Action, ArmTag
    from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz

    target = matrix_to_pose_wxyz(matrix)
    moved = bool(task.move((ArmTag(arm), [Action(ArmTag(arm), "move", target_pose=target)])))
    decision = handoff_ready(
        task.get_arm_pose(arm), target, moved and task.plan_success,
        position_tolerance_m=tolerance,
    )
    result.setdefault("actions", []).append(dict(
        action=stage, target_pose=target.tolist(), arrival=decision.to_dict()
    ))
    if required and not decision.ready:
        raise RuntimeError(f"Adaptive placement failed at {stage}: {decision.reason}")
    return decision


def finish_adaptive_support_placement(
        task, perception, arm, reference_path, result, *,
        registration_config=None, reobserve=False, candidate_budget=6,
        release_gap_m=None, rule=None):
    """Place a held rigid object on a measured support and release safely.

    ``release_gap_m`` is accepted only for drop-in compatibility with the old
    runner.  It is deliberately ignored; the gap is inferred from uncertainty.
    """
    from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix

    if rule is not None:
        candidate_budget = rule.candidate_budget
        if rule.fixed_release_gap is not None:
            raise ValueError("Adaptive placement rule cannot specify a fixed release gap")
    reference_path = Path(reference_path)
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_points = _finite_cloud(reference["cup_points_world"], "reference object")
        bounds = np.asarray(reference["cup_bounds_world"], dtype=float)
        observation = json.loads(str(reference["observation_audit_json"]))
        saved_target = (reference["coaster_points_world"].copy()
                        if "coaster_points_world" in reference else np.empty((0, 3)))
    config = registration_config or pointcloud_registration.RegistrationConfig()
    # Derive registration scales from the measured object cloud instead of a
    # global constant, so small objects are not rejected by large thresholds.
    resolution = float(measured_resolution(reference_points))
    if np.isfinite(resolution) and resolution > 0:
        config = pointcloud_registration.RegistrationConfig(
            voxel_m=float(np.clip(0.5 * resolution, 0.001, 0.004)),
            inlier_m=float(np.clip(1.5 * resolution, 0.005, 0.015)),
            correspondence_m=float(np.clip(3.0 * resolution, 0.02, 0.04)),
            min_points=max(10, int(0.02 * len(reference_points))),
            min_observation_overlap=0.45,
            min_reference_overlap=0.35,
        )
    if reobserve:
        from adapters.reobservation import observe_for_placement
        result["reobservation"] = {}
        estimate, motion, registration = observe_for_placement(
            task, perception, arm, reference_path, config, result["reobservation"]
        )
    else:
        if hasattr(perception, "placement_observation"):
            estimate = perception.placement_observation(task, arm)
        else:
            camera = task.get_camera_observation("head_camera")
            estimate = perception.estimate(
                camera["rgb"], camera["depth"], camera["intrinsic_cv"], camera["extrinsic_cv"]
            )
        motion, registration = pointcloud_registration.estimate_pointcloud_motion(
            reference_points, estimate.cup_points_world, config=config
        )
    result.update(
        complete=False,
        geometry_source="adaptive_measured_RGBD_support_placement",
        uses_prepared_object_model=False,
        uses_foundationpose=False,
        uses_object_truth_for_planning=False,
        fixed_release_gap_ignored=release_gap_m,
        candidate_budget=int(candidate_budget),
        task_rule=(None if rule is None else rule.to_dict()),
        registration=registration,
        timing_s=estimate.timings_s,
        actions=[],
    )
    registration_fallback = None
    if motion is None:
        fallback_pose = getattr(estimate, "object_pose_world", None)
        if fallback_pose is None:
            raise RuntimeError("Adaptive placement rejected unreliable surface registration")
        fallback_pose = np.asarray(fallback_pose, dtype=float)
        if fallback_pose.shape != (4, 4) or not np.isfinite(fallback_pose).all():
            raise RuntimeError("Adaptive placement rejected unreliable surface registration")
        motion = fallback_pose
        registration_fallback = dict(source="cad_verified_object_pose", uses_simulator_truth=False)
    result["registration_fallback"] = registration_fallback

    support_clouds = [saved_target, getattr(estimate, "coaster_points_world", None)]
    target_points, support_fusion = fuse_support_clouds(support_clouds)
    target_points = _finite_cloud(target_points, "support target")
    if registration_fallback is not None:
        current_points = _finite_cloud(estimate.cup_points_world, "current object")
    else:
        current_points = reference_points @ motion[:3, :3].T + motion[:3, 3]
    initial_anchor = bounds.mean(axis=0)
    initial_anchor[2] = float(observation["table_height_m"])
    ee_world = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    hand_to_object = np.linalg.inv(ee_world) @ motion
    selected = registration.get("selected_attempt")
    registration_rmse = []
    if selected is not None:
        registration_rmse.append(registration["attempts"][selected]["trimmed_rmse_m"])
    uncertainty = combined_uncertainty(current_points, target_points, registration_rmse)
    fit = support_fit(reference_points, initial_anchor, target_points, uncertainty)
    requires_containment = (rule is None or "footprint_contained" in rule.release_requirements)
    support_reobservation = []
    if requires_containment and not fit["geometry_can_contain_object"] and hasattr(perception, "support_only"):
        # The head view can miss support edges behind the hand/object.  Reuse
        # the two already-mounted wrist cameras without adding observation motion.
        for camera_name in (f"{arm}_camera",
                            "right_camera" if str(arm) == "left" else "left_camera"):
            item = dict(camera=camera_name)
            support_reobservation.append(item)
            try:
                camera = task.get_camera_observation(camera_name)
                observed = perception.support_only(camera, f"support_recovery_{camera_name}")
                support_clouds.append(observed.coaster_points_world)
                item.update(accepted=True, points=int(len(observed.coaster_points_world)))
            except (RuntimeError, ValueError) as error:
                item["error"] = f"{type(error).__name__}: {error}"
        target_points, support_fusion = fuse_support_clouds(support_clouds)
        uncertainty = combined_uncertainty(current_points, target_points, registration_rmse)
        fit = support_fit(reference_points, initial_anchor, target_points, uncertainty)
    target_point = np.r_[np.asarray(fit["target_center_xy"], float),
                         float(np.median(target_points[:, 2]))]
    rgbd_path = reference_path.parent / "rgbd.npz"
    if not rgbd_path.exists():
        raise RuntimeError("Saved pre-grasp RGB-D frame is required for safe-corridor geometry")
    scene_points = scene_points_from_rgbd(rgbd_path)
    state = ManipulationGeometryState(
        tcp_world=ee_world,
        hand_to_object=hand_to_object,
        functional_point_object=initial_anchor,
        target_point_world=target_point,
        target_normal_world=np.array([0., 0., 1.]),
        up_world=np.array([0., 0., 1.]),
        object_points_world=current_points,
        target_points_world=target_points,
        scene_points_world=scene_points,
        uncertainty_m=uncertainty,
    )
    if requires_containment and not fit["geometry_can_contain_object"]:
        raise RuntimeError("Measured support is too small for the observed object footprint")
    escape_tcp, escape_audit = minimum_safe_escape(state)
    desired_rotation = np.eye(3)
    if rule is not None:
        if rule.desired_orientation == "fixed_quaternion":
            desired_rotation = pose_wxyz_to_matrix(
                np.array([0.0, 0.0, 0.0, *rule.desired_quaternion_wxyz], dtype=float))[:3, :3]
        elif rule.desired_orientation != "initial_world":
            raise ValueError("Unsupported desired orientation in task rule")
    targets = interaction_targets(
        state, desired_object_rotation_world=desired_rotation,
        candidate_budget=candidate_budget,
    )
    result["adaptive_geometry"] = dict(
        uncertainty_m=uncertainty,
        uncertainty_sources="object/support RGB-D spacing plus selected ICP residual",
        initial_support_anchor_world=initial_anchor.tolist(),
        target_support_point_world=target_point.tolist(),
        support_fit={key: (value.tolist() if isinstance(value, np.ndarray) else value)
                     for key, value in fit.items()},
        support_fusion=support_fusion,
        support_reobservation=support_reobservation,
        escape=escape_audit,
        interaction=targets["audit"],
        task_specific_distance_constants=False,
    )

    # Conservative arrival gate: fused uncertainty is an RSS estimate and
    # can undershoot the real tracking residual, so apply a measured floor.
    tolerance = max(uncertainty, 0.005)
    arrivals = []
    attachment_confirmed = False
    hand_before_transport = ee_world.copy()
    if escape_audit["escape_distance_m"] > 0:
        decision = _move(task, arm, escape_tcp, "adaptive_minimum_safe_escape",
                         result, tolerance)
        arrivals.append(decision.position_error_m)
        confirmation = confirm_after_natural_motion(
            task, perception, arm, current_points, hand_before_transport,
            "attachment_after_safe_escape")
        result["attachment_confirmation"] = confirmation
        attachment_confirmed = bool(confirmation["accepted"])  # record only, never abort
    chosen_index = None
    chosen_precontact = None
    for index, candidate in enumerate(targets["precontact_tcp_candidates"]):
        decision = _move(
            task, arm, candidate, f"adaptive_precontact_candidate_{index}",
            result, tolerance, required=False,
        )
        if decision.ready:
            chosen_index, chosen_precontact = index, candidate
            arrivals.append(decision.position_error_m)
            break
        if decision.reason != "traditional_planning_failed":
            break
    result["adaptive_geometry"]["selected_precontact_candidate"] = chosen_index
    if chosen_precontact is None:
        raise RuntimeError("No geometry-derived pre-contact candidate was feasible and reached")
    if not attachment_confirmed:
        confirmation = confirm_after_natural_motion(
            task, perception, arm, current_points, hand_before_transport,
            "attachment_after_precontact_move")
        result["attachment_confirmation"] = confirmation  # record only, never abort

    baseline = float(np.sqrt(np.mean(np.square(arrivals)))) if arrivals else 0.0
    _move(task, arm, targets["surface_tcp"], "adaptive_estimated_support_surface",
          result, tolerance)
    actual = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    feedback = contact_feedback(state, actual, targets["surface_tcp"], baseline)
    result["adaptive_geometry"]["surface_feedback"] = feedback
    if not feedback["relation_observed"]:
        _move(task, arm, targets["contact_limit_tcp"], "adaptive_bounded_support_contact",
              result, tolerance)
        actual = pose_wxyz_to_matrix(task.get_arm_pose(arm))
        feedback = contact_feedback(state, actual, targets["contact_limit_tcp"], baseline)
        result["adaptive_geometry"]["bounded_contact_feedback"] = feedback
    release = release_decision(state, actual, fit, feedback,
                               require_containment=requires_containment)
    result["release_decision"] = release
    if not release["ready"]:
        raise RuntimeError("Adaptive support relation did not authorize release")

    opened = bool(task.move(task.open_gripper(arm, pos=1.0)))
    result["actions"].append(dict(action="release", success=opened))
    if not opened or not task.plan_success:
        raise RuntimeError("Adaptive placement failed to release")
    if rule is not None and rule.retreat != "selected_precontact":
        raise ValueError("Unsupported retreat policy in task rule")
    _move(task, arm, chosen_precontact, "retreat_to_verified_precontact",
          result, tolerance)
    result["complete"] = True

