"""Separate known-CAD bottle experiment; scene poses come only from RGB-D.

Legacy cup/coaster keys mean bottle/pad. No live object actor is used here.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adapters.perception import PerceptionWorker
from adapters.pillbottle import PillbottlePerception, observation_quality, projected_object_prompt, approach_bottle
from adapters.pillbottle_control import settle_at_target
from adapters.early_grasp import slower_trajectory
from adapters.handoff import handoff_ready


def support_point(pose, metadata):
    local = np.asarray(metadata["functional_matrix"][0], dtype=float).copy()
    local[:3, 3] *= np.asarray(metadata["scale"])
    return (np.asarray(pose) @ local)[:3, 3]


def central_contact_group(metadata):
    """Pick the annotated group nearest the scaled CAD center, not scene truth.

    For the known bottle this favors body contacts over cap contacts. It does
    not infer a semantic label or certify that a resulting grasp is feasible.
    """
    contacts = np.asarray(metadata["contact_points_pose"], dtype=float)
    center = np.asarray(metadata["center"], dtype=float)
    scale = np.asarray(metadata["scale"], dtype=float)
    if (contacts.ndim != 3 or contacts.shape[1:] != (4, 4)
            or center.shape != (3,) or scale.shape != (3,)
            or not all(np.isfinite(v).all() for v in (contacts, center, scale))
            or np.any(scale <= 0)):
        raise ValueError("Invalid known-CAD contact geometry")
    groups = metadata.get("contact_points_group", [])
    masks = metadata.get("contact_points_mask", [True]*len(groups))
    candidates = []
    for index, group in enumerate(groups):
        if index >= len(masks) or not masks[index] or not group:
            continue
        if any(not isinstance(i, int) or i < 0 or i >= len(contacts) for i in group):
            raise ValueError("Invalid known-CAD contact group")
        distance = np.linalg.norm((contacts[group, :3, 3]-center)*scale, axis=1)
        candidates.append((float(np.mean(distance)), index, list(group)))
    if not candidates:
        raise ValueError("No enabled known-CAD contact group")
    return min(candidates, key=lambda item: item[:2])[2]


def centered_camera_pose(world_pose, world_to_camera, centering_transform):
    """FP tracking's internal centered mesh frame, accepting 3x4 CV extrinsics."""
    from traditional_place_empty_cup.geometry import homogeneous_transform
    return (homogeneous_transform(world_to_camera) @ homogeneous_transform(world_pose)
            @ np.linalg.inv(homogeneous_transform(centering_transform)))


def pose_fit(points, pose, model_points):
    """One-sided depth agreement: hidden CAD surfaces need not be observed."""
    from scipy.spatial import cKDTree
    values = np.asarray(points, dtype=float)
    matrix = np.asarray(pose, dtype=float)
    model = np.asarray(model_points, dtype=float)
    if (matrix.shape != (4, 4) or values.ndim != 2 or values.shape[1:] != (3,)
            or model.ndim != 2 or model.shape[1:] != (3,) or min(len(values), len(model)) < 40
            or not all(np.isfinite(v).all() for v in (values, matrix, model))
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=.01)
            or np.linalg.det(matrix[:3, :3]) < .99):
        raise ValueError("Invalid CAD registration geometry")
    world = model @ matrix[:3, :3].T + matrix[:3, 3]
    distance = cKDTree(world).query(values)[0]
    median = float(np.median(distance))
    fraction = float(np.mean(distance <= .015))
    # Thresholds are per-object-class: a flat, low-profile object covers far less
    # of its CAD from a single view than a cup, so the one-sided fit is coarser by
    # construction.  Defaults are unchanged; callers may widen them.
    import os as _os
    max_median = float(_os.environ.get("ULA_POSE_FIT_MAX_MEDIAN_MM", "8")) / 1000.0
    min_fraction = float(_os.environ.get("ULA_POSE_FIT_MIN_FRACTION", "0.65"))
    return dict(accepted=median <= max_median and fraction >= min_fraction,
        median_depth_to_model_m=median, fraction_within_15mm=fraction,
        maximum_median_m=max_median, minimum_fraction=min_fraction,
        limitation="visible_surface_agreement_does_not_resolve_symmetry_or_prove_grasp")


class KnownWorker(PerceptionWorker):
    def __init__(self, root, output, *, model_id=1, object_prompt="pill bottle",
                 target_prompt="blue square pad", fp_iterations=5):
        super().__init__(root, output, backend="foundationpose", fp_iterations=fp_iterations)
        self.model_id, self.object_prompt, self.target_prompt = model_id, object_prompt, target_prompt

    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        launch = [command[0], str(self.root/"scripts/perceive_pillbottle_known_model.py"),
            "--model-id", str(self.model_id), "--debug-dir", str(self.output/"debug"),
            "--object-prompt", self.object_prompt, "--target-prompt", self.target_prompt,
            "--fp-iterations", str(self.fp_iterations)]
        if getattr(self, "placement_policy", "legacy") == "visual":
            launch.append("--geometry-guard")
        return launch, cwd, environment


class KnownPerception(PillbottlePerception):
    def resolve_initial_pad(self, task, value):
        """Reject elevated semantic targets; recover from real blue table points."""
        from adapters.pillbottle_pad_target import pad_height_check, recover_blue_pad
        table_z = float(value.observation_audit["table_height_m"])
        audit = dict(initial_check=pad_height_check(value.coaster_world_center, table_z))
        if not audit["initial_check"]["accepted"]:
            cameras = {}
            attempts = audit["recovery_attempts"] = []
            for name in ("head_camera", "left_camera", "right_camera"):
                cameras[name] = task.get_camera_observation(name)
                attempt = dict(available_cameras=list(cameras), accepted=False)
                attempts.append(attempt)
                try:
                    center, recovery = recover_blue_pad(cameras, table_z)
                    value.coaster_world_center = center
                    attempt.update(accepted=True, recovery=recovery)
                    break
                except (RuntimeError, ValueError) as error:
                    attempt["error"] = str(error)
            else:
                raise RuntimeError(f"Initial placement target rejected after visual pad recovery: {audit}")
        audit.update(final_target=value.coaster_world_center.tolist(), uses_object_truth=False)
        value.observation_audit["pad_target_validation"] = audit
        self.worker.pad_target_validation = audit
        self.worker.reference_table_z = table_z
        self.select_reference(value)
        return audit

    def estimate(self, rgb, depth, intrinsic, world_to_camera):
        try:
            return super().estimate(rgb, depth, intrinsic, world_to_camera)
        except (RuntimeError, ValueError) as error:
            if getattr(self.worker, "placement_policy", "legacy") != "visual":
                raise
            source = self.output/"estimate.npz"
            if not source.is_file():
                raise
            with np.load(source, allow_pickle=False) as data:
                points = data["cup_points_world"].copy()
            # Old mask is only a current-frame search hint, not accepted geometry.
            camera = dict(rgb=rgb, depth=depth, intrinsic_cv=intrinsic, extrinsic_cv=world_to_camera)
            lo, hi = np.quantile(points, [.03, .97], axis=0)
            points = points[np.all((points >= lo) & (points <= hi), axis=1)]
            box, point = projected_object_prompt(points, camera)
            value = self.supplementary(camera, "initial_projected_retry", cup_only=True,
                refine_cup=True, coarse_only=False, object_roi_xyxy=box, object_point_xy=point)
            quality = observation_quality(value)
            if not quality["geometry_ok"]:
                raise RuntimeError(f"Initial visual retry rejected: {quality}") from error
            value.coaster_world_center = self.pad_only(camera).coaster_world_center.copy()
            value.observation_audit["initial_projected_retry"] = True
            self.select_reference(value)
            return value

    def _request(self, camera, label, *, cup_only=True, refine_cup=False,
                 coarse_only=False, camera_name="head_camera", object_roi_xyxy=None,
                 object_point_xy=None, pose_hint_world=None):
        directory = self.output/label
        directory.mkdir(parents=True, exist_ok=True)
        request, result = directory/"rgbd.npz", directory/"estimate.npz"
        payload = dict(rgb=camera["rgb"], depth=camera["depth"],
            intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"],
            cup_only=np.asarray(cup_only), refine_cup=np.asarray(refine_cup),
            coarse_only=np.asarray(coarse_only), camera_name=np.asarray(camera_name))
        if object_roi_xyxy is not None:
            payload.update(object_roi_xyxy=object_roi_xyxy, object_point_xy=object_point_xy)
        if pose_hint_world is not None:
            payload["pose_hint_world"] = np.asarray(pose_hint_world, dtype=float)
        np.savez_compressed(request, **payload)
        return self.worker.estimate_files(request, result)

    def supplementary(self, camera, label, *, cup_only=True, refine_cup=False,
                      coarse_only=None, camera_name=None, **kwargs):
        if coarse_only is None:
            coarse_only = label.startswith("close_")
        if camera_name is None:
            camera_name = next((name for name in ("left_camera", "right_camera") if name in label), "head_camera")
        attempts, value = [], None
        try:
            value = self._request(camera, label, cup_only=cup_only, refine_cup=refine_cup,
                coarse_only=coarse_only, camera_name=camera_name, **kwargs)
            if not coarse_only or refine_cup:
                return value
            quality = observation_quality(value)
            if quality["geometry_ok"]:
                return value
            attempts.append(dict(prompt="raw_box", quality=quality))
        except (RuntimeError, ValueError) as error:
            if not coarse_only or refine_cup:
                raise
            attempts.append(dict(prompt="raw_box", error=str(error)))
        for prompt in ("projected_reference", "depth_refined_box"):
            attempt = dict(prompt=prompt)
            attempts.append(attempt)
            try:
                hints = {}
                if prompt == "projected_reference":
                    points = getattr(self.worker, "reference_points", None)
                    if points is None:
                        continue
                    box, point = projected_object_prompt(points, camera)
                    hints.update(object_roi_xyxy=box, object_point_xy=point)
                candidate = self._request(camera, f"{label}_{prompt}", cup_only=True,
                    refine_cup=True, coarse_only=True, camera_name=camera_name, **hints)
                attempt["quality"] = observation_quality(candidate)
                if attempt["quality"]["geometry_ok"]:
                    candidate.observation_audit["coarse_reobservation_attempts"] = attempts
                    return candidate
            except (RuntimeError, ValueError) as error:
                attempt["error"] = str(error)
        if value is None:
            raise RuntimeError(f"No credible coarse observation after bounded prompts: {attempts}")
        value.observation_audit["coarse_reobservation_attempts"] = attempts
        return value

    def localize_held(self, task, arm, audit, *, pose_hint_world=None, candidate_validator=None):
        """Fresh localization; a placement-rejected head pose must not hide the wrist.

        The optional validator uses only the current visual estimate and robot
        state. Legacy callers keep their original surface-fit acceptance rule.
        """
        head = task.get_camera_observation("head_camera")
        target = self.pad_only(head)
        if not np.isfinite(target.coaster_world_center).all():
            raise RuntimeError("No finite current visual placement target")
        attempts = audit.setdefault("precision_observation_attempts", [])
        for name in ("head_camera", f"{arm}_camera"):
            frame = head if name == "head_camera" else task.get_camera_observation(name)
            prompts = (("projected_reference", "raw_box", "depth_refined_box") if pose_hint_world is not None else
                       ("raw_box", "depth_refined_box", "projected_reference"))
            for prompt in prompts:
                refined = prompt != "raw_box"
                attempt = dict(camera=name, refined=refined, prompt=prompt, accepted=False)
                attempts.append(attempt)
                try:
                    hints = {} if pose_hint_world is None else dict(pose_hint_world=pose_hint_world)
                    if prompt == "projected_reference":
                        points = getattr(self.worker, "reference_points", None)
                        if points is None:
                            continue
                        reference_pose = getattr(self.worker, "reference_pose", None)
                        if pose_hint_world is not None and reference_pose is not None:
                            motion = np.asarray(pose_hint_world)@np.linalg.inv(reference_pose)
                            points = points@motion[:3, :3].T+motion[:3, 3]
                        box, point = projected_object_prompt(points, frame)
                        hints.update(object_roi_xyxy=box, object_point_xy=point)
                    stage = audit.get("stage", "held")
                    value = self.supplementary(frame, f"{stage}_{name}_{prompt}", cup_only=True,
                        refine_cup=refined, coarse_only=False, camera_name=name, **hints)
                    pose = np.asarray(value.cup_world_pose, dtype=float)
                    if pose.shape != (4, 4) or not np.isfinite(pose).all():
                        raise RuntimeError("Precise placement requires a finite CAD pose")
                    attempt["observation"] = value.observation_audit
                    if candidate_validator is not None:
                        validation = candidate_validator(value)
                        attempt["candidate_validation"] = validation
                        if not validation["accepted"]:
                            attempt.update(error=validation["reason"],
                                rejection_stage="placement_geometry", next_action="try_next_camera")
                            # Do not stop the whole placement or keep trying
                            # this camera after a geometrically implausible fit.
                            break
                    attempt["accepted"] = True
                    value.coaster_world_center = target.coaster_world_center.copy()
                    value.timings_s.update({f"target_{k}": v for k, v in target.timings_s.items()})
                    return value
                except (ValueError, RuntimeError) as error:
                    attempt["error"] = str(error)
        raise RuntimeError(f"No credible held CAD pose after head/wrist resegmentation: {attempts}")

    def select_reference(self, estimate):
        super().select_reference(estimate)
        # Only a search hint in later images. Every returned point remains a
        # CURRENT measured depth; the initial surface is never a fake observation.
        self.worker.reference_points = np.asarray(estimate.cup_points_world).copy()
        if self.output.name == "perception":
            self.worker.reference_pad = np.asarray(estimate.coaster_world_center).copy()
            self.worker.reference_pose = np.asarray(estimate.cup_world_pose).copy()
        with np.load(self.reference_path, allow_pickle=False) as data:
            payload = dict(data)
        payload["cup_world_pose"] = estimate.cup_world_pose
        np.savez(self.reference_path, **payload)


def execute_checked(task, arm, pose, plan, audit, *, slowdown=2., settle_steps=200,
                    allow_measured_handoff=False):
    if plan["status"] != "Success":
        raise RuntimeError(f"Known-model planning rejected: {plan['status']}")
    controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
    controls[f"{arm}_arm"] = slower_trajectory(plan, slowdown)
    audit.update(final_joint_target=np.asarray(plan["position"][-1]).tolist())
    moved = bool(task.take_dense_action(controls))
    audit.update(settling={}, slowdown=slowdown)
    settled = settle_at_target(task, arm, pose, plan["position"][-1], audit["settling"],
                               max_steps=settle_steps)
    if hasattr(task, "robot"):
        audit["measured_joint_state_after_hold"] = list(getattr(task.robot, f"get_{arm}_arm_real_jointState")())
    if moved and not settled and allow_measured_handoff:
        from adapters.pillbottle_known_handoff import settle_measured_handoff
        audit["strict_joint_target_reached"] = False
        audit["measured_state_handoff"] = {}
        settled = settle_measured_handoff(task, arm, pose, audit["measured_state_handoff"])
    if not moved or not settled:
        raise RuntimeError("Known-model motion did not reach measured robot target")


def joint_limit_audit(robot, arm, plan):
    """Check the actual robot's calibrated joint limits, not an object truth."""
    entity = getattr(robot, f"{arm}_entity")
    names = [joint.get_name() for joint in entity.get_active_joints()]
    indices = [names.index(name) for name in getattr(robot, f"{arm}_arm_joints_name")]
    limits = np.asarray(entity.get_qlimits(), dtype=float)[indices]
    position = np.asarray(plan["position"], dtype=float)
    if position.ndim != 2 or position.shape[1] != len(indices) or not np.isfinite(position).all():
        raise ValueError("Invalid planned arm trajectory")
    violation = np.maximum(limits[:, 0]-position, 0.)+np.maximum(position-limits[:, 1], 0.)
    maximum = float(np.max(violation))
    return dict(accepted=maximum <= 1e-4, maximum_violation_rad=maximum,
        physical_joint_limits=limits.tolist(), source="robot_joint_model_not_object_state")


def diagonal_approach(task, perception, args):
    # Reuse the already tested diagonal RGB-D approach without changing its
    # source. Known CAD registration still supplies the initial placement frame.
    saved = []
    try:
        for arm in ("left", "right"):
            name = f"{arm}_plan_path"
            original = getattr(task.robot, name)
            saved.append((name, name in vars(task.robot), original))
            def checked(*a, _original=original, _arm=arm, **kw):
                plan = _original(*a, **kw)
                if plan["status"] == "Success":
                    limits = joint_limit_audit(task.robot, _arm, plan)
                    plan = dict(plan, physical_joint_limit_check=limits)
                    if not limits["accepted"]:
                        plan.update(status="Fail", planner_status="PHYSICAL_JOINT_LIMIT")
                return plan
            setattr(task.robot, name, checked)
        handoff = approach_bottle(task, perception, args)
    finally:
        for name, owned, original in reversed(saved):
            if owned:
                setattr(task.robot, name, original)
            else:
                delattr(task.robot, name)
    with np.load(perception.reference_path, allow_pickle=False) as data:
        handoff["estimated_bottle_world_pose"] = data["cup_world_pose"].tolist()
    handoff["known_model_usage"] = "visual_initial_frame_and_placement_not_contact_approach"
    return handoff


def legacy_pregrasp(task, actor, arm, metadata, audit, allowed):
    """Frozen exhaustive selection retained as the old comparison path."""
    contact_args = {} if allowed is None else dict(contact_point_id=allowed)
    target, _ = task.choose_grasp_pose(actor, arm_tag=arm, pre_dis=.06, target_dis=0., **contact_args)
    if target is None or np.asarray(target).shape != (7,):
        raise RuntimeError("No feasible annotated visual pre-grasp")
    order = sorted(allowed if allowed is not None else range(len(metadata["contact_points_pose"])),
        key=lambda index: actor.get_contact_point(index, "matrix")[2, 3], reverse=True)
    for index in (None, *order):
        if index is not None:
            target, _ = task.choose_grasp_pose(actor, arm_tag=arm, pre_dis=.06,
                                               target_dis=0., contact_point_id=[index])
            if target is None:
                continue
        candidate = getattr(task.robot, f"{arm}_plan_path")(target)
        attempt = dict(contact_id=index, status=candidate["status"])
        audit["planning_attempts"].append(attempt)
        if candidate["status"] != "Success":
            continue
        limits = joint_limit_audit(task.robot, arm, candidate)
        attempt["physical_joint_limit_check"] = limits
        if limits["accepted"]:
            return target, candidate
    raise RuntimeError("No annotated pre-grasp within robot joint limits")


def approach(task, perception, args):
    if getattr(args, "known_approach", "annotated") == "diagonal":
        return diagonal_approach(task, perception, args)
    from envs.utils import ArmTag
    from traditional_place_empty_cup.controller import EstimatedActor
    if getattr(getattr(perception, "worker", None), "placement_policy", "legacy") == "visual":
        perception.worker.reference_pad = None
        perception.worker.reference_points = None
        perception.worker.reference_pose = None
        perception.worker.reference_table_z = None
        perception.worker.pad_target_validation = None
    camera = task.get_camera_observation("head_camera")
    value = perception.estimate(camera["rgb"], camera["depth"],
                                camera["intrinsic_cv"], camera["extrinsic_cv"])
    if getattr(getattr(perception, "worker", None), "placement_policy", "legacy") == "visual":
        perception.resolve_initial_pad(task, value)
    arm = ArmTag("left" if value.cup_world_pose[0, 3] < 0 else "right")
    metadata = json.loads(Path(args.known_metadata).read_text())
    if not task.move(task.open_gripper(arm, pos=1.)):
        raise RuntimeError("Known-model gripper preparation failed")
    actor = EstimatedActor(value.cup_world_pose, metadata)
    region = getattr(args, "known_contact_region", "all")
    allowed = central_contact_group(metadata) if region == "central" else None
    search = getattr(args, "known_approach_search", "ranked")
    audit = dict(source="known_contact_annotations_transformed_by_foundationpose",
                 uses_object_truth=False, planning_attempts=[], contact_region=region,
                 allowed_contact_ids=allowed, search_mode=search)
    if search == "legacy":
        target, plan = legacy_pregrasp(task, actor, arm, metadata, audit, allowed)
    elif search == "ranked":
        from adapters.pillbottle_pregrasp_search import select_ranked_pregrasp
        target, plan = select_ranked_pregrasp(task, actor, arm, audit, allowed)
    else:
        raise ValueError("Unknown known-model pregrasp search mode")
    moved = True
    try:
        execute_checked(task, arm, target, plan, audit, slowdown=args.approach_slowdown,
                        settle_steps=args.settle_steps, allow_measured_handoff=True)
    except RuntimeError as error:
        moved = False
        audit["execution_error"] = str(error)
    return dict(**handoff_ready(task.get_arm_pose(arm), target, moved and task.plan_success).to_dict(),
        arm=str(arm), target_ee_pose=list(target), actual_ee_pose=list(task.get_arm_pose(arm)),
        estimated_bottle_world_pose=value.cup_world_pose.tolist(), uses_contact_annotations=True,
        timing_s=value.timings_s, geometry_approach=audit,
        pad_target_validation=getattr(getattr(perception, "worker", None), "pad_target_validation", None))


def direct_handoff(task, perception, arm, result, config, **unused):
    """Authorize transport after the caller's close/near candidate, with NO motion.

    The shared runner's legacy verification slot is deliberately a no-op here.
    Authorization is not evidence of a physical grasp or of bottle/hand co-motion.
    """
    result.update(skipped=True, complete=False, handoff_authorized=True,
        trial_lift_enabled=False, lifted_grasp_verified=False, segments=[], observations=[],
        uses_object_truth=False, source="closed_near_candidate_without_trial_lift",
        reason="user_requested_direct_traditional_placement")
    return dict(ready=True, detector="closed_near_candidate_direct_handoff",
        trial_lift_enabled=False, lifted_grasp_verified=False, uses_object_truth=False)


def place(task, perception, arm, reference_path, verification, result, *,
          release_gap_m=.003, registration_config=None):
    from adapters.pillbottle_known_placement import finish_placement
    if not verification.get("handoff_authorized"):
        raise RuntimeError("Direct placement requires the close/near candidate handoff")
    result.update(trial_lift_enabled=False, lifted_grasp_verified=False,
        support_rise_required=False, assumes_bottle_held_after_closed_near_candidate=True)
    with np.load(reference_path, allow_pickle=False) as data:
        initial = data["cup_world_pose"].copy()
    if getattr(perception.worker, "placement_policy", "legacy") == "visual":
        from adapters.pillbottle_closed_loop import finish_visual_placement
        finish_visual_placement(task, perception, arm, perception.worker.metadata_path,
            result, upright_reference=initial, release_gap_m=release_gap_m)
        return
    original_estimate = perception.estimate
    original_execute = task.take_dense_action
    metadata = json.loads(Path(perception.worker.metadata_path).read_text())
    def estimate(*a, **kw):
        value = (perception.localize_held(task, arm, result) if hasattr(perception, "localize_held")
                 else original_estimate(*a, **kw))
        rise = support_point(value.cup_world_pose, metadata)[2]-support_point(initial, metadata)[2]
        result["initial_to_held_support_rise_m"] = float(rise)
        # Diagnostic only. A bottle still on the table must be allowed to enter
        # transport; requiring a rise here would silently reinstate the lift gate.
        result["support_rise_role"] = "diagnostic_only_not_a_placement_gate"
        value.coaster_world_center[2] += release_gap_m
        return value
    def execute(controls, *a, **kw):
        plan = controls.get(f"{arm}_arm")
        if plan is None or len(result["actions"]) >= 2:
            return original_execute(controls, *a, **kw)
        target = np.asarray(result["target_ee_pose"], dtype=float).copy()
        if not result["actions"]:
            target[2] += result["chosen_clearance_m"]
        altered = dict(controls)
        altered[f"{arm}_arm"] = slower_trajectory(plan, 2.)
        moved = bool(original_execute(altered, *a, **kw))
        audit = {}
        result.setdefault("arrival_checks", []).append(audit)
        if not moved or not settle_at_target(task, arm, target, plan["position"][-1], audit):
            raise RuntimeError("Known-model placement target not reached; do not release")
        return True
    perception.estimate, task.take_dense_action = estimate, execute
    try:
        finish_placement(task, perception, arm, perception.worker.metadata_path, result,
                         upright_reference=initial)
        result.update(uses_prepared_object_model=True, release_gap_m=release_gap_m)
    finally:
        result["geometry_source"] = "fresh_rgbd_after_close_candidate_without_trial_lift"
        perception.estimate, task.take_dense_action = original_estimate, original_execute
