"""Independent model-free pill-bottle adapter.

The legacy cup/coaster field names are an interface to the existing, unmodified
RGB-D/ICP/placement utilities; they mean bottle/pad here, not cup assets.
"""

import os
import time
from types import SimpleNamespace

import numpy as np

from adapters.perception import ExternalPerception, PerceptionWorker
from adapters.pointcloud_geometry import estimate_visible_bounds
from adapters.pointcloud_registration import estimate_pointcloud_motion
from adapters.rgbd_perception import GroundedPointCloudPerception, depth_foreground_seed
from adapters.visual_grasp import ClosedCommandMonitor, visual_comotion
from adapters.grasp import GraspComplete
from adapters.early_grasp import slower_trajectory
from adapters.handoff import handoff_ready
from adapters.approach_geometry import GeometryApproachConfig, generate_bounds_candidates, _execute_candidates
from adapters.adaptive_interaction import measured_resolution
from adapters.demo_relative_approach import load_demo_approach_prior, prepend_demo_candidates
from adapters.pillbottle_control import audit_topp, robot_state, settle_at_target


def projected_object_prompt(points, camera, *, displacement=None, padding_px=8):
    """Project a measured surface for SAM search, not a synthetic observation.

    Displacement is only a search hypothesis. ICP must still measure motion
    from fresh depths, and the independent co-motion gate must still pass.
    """
    from traditional_place_empty_cup.geometry import homogeneous_transform
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all():
        raise ValueError("Invalid measured reference surface")
    if displacement is not None:
        values = values+np.asarray(displacement, dtype=float)
    extrinsic = homogeneous_transform(camera["extrinsic_cv"])
    xyz = values@extrinsic[:3, :3].T+extrinsic[:3, 3]
    xyz = xyz[xyz[:, 2] > .01]
    if len(xyz) < 20:
        raise RuntimeError("Reference surface is behind this camera")
    pixels = xyz@np.asarray(camera["intrinsic_cv"], dtype=float).T
    pixels = pixels[:, :2]/pixels[:, 2:]
    h, w = camera["rgb"].shape[:2]
    lower, upper = np.quantile(pixels, [.01, .99], axis=0)
    lower = np.maximum(lower-padding_px, [0., 0.])
    upper = np.minimum(upper+padding_px, [w-1., h-1.])
    if np.any(upper-lower < 5):
        raise RuntimeError("Projected object search region is outside this camera")
    seed = np.median(pixels, axis=0)
    seed = np.clip(seed, lower+1, upper-1)
    return np.r_[lower, upper], seed


def observation_quality(estimate, reference_bounds=None):
    bounds = np.asarray(estimate.cup_bounds_world, dtype=float)
    points = np.asarray(estimate.cup_points_world, dtype=float)
    reasons = []
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 40
            or not np.isfinite(points).all()):
        raise RuntimeError("Invalid measured bottle surface")
    spans = bounds[1]-bounds[0]
    if np.min(spans[:2]) <= 0 or spans[2] < 0:
        raise RuntimeError("Invalid measured bottle extent")
    aspect = float(spans[2]/max(min(spans[:2]), .001))
    # Explicit bottle-class prior, not dimensions read from a simulator asset.
    if aspect > 4.0:
        reasons.append("vertical_aspect_over_bottle_prior_4")
    growth = None
    if reference_bounds is not None:
        ref = np.asarray(reference_bounds, dtype=float)
        growth = float(np.linalg.norm(spans)/max(np.linalg.norm(ref[1]-ref[0]), .001))
        if growth > 1.6:
            reasons.append("extent_growth_over_1.6")
    if estimate.observation_audit.get("cup_mask_touches_border", False):
        reasons.append("object_mask_clipped")
    return dict(geometry_ok=not reasons, reasons=reasons, spans_m=spans.tolist(),
                vertical_aspect=aspect, vertical_aspect_limit=4.0, extent_growth_ratio=growth)


class PillbottleWorker(PerceptionWorker):
    def __init__(self, root, output, *, object_prompt="pill bottle", target_prompt="blue square pad"):
        super().__init__(root, output, backend="pointcloud")
        self.object_prompt, self.target_prompt = object_prompt, target_prompt

    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command = [command[0], str(self.root/"scripts/perceive_pillbottle_rgbd.py"),
                   "--object-prompt", self.object_prompt, "--target-prompt", self.target_prompt]
        environment["PILLBOTTLE_VISION_TORCH_FRACTION"] = os.environ.get(
            "PILLBOTTLE_VISION_TORCH_FRACTION", ".08")
        return command, cwd, environment


class PillbottleVision(GroundedPointCloudPerception):
    def __init__(self, root, object_prompt="pill bottle", target_prompt="blue square pad"):
        super().__init__(root)
        self.prompts = dict(cup=object_prompt, coaster=target_prompt)

    def _detect_both(self, rgb):
        # Query each semantic class separately; the cup implementation's phrase
        # matcher deliberately recognizes only the literal words cup/coaster.
        selected = {}
        for key, prompt in self.prompts.items():
            found, scores, _ = self._run_dino(rgb, f"{prompt} .")
            if not len(found):
                raise RuntimeError(f"GroundingDINO did not detect {prompt!r}")
            index = int(np.argmax(scores.detach().cpu().numpy().reshape(-1)))
            selected[key] = self._xyxy(found[index].detach().cpu().numpy(), rgb.shape[1], rgb.shape[0])
        return selected

    def estimate(self, rgb, depth, intrinsic, world_to_camera, cup_only=False, refine_cup=False,
                 object_roi_xyxy=None, object_point_xy=None, target_only=False):
        from traditional_place_empty_cup.geometry import depth_to_metres, robust_mask_center_world

        start = time.perf_counter()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        if target_only:
            found, scores, _ = self._run_dino(rgb, f"{self.prompts['coaster']} .")
            if not len(found):
                raise RuntimeError("No pad detected in current RGB-D")
            index = int(np.argmax(scores.detach().cpu().numpy().reshape(-1)))
            box = self._xyxy(found[index].detach().cpu().numpy(), rgb.shape[1], rgb.shape[0])
            with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                self.sam.set_image(rgb)
                masks, _, _ = self.sam.predict(box=np.asarray([box]), multimask_output=False)
            mask = np.asarray(masks).reshape(-1, *rgb.shape[:2])[0].astype(bool)
            centre = robust_mask_center_world(mask, depth_to_metres(depth), intrinsic, world_to_camera)
            return SimpleNamespace(cup_bounds_world=np.full((2, 3), np.nan),
                cup_points_world=np.empty((0, 3)), coaster_world_center=centre,
                observation_audit=dict(target_only=True, uses_object_truth=False),
                timings_s=dict(perception_total=time.perf_counter()-start))
        stage = time.perf_counter()
        if object_roi_xyxy is not None:
            if not cup_only:
                raise ValueError("Projected object prompts are object-only")
            box = np.asarray(object_roi_xyxy, dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError("Invalid projected image box")
            boxes = dict(cup=box)
        elif cup_only:
            found, scores, _ = self._run_dino(rgb, f"{self.prompts['cup']} .")
            if not len(found):
                raise RuntimeError("No bottle detected in supplementary view")
            index = int(np.argmax(scores.detach().cpu().numpy().reshape(-1)))
            boxes = dict(cup=self._xyxy(found[index].detach().cpu().numpy(), rgb.shape[1], rgb.shape[0]))
        else:
            boxes = self._detect_both(rgb)
        self.torch.cuda.synchronize()
        timings = dict(detection=time.perf_counter()-stage)
        stage = time.perf_counter()
        if cup_only:
            with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                self.sam.set_image(rgb)
                values, _, _ = self.sam.predict(box=np.asarray([boxes["cup"]]), multimask_output=False)
            mask = np.asarray(values).reshape(-1, *rgb.shape[:2])[0].astype(bool)
            masks = dict(cup=mask, coaster=np.zeros_like(mask))
        else:
            masks = self._segment_both(rgb, boxes)

        def measure(mask):
            bounds, audit, points = estimate_visible_bounds(mask, masks["coaster"], depth,
                intrinsic, world_to_camera, return_points=True)
            rows, cols = np.nonzero(mask)
            audit.update(cup_mask_touches_border=bool(np.any(rows < 2) or np.any(cols < 2)
                or np.any(rows >= rgb.shape[0]-2) or np.any(cols >= rgb.shape[1]-2)),
                cup_mask_bbox_pixels=[int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())])
            return SimpleNamespace(cup_bounds_world=bounds, cup_points_world=points, observation_audit=audit)

        if refine_cup:
            seed = None if object_point_xy is None else np.asarray(object_point_xy, dtype=float)
            try:
                initial = measure(masks["cup"])
                if seed is None:
                    seed = depth_foreground_seed(masks["cup"], depth, intrinsic, world_to_camera,
                        initial.observation_audit["table_height_m"], boxes["cup"])
            except (RuntimeError, ValueError):
                pass
            def valid(mask):
                try:
                    return observation_quality(measure(mask))["geometry_ok"]
                except (RuntimeError, ValueError):
                    return False
            masks["cup"] = self._refine_cup_mask(rgb, boxes["cup"], seed, valid)
        self.torch.cuda.synchronize()
        timings["segmentation"] = time.perf_counter()-stage
        stage = time.perf_counter()
        result = measure(masks["cup"])
        result.coaster_world_center = (np.full(3, np.nan) if cup_only else robust_mask_center_world(
            masks["coaster"], depth_to_metres(depth), intrinsic, world_to_camera))
        result.observation_audit.update(object_semantics="pill_bottle", target_semantics="blue_square_pad",
            image_size_hw=list(rgb.shape[:2]), coaster_observed=not cup_only,
            cup_segmentation_variant="roi_box_and_positive_point" if refine_cup else "box_only",
            geometry_quality=observation_quality(result), uses_robot_identity_mask=False,
            object_prompt=self.prompts["cup"], target_prompt=self.prompts["coaster"])
        result.observation_audit["projected_reference_prompt"] = object_roi_xyxy is not None
        timings.update(pointcloud_geometry=time.perf_counter()-stage, perception_total=time.perf_counter()-start)
        result.timings_s = timings
        return result


class PillbottlePerception(ExternalPerception):
    """At most original head mask + changed-prompt head mask, no arm motion."""
    def estimate(self, rgb, depth, intrinsic, world_to_camera):
        attempts = []
        camera = dict(rgb=rgb, depth=depth, intrinsic_cv=intrinsic, extrinsic_cv=world_to_camera)
        for refined in (False, True):
            attempt = dict(refined=refined)
            attempts.append(attempt)
            try:
                value = (super().supplementary(camera, "head_refined", cup_only=False, refine_cup=True)
                         if refined else super().estimate(rgb, depth, intrinsic, world_to_camera))
                attempt["quality"] = observation_quality(value)
                if not attempt["quality"]["geometry_ok"]:
                    continue
                if not np.isfinite(value.coaster_world_center).all():
                    raise RuntimeError("No finite visual pad target")
                value.observation_audit["head_observation_attempts"] = attempts
                self.select_reference(value)
                return value
            except (RuntimeError, ValueError) as error:
                attempt["error"] = str(error)
        raise RuntimeError(f"Bottle/pad observation rejected after bounded resegmentation: {attempts}")

    def hinted(self, camera, label, points, *, displacement=None):
        directory = self.output/label
        directory.mkdir(parents=True, exist_ok=True)
        box, seed = projected_object_prompt(points, camera, displacement=displacement)
        request, result = directory/"rgbd.npz", directory/"estimate.npz"
        np.savez_compressed(request, rgb=camera["rgb"], depth=camera["depth"],
            intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"],
            cup_only=np.asarray(True), refine_cup=np.asarray(True), object_roi_xyxy=box, object_point_xy=seed)
        estimate = self.worker.estimate_files(request, result)
        estimate.observation_audit.update(search_motion_hypothesis_only=True,
            observed_points_are_fresh_depth=True, object_search_box=box.tolist())
        return estimate

    def pad_only(self, camera):
        self.output.mkdir(parents=True, exist_ok=True)
        request, result = self.output/"pad_rgbd.npz", self.output/"pad_estimate.npz"
        np.savez_compressed(request, rgb=camera["rgb"], depth=camera["depth"],
            intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"], target_only=np.asarray(True))
        return self.worker.estimate_files(request, result)


def geometry_pitch_candidates(bounds, table_height, requested=()):
    """Generate orientation coverage from the measured object aspect ratio.

    User-supplied values remain compatibility seeds, but the search no longer
    depends on them being sufficient.  Extra angles are derived from observed
    horizontal radius versus height and are therefore shared by every object.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Expected finite measured object bounds")
    spans = np.maximum(bounds[1] - bounds[0], np.finfo(float).eps)
    height = max(float(bounds[1, 2] - table_height), float(np.min(spans)))
    radius = float(np.linalg.norm(spans[:2]) / 2.)
    natural = float(np.degrees(np.arctan2(radius, height)))
    derived = (0., -natural, natural, -2. * natural, 2. * natural)
    values = []
    for angle in (*requested, *derived):
        angle = float(np.clip(angle, -60., 60.))
        if not any(abs(angle - old) < 1e-6 for old in values):
            values.append(angle)
    return values


def approach_bottle(task, perception, args):
    """Observed bottle bounds, robot TCP calibration, bounded pitch/yaw IK search."""
    from envs.utils import ArmTag
    from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix

    camera = task.get_camera_observation("head_camera")
    estimate = perception.estimate(camera["rgb"], camera["depth"], camera["intrinsic_cv"], camera["extrinsic_cv"])
    bounds = np.asarray(estimate.cup_bounds_world)
    arm = ArmTag("left" if bounds[:, 0].mean() < 0 else "right")
    if getattr(args, "approach_style", "side") == "home-diagnostic":
        return dict(ready=True, reason="diagnostic_no_traditional_approach", arm=str(arm),
            diagnostic_only=True, estimated_bottle_bounds_world=bounds.tolist(),
            estimated_pad_world_center=estimate.coaster_world_center.tolist(),
            geometry_approach=dict(observation=estimate.observation_audit, planning_attempts=[]))
    config = GeometryApproachConfig(args.geometry_top_clearance, args.geometry_edge_fraction,
                                    tuple(args.geometry_yaw_offsets))
    style = getattr(args, "approach_style", "top")
    table_z = float(estimate.observation_audit.get("table_height_m", bounds[0, 2]))
    pitches = geometry_pitch_candidates(bounds, table_z, args.approach_pitch_deg)

    def build(candidate_arm):
        ee = pose_wxyz_to_matrix(task.get_arm_pose(candidate_arm))
        tcp = pose_wxyz_to_matrix(getattr(task.robot, f"get_{candidate_arm}_tcp_pose")())
        offset = np.linalg.solve(ee, tcp)
        base, local_audit = generate_bounds_candidates(
            bounds, str(candidate_arm), offset, config,
            reference_tcp_world=tcp if style != "side" else None)
        if style == "diagonal":
            side = 1. if str(candidate_arm) == "left" else -1.
            angle = np.radians(15.)
            forward = np.array([side*np.sin(angle)/np.sqrt(2), np.sin(angle)/np.sqrt(2), -np.cos(angle)])
            jaw = np.array([-1., side, 0.])/np.sqrt(2)
            canonical = np.column_stack((forward, jaw, np.cross(forward, jaw)))
            for candidate in base:
                yaw = candidate["yaw_offset_rad"]
                c, s = np.cos(yaw), np.sin(yaw)
                candidate["tcp_world"][:3, :3] = np.array(
                    [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ canonical
        if style == "side":
            position = bounds.mean(axis=0)
            position[1] = bounds[0, 1] - args.side_standoff
            position[2] = table_z + .5 * (bounds[1, 2] - table_z)
            canonical = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            for candidate in base:
                angle = candidate["yaw_offset_rad"]
                c, s = np.cos(angle), np.sin(angle)
                candidate["tcp_world"][:3, :3] = np.array(
                    [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ canonical
                candidate["tcp_world"][:3, 3] = position
        candidates = []
        for pitch in pitches:
            a = np.radians(pitch)
            tilt = np.array([[1., 0., 0.], [0., np.cos(a), -np.sin(a)],
                             [0., np.sin(a), np.cos(a)]])
            for candidate in base:
                matrix = candidate["tcp_world"].copy()
                matrix[:3, :3] = tilt @ matrix[:3, :3]
                candidates.append(dict(
                    candidate, tcp_world=matrix, ee_world=matrix @ np.linalg.inv(offset),
                    approach_pitch_deg=float(pitch)))
        local_audit.update(
            source="rgbd_bounds_with_geometry_derived_orientation_and_dual_arm_reachability",
            uses_prepared_object_model=False, uses_foundationpose=False,
            observation=estimate.observation_audit,
            pitch_candidates_deg=pitches,
            pitch_candidates_source="compatibility_seeds_plus_measured_object_aspect_ratio",
            pitch_axis="world_X_toward_positive_Y_for_robot_reach",
            candidate_count=len(candidates), approach_style=style,
            side_standoff_m=getattr(args, "side_standoff", None),
            orientation_source=("horizontal_front_body_grasp_with_pitch_yaw_candidates"
                if style == "side" else "robot_mirrored_diagonal_jaw_15deg_downward_approach"
                if style == "diagonal" else local_audit["orientation_source"]),
        )
        prior_path = getattr(args, "demo_approach_prior", None)
        if prior_path is not None:
            candidates, local_audit = prepend_demo_candidates(
                candidates, local_audit, load_demo_approach_prior(prior_path), bounds,
                str(candidate_arm), offset,
                object_pose_world=getattr(estimate, "object_pose_world", None),
                uncertainty_m=measured_resolution(estimate.cup_points_world),
            )
        return candidates, local_audit

    def execute_for_arm(candidate_arm):
        if not task.move(task.close_gripper(
                candidate_arm, pos=getattr(args, "approach_gripper", .6))):
            raise RuntimeError("Bottle gripper preparation failed")
        candidates, local_audit = build(candidate_arm)
        if getattr(args, "settle_steps", 0):
            original_execute = task.take_dense_action
            def execute(controls, *a, **kw):
                altered = dict(controls)
                plan = controls[f"{candidate_arm}_arm"]
                altered[f"{candidate_arm}_arm"] = slower_trajectory(plan, args.approach_slowdown)
                moved = original_execute(altered, *a, **kw)
                local_audit["trajectory_execution"] = dict(
                    slowdown=args.approach_slowdown,
                    final_joint_target=np.asarray(plan["position"][-1]).tolist(),
                    before_settle_robot_state=robot_state(task.robot), settling={})
                target_pose = candidates[local_audit["selected_candidate"]]["ee_world"]
                from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz
                settled = settle_at_target(
                    task, candidate_arm, matrix_to_pose_wxyz(target_pose), plan["position"][-1],
                    local_audit["trajectory_execution"]["settling"], max_steps=args.settle_steps)
                local_audit["trajectory_execution"]["after_settle_robot_state"] = robot_state(task.robot)
                return bool(moved and settled)
            task.take_dense_action = execute
            try:
                target, moved, local_audit = _execute_candidates(
                    task, candidate_arm, candidates, local_audit)
            finally:
                task.take_dense_action = original_execute
        else:
            target, moved, local_audit = _execute_candidates(
                task, candidate_arm, candidates, local_audit)
        for attempt in local_audit["planning_attempts"]:
            attempt["approach_pitch_deg"] = candidates[
                attempt["candidate_index"]]["approach_pitch_deg"]
        return target, moved, local_audit, candidates

    preferred = str(arm)
    arm_attempts = []
    target, moved, audit, candidates = execute_for_arm(arm)
    arm_attempts.append(dict(arm=preferred, selected_candidate=audit["selected_candidate"],
                             failure_reason=audit.get("failure_reason")))
    # Only change arms if no trajectory was executed.  This is reachability
    # selection, not a retry after physical motion.
    if not moved and audit["selected_candidate"] is None:
        alternate = ArmTag("right" if preferred == "left" else "left")
        alt_target, alt_moved, alt_audit, alt_candidates = execute_for_arm(alternate)
        arm_attempts.append(dict(arm=str(alternate), selected_candidate=alt_audit["selected_candidate"],
                                 failure_reason=alt_audit.get("failure_reason")))
        if alt_moved or alt_audit["selected_candidate"] is not None:
            arm, target, moved, audit, candidates = (
                alternate, alt_target, alt_moved, alt_audit, alt_candidates)
    audit["arm_reachability_attempts"] = arm_attempts
    audit["arm_selection_source"] = "bounded_motion_planning_before_execution"
    decision = handoff_ready(task.get_arm_pose(arm), target, moved and task.plan_success)
    return dict(**decision.to_dict(), arm=str(arm), target_ee_pose=target.tolist(),
        actual_ee_pose=list(task.get_arm_pose(arm)), estimated_bottle_bounds_world=bounds.tolist(),
        estimated_pad_world_center=estimate.coaster_world_center.tolist(), geometry_approach=audit,
        timing_s=estimate.timings_s, uses_contact_annotations=False)


def rollout_bottle(task, policy, perception, arm, record, *, proximity_m=.015, trace=False,
                   reference_path=None, synchronize_handoff=False, stable_steps=3,
                   gripper_threshold=.05, require_visual_proximity=True,
                   sim_monitor=None):
    """Run local VLA grasp until the selected handoff detector is satisfied.

    Legacy callers retain the visual-proximity gate.  The rule-driven ULA
    entry can instead use the declarative stable closed-command detector; that
    mode deliberately performs no extra RGB-D query during the VLA rollout.
    Neither mode claims that a physical grasp has been verified.
    """
    if not np.isfinite(proximity_m) or not 0 < proximity_m <= .03:
        raise ValueError("Invalid bottle close-candidate proximity")
    monitor = ClosedCommandMonitor(
        stable_steps=stable_steps, gripper_threshold=gripper_threshold)
    original = task.take_action
    record["close_proximity_checks"] = checks = []
    last_check = -3
    reference = None
    # The grasp-only path never calls select_reference(), so the close-proximity
    # check must tolerate a missing reference file instead of aborting the episode.
    if reference_path is not None and os.path.exists(str(reference_path)):
        with np.load(reference_path, allow_pickle=False) as data:
            reference = SimpleNamespace(cup_points_world=data["cup_points_world"].copy(),
                cup_bounds_world=data["cup_bounds_world"].copy())
    # Object-scale proximity instead of a fixed 1.5 cm: half of the smallest
    # observed dimension, so a 3 cm bottle is not held to a cup-sized absolute
    # gate while a large object is not over-constrained.  The visual trial lift
    # remains the real verdict; this is only the close-candidate heuristic.
    if reference is not None:
        extents = (np.asarray(reference.cup_bounds_world, dtype=float)[1]
                   - np.asarray(reference.cup_bounds_world, dtype=float)[0])
        usable = extents[np.isfinite(extents) & (extents > 0)]
        if usable.size:
            proximity_m = float(max(0.5 * float(np.min(usable)), 0.01))
    record["close_proximity_m"] = float(proximity_m)
    if trace:
        record["action_trace"] = []
    if synchronize_handoff:
        record["handoff_state_sync"] = dict(before=robot_state(task.robot), uses_object_truth=False,
            policy_state_semantics="training_command_state_after_measured_handoff_sync",
            action_interpolation_start="measured_joint_angles")
        for name in ("left", "right"):
            actual = np.asarray(getattr(task.robot, f"get_{name}_arm_real_jointState")(), dtype=float)[:6]
            task.robot.set_arm_joints(actual, np.zeros(6), name)
        record["handoff_state_sync"]["after"] = robot_state(task.robot)

    def action(value, *a, **kw):
        nonlocal last_check
        values = np.asarray(value)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("WorldLA must return a finite 14-D joint action")
        if trace:
            item = dict(action=values.tolist(), before=robot_state(task.robot), topp=[])
            record["action_trace"].append(item)
            with audit_topp(task.robot, item["topp"]):
                if synchronize_handoff:
                    from adapters.pillbottle_control import measured_joint_input
                    with measured_joint_input(task.robot):
                        returned = original(value, *a, **kw)
                else:
                    returned = original(value, *a, **kw)
            item.update(step=int(task.take_action_cnt), after=robot_state(task.robot),
                ee_pose=list(task.get_arm_pose(arm)),
                tcp_pose=list(getattr(task.robot, f"get_{arm}_tcp_pose")()))
        else:
            returned = original(value, *a, **kw)
        if sim_monitor is not None:
            from adapters.grasp import active_gripper_contacts
            from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
            sim = sim_monitor.update(
                task.cup.get_pose().to_transformation_matrix(),
                pose_wxyz_to_matrix(task.get_arm_pose(arm)),
                getattr(task.robot, f"get_{arm}_gripper_val")(),
                active_gripper_contacts(task, arm), task.take_action_cnt,
            )
            if sim["ready"]:
                sim["ee_pose"] = list(task.get_arm_pose(arm))
                record["candidate_handoff"] = sim
                raise GraspComplete()
        candidate = monitor.update(getattr(task.robot, f"get_{arm}_gripper_val")(), task.take_action_cnt)
        if not candidate["ready"] or task.take_action_cnt-last_check < 3:
            return returned
        last_check = task.take_action_cnt
        if not require_visual_proximity:
            candidate.update(
                ee_pose=list(task.get_arm_pose(arm)),
                detector="stable_closed_gripper_command",
                visual_proximity_required=False,
                lifted_grasp_verified=False,
            )
            record["candidate_handoff"] = candidate
            raise GraspComplete()
        if reference is not None:
            from adapters.pillbottle_close import observe_close_candidate
            observed = observe_close_candidate(task, perception, arm, task.take_action_cnt,
                reference, checks, threshold=proximity_m)
            if observed is not None:
                candidate.update(visual_proximity=observed, ee_pose=list(task.get_arm_pose(arm)),
                    observation_reference=observed["observation_reference"],
                    detector="closed_command_and_registered_current_rgbd_reference_proximity",
                    lifted_grasp_verified=False)
                record["candidate_handoff"] = candidate
                raise GraspComplete()
            return returned
        for camera_name in ("head_camera", f"{arm}_camera"):
            audit = dict(vla_step=int(task.take_action_cnt), camera=camera_name, threshold_m=proximity_m,
                         uses_object_truth=False)
            checks.append(audit)
            try:
                frame = task.get_camera_observation(camera_name)
                estimate = perception.supplementary(frame, f"close_{task.take_action_cnt}_{camera_name}",
                    cup_only=True, refine_cup=False)
                audit["quality"] = observation_quality(estimate)
                if not audit["quality"]["geometry_ok"]:
                    continue
                tcp = np.asarray(getattr(task.robot, f"get_{arm}_tcp_pose")(), dtype=float)[:3]
                distances = np.linalg.norm(estimate.cup_points_world-tcp, axis=1)
                # TCP is the pinch reference, not a point on the bottle skin.
                # Interior TCPs must not fail simply because the bottle is wide.
                bounds = np.asarray(estimate.cup_bounds_world, dtype=float)
                outside = np.maximum(bounds[0]-tcp, 0.)+np.maximum(tcp-bounds[1], 0.)
                bounds_distance = float(np.linalg.norm(outside))
                audit.update(tcp_world=tcp.tolist(), nearest_surface_distance_m=float(distances.min()),
                    distance_to_visible_bounds_m=bounds_distance, visible_bounds_world=bounds.tolist(),
                    proximity_geometry="distance_to_observed_AABB_not_surface", accepted=bounds_distance <= proximity_m)
                if audit["accepted"]:
                    candidate.update(visual_proximity=audit, ee_pose=list(task.get_arm_pose(arm)),
                        detector="closed_command_and_current_visual_surface_proximity",
                        lifted_grasp_verified=False)
                    record["candidate_handoff"] = candidate
                    raise GraspComplete()
                # A valid head observation is sufficient to reject proximity;
                # only try the wrist if the head geometry itself was invalid.
                break
            except (RuntimeError, ValueError) as error:
                audit["error"] = str(error)
        return returned

    task.take_action = action
    try:
        policy.reset()
        while task.take_action_cnt < task.step_lim:
            before = task.take_action_cnt
            policy.step(task, task.get_obs())
            print(f"[ula-pillbottle] vla_step={task.take_action_cnt}/{task.step_lim}", flush=True)
            if before == task.take_action_cnt:
                raise RuntimeError("WorldLA executed no actions")
    except GraspComplete:
        print(f"[ula-pillbottle] visual close candidate at step={task.take_action_cnt}", flush=True)
    finally:
        task.take_action = original
    if not record.get("candidate_handoff"):
        requirement = ("stable closed-gripper command" if not require_visual_proximity
                       else "visually near closed-gripper candidate")
        raise RuntimeError(f"No {requirement} within VLA budget")


def verify_bottle_lift(task, perception, arm, result, config, *, lift_m=.05,
                       candidate_reference_path=None):
    """Same visual co-motion criterion as the cup pilot, bottle-specific quality."""
    from copy import deepcopy

    if not np.isfinite(lift_m) or not .03 <= lift_m <= .10:
        raise ValueError("Visual trial lift must be finite and within 3-10 cm")
    result.update(complete=False, observations=[], segments=[], uses_object_truth=False,
        slip_gate_enabled=False, failure_policy="stop_without_release", minimum_visual_lift_m=.03)
    names = ("head_camera", f"{arm}_camera")
    frames = {name: deepcopy(task.get_camera_observation(name)) for name in names}
    refs = {}
    candidate_points = None
    if candidate_reference_path is not None:
        with np.load(candidate_reference_path, allow_pickle=False) as data:
            candidate_points = data["cup_points_world"].copy()
    def observe(frame, label, refined, ref=None, hint=None, displacement=None):
        value = (perception.hinted(frame, label, hint, displacement=displacement) if hint is not None else
                 perception.supplementary(frame, label, cup_only=True, refine_cup=refined))
        quality = observation_quality(value, ref)
        if not quality["geometry_ok"]:
            raise RuntimeError(f"Untrusted bottle geometry: {quality}")
        return value
    for name in names:
        try:
            refs[name] = observe(frames[name], f"before_{name}", True,
                hint=candidate_points if hasattr(perception, "hinted") else None)
            result["observations"].append(dict(stage="baseline", camera=name, accepted=True))
            break
        except (RuntimeError, ValueError) as error:
            result["observations"].append(dict(stage="baseline", camera=name, error=str(error)))
    if not refs:
        raise RuntimeError("No credible bottle baseline; no trial lift executed")
    start = np.asarray(task.get_arm_pose(arm), dtype=float)
    for index in range(3):
        target = start.copy()
        target[2] += lift_m*(index+1)/3
        plan = getattr(task.robot, f"{arm}_plan_path")(target)
        audit = dict(target_ee_pose=target.tolist(), status=plan["status"])
        result["segments"].append(audit)
        if plan["status"] != "Success":
            raise RuntimeError("Bottle trial-lift planning failed")
        controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
        controls[f"{arm}_arm"] = slower_trajectory(plan, 2)
        moved = bool(task.take_dense_action(controls))
        decision = handoff_ready(task.get_arm_pose(arm), target, moved and task.plan_success)
        audit["arrival"] = decision.to_dict()
        if not decision.ready:
            raise RuntimeError("Bottle trial-lift arrival rejected")
    displacement = np.asarray(task.get_arm_pose(arm))[:3]-start[:3]
    for name in names:
        try:
            if name not in refs:
                # Same-time head geometry can prompt a wrist image without
                # claiming that these are the same camera reference surfaces.
                head = refs.get("head_camera")
                refs[name] = observe(frames[name], f"before_{name}", True,
                    hint=(head.cup_points_world if head is not None and hasattr(perception, "hinted") else None))
            ref = refs[name]
        except (RuntimeError, ValueError) as error:
            result["observations"].append(dict(stage="fallback_baseline", camera=name, error=str(error)))
            continue
        for refined in (False, True):
            audit = dict(stage="after_lift", camera=name, refined=refined)
            result["observations"].append(audit)
            try:
                after = observe(task.get_camera_observation(name), f"after_{name}_{int(refined)}",
                    refined, ref.cup_bounds_world,
                    hint=(ref.cup_points_world if refined and hasattr(perception, "hinted") else None),
                    displacement=displacement)
                motion, audit["registration"] = estimate_pointcloud_motion(ref.cup_points_world,
                    after.cup_points_world, config)
                if motion is None:
                    continue
                decision = visual_comotion(ref.cup_points_world, motion, displacement)
                audit["comotion"] = decision
                if decision["ready"]:
                    if hasattr(perception, "select_reference"):
                        perception.select_reference(after)
                        result.update(verified_surface_path=str(perception.reference_path),
                            verified_surface_ee_pose=list(task.get_arm_pose(arm)),
                            verified_surface_camera=name)
                    result.update(complete=True, selected_camera=name)
                    return dict(decision, lifted_grasp_verified=True, camera=name)
            except (RuntimeError, ValueError) as error:
                audit["error"] = str(error)
    raise RuntimeError("Visual trial lift did not establish bottle/hand co-motion")
