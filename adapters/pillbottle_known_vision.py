"""Separate surface-only handoff perception from strict known-CAD localization."""
import time

import numpy as np

from traditional_place_empty_cup.perception import GroundedFoundationPerception
from traditional_place_empty_cup.geometry import depth_to_metres, homogeneous_transform
from adapters.pillbottle import PillbottleVision
from adapters.rgbd_perception import GroundedPointCloudPerception
from adapters.pillbottle_known_model import centered_camera_pose, pose_fit


def surface_mask(points, intrinsic, world_to_camera, shape):
    """Project measured, filtered object pixels, not a rendered CAD silhouette."""
    transform = homogeneous_transform(world_to_camera)
    xyz = np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
    xyz = xyz[xyz[:, 2] > .01]
    pixels = xyz @ np.asarray(intrinsic).T
    xy = np.rint(pixels[:, :2]/pixels[:, 2:]).astype(int)
    h, w = shape
    xy = xy[(xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)]
    mask = np.zeros((h, w), dtype=bool)
    mask[xy[:, 1], xy[:, 0]] = True
    if mask.sum() < 40:
        raise RuntimeError("Insufficient measured object pixels for CAD localization")
    return mask


class KnownBottleVision(GroundedFoundationPerception):
    _detect_both = PillbottleVision._detect_both
    _refine_cup_mask = GroundedPointCloudPerception._refine_cup_mask

    def estimate(self, rgb, depth, intrinsic, world_to_camera, cup_only=False,
                 refine_cup=False, object_roi_xyxy=None, object_point_xy=None,
                 coarse_only=False, target_only=False, camera_name="head_camera",
                 pose_hint_world=None):
        value = PillbottleVision.estimate(self, rgb, depth, intrinsic, world_to_camera,
            cup_only=cup_only, refine_cup=refine_cup, object_roi_xyxy=object_roi_xyxy,
            object_point_xy=object_point_xy, target_only=target_only)
        value.cup_world_pose = None
        value.observation_audit.update(perception_role="target_only" if target_only else
            "coarse_handoff_surface" if coarse_only else "precise_cad_localization",
            camera_name=str(camera_name), cad_pose_required=not (coarse_only or target_only))
        # Do not call FoundationPose, read its last pose, or mutate tracking state
        # for a close candidate. These observations must work even if CAD fails.
        if coarse_only or target_only:
            return value
        started = time.perf_counter()
        transform = homogeneous_transform(world_to_camera)
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        metres = depth_to_metres(depth)
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        mask = surface_mask(value.cup_points_world, intrinsic, transform, rgb.shape[:2])
        guarded = getattr(self, "geometry_guard", False)
        if guarded:
            from adapters.pillbottle_scene_geometry import table_clearance
            if not cup_only or getattr(self, "table_height_world", None) is None:
                self.table_height_world = float(value.observation_audit["table_height_m"])
            luminance = rgb[..., :3].astype(float)@np.array([.2126, .7152, .0722])/255.
            if not cup_only:
                # Appearance is learned from this episode's unobstructed object,
                # never robot pixel identities or a hard-coded bottle color.
                self.object_luminance_floor = max(0., float(np.quantile(luminance[mask], .1))-.12)
            elif getattr(self, "object_luminance_floor", 0.) > 0:
                from adapters.pointcloud_geometry import estimate_visible_bounds
                filtered = mask & (luminance >= self.object_luminance_floor)
                bounds, appearance_audit, points = estimate_visible_bounds(filtered,
                    np.zeros_like(filtered), depth, intrinsic, world_to_camera, return_points=True)
                value.cup_points_world, value.cup_bounds_world = points, bounds
                value.observation_audit.update(appearance_audit,
                    appearance_source="initial_rgbd_bottle_luminance_not_robot_identity",
                    learned_luminance_floor=self.object_luminance_floor,
                    appearance_retained_fraction=float(filtered.sum()/max(mask.sum(), 1)))
                mask = surface_mask(points, intrinsic, transform, rgb.shape[:2])
            # The refinement network must not align the bottle to nearby fingers
            # or background depth. RGB is retained, non-object depth is unknown.
            metres = np.where(mask, metres, 0.).astype(np.float32)
        attempts = []
        accepted = None
        methods = (["robot_transport_hypothesis_refined_by_rgbd"] if guarded and pose_hint_world is not None else [])
        methods += (["visual_previous_pose_tracking"] if self.last_accepted_world_pose is not None else [])
        methods += ["global_registration"]
        for method in methods:
            attempt = dict(method=method)
            attempts.append(attempt)
            try:
                with self.torch.inference_mode():
                    if method != "global_registration":
                        centered = self.foundationpose.get_tf_to_centered_mesh().cpu().numpy()
                        prior = pose_hint_world if method.startswith("robot_transport") else self.last_accepted_world_pose
                        if guarded:
                            prior_fit = pose_fit(value.cup_points_world, prior, self.model_samples)
                            prior_fit["table_clearance"] = table_clearance(prior, self.model_samples, self.table_height_world)
                            if prior_fit["accepted"] and prior_fit["table_clearance"]["accepted"]:
                                accepted = np.asarray(prior, dtype=float).copy()
                                attempt.update(prior_fit, fresh_surface_validated_prior=True)
                                break
                        self.foundationpose.pose_last = self.torch.as_tensor(
                            centered_camera_pose(prior, transform, centered),
                            dtype=self.torch.float32, device=self.device)
                        camera_pose = self.foundationpose.track_one(rgb=rgb, depth=metres,
                            K=intrinsic, iteration=self.fp_iterations)
                    else:
                        if guarded:
                            self.foundationpose.poses = None
                        camera_pose = self.foundationpose.register(rgb=rgb, depth=metres,
                            K=intrinsic, ob_mask=mask, iteration=self.fp_iterations)
                hypotheses = [np.asarray(camera_pose)]
                if guarded and method == "global_registration" and getattr(self.foundationpose, "poses", None) is not None:
                    centered = self.foundationpose.get_tf_to_centered_mesh().cpu().numpy()
                    hypotheses = [p@centered for p in self.foundationpose.poses[:16].detach().cpu().numpy()]
                attempt["hypotheses"] = []
                for rank, camera_hypothesis in enumerate(hypotheses):
                    pose = np.linalg.inv(transform) @ camera_hypothesis
                    fit = pose_fit(value.cup_points_world, pose, self.model_samples)
                    if guarded:
                        fit["table_clearance"] = table_clearance(pose, self.model_samples, self.table_height_world)
                        fit["accepted"] = fit["accepted"] and fit["table_clearance"]["accepted"]
                    attempt["hypotheses"].append(dict(rank=rank, **fit))
                    attempt.update(fit)
                    if fit["accepted"]:
                        accepted = pose
                        attempt["selected_hypothesis_rank"] = rank
                        break
                if accepted is not None:
                    break
            except (RuntimeError, ValueError) as error:
                attempt.update(accepted=False, error=str(error))
        if accepted is None:
            raise RuntimeError(f"Visual CAD/depth agreement rejected: {attempts}")
        self.last_accepted_world_pose = accepted.copy()
        value.cup_world_pose = accepted
        value.observation_audit.update(uses_prepared_object_model=True, uses_foundationpose=True,
            foundationpose_attempts=attempts, foundationpose_depth_fit=attempts[-1],
            refine_request_changes_mask=bool(refine_cup))
        value.timings_s["foundationpose"] = time.perf_counter()-started
        value.timings_s["perception_total"] += value.timings_s["foundationpose"]
        return value
