"""Microwave-specific image prompts and semantic protocol, cup code read-only."""

import json
from pathlib import Path
import time

import numpy as np

from adapters.perception import PerceptionWorker
from adapters.microwave.geometry import (estimate_hinge, fit_panel, protruding_handle,
    partial_handle, isolate_door_surface, panel_corners, update_panel_corners, depth_panel_candidates)


def validate_masks(door, handle):
    door, handle = np.asarray(door, dtype=bool), np.asarray(handle, dtype=bool)
    if door.ndim != 2 or door.shape != handle.shape or not door.any() or not handle.any():
        raise RuntimeError("Missing door or handle mask")
    area_ratio = float(handle.sum()/door.sum())
    iou = float((door & handle).sum()/(door | handle).sum())
    if iou > .45 or area_ratio > .45:
        raise RuntimeError(f"Door/handle segmentation ambiguous: mask IoU={iou:.3f}, handle/door area={area_ratio:.3f}")
    rows, cols = np.nonzero(door)
    lateral_clipped = bool(np.any(cols < 2) or np.any(cols >= door.shape[1]-2))
    vertical_clipped = bool(np.any(rows < 2) or np.any(rows >= door.shape[0]-2))
    if lateral_clipped:
        raise RuntimeError("Door mask laterally clipped at image border; cannot infer opposite hinge edge")
    # A vertical hinge needs the lateral edge, not the hidden panel top/bottom.
    # Keep vertical clipping explicit; fit_panel still requires enough height
    # and a well-supported plane. This does NOT complete the missing surface.
    return dict(mask_iou=iou, handle_door_area_ratio=area_ratio,
                door_lateral_clipped=False, door_vertical_clipped=vertical_clipped)


def select_handle_box(candidates, scores, door_box):
    """Reject whole-appliance detections before selecting a small-part prompt."""
    candidates, scores = np.asarray(candidates), np.asarray(scores)
    door_box = np.asarray(door_box)
    door_area = float(np.prod(np.maximum(door_box[2:]-door_box[:2], 0)))
    if candidates.ndim != 2 or candidates.shape[1:] != (4,) or scores.shape != (len(candidates),):
        raise ValueError("Invalid handle detection candidates")
    if not np.isfinite(candidates).all() or not np.isfinite(scores).all() or door_area <= 0:
        raise ValueError("Invalid finite part boxes")
    areas = np.prod(np.maximum(candidates[:, 2:]-candidates[:, :2], 0), axis=1)
    keep = np.flatnonzero((areas > 0) & (areas <= .45*door_area))
    if not len(keep):
        raise RuntimeError("Handle detector only returned whole-door/appliance-sized boxes")
    return candidates[keep[np.argmax(scores[keep])]]


class MicrowaveWorker(PerceptionWorker):
    def __init__(self, root, output, *, door_prompt="microwave door", handle_prompt="microwave handle",
                 vision_gpu=None, torch_fraction=.10):
        super().__init__(root, output, backend="pointcloud")
        self.door_prompt, self.handle_prompt = door_prompt, handle_prompt
        if not np.isfinite(torch_fraction) or not 0 < torch_fraction <= 1:
            raise ValueError("Invalid vision allocator fraction")
        self.vision_gpu, self.torch_fraction = vision_gpu, float(torch_fraction)

    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command = [command[0], str(self.root/"scripts/perceive_microwave_rgbd.py"),
                   "--door-prompt", self.door_prompt, "--handle-prompt", self.handle_prompt]
        if self.vision_gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(self.vision_gpu)
        environment["MICROWAVE_VISION_MEMORY_FRACTION"] = str(self.torch_fraction)
        return command, cwd, environment

    def estimate_frame(self, camera, output, label, *, refined=False,
                       door_prompt_override=None, handle_prompt_override=None):
        directory = Path(output).resolve()/label
        directory.mkdir(parents=True, exist_ok=True)
        request, result = directory/"rgbd.npz", directory/"observation.npz"
        overrides = {}
        if door_prompt_override is not None:
            overrides["door_prompt_override"] = np.asarray(door_prompt_override)
        if handle_prompt_override is not None:
            overrides["handle_prompt_override"] = np.asarray(handle_prompt_override)
        np.savez_compressed(request, rgb=camera["rgb"], depth=camera["depth"],
            intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"],
            refined=np.asarray(refined), **overrides)
        with self._lock:
            start = time.perf_counter()
            try:
                self._start()
                self._stats["requests"] += 1
                number = self._stats["requests"]
                self._send(dict(op="estimate", id=number, input=str(request), output=str(result)))
                reply = self._receive()
                if reply.get("id") != number or reply.get("status") not in ("ok", "error"):
                    raise RuntimeError("Invalid microwave vision protocol response")
            except BaseException:
                self.close()
                raise
            if reply["status"] == "error":
                raise RuntimeError(f"Microwave observation failed: {reply['error']}")
            try:
                with np.load(result, allow_pickle=False) as data:
                    observation = dict(door_points=data["door_points"].copy(),
                        handle_world=data["handle_world"].copy(), camera_world=data["camera_world"].copy(),
                        vision_audit=json.loads(str(data["audit_json"])),
                        timings_s=json.loads(str(data["timings_json"])))
                self._stats["completed_requests"] += 1
            except BaseException:
                self.close()
                raise
        observation["timings_s"]["worker_wall_s"] = time.perf_counter()-start
        return observation

    def summary(self):
        return dict(super().summary(), task="open_microwave", door_prompt=self.door_prompt,
                    handle_prompt=self.handle_prompt, vision_gpu=self.vision_gpu,
                    torch_allocator_fraction=self.torch_fraction, uses_prepared_object_model=False)


def initial_depth_frame(camera, door_box):
    """Recover a recessed front from several depth layers; no saved object model."""
    from traditional_place_empty_cup.geometry import depth_to_metres, homogeneous_transform

    depth = depth_to_metres(camera["depth"])
    intrinsic = np.asarray(camera["intrinsic_cv"])
    world = np.linalg.inv(homogeneous_transform(camera["extrinsic_cv"]))
    yy, xx = np.indices(depth.shape)
    rays = np.linalg.solve(intrinsic, np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size))))
    grid = ((rays*depth.reshape(1, -1)).T@world[:3, :3].T+world[:3, 3]).reshape(*depth.shape, 3)
    grid[~np.isfinite(depth) | (depth <= 0)] = np.nan
    box = np.asarray(door_box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Invalid detected door box")
    roi = ((xx >= box[0]) & (xx <= box[2]) & (yy >= box[1]) & (yy <= box[3])
           & np.isfinite(depth) & (depth > 0))
    if roi.sum() < 60:
        raise RuntimeError("No measured appliance ROI")
    hint = world[:3, 3]-np.median(grid[roi], axis=0)
    proposals = depth_panel_candidates(grid, roi, hint)
    accepted, rejected = [], []
    for panel, mask, support in proposals:
        try:
            handle, handle_mask, audit = protruding_handle(grid, roi, panel)
            cloud, _ = isolate_door_surface(grid[mask], panel, handle)
            fit_panel(cloud, hint)
            validate_masks(mask, handle_mask)
            accepted.append((panel, mask, handle, handle_mask, audit, support))
        except (RuntimeError, ValueError) as error:
            rejected.append(str(error))
    if not accepted:
        raise RuntimeError(f"No depth-layer panel/handle pair: candidates={len(proposals)}, rejected={rejected}")
    if any(np.linalg.norm(a[2]-accepted[0][2]) > .02 for a in accepted[1:]):
        raise RuntimeError("Different depth layers suggest incompatible handle positions")
    selected = min(accepted, key=lambda a: a[4]["protrusion_m"])
    panel, mask, handle, handle_mask, audit, support = selected
    audit.update(source="segmented_depth_planes", measured_plane_candidates=len(proposals),
                 compatible_pairs=len(accepted), selected_raw_roi_fraction=support,
                 original_protrusion_source="depth_protrusion")
    return dict(door_points=grid[mask], handle_world=handle, camera_world=world[:3, 3],
                door_mask=mask, handle_mask=handle_mask, handle_audit=audit)


def reference_depth_frame(camera, reference, expected_motion=None):
    """Old measured surface selects an ROI; ALL returned geometry is fresh depth.

    Search the union of unmoved and commanded projections: a command must not
    force the observation to report that the door actually moved.
    """
    from traditional_place_empty_cup.geometry import depth_to_metres, homogeneous_transform

    start = time.perf_counter()
    points = np.asarray(reference["door_points"], dtype=float)
    motion = np.eye(4) if expected_motion is None else np.asarray(expected_motion, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 60
            or not np.isfinite(points).all() or motion.shape != (4, 4)
            or not np.isfinite(motion).all()
            or not np.allclose(motion[3], [0, 0, 0, 1])
            or not np.allclose(motion[:3, :3].T@motion[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(motion[:3, :3]), 1)):
        raise ValueError("Invalid measured reference or search motion")
    depth = depth_to_metres(camera["depth"])
    intrinsic = np.asarray(camera["intrinsic_cv"])
    extrinsic = homogeneous_transform(camera["extrinsic_cv"])
    world = np.linalg.inv(extrinsic)
    corners = np.asarray(reference.get("tracking_corners", panel_corners(reference["panel"])), dtype=float)
    if corners.shape != (4, 3) or not np.isfinite(corners).all():
        raise ValueError("Invalid fixed measured door footprint")
    search = np.vstack((corners, corners@motion[:3, :3].T+motion[:3, 3]))
    projected = search@extrinsic[:3, :3].T+extrinsic[:3, 3]
    projected = projected[projected[:, 2] > .01]
    if len(projected) < 4:
        raise RuntimeError("Reference surface is outside current camera")
    pixels = projected@intrinsic.T
    pixels = pixels[:, :2]/pixels[:, 2:]
    low, high = np.floor(pixels.min(axis=0)-4), np.ceil(pixels.max(axis=0)+4)
    yy, xx = np.indices(depth.shape)
    roi = (xx >= low[0]) & (xx <= high[0]) & (yy >= low[1]) & (yy <= high[1])
    valid = roi & np.isfinite(depth) & (depth > 0)
    rays = np.linalg.solve(intrinsic, np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size))))
    grid = ((rays*depth.reshape(1, -1)).T@world[:3, :3].T+world[:3, 3]).reshape(*depth.shape, 3)
    grid[~np.isfinite(depth) | (depth <= 0)] = np.nan
    hint = motion[:3, :3]@np.asarray(reference["panel"]["normal"])
    # The gripper may occlude most ROI pixels. A depth band about either the
    # unmoved or commanded prior plane excludes far foreground/background;
    # the surviving CURRENT measurements must still support a fresh plane.
    old_panel = reference["panel"]
    old_centre = np.asarray(old_panel["centre"])
    predicted_centre = motion[:3, :3]@old_centre+motion[:3, 3]
    plane_distance = np.minimum(np.abs((grid-old_centre)@old_panel["normal"]),
                                np.abs((grid-predicted_centre)@hint))
    band_rejections = []
    for depth_band in (.025, .012):
        near_plane = valid & (plane_distance < depth_band)
        try:
            panel = fit_panel(grid[near_plane], hint, normal_cone_deg=12.)
            break
        except (RuntimeError, ValueError) as error:
            band_rejections.append(dict(depth_band_m=depth_band, error=str(error)))
    else:
        raise RuntimeError(f"No measured tracked panel in either depth band: {band_rejections}")
    # Remove measured foreground occluders, not robot identity labels.
    surface = valid & (np.abs((grid-panel["centre"])@panel["normal"]) <= .008)
    try:
        handle, handle_mask, handle_audit = protruding_handle(grid, valid, panel)
        handle_audit.update(reference_height_anchor=True, hidden_grasp_point_measured=False,
                            measured_visible_centroid=handle.tolist())
        handle[2] = reference["handle_world"][2]
    except RuntimeError as error:
        try:
            handle, handle_mask, handle_audit = partial_handle(grid, valid, panel, reference, motion,
                                                               camera.get("rgb"))
        except RuntimeError as partial_error:
            raise RuntimeError(f"{error}; partial tracking: {partial_error}") from partial_error
        handle_audit["full_bar_rejection"] = str(error)
    old_handle = np.asarray(reference["handle_world"])
    predicted_handle = motion[:3, :3]@old_handle+motion[:3, 3]
    if min(np.linalg.norm(handle-old_handle), np.linalg.norm(handle-predicted_handle)) > .04:
        raise RuntimeError("Observed handle jumped outside local reference search")
    updated_corners = update_panel_corners(corners, reference["panel"], panel, old_handle, handle)
    lateral_coordinates = grid@panel["lateral"]
    bounds = updated_corners@panel["lateral"]
    surface &= ((lateral_coordinates >= bounds.min()-.008) & (lateral_coordinates <= bounds.max()+.008)
                & (grid[:, :, 2] >= updated_corners[:, 2].min()-.008)
                & (grid[:, :, 2] <= updated_corners[:, 2].max()+.008))
    quality = validate_masks(surface, handle_mask)
    return dict(door_points=grid[surface], handle_world=handle, camera_world=world[:3, 3],
        tracking_corners=updated_corners,
        vision_audit=dict(source="reference_roi_fresh_depth", uses_object_truth=False,
            uses_prepared_object_model=False, robot_identity_mask=False,
            reference_is_search_region_only=True, current_depth_required=True,
            search_includes_unmoved_surface=True, projected_roi_xyxy=[*low.tolist(), *high.tolist()],
            bounded_rigid_footprint=True,
            filtered_plane_support=panel["inlier_fraction"], mask_quality=quality,
            raw_roi_near_plane_fraction=float(near_plane.sum()/max(valid.sum(), 1)),
            depth_band_m=depth_band, depth_band_rejections=band_rejections,
            measured_near_plane_pixels=int(near_plane.sum()), normal_search_cone_deg=12.,
            handle_detection=handle_audit, handle_depth_points=int(handle_mask.sum())),
        timings_s=dict(reference_depth_s=time.perf_counter()-start))


def observe(task, worker, output, label, audit, *, normal_hint=None, refined=False, timer=None,
            reference=None, expected_motion=None):
    """At most two fixed cameras; no actor/contact/joint read, no view motion."""
    from contextlib import nullcontext

    def span(name):
        return timer.span(name) if timer is not None else nullcontext()
    candidates = ["head_camera", "left_camera"]
    for camera_name in candidates:
        attempt = dict(label=label, camera=camera_name, refined=refined)
        audit.append(attempt)
        try:
            with span("rgbd_acquisition"):
                camera = task.get_camera_observation(camera_name)
            with span("detection_segmentation"):
                result = None
                if reference is not None:
                    try:
                        result = reference_depth_frame(camera, reference, expected_motion)
                        directory = Path(output)/f"{label}_{camera_name}"
                        directory.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(directory/"rgbd.npz", rgb=camera["rgb"], depth=camera["depth"],
                            intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"])
                        np.savez_compressed(directory/"observation.npz", door_points=result["door_points"],
                            handle_world=result["handle_world"], camera_world=result["camera_world"],
                            tracking_corners=result["tracking_corners"],
                            audit_json=np.asarray(json.dumps(result["vision_audit"])),
                            timings_json=np.asarray(json.dumps(result["timings_s"])))
                    except (RuntimeError, ValueError) as error:
                        attempt["reference_depth_rejection"] = str(error)
                if result is None:
                    result = worker.estimate_frame(camera, output, f"{label}_{camera_name}", refined=refined)
            cloud = result["door_points"]
            hint = (np.asarray(normal_hint) if normal_hint is not None else
                    result["camera_world"]-np.median(cloud, axis=0))
            with span("panel_geometry"):
                result["panel"] = fit_panel(cloud, hint)
                if reference is None:
                    cloud, selection = isolate_door_surface(cloud, result["panel"], result["handle_world"])
                    result["door_points"] = cloud
                    result["panel"] = fit_panel(cloud, hint)
                    result["vision_audit"]["door_surface_selection"] = selection
                estimate_hinge(result["panel"], result["handle_world"])
                result["tracking_corners"] = (panel_corners(result["panel"]) if reference is None else
                    update_panel_corners(reference.get("tracking_corners", panel_corners(reference["panel"])),
                        reference["panel"], result["panel"], reference["handle_world"], result["handle_world"]))
                result["vision_audit"]["tracking_corners_world"] = result["tracking_corners"].tolist()
            result["camera"] = camera_name
            attempt.update(accepted=True, vision=result["vision_audit"], panel=result["panel"])
            return result
        except (RuntimeError, ValueError) as error:
            attempt.update(accepted=False, error=str(error))
    raise RuntimeError("Microwave observation rejected by both head/wrist views")


class MicrowaveVision:
    """Uses existing DINO/SAM loading only; no FoundationPose/CAD initialization."""
    def __init__(self, root, door_prompt, handle_prompt):
        import os
        import torch
        from adapters.rgbd_perception import GroundedPointCloudPerception

        fraction = float(os.environ.get("MICROWAVE_VISION_MEMORY_FRACTION", ".10"))
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Invalid vision allocator fraction")
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        self.models = GroundedPointCloudPerception(root)
        self.load_timings_s = self.models.load_timings_s
        self.prompts = {"door": door_prompt, "handle": handle_prompt}

    def estimate(self, rgb, depth, intrinsic, world_to_camera, refined=False,
                 door_prompt_override=None, handle_prompt_override=None):
        from traditional_place_empty_cup.geometry import backproject_mask_to_world, depth_to_metres, homogeneous_transform

        start = time.perf_counter()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        masks, boxes = {}, {}
        m = self.models
        prompts = dict(self.prompts)
        if door_prompt_override is not None:
            prompts["door"] = str(door_prompt_override)
        if handle_prompt_override is not None:
            prompts["handle"] = str(handle_prompt_override)
        depth_audit = None
        for name, prompt in prompts.items():
            if name == "handle":
                # Door localization is semantic; the thin handle itself need not
                # be recognized by DINO. Search measured depth inside its ROI.
                metric = depth_to_metres(depth)
                cloud = backproject_mask_to_world(masks["door"], metric, intrinsic, world_to_camera)
                camera = np.linalg.inv(homogeneous_transform(world_to_camera))[:3, 3]
                try:
                    panel = fit_panel(cloud, camera-np.median(cloud, axis=0))
                    yy, xx = np.indices(metric.shape)
                    rays = np.linalg.solve(intrinsic, np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size))))
                    transform = np.linalg.inv(homogeneous_transform(world_to_camera))
                    points = (rays*metric.reshape(1, -1)).T@transform[:3, :3].T+transform[:3, 3]
                    grid = points.reshape(*metric.shape, 3)
                    grid[~np.isfinite(metric) | (metric <= 0)] = np.nan
                    box = boxes["door"]
                    roi = (xx >= box[0]) & (xx <= box[2]) & (yy >= box[1]) & (yy <= box[3])
                    handle, mask, depth_audit = protruding_handle(grid, roi, panel)
                    validate_masks(masks["door"], mask)
                    # A whole-appliance plane can pass while its hinge-side
                    # subset is still a mixture of bezel and recessed window.
                    # Validate the SAME subset the controller will use before
                    # bypassing the segmented-depth fallback.
                    surface, _ = isolate_door_surface(cloud, panel, handle)
                    fit_panel(surface, panel["normal"])
                    masks[name], boxes[name] = mask, depth_audit["box"]
                    continue
                except (RuntimeError, ValueError) as error:
                    depth_audit = dict(source="text_detection_fallback", depth_rejection=str(error))
                    try:
                        layers = initial_depth_frame(dict(depth=depth, intrinsic_cv=intrinsic,
                            extrinsic_cv=world_to_camera), boxes["door"])
                        masks["door"], masks["handle"] = layers["door_mask"], layers["handle_mask"]
                        depth_audit = layers["handle_audit"]
                        depth_audit["single_plane_rejection"] = str(error)
                        boxes["handle"] = depth_audit["box"]
                        continue
                    except (RuntimeError, ValueError) as layer_error:
                        depth_audit["depth_layer_rejection"] = str(layer_error)
            view, origin = rgb, np.array([0, 0, 0, 0])
            if refined and name == "handle":
                door_box = np.asarray(boxes["door"])
                pad = max(4, int(max(door_box[2:]-door_box[:2])*.15))
                x0, y0 = np.maximum(np.floor(door_box[:2]).astype(int)-pad, 0)
                x1, y1 = np.minimum(np.ceil(door_box[2:]).astype(int)+pad+1, [rgb.shape[1], rgb.shape[0]])
                view = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                origin = np.array([x0, y0, x0, y0])
            found, logits, _ = m._run_dino(view, prompt+" .")
            if len(found) == 0:
                raise RuntimeError(f"No detected microwave {name}")
            candidates = np.asarray([m._xyxy(value, view.shape[1], view.shape[0])+origin
                                     for value in found.detach().cpu().numpy()])
            scores = logits.detach().cpu().numpy().reshape(-1)
            box = (select_handle_box(candidates, scores, boxes["door"]) if name == "handle" else
                   candidates[int(np.argmax(scores))])
            boxes[name] = box.tolist()
            if refined:
                # Generic ROI + positive centre point; no cup/table heuristics.
                mask = m._refine_cup_mask(rgb, box)
            else:
                with m.torch.inference_mode(), m.torch.autocast("cuda", dtype=m.torch.bfloat16):
                    m.sam.set_image(rgb)
                    value, _, _ = m.sam.predict(box=np.asarray([box]), multimask_output=False)
                mask = np.asarray(value).reshape(-1, *rgb.shape[:2])[0].astype(bool)
            masks[name] = mask
        m.torch.cuda.synchronize()
        inference_s = time.perf_counter()-start
        mask_quality = validate_masks(masks["door"], masks["handle"])
        metric_depth = depth_to_metres(depth)
        clouds = {name: backproject_mask_to_world(mask, metric_depth, intrinsic, world_to_camera)
                  for name, mask in masks.items()}
        if len(clouds["door"]) < 60 or len(clouds["handle"]) < 12:
            raise RuntimeError("Too few measured door or handle depth pixels")
        handle = np.median(clouds["handle"], axis=0)
        handle_extent = np.quantile(clouds["handle"], .95, axis=0)-np.quantile(clouds["handle"], .05, axis=0)
        if np.linalg.norm(handle_extent) > .5:
            raise RuntimeError("Handle mask is too large for a local grasp target")
        camera = np.linalg.inv(homogeneous_transform(world_to_camera))[:3, 3]
        return dict(door_points=clouds["door"], handle_world=handle, camera_world=camera,
                    audit=dict(boxes=boxes, mask_quality=mask_quality, prompts=prompts,
                        handle_detection=depth_audit,
                        handle_detection_on_door_roi=bool(refined or
                            depth_audit and depth_audit["source"] in ("depth_protrusion", "segmented_depth_planes")),
                        handle_extent_m=handle_extent.tolist(),
                        door_depth_points=len(clouds["door"]), handle_depth_points=len(clouds["handle"]),
                        refined=bool(refined), uses_object_truth=False, robot_identity_mask=False,
                        uses_prepared_object_model=False, uses_foundationpose=False),
                    timings_s=dict(inference_s=inference_s, total_s=time.perf_counter()-start))
