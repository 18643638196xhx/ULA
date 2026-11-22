"""Opt-in appliance-first localization and fresh-depth door-plane tracking.

No CAD, actor pose, contacts, simulator pixel identities or command-as-evidence.
An occluded handle anchor is INFERRED from the measured plane angle and initial
visual hinge approximation; it is never reported as an observed handle.
"""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from adapters.microwave.geometry import (
    estimate_hinge, fit_panel, hinge_motion, opening_angle, panel_corners)
from adapters.microwave.perception import MicrowaveVision, MicrowaveWorker, initial_depth_frame


class ApplianceFirstVision(MicrowaveVision):
    def estimate(self, rgb, depth, intrinsic, world_to_camera, refined=False,
                 door_prompt_override=None, handle_prompt_override=None):
        start = time.perf_counter()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        prompt = "microwave oven ." if refined else "microwave ."
        found, scores, _ = self.models._run_dino(rgb, prompt)
        attempts = []
        order = np.argsort(scores.detach().cpu().numpy().reshape(-1))[::-1][:3]
        camera = dict(rgb=rgb, depth=depth, intrinsic_cv=intrinsic,
                      extrinsic_cv=world_to_camera)
        accepted = []
        for index in order:
            box = self.models._xyxy(found[int(index)].detach().cpu().numpy(),
                                   rgb.shape[1], rgb.shape[0])
            attempt = dict(box=np.asarray(box).tolist(), accepted=False)
            attempts.append(attempt)
            try:
                result = initial_depth_frame(camera, box)
                hint = result["camera_world"]-np.median(result["door_points"], axis=0)
                panel = fit_panel(result["door_points"], hint)
                hinge = estimate_hinge(panel, result["handle_world"])
                attempt.update(accepted=True, radius_m=hinge["radius_m"])
                accepted.append((result, box))
            except (RuntimeError, ValueError) as error:
                attempt["error"] = str(error)
        if accepted:
            if any(np.linalg.norm(a[0]["handle_world"]-accepted[0][0]["handle_world"]) > .02
                   for a in accepted[1:]):
                raise RuntimeError("Appliance boxes suggest competing handle identities")
            result, box = accepted[0]
            handle_audit = result["handle_audit"]
            return dict(door_points=result["door_points"], handle_world=result["handle_world"],
                        camera_world=result["camera_world"],
                        audit=dict(source="appliance_box_then_measured_depth_layers",
                                   boxes=dict(appliance=np.asarray(box).tolist()),
                                   appliance_prompt=prompt, appliance_attempts=attempts,
                                   handle_detection=handle_audit,
                                   handle_measured=True, uses_object_truth=False,
                                   uses_prepared_object_model=False, robot_identity_mask=False),
                        timings_s=dict(total_s=time.perf_counter()-start))
        # Keep the old bounded semantic-door path as a fallback, not a source
        # of guessed geometry. Every candidate still needs measured depth.
        try:
            result = super().estimate(rgb, depth, intrinsic, world_to_camera, refined,
                                      door_prompt_override, handle_prompt_override)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"Appliance-depth attempts={attempts}; semantic door fallback: {error}") from error
        result["audit"].update(appliance_attempts=attempts,
                                initial_localization_source="semantic_door_fallback",
                                handle_measured=True)
        return result


class PanelWorker(MicrowaveWorker):
    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(self.root/"scripts/perceive_microwave_panel.py")
        return command, cwd, environment

    def summary(self):
        return dict(super().summary(), appliance_first=True,
                    postgrasp_evidence="fresh_measured_door_plane_not_visible_handle")


def attach_reference(result):
    """Freeze initial measured identity/footprint; never grow it with occluders."""
    result["tracking_root"] = dict(
        panel=deepcopy(result["panel"]), hinge=estimate_hinge(result["panel"], result["handle_world"]),
        corners=panel_corners(result["panel"]), handle=np.asarray(result["handle_world"]).copy())
    result["vision_audit"].update(handle_measured=True,
                                  hidden_handle_anchor_inferred=False)
    return result


def track_panel(camera, reference, expected_motion=None, *, refined=False):
    """Fit CURRENT depth in old+predicted ROIs; infer anchor only after fitting.

    Command motion changes the search region, never the measured rotation.
    The fixed initial footprint and hinge are priors, not a measured hidden CAD.
    """
    from traditional_place_empty_cup.geometry import depth_to_metres, homogeneous_transform

    start = time.perf_counter()
    root = reference["tracking_root"]
    motion = np.eye(4) if expected_motion is None else np.asarray(expected_motion, dtype=float)
    if (motion.shape != (4, 4) or not np.isfinite(motion).all()
            or not np.allclose(motion[3], [0, 0, 0, 1])
            or not np.allclose(motion[:3, :3].T@motion[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(motion[:3, :3]), 1)):
        raise ValueError("Invalid bounded rigid tracking search")
    old = reference["panel"]
    depth = depth_to_metres(camera["depth"])
    intrinsic = np.asarray(camera["intrinsic_cv"])
    extrinsic = homogeneous_transform(camera["extrinsic_cv"])
    world = np.linalg.inv(extrinsic)
    corners = np.asarray(reference["tracking_corners"])
    search = np.vstack((corners, corners@motion[:3, :3].T+motion[:3, 3]))
    projected = search@extrinsic[:3, :3].T+extrinsic[:3, 3]
    if np.any(projected[:, 2] <= .01):
        raise RuntimeError("Door footprint crosses the current camera near plane")
    pixels = projected@intrinsic.T
    pixels = pixels[:, :2]/pixels[:, 2:]
    low, high = np.floor(pixels.min(0)-4), np.ceil(pixels.max(0)+4)
    yy, xx = np.indices(depth.shape)
    rays = np.linalg.solve(intrinsic, np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size))))
    grid = ((rays*depth.reshape(1, -1)).T@world[:3, :3].T+world[:3, 3]).reshape(*depth.shape, 3)
    valid = (np.isfinite(depth) & (depth > 0) & np.isfinite(grid).all(2)
             & (xx >= low[0]) & (xx <= high[0]) & (yy >= low[1]) & (yy <= high[1]))
    if valid.sum() < 80:
        raise RuntimeError("Insufficient current door depth in either camera search")
    old_normal, old_centre = np.asarray(old["normal"]), np.asarray(old["centre"])
    predicted_normal = motion[:3, :3]@old_normal
    predicted_centre = motion[:3, :3]@old_centre+motion[:3, 3]
    distance = np.minimum(np.abs((grid-old_centre)@old_normal),
                          np.abs((grid-predicted_centre)@predicted_normal))
    rejected = []
    for band in ((.004, .008, .012) if refined else (.006, .012)):
        selected = valid & (distance < band)
        try:
            panel = fit_panel(grid[selected], predicted_normal,
                              tolerance_m=.003, min_points=80, normal_cone_deg=12.)
            angle = opening_angle(root["hinge"], panel)
            if not -3. <= angle <= 85.:
                raise RuntimeError("Measured door angle outside the allowed opening sector")
            observed_motion = hinge_motion(root["hinge"]["hinge_world"],
                np.radians(angle)*root["hinge"]["opening_sign"])
            root_centre = np.asarray(root["panel"]["centre"])
            rigid_centre = observed_motion[:3, :3]@root_centre+observed_motion[:3, 3]
            offset = float(abs((panel["centre"]-rigid_centre)@panel["normal"]))
            if offset > .015:
                raise RuntimeError("Measured plane is incompatible with the initial visual hinge")
            updated = np.asarray(root["corners"])@observed_motion[:3, :3].T+observed_motion[:3, 3]
            lateral = grid@panel["lateral"]
            limits = updated@panel["lateral"]
            surface = (selected & (np.abs((grid-panel["centre"])@panel["normal"]) <= .004)
                       & (lateral >= limits.min()-.004) & (lateral <= limits.max()+.004)
                       & (grid[:, :, 2] >= updated[:, 2].min()-.004)
                       & (grid[:, :, 2] <= updated[:, 2].max()+.004))
            panel = fit_panel(grid[surface], panel["normal"], tolerance_m=.003,
                              min_points=80, normal_cone_deg=3.)
            # Use the final CURRENT fit for all angle-derived geometry.
            angle = opening_angle(root["hinge"], panel)
            observed_motion = hinge_motion(root["hinge"]["hinge_world"],
                np.radians(angle)*root["hinge"]["opening_sign"])
            rigid_centre = observed_motion[:3, :3]@root_centre+observed_motion[:3, 3]
            offset = float(abs((panel["centre"]-rigid_centre)@panel["normal"]))
            if not -3. <= angle <= 85. or offset > .015:
                raise RuntimeError("Final measured surface left the visual hinge envelope")
            handle = observed_motion[:3, :3]@root["handle"]+observed_motion[:3, 3]
            updated = np.asarray(root["corners"])@observed_motion[:3, :3].T+observed_motion[:3, 3]
            original_area = root["panel"]["width_m"]*root["panel"]["height_m"]
            fraction = panel["width_m"]*panel["height_m"]/original_area
            if not .20 <= fraction <= 1.4:
                raise RuntimeError("Measured door patch has insufficient fixed-footprint coverage")
            audit = dict(source="fixed_identity_fresh_depth_plane", uses_object_truth=False,
                         uses_prepared_object_model=False, robot_identity_mask=False,
                         handle_measured=False, hidden_handle_anchor_inferred=True,
                         handle_position_source="initial_visual_anchor_rotated_by_measured_plane_angle",
                         command_is_only_search_hint=True, current_depth_required=True,
                         search_includes_unmoved_surface=True, fixed_initial_footprint=True,
                         measured_points=int(surface.sum()), measured_angle_deg=angle,
                         plane_rmse_m=panel["plane_rmse_m"], rigid_plane_offset_m=offset,
                         visible_area_fraction=float(fraction), projected_roi_xyxy=[*low, *high],
                         search_band_m=band, rejected_bands=rejected)
            return dict(door_points=grid[surface], handle_world=handle,
                        camera_world=world[:3, 3], panel=panel, tracking_root=root,
                        tracking_corners=updated, vision_audit=audit,
                        timings_s=dict(panel_tracking_s=time.perf_counter()-start))
        except (RuntimeError, ValueError) as error:
            rejected.append(dict(band_m=band, error=str(error)))
    raise RuntimeError(f"No credible current door patch: {rejected}")


def observe_panel(task, worker, output, label, audit, *, normal_hint=None, refined=False,
                  timer=None, reference=None, expected_motion=None):
    from adapters.microwave.perception import observe

    if reference is None:
        return attach_reference(observe(task, worker, output, label, audit,
                                       normal_hint=normal_hint, refined=refined, timer=timer))
    for camera_name in ("head_camera", "left_camera"):
        attempt = dict(label=label, camera=camera_name, refined=refined,
                       evidence="fresh_door_patch_not_visible_handle")
        audit.append(attempt)
        try:
            with timer.span("rgbd_acquisition") if timer else nullcontext():
                camera = task.get_camera_observation(camera_name)
            with timer.span("panel_geometry") if timer else nullcontext():
                result = track_panel(camera, reference, expected_motion, refined=refined)
            result["camera"] = camera_name
            attempt.update(accepted=True, vision=result["vision_audit"], panel=result["panel"])
            directory = Path(output)/f"{label}_{camera_name}"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory/"rgbd.npz", rgb=camera["rgb"], depth=camera["depth"],
                                intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"])
            np.savez_compressed(directory/"observation.npz", door_points=result["door_points"],
                                handle_world=result["handle_world"],
                                audit_json=np.asarray(json.dumps(result["vision_audit"], default=lambda a:
                                    a.item() if isinstance(a, np.generic) else a.tolist())))
            return result
        except (RuntimeError, ValueError) as error:
            attempt.update(accepted=False, error=str(error))
    raise RuntimeError("Current door-plane tracking rejected by both head/wrist views")


def verify_panel_step(hinge, before, after, expected_motion, step_deg):
    """Explicit replacement evidence: measured plane, NOT an invented bar track."""
    audit = after.get("vision_audit", {})
    if audit.get("source") != "fixed_identity_fresh_depth_plane":
        raise RuntimeError("Panel-only opening requires fresh plane tracking evidence")
    angle0, angle1 = (opening_angle(hinge, frame["panel"]) for frame in (before, after))
    progress = angle1-angle0
    ready = (max(.4, step_deg*.20) <= progress <= step_deg*2+.5
             and audit["measured_points"] >= 80 and audit["plane_rmse_m"] <= .003
             and audit["rigid_plane_offset_m"] <= .015
             and .20 <= audit["visible_area_fraction"] <= 1.4)
    return dict(ready=bool(ready), opening_angle_deg=angle1, step_progress_deg=progress,
                expected_step_deg=step_deg, detector="fresh_door_plane_rotation_and_rigid_offset",
                handle_measured=False, uses_object_truth=False, physical_grasp_certificate=False,
                commanded_motion_is_not_measurement=True,
                plane_rmse_m=audit["plane_rmse_m"],
                rigid_plane_offset_m=audit["rigid_plane_offset_m"],
                measured_points=audit["measured_points"],
                visible_area_fraction=audit["visible_area_fraction"])


def open_panel_door(task, observer, hinge, baseline, result, **kwargs):
    from adapters.microwave import control

    result.update(observation_evidence="current_depth_door_plane",
                  independently_observed_handle_required=False,
                  hidden_handle_anchor_inferred=True)
    try:
        return control.open_door(task, observer, hinge, baseline, result, **kwargs)
    except RuntimeError as error:
        if str(error) == "No credible door rotation/handle co-motion after traditional pull":
            raise RuntimeError("No credible measured door-plane rotation after traditional pull") from error
        raise
