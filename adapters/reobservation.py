"""Bounded pointcloud reobservation, without CAD, robot masks or object truth.

Changed SAM prompts/ROI are used for retries; repeat identical inference is not
treated as new evidence. Wrist ICP always uses a saved pre-VLA wrist reference.
No new observation motion is initiated from an untrusted object estimate.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adapters import pointcloud_registration


class ReobservationRejected(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.reobservation_audit = audit


def observation_quality(estimate, reference_bounds=None):
    reasons, warnings = [], []
    bounds = np.asarray(estimate.cup_bounds_world, dtype=float)
    points = np.asarray(estimate.cup_points_world, dtype=float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or points.ndim != 2 or points.shape[1:] != (3,)
            or len(points) < 40 or not np.isfinite(points).all()):
        return dict(geometry_ok=False, reasons=["invalid_bounds_or_surface"], warnings=[])
    spans = bounds[1]-bounds[0]
    if np.any(spans < 0) or np.min(spans[:2]) <= 0:
        return dict(geometry_ok=False, reasons=["invalid_spans"], warnings=[])
    # Vertical aspect is NOT a validity criterion: bowls, containers and shoes
    # legitimately exceed a cup-like ratio (a container measured 2.296 and was
    # wrongly rejected).  Keep it as a diagnostic warning only.
    aspect = float(spans[2]/max(min(spans[:2]), .001))
    if aspect > 2.2:
        warnings.append("elongated_vertical_extent")
    growth = None
    if reference_bounds is not None:
        ref = np.asarray(reference_bounds, dtype=float)
        growth = float(np.linalg.norm(spans)/max(np.linalg.norm(ref[1]-ref[0]), .001))
        if growth > 1.6:
            reasons.append("extent_growth_from_episode_reference")
    # Positive CAD check: when the frame carries a verified CAD pose, compare the
    # observed spans with the CAD-predicted extents instead of using shape priors.
    cad_spans = None
    audit = estimate.observation_audit or {}
    predicted = audit.get("cad_predicted_spans_m")
    if predicted is not None:
        cad_spans = np.asarray(predicted, dtype=float)
        if cad_spans.shape == (3,) and np.isfinite(cad_spans).all():
            ratio = spans / np.maximum(cad_spans, 1e-3)
            if np.max(np.abs(np.log(np.maximum(ratio, 1e-3)))) > np.log(1.6):
                reasons.append("observed_spans_differ_from_CAD_prediction")
    if (estimate.observation_audit or {}).get("cup_mask_touches_border", False):
        warnings.append("cup_mask_touches_border")
    return dict(geometry_ok=not reasons, reasons=reasons, warnings=warnings,
                spans_m=spans.tolist(), vertical_aspect=aspect, extent_growth_ratio=growth,
                cad_predicted_spans_m=(None if cad_spans is None else cad_spans.tolist()))


def _camera(task, name):
    return task.get_camera_observation(name)


def _copy(estimate, **audit):
    combined = SimpleNamespace(**vars(estimate))
    combined.observation_audit = dict(estimate.observation_audit or {}, **audit)
    return combined


def observe_before_approach(task, perception, arm, audit):
    """At most initial head + refined head + refined wrist, before any move."""
    audit.update(stage="before_approach", attempts=[], resolved=False,
                 robot_region_filtering=False, observation_motion=False)
    head = None
    for index, (camera_name, refined) in enumerate([
            ("head_camera", False), ("head_camera", True),
            (f"{arm}_camera", True)]):
        if arm == "auto" and index == 2:
            break  # Never choose an arm from an invalid estimate.
        attempt = dict(camera=camera_name, refined=refined)
        audit["attempts"].append(attempt)
        try:
            camera = _camera(task, camera_name)
            if index == 0:
                estimate = perception.estimate(camera["rgb"], camera["depth"],
                                               camera["intrinsic_cv"], camera["extrinsic_cv"])
            else:
                estimate = perception.supplementary(camera, f"reobserve_pre_{index}",
                    cup_only=camera_name != "head_camera", refine_cup=True)
            if camera_name == "head_camera":
                head = estimate
            quality = observation_quality(estimate)
            attempt["quality"] = quality
            if not quality["geometry_ok"]:
                continue
            # A border-only head view first gets a changed segmentation prompt.
            if index == 0 and quality["warnings"]:
                continue
            if camera_name != "head_camera":
                if head is None or not np.isfinite(head.coaster_world_center).all():
                    attempt["rejected"] = "no_valid_head_coaster_target"
                    continue
                if quality["warnings"]:
                    attempt["rejected"] = "wrist_view_still_clipped"
                    continue
                table_difference = abs(estimate.observation_audit["table_height_m"]
                                       -head.observation_audit["table_height_m"])
                if table_difference > .02:
                    attempt["rejected"] = "inconsistent_observed_table_height"
                    continue
                estimate = _copy(estimate, coaster_camera="head_camera")
                estimate.coaster_world_center = head.coaster_world_center.copy()
            # A refined but clipped head can serve ONLY coarse approach. The
            # existing wrist gate must still resolve clipping before VLA starts.
            audit.update(resolved=True, selected_camera=camera_name,
                         selected_attempt=index, requires_postapproach_wrist=bool(quality["warnings"]))
            return _copy(estimate, registration_reference_camera=camera_name,
                         reobservation_before_approach=audit)
        except (RuntimeError, ValueError) as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
    raise ReobservationRejected("Pre-approach reobservation exhausted: no credible cup geometry", audit)


def _saved_wrist(path, arm):
    with np.load(path, allow_pickle=False) as data:
        if str(data["camera_name"]) != f"{arm}_camera":
            raise RuntimeError("Saved wrist reference belongs to another camera")
        return dict(rgb=data["rgb"].copy(), depth=data["depth"].copy(),
                    intrinsic_cv=data["intrinsic"].copy(), extrinsic_cv=data["world_to_camera"].copy())


def observe_for_placement(task, perception, arm, reference_path, config, audit):
    """Validate geometry and ICP before planning; never mix camera references."""
    path = Path(reference_path)
    with np.load(path, allow_pickle=False) as data:
        primary_points = data["cup_points_world"].copy()
        primary_bounds = data["cup_bounds_world"].copy()
        reference_audit = json.loads(str(data["observation_audit_json"]))
    primary_camera = reference_audit.get("registration_reference_camera", "head_camera")
    if primary_camera not in ("head_camera", f"{arm}_camera"):
        raise RuntimeError("Unknown primary reference camera")
    audit.update(stage="before_placement", attempts=[], resolved=False,
                 robot_region_filtering=False, observation_motion=False,
                 primary_reference_camera=primary_camera)
    # Head supplies coaster depth even if wrist supplies the cup surface.
    head = None
    try:
        camera = _camera(task, "head_camera")
        head = perception.estimate(camera["rgb"], camera["depth"],
                                   camera["intrinsic_cv"], camera["extrinsic_cv"])
    except (RuntimeError, ValueError) as error:
        audit["initial_head_error"] = str(error)
    candidates = [(primary_camera, False), (primary_camera, True)]
    if primary_camera == "head_camera":
        candidates.append((f"{arm}_camera", True))
    for index, (name, refined) in enumerate(candidates):
        attempt = dict(camera=name, refined=refined)
        audit["attempts"].append(attempt)
        try:
            reference_points = primary_points
            if index == 2:
                saved = path.parent / "wrist_reference_frame.npz"
                if not saved.exists():
                    attempt["rejected"] = "no_matching_pregrasp_wrist_reference"
                    continue
                ref = perception.supplementary(_saved_wrist(saved, arm),
                    "reobserve_saved_wrist_reference", refine_cup=True)
                ref_quality = observation_quality(ref)
                attempt["reference_quality"] = ref_quality
                if not ref_quality["geometry_ok"]:
                    continue
                reference_points = ref.cup_points_world
            if name == "head_camera" and not refined:
                if head is None:
                    continue
                estimate = head
            else:
                estimate = perception.supplementary(_camera(task, name),
                    f"reobserve_place_{index}", cup_only=name != "head_camera", refine_cup=refined)
                if name == "head_camera":
                    head = estimate
            quality = observation_quality(estimate, primary_bounds)
            attempt["quality"] = quality
            if not quality["geometry_ok"]:
                continue
            # Border triggers one changed-prompt head retry. On a partial retry,
            # the unchanged bidirectional ICP gates provide extra evidence;
            # clipping is still recorded and never claimed as full visibility.
            if index == 0 and quality["warnings"]:
                continue
            motion, registration = pointcloud_registration.estimate_pointcloud_motion(
                reference_points, estimate.cup_points_world, config)
            attempt["registration"] = registration
            if motion is None:
                continue
            if head is None or not np.isfinite(head.coaster_world_center).all():
                attempt["rejected"] = "no_fresh_head_coaster_target"
                continue
            estimate = _copy(estimate, selected_cup_camera=name, coaster_camera="head_camera")
            estimate.coaster_world_center = head.coaster_world_center.copy()
            audit.update(resolved=True, selected_attempt=index, selected_camera=name,
                         matching_reference_camera=name)
            return estimate, motion, registration
        except (RuntimeError, ValueError) as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
    raise ReobservationRejected("Placement reobservation exhausted: no credible registered cup observation", audit)
