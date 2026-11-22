"""Wrist-camera grasp contact check.

The trial lift relied on tracking the object's surface motion after a 5 cm lift.
For small objects (3 cm bottle) the gripper occludes most of the surface and the
ICP routinely latched onto background, producing 27-30 cm spurious displacements.
Meanwhile a fully-closed gripper on empty air is accepted as a grasp because the
closed-command monitor cannot see position.

This check looks at the wrist RGB-D *between the fingers* right after closing:
a real grasp leaves a visible surface patch whose extent matches the object's CAD
size, and the fingers retain a non-zero opening instead of closing completely.

No simulator truth is used: only the wrist camera's own depth image, the arm's
measured TCP pose, and the object's CAD extents from its metadata.
"""
from __future__ import annotations

import numpy as np

# Fraction of the object's narrowest extent that the fingers must retain when
# holding it.  A 3 cm bottle forces ~2.4 cm; a fully closed empty gripper is ~0.
MIN_RETAINED_GAP_RATIO = 0.55
# Depth samples required between the fingers for a credible surface patch.
MIN_SURFACE_POINTS = 25
# Fraction of the CAD narrowest extent the patch extent must reach.
MIN_PATCH_RATIO = 0.35


def _cad_extents(metadata):
    """Object extents in metres from its CAD metadata."""
    if not metadata:
        return None
    extents = metadata.get("extents")
    scale = metadata.get("scale")
    if extents is None or scale is None:
        return None
    values = np.abs(np.asarray(extents, dtype=float) * np.asarray(scale, dtype=float))
    values = values[np.isfinite(values) & (values > 1e-6)]
    return values if values.size else None


def wrist_contact_verified(task, arm, metadata=None, *, extents=None, camera_name=None,
                           min_points=MIN_SURFACE_POINTS,
                           min_patch_ratio=MIN_PATCH_RATIO,
                           min_gap_ratio=MIN_RETAINED_GAP_RATIO):
    """Return an audit dict; ``accepted`` is True when a grasp is plausible.

    Checks, in order:
      1. the gripper retains a non-trivial opening (not a fully closed empty jaw);
      2. the wrist depth image contains enough valid samples between the fingers;
      3. those samples span an extent consistent with the object's CAD size.
    """
    audit = dict(detector="wrist_rgbd_cad_patch", uses_simulator_truth=False,
                 camera=camera_name or f"{arm}_camera", accepted=False, checks={})

    # 1) retained gap -------------------------------------------------------
    try:
        aperture = float(getattr(task.robot, f"get_{arm}_gripper_val")())
    except Exception as error:
        audit["error"] = f"gripper read failed: {error}"
        return audit
    audit["gripper_aperture"] = aperture

    values = extents if extents is not None else _cad_extents(metadata)
    if values is None:
        audit["error"] = "no CAD extents provided"
        return audit
    extents = np.abs(np.asarray(values, dtype=float))
    extents = extents[np.isfinite(extents) & (extents > 1e-6)]
    if extents.size == 0:
        audit["error"] = "invalid CAD extents"
        return audit
    narrowest = float(np.min(extents))
    audit["cad_extents_m"] = np.round(extents, 4).tolist()

    # The gripper value is normalised [0, 1]: 1 = fully open.  A held object of
    # width w leaves roughly w / max_opening, so scale by the narrowest extent.
    required_gap = narrowest * min_gap_ratio
    audit["checks"]["retained_gap"] = dict(
        aperture=aperture, required=required_gap,
        passed=bool(aperture * 0.08 >= required_gap))
    if not audit["checks"]["retained_gap"]["passed"]:
        audit["reason"] = "gripper closed past the object width (empty jaw)"
        return audit

    # 2) depth samples between the fingers ---------------------------------
    name = camera_name or f"{arm}_camera"
    try:
        frame = task.get_camera_observation(name)
        depth = np.asarray(frame["depth"], dtype=float)
        intrinsic = np.asarray(frame["intrinsic_cv"], dtype=float)
    except Exception as error:
        audit["error"] = f"wrist frame failed: {error}"
        return audit

    if depth.ndim != 2:
        audit["error"] = "unexpected depth shape"
        return audit
    # Depth is published in millimetres by RoboTwin; convert to metres.
    metres = depth / 1000.0 if np.nanmax(depth) > 20.0 else depth.copy()
    valid = np.isfinite(metres) & (metres > 0.02) & (metres < 0.60)

    # Central band of the wrist image: that is where the fingers close.
    h, w = metres.shape
    band = np.zeros_like(valid)
    band[int(h * 0.30):int(h * 0.85), int(w * 0.30):int(w * 0.85)] = True
    mask = valid & band
    count = int(mask.sum())
    audit["checks"]["surface_points"] = dict(count=count, required=min_points,
                                             passed=bool(count >= min_points))
    if count < min_points:
        audit["reason"] = "no object surface visible between the fingers"
        return audit

    # 3) patch extent vs CAD ----------------------------------------------
    ys, xs = np.nonzero(mask)
    zs = metres[ys, xs]
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    xs_m = (xs - cx) / fx * zs
    ys_m = (ys - cy) / fy * zs
    span_x = float(np.percentile(xs_m, 98) - np.percentile(xs_m, 2))
    span_y = float(np.percentile(ys_m, 98) - np.percentile(ys_m, 2))
    patch = float(np.hypot(span_x, span_y))
    required_patch = float(np.max(extents)) * min_patch_ratio
    audit["checks"]["patch_extent"] = dict(
        span_x_m=round(span_x, 4), span_y_m=round(span_y, 4),
        patch_m=round(patch, 4), required=round(required_patch, 4),
        passed=bool(patch >= required_patch))
    if patch < required_patch:
        audit["reason"] = "surface patch smaller than the object's CAD footprint"
        return audit

    audit["accepted"] = True
    return audit
