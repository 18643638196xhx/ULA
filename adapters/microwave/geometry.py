"""CPU-only geometry for a vertical, rectangular, hinged-door pilot.

The hinge is estimated at the observed panel edge opposite the handle. This is
an explicit approximation, NOT a recovered hidden joint or a general hinge
identification algorithm. No actor/model/contact API is accepted here.
"""

import numpy as np


def vector(value, name):
    out = np.asarray(value, dtype=float)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return out


def unit(value, name):
    out = vector(value, name)
    length = np.linalg.norm(out)
    if length < 1e-8:
        raise ValueError(f"{name} has no direction")
    return out/length


def fit_panel(points, normal_hint, *, tolerance_m=.008, min_points=60, normal_cone_deg=None):
    """Fit a measured vertical plane directly, rather than tilting then projecting.

    The explicit vertical-hinge prior reduces this to a robust XY line fit.
    A tracking cone rejects unrelated foreground planes but does not prescribe
    the measured angle. Support, 2-D surface spread and residuals remain required.
    """
    points = np.asarray(points, dtype=float)
    hint = unit(normal_hint, "visible normal hint")
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < min_points
            or not np.isfinite(points).all() or tolerance_m <= 0
            or normal_cone_deg is not None and (not np.isfinite(normal_cone_deg) or not 1 <= normal_cone_deg <= 45)):
        raise ValueError("Insufficient or invalid door surface points")
    sample = points[np.linspace(0, len(points)-1, min(1600, len(points))).astype(int)]
    rng = np.random.default_rng(0)
    best = np.zeros(len(sample), dtype=bool)
    for _ in range(150):
        p, q = sample[rng.choice(len(sample), 2, replace=False)]
        delta = q-p
        normal = np.array([-delta[1], delta[0], 0.])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if normal_cone_deg is not None and abs(normal@hint) < np.cos(np.radians(normal_cone_deg)):
            continue
        keep = np.abs((sample-p)@normal) <= tolerance_m
        if keep.sum() > best.sum():
            best = keep
    if best.sum() < min_points or best.mean() < .60:
        raise RuntimeError("No credible vertical panel plane")
    surface = sample[best]
    for _ in range(3):
        centre = surface.mean(axis=0)
        _, _, vt = np.linalg.svd(surface[:, :2]-centre[:2], full_matrices=False)
        normal = np.array([*vt[-1], 0.])
        if normal@hint < 0:
            normal = -normal
        if normal_cone_deg is not None and normal@hint < np.cos(np.radians(normal_cone_deg)):
            raise RuntimeError("Measured panel normal left the local tracking cone")
        keep = np.abs((points-centre)@normal) <= tolerance_m
        if keep.sum() < min_points or keep.mean() < .60:
            raise RuntimeError("Vertical panel plane lost measured support")
        surface = points[keep]
    centre = surface.mean(axis=0)
    lateral = np.cross([0., 0., 1.], normal)
    lo, hi = np.quantile((surface-centre)@lateral, [.02, .98])
    zlo, zhi = np.quantile(surface[:, 2], [.02, .98])
    width, height = float(hi-lo), float(zhi-zlo)
    if not .08 <= width <= .9 or not .08 <= height <= .9:
        raise RuntimeError("Panel extent is implausible for this tabletop pilot")
    coordinates = np.column_stack(((surface-centre)@lateral, surface[:, 2]-centre[2]))
    spread = np.linalg.svd(coordinates, compute_uv=False)
    if spread[-1]/np.sqrt(len(surface)) < .005:
        raise RuntimeError("Measured points are a line, not a supported panel surface")
    rmse = float(np.sqrt(np.mean(((surface-centre)@normal)**2)))
    return dict(centre=centre, normal=normal, lateral=lateral, lateral_limits=[lo, hi],
                z_limits=[zlo, zhi], width_m=width, height_m=height,
                inlier_fraction=float(keep.mean()), plane_rmse_m=rmse,
                uses_object_truth=False, vertical_axis_assumed=True,
                normal_search_cone_deg=normal_cone_deg)


def estimate_hinge(panel, handle):
    handle = vector(handle, "observed handle")
    centre, normal, lateral = (np.asarray(panel[k]) for k in ("centre", "normal", "lateral"))
    lo, hi = panel["lateral_limits"]
    location = float((handle-centre)@lateral)
    if (abs(location-(lo+hi)/2) < .20*(hi-lo)
            or not lo-.06 <= location <= hi+.06
            or abs((handle-centre)@normal) > .12
            or not panel["z_limits"][0]-.05 <= handle[2] <= panel["z_limits"][1]+.05):
        raise RuntimeError("Handle/panel geometry cannot identify an opposite hinge edge")
    hinge = centre+lateral*(lo if location > (lo+hi)/2 else hi)
    hinge[2] = handle[2]
    radial = handle-hinge
    radius = float(np.linalg.norm(radial[:2]))
    if not .07 <= radius <= .9:
        raise RuntimeError("Implausible observed hinge-to-handle distance")
    tangent = np.cross([0., 0., 1.], radial)
    outward = float(tangent@normal)
    if abs(outward) < .04:
        raise RuntimeError("Ambiguous opening direction")
    return dict(hinge_world=hinge, axis_world=np.array([0., 0., 1.]),
                opening_sign=1. if outward > 0 else -1., initial_normal=normal.copy(),
                initial_panel=panel, radius_m=radius, initial_handle=handle.copy(),
                source="observed_panel_edge_opposite_handle", uses_object_truth=False,
                hidden_hinge_measured=False, hinge_at_panel_edge_assumed=True)


def isolate_door_surface(points, panel, handle):
    """Exclude the static control strip beyond the free-edge handle.

    This uses the same visible handle/opposite-edge prior as the hinge pilot,
    not an appliance CAD or actor label. Refit the retained measured points.
    """
    estimate_hinge(panel, handle)
    points, handle = np.asarray(points), np.asarray(handle)
    lateral = np.asarray(panel["lateral"])
    lo, hi = panel["lateral_limits"]
    location = float((handle-panel["centre"])@lateral)
    right_handle = location > (lo+hi)/2
    signed = (points-handle)@lateral
    keep = signed <= .003 if right_handle else signed >= -.003
    retained = points[keep]
    if len(retained) < 60:
        raise RuntimeError("Insufficient measured door surface after control-strip exclusion")
    return retained, dict(source="measured_surface_on_hinge_side_of_handle",
        handle_near_free_edge_assumed=True, margin_m=.003,
        input_points=int(len(points)), retained_points=int(len(retained)), uses_object_truth=False)


def panel_corners(panel):
    """Rectangle of INITIAL measured support, not CAD or hidden door completion."""
    centre, lateral = np.asarray(panel["centre"]), np.asarray(panel["lateral"])
    return np.asarray([[*(centre+lateral*x)[:2], z]
                       for x in panel["lateral_limits"] for z in panel["z_limits"]])


def depth_panel_candidates(grid, roi, normal_hint):
    """Segment up to four measured vertical depth layers inside a detected box.

    A recessed window need not share the bezel plane. Each proposal must cover
    >=12% of the original ROI and pass the unchanged final plane/extent checks.
    The caller additionally requires a compatible measured protruding handle.
    """
    flat = np.flatnonzero(np.asarray(roi) & np.isfinite(grid).all(axis=-1))
    remaining = flat.copy()
    rng = np.random.default_rng(0)
    proposals = []
    for _ in range(4):
        if len(remaining) < max(60, .12*len(flat)):
            break
        points = grid.reshape(-1, 3)[remaining]
        sample = points[np.linspace(0, len(points)-1, min(1200, len(points))).astype(int)]
        best = np.zeros(len(sample), bool)
        origin = normal = None
        for _ in range(180):
            p, q = sample[rng.choice(len(sample), 2, replace=False)]
            delta = q-p
            candidate = np.array([-delta[1], delta[0], 0.])
            length = np.linalg.norm(candidate)
            if length < 1e-8:
                continue
            candidate /= length
            keep = np.abs((sample-p)@candidate) <= .004
            if keep.sum() > best.sum():
                best, origin, normal = keep, p, candidate
        if origin is None:
            break
        keep = np.abs((points-origin)@normal) <= .004
        selected = remaining[keep]
        remaining = remaining[~keep]
        if len(selected) < max(60, .12*len(flat)):
            continue
        try:
            panel = fit_panel(grid.reshape(-1, 3)[selected], normal_hint)
        except (RuntimeError, ValueError):
            continue
        mask = np.zeros(roi.shape, bool)
        mask.flat[selected] = True
        proposals.append((panel, mask, float(len(selected)/len(flat))))
    return proposals


def update_panel_corners(corners, old_panel, new_panel, old_handle, new_handle):
    """Rigidly transport a fixed footprint using measured normal and bar XY."""
    a, b = np.asarray(old_panel["normal"]), np.asarray(new_panel["normal"])
    angle = np.arctan2(np.cross(a, b)[2], a@b)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (np.asarray(corners)-old_handle)@rotation.T+new_handle


def protruding_handle(world_grid, roi, panel):
    """Find a measured vertical bar in front of the visible door plane.

    Only measured depth and an image ROI enter this function. It deliberately
    rejects short knobs, broad blobs and similarly plausible competing bars.
    This is a vertical-bar prior, not a general handle detector.
    """
    from scipy.ndimage import label

    grid, roi = np.asarray(world_grid, dtype=float), np.asarray(roi, dtype=bool)
    if grid.shape != (*roi.shape, 3) or roi.ndim != 2:
        raise ValueError("Invalid organized depth cloud/ROI")
    centre, normal, lateral = (np.asarray(panel[k]) for k in ("centre", "normal", "lateral"))
    residual = (grid-centre)@normal
    threshold = max(.006, 3*panel["plane_rmse_m"])
    foreground = roi & np.isfinite(grid).all(axis=-1) & (residual > threshold) & (residual < .08)
    components, count = label(foreground, structure=np.ones((3, 3)))
    candidates = []
    lo, hi = panel["lateral_limits"]
    for number in range(1, count+1):
        mask = components == number
        if mask.sum() < 12:
            continue
        points = grid[mask]
        width = float(np.ptp(points@lateral))
        height = float(np.ptp(points[:, 2]))
        handle = np.median(points, axis=0)
        location = float((handle-centre)@lateral)
        if (not .003 <= width <= .05 or not .045 <= height <= .40
                or height/max(width, .003) < 2.5
                or not lo-.02 <= location <= hi+.02):
            continue
        try:
            estimate_hinge(panel, handle)
        except RuntimeError:
            continue
        y, x = np.nonzero(mask)
        candidates.append(dict(mask=mask, handle=handle, height_m=height, width_m=width,
            protrusion_m=float(np.median(residual[mask])), pixels=int(mask.sum()),
            box=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
            score=height/max(width, .003)))
    candidates.sort(key=lambda c: c["score"], reverse=True)
    if not candidates:
        raise RuntimeError("No credible depth-protruding vertical handle")
    if len(candidates) > 1 and candidates[1]["score"] >= .8*candidates[0]["score"]:
        raise RuntimeError("Ambiguous competing depth-protruding handles")
    selected = candidates[0]
    audit = {k: v for k, v in selected.items() if k not in ("mask", "handle")}
    audit.update(source="depth_protrusion", threshold_m=threshold, candidates=len(candidates),
                 vertical_bar_prior=True, uses_object_truth=False)
    return selected["handle"], selected["mask"], audit


def partial_handle(world_grid, roi, panel, reference, motion, rgb=None):
    """Track fresh visible pieces of an ALREADY recognized vertical bar.

    XY comes from current depth. Z is a fixed reference-height anchor along
    the assumed vertical bar, not a claim that the hidden grasp point was seen.
    """
    from scipy.ndimage import label

    grid = np.asarray(world_grid)
    old = np.asarray(reference["handle_world"])
    predicted = motion[:3, :3]@old+motion[:3, 3]
    normal, lateral = np.asarray(panel["normal"]), np.asarray(panel["lateral"])
    protrusion = float((old-reference["panel"]["centre"])@reference["panel"]["normal"])
    if not .006 <= protrusion <= .06:
        raise RuntimeError("Reference handle has no credible measured protrusion")
    offset = (grid-panel["centre"])@normal
    side = np.minimum(np.abs((grid-old)@lateral), np.abs((grid-predicted)@lateral))
    mask = (roi & np.isfinite(grid).all(axis=-1) & (side < .006)
            & (np.abs(offset-protrusion) < .006) & (offset > .006)
            & (np.abs(grid[:, :, 2]-old[2]) < .08))
    labels, count = label(mask, np.ones((3, 3)))
    candidates, components = [], []
    for number in range(1, count+1):
        visible = labels == number
        points = grid[visible]
        if len(points) < 12:
            continue
        height, width = float(np.ptp(points[:, 2])), float(np.ptp(points@lateral))
        if len(components) < 5:
            components.append(dict(pixels=len(points), height_m=round(height, 5), width_m=round(width, 5),
                                   centroid_xy=np.median(points[:, :2], axis=0).tolist()))
            if rgb is not None:
                components[-1]["rgb_median"] = np.median(np.asarray(rgb)[visible], axis=0).tolist()
        if not .015 <= height <= .3 or not .003 <= width <= .018 or height/max(width, .003) < 2:
            continue
        measured = np.median(points, axis=0)
        if min(np.linalg.norm(measured[:2]-p[:2]) for p in (old, predicted)) > .02:
            continue
        anchor = measured.copy()
        anchor[2] = old[2]
        candidates.append((anchor, visible, measured, height, width))
    fragment_count = len(candidates)
    if 1 < fragment_count <= 10:
        # Occlusion/depth holes can split ONE vertical bar into several image
        # components. Merge only if their measured axes and appearance agree;
        # never resolve spatially competing handles by picking the largest.
        centres = np.asarray([c[2][:2] for c in candidates])
        axis_spread = np.linalg.norm(centres[:, None]-centres[None, :], axis=-1).max()
        colour_ok = True
        if rgb is not None:
            colours = np.asarray([np.median(np.asarray(rgb)[c[1]], axis=0) for c in candidates])
            colour_ok = bool(np.max(np.abs(colours[:, None]-colours[None, :])) <= 40)
            if not colour_ok:
                # The same neutral handle can have a lit and a shaded piece.
                # Compare chromaticity as well, but do not normalize near-black
                # gripper pixels into an apparently matching grey material.
                brightness = colours.mean(axis=1)
                chroma = colours/np.maximum(colours.sum(axis=1, keepdims=True), 1.)
                colour_ok = bool(brightness.min() >= 20 and brightness.max() <= 3*brightness.min()
                    and np.max(np.abs(chroma[:, None]-chroma[None, :])) <= .12)
        if axis_spread <= .008 and colour_ok:
            visible = np.logical_or.reduce([c[1] for c in candidates])
            measured = np.median(grid[visible], axis=0)
            anchor = measured.copy()
            anchor[2] = old[2]
            width = float(np.ptp(grid[visible]@lateral))
            height = float(np.ptp(grid[visible][:, 2]))
            if width <= .018:
                candidates = [(anchor, visible, measured, height, width)]
    if len(candidates) != 1:
        raise RuntimeError(f"Missing or ambiguous locally tracked handle fragment: "
                           f"accepted={len(candidates)}, measured_components={components}")
    anchor, visible, measured, height, width = candidates[0]
    return anchor, visible, dict(source="reference_vertical_bar_partial_depth",
        measured_visible_centroid=measured.tolist(), height_m=height, width_m=width,
        pixels=int(visible.sum()), protrusion_m=float(np.median(offset[visible])),
        reference_height_anchor=True, hidden_grasp_point_measured=False,
        lateral_track_half_width_m=.006,
        merged_visible_fragments=fragment_count, fragment_merge_max_xy_m=.008,
        fragment_appearance_checked=rgb is not None and fragment_count > 1,
        fragment_appearance_metric="rgb_or_chromaticity_with_bounded_brightness_ratio",
        vertical_bar_prior=True, uses_object_truth=False)


def hinge_motion(hinge, signed_angle_rad):
    hinge = vector(hinge, "hinge")
    if not np.isfinite(signed_angle_rad):
        raise ValueError("Invalid hinge angle")
    c, s = np.cos(signed_angle_rad), np.sin(signed_angle_rad)
    motion = np.eye(4)
    motion[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    motion[:3, 3] = hinge-motion[:3, :3]@hinge
    return motion


def opening_angle(hinge, panel):
    a = unit(hinge["initial_normal"], "reference normal")
    b = unit(panel["normal"], "current normal")
    return float(np.degrees(np.arctan2(np.cross(a, b)[2], a@b))*hinge["opening_sign"])


def approach_target(hinge, handle, ee_to_tcp, clearance_m=.08, pitch_deg=0.):
    if not np.isfinite(clearance_m) or not .005 <= clearance_m <= .20:
        raise ValueError("Approach clearance must be 0.5–20 cm")
    if not np.isfinite(pitch_deg) or not -30 <= pitch_deg <= 30:
        raise ValueError("Invalid bounded approach pitch")
    offset = np.asarray(ee_to_tcp, dtype=float)
    if offset.shape != (4, 4) or not np.isfinite(offset).all():
        raise ValueError("Invalid robot EE/TCP transform")
    normal = np.asarray(hinge["initial_normal"])
    pitch = np.radians(pitch_deg)
    forward = -normal*np.cos(pitch)-np.array([0., 0., 1.])*np.sin(pitch)
    closing = unit(np.cross([0., 0., 1.], forward), "horizontal closing direction")
    tcp = np.eye(4)
    tcp[:3, :3] = np.column_stack((forward, closing, np.cross(forward, closing)))
    tcp[:3, 3] = vector(handle, "handle")-forward*clearance_m
    return tcp@np.linalg.inv(offset)


def verify_step(hinge, before, after, expected_motion, step_deg):
    """Visual evidence only: the panel must rotate AND its handle must follow."""
    old, new = before["panel"], after["panel"]
    angle0, angle1 = opening_angle(hinge, old), opening_angle(hinge, new)
    progress = angle1-angle0
    handle0, handle1 = (vector(o["handle_world"], "handle") for o in (before, after))
    predicted = expected_motion[:3, :3]@handle0+expected_motion[:3, 3]
    measured_delta, expected_delta = handle1-handle0, predicted-handle0
    observed_distance, expected_distance = np.linalg.norm(measured_delta), np.linalg.norm(expected_delta)
    ratio = float(observed_distance/max(expected_distance, 1e-9))
    direction = float(measured_delta@expected_delta/max(observed_distance*expected_distance, 1e-9))
    error = float(np.linalg.norm(handle1-predicted))
    centre0, centre1 = (np.asarray(p["centre"]) for p in (old, new))
    predicted_centre = expected_motion[:3, :3]@centre0+expected_motion[:3, 3]
    centre_error = float(np.linalg.norm(centre1-predicted_centre))
    size_ok = (.5 <= new["width_m"]/old["width_m"] <= 1.6
               and .6 <= new["height_m"]/old["height_m"] <= 1.6)
    ready = (max(.4, step_deg*.20) <= progress <= step_deg*2+.5
             and observed_distance >= .002 and .25 <= ratio <= 1.8
             and direction >= .6 and error <= .025 and centre_error <= .06 and size_ok)
    return dict(ready=bool(ready), opening_angle_deg=angle1, step_progress_deg=progress,
                expected_step_deg=step_deg, observed_handle_displacement_m=observed_distance,
                handle_motion_ratio=ratio, handle_direction_cosine=direction,
                handle_prediction_error_m=error, panel_centre_prediction_error_m=centre_error,
                panel_size_consistent=bool(size_ok), uses_object_truth=False,
                detector="visual_panel_rotation_and_handle_comotion",
                physical_grasp_certificate=False)
