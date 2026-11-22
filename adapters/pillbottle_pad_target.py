"""Visual recovery for this task's blue horizontal square pad, never actor state.

No pad CAD or dimensions are read. Side lengths are estimated from observed
blue depth points. Incomplete, non-square extents are rejected, not completed.
"""
import numpy as np


def pad_height_check(center, table_z, tolerance=.01):
    center = np.asarray(center, dtype=float)
    valid = (center.shape == (3,) and np.isfinite(center).all()
             and np.isfinite(table_z))
    error = float(center[2]-table_z) if valid else None
    return dict(accepted=bool(valid and abs(error) <= tolerance),
        reason="near_visual_table" if valid and abs(error) <= tolerance else "pad_not_on_visual_table",
        estimated_pad_world=center.tolist(), visual_table_height_m=float(table_z),
        height_error_m=error, maximum_height_error_m=tolerance,
        uses_object_truth=False)


def square_bounds(points):
    """Minimum-area rectangle of measured XY points; no hidden-edge completion."""
    from scipy.spatial import ConvexHull
    xy = np.asarray(points, dtype=float)[:, :2]
    hull = xy[ConvexHull(xy).vertices]
    edges = np.roll(hull, -1, axis=0)-hull
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi/2))
    best = None
    for angle in angles:
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local = xy@rotation
        lower, upper = np.min(local, axis=0), np.max(local, axis=0)
        spans = upper-lower
        area = float(np.prod(spans))
        if best is None or area < best[0]:
            best = area, ((lower+upper)/2)@rotation.T, spans, angle
    if best is None:
        raise RuntimeError("No measured pad boundary")
    _, center, spans, angle = best
    if np.min(spans) < .025 or np.max(spans) > .15 or np.min(spans)/np.max(spans) < .75:
        raise RuntimeError(f"Incomplete or implausible blue square extent: {spans.tolist()}")
    return center, dict(measured_side_lengths_m=spans.tolist(), heading_rad=float(angle),
        minimum_side_m=.025, maximum_side_m=.15, minimum_side_ratio=.75)


def recover_blue_pad(cameras, table_z):
    from scipy.ndimage import label
    from traditional_place_empty_cup.geometry import backproject_mask_to_world, depth_to_metres
    if not np.isfinite(table_z):
        raise ValueError("No visual table height for pad recovery")
    clouds, views = [], []
    for name, frame in cameras.items():
        item = dict(camera=name, accepted=False)
        views.append(item)
        try:
            rgb = np.asarray(frame["rgb"])[..., :3].astype(float)/255.
            blue = ((rgb[..., 2] > .25) & (rgb[..., 2] > rgb[..., 0]+.15)
                    & (rgb[..., 2] > rgb[..., 1]+.15))
            components, count = label(blue)
            if not count:
                raise RuntimeError("No visible blue pad pixels")
            # Occlusion can split a single pad into disconnected visible strips.
            # Keep supported fragments, not just the largest strip. This task
            # has one blue target; multiple blue objects are not disambiguated.
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            mask = np.isin(components, np.flatnonzero(sizes >= 5))
            points = backproject_mask_to_world(mask, depth_to_metres(frame["depth"]),
                frame["intrinsic_cv"], frame["extrinsic_cv"])
            points = points[np.isfinite(points).all(axis=1)]
            points = points[np.abs(points[:, 2]-table_z) <= .01]
            if len(points) < 20:
                raise RuntimeError("Too few measured blue points near visual table")
            clouds.append(points)
            item.update(accepted=True, points=len(points))
        except (ValueError, RuntimeError) as error:
            item["error"] = str(error)
    if not clouds:
        raise RuntimeError(f"No credible blue pad depth from available cameras: {views}")
    points = np.concatenate(clouds)
    center, shape = square_bounds(points)
    result = np.r_[center, np.median(points[:, 2])]
    check = pad_height_check(result, table_z)
    if not check["accepted"]:
        raise RuntimeError("Recovered blue target is not on visual table")
    return result, dict(accepted=True, source="measured_blue_horizontal_square_rgbd_bounds",
        cameras=views, observed_points=len(points), shape=shape, height_check=check,
        assumptions=["single_blue_horizontal_square_pad", "world_Z_table_up",
            "measured_bounds_approximately_cover_both_square_axes"],
        shape_completion=False, uses_pad_cad=False, uses_object_truth=False)
