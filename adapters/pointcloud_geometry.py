"""Model-free visible-surface bounds from segmented RGB-D, for a horizontal table."""

import numpy as np


def estimate_visible_bounds(cup_mask, coaster_mask, depth, intrinsic, world_to_camera,
                            min_points=40, trim_quantile=0.01, return_points=False):
    from traditional_place_empty_cup.geometry import backproject_mask_to_world, depth_to_metres

    cup_mask = np.asarray(cup_mask, dtype=bool)
    coaster_mask = np.asarray(coaster_mask, dtype=bool)
    if cup_mask.ndim != 2 or coaster_mask.shape != cup_mask.shape:
        raise ValueError("Expected matching 2-D cup/coaster masks")
    if min_points < 3 or not 0 <= trim_quantile < 0.25:
        raise ValueError("Invalid point-cloud filtering parameters")
    rows, cols = np.nonzero(cup_mask)
    if len(rows) < min_points:
        raise RuntimeError("Too few segmented cup pixels")
    metres = depth_to_metres(depth)
    # One-pixel erosion reduces mixed object/background depth at the silhouette.
    interior = np.zeros_like(cup_mask)
    interior[1:-1, 1:-1] = (cup_mask[1:-1, 1:-1] & cup_mask[:-2, 1:-1]
                           & cup_mask[2:, 1:-1] & cup_mask[1:-1, :-2] & cup_mask[1:-1, 2:])
    points = backproject_mask_to_world(interior, metres, intrinsic, world_to_camera)
    eroded = True
    if len(points) < min_points:
        points = backproject_mask_to_world(cup_mask, metres, intrinsic, world_to_camera)
        eroded = False
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < min_points:
        raise RuntimeError("Too few valid cup depth points")
    raw_count = len(points)
    # Estimate the local horizontal support height from an expanded image region.
    padding = max(8, int(max(np.ptp(rows), np.ptp(cols)) * 0.4))
    ring = np.zeros_like(cup_mask)
    ring[max(0, rows.min()-padding):rows.max()+padding+1,
         max(0, cols.min()-padding):cols.max()+padding+1] = True
    ring &= ~(cup_mask | coaster_mask)
    background = backproject_mask_to_world(ring, metres, intrinsic, world_to_camera)
    background = background[np.isfinite(background).all(axis=1)]
    if len(background) < min_points:
        raise RuntimeError("Too few local background depth points to estimate table height")
    # Height-mode prior is explicit: this pilot assumes world +Z table-up.
    bins = np.floor(background[:, 2] / 0.005).astype(np.int64)
    labels, counts = np.unique(bins, return_counts=True)
    selected = labels[np.argmax(counts)]
    support = background[np.abs(background[:, 2] - (selected + 0.5) * 0.005) <= 0.0075]
    if len(support) < min_points:
        raise RuntimeError("No sufficiently supported local horizontal table-height cluster")
    table_z = float(np.median(support[:, 2]))
    points = points[points[:, 2] > table_z + 0.003]
    if len(points) < min_points:
        raise RuntimeError("Too few cup points above the estimated table")
    lower = np.quantile(points, trim_quantile, axis=0)
    upper = np.quantile(points, 1-trim_quantile, axis=0)
    spans = upper-lower
    if np.any(spans[:2] < 0.005) or np.any(spans > 0.5) or upper[2] < table_z + 0.01:
        raise RuntimeError("Observed object bounds failed generic workspace/size sanity checks")
    audit = dict(
        source="segmented_rgbd_visible_surface_quantiles", uses_prepared_object_model=False,
        uses_foundationpose=False, uses_contact_annotations=False,
        uses_simulator_truth_for_target=False, visible_surface_only=True,
        shape_completion=False, world_up_axis="+Z", table_height_m=table_z,
        table_height_source="local_background_depth_height_mode",
        table_support_points=len(support), raw_valid_cup_points=raw_count,
        cup_points_above_table=len(points), mask_eroded=eroded,
        trim_quantile=float(trim_quantile), bbox_world_min=lower.tolist(),
        bbox_world_max=upper.tolist(),
    )
    if return_points:
        # Keep measured surfaces, not a completed mesh. Registration has its own outlier filter.
        return np.stack((lower, upper)), audit, points
    return np.stack((lower, upper)), audit
