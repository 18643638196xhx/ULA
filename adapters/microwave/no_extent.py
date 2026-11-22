"""Isolated ablation: disable ONLY fixed panel width/height bounds in tracking.

Initial localization remains unchanged. RANSAC support, point count, 2-D
spread, normal cone, hinge consistency, relative coverage and progress tests
remain enabled. No saved result or original module source is rewritten.
"""
from contextlib import contextmanager

import numpy as np

from adapters.microwave.geometry import unit


def fit_panel_without_extent(points, normal_hint, *, tolerance_m=.008, min_points=60,
                             normal_cone_deg=None):
    """Same deterministic fit as geometry.fit_panel, minus its extent gate."""
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
    # Deliberately OMIT the old 0.08 <= width,height <= 0.9 gate.
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


@contextmanager
def tracking_without_extent():
    """Patch only main-process postgrasp tracking; restore even on failure."""
    from adapters.microwave import panel_tracking as tracking

    original_track, original_fit = tracking.track_panel, tracking.fit_panel

    def track(*args, **kwargs):
        tracking.fit_panel = fit_panel_without_extent
        try:
            result = original_track(*args, **kwargs)
            result["vision_audit"].update(fixed_panel_extent_check_enabled=False,
                                          relative_coverage_check_enabled=True)
            return result
        finally:
            tracking.fit_panel = original_fit

    tracking.track_panel = track
    try:
        yield
    finally:
        tracking.track_panel, tracking.fit_panel = original_track, original_fit
