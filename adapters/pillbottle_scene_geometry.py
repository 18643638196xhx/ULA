"""Known-CAD geometry checks against a visually measured horizontal table."""
import numpy as np


def table_clearance(pose, model_points, table_z, tolerance=.008):
    pose, model = np.asarray(pose, dtype=float), np.asarray(model_points, dtype=float)
    if (pose.shape != (4, 4) or model.ndim != 2 or model.shape[1] != 3 or len(model) < 8
            or not np.isfinite(pose).all() or not np.isfinite(model).all()
            or not np.isfinite(table_z) or tolerance < 0):
        raise ValueError("Invalid CAD/table geometry")
    lowest = float(np.min(model@pose[2, :3]+pose[2, 3]))
    return dict(accepted=lowest >= table_z-tolerance,
        lowest_model_world_z_m=lowest, visual_table_height_m=float(table_z),
        maximum_table_penetration_m=tolerance,
        source="known_cad_and_rgbd_table_not_simulator_pose")


def gravity_upright(reference):
    """Preserve visual heading, remove initial FP roll/pitch estimation noise."""
    value = np.asarray(reference, dtype=float).copy()
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("No credible visual bottle heading")
    heading = value[:3, 2].copy()
    heading[2] = 0.
    if np.linalg.norm(heading) < .1:
        raise ValueError("No credible visual bottle heading")
    heading /= np.linalg.norm(heading)
    up = np.array([0., 0., 1.])
    value[:3, :3] = np.column_stack((np.cross(up, heading), up, heading))
    return value
