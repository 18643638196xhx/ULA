"""Geometric handoff gate. Inputs use RoboTwin world-frame EE poses (wxyz)."""

import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HandoffDecision:
    ready: bool
    reason: str
    position_error_m: float
    orientation_error_deg: float

    def to_dict(self):
        return asdict(self)


def handoff_ready(current_ee_pose, pregrasp_pose, plan_success,
                  position_tolerance_m=None, orientation_tolerance_deg=None):
    """Use measured robot state and a vision-derived target, never object truth.

    The pre-grasp pose is a staging point: the VLA then closes the remaining gap
    itself, so a couple of centimetres of arrival slack is harmless.  Larger,
    differently shaped objects (fan, stapler, shoe) stop a few mm short of the
    2 cm window that was tuned on the cup, where arrival errors are ~2 mm anyway.
    """
    if position_tolerance_m is None:
        position_tolerance_m = float(os.environ.get("ULA_HANDOFF_POSITION_TOL", "0.03"))
    if orientation_tolerance_deg is None:
        orientation_tolerance_deg = float(
            os.environ.get("ULA_HANDOFF_ORIENTATION_TOL", "25.0"))
    if position_tolerance_m <= 0 or not 0 < orientation_tolerance_deg <= 180:
        raise ValueError("Handoff tolerances must be positive (orientation <= 180)")
    current = np.asarray(current_ee_pose, dtype=float)
    target = np.asarray(pregrasp_pose, dtype=float)
    if (current.shape != (7,) or target.shape != (7,)
            or not np.isfinite(current).all() or not np.isfinite(target).all()):
        raise ValueError("Expected finite [x,y,z,qw,qx,qy,qz] poses")
    norms = np.linalg.norm(current[3:]), np.linalg.norm(target[3:])
    if min(norms) < 1e-12:
        raise ValueError("Quaternion has zero norm")
    position = float(np.linalg.norm(current[:3] - target[:3]))
    dot = abs(float(np.dot(current[3:] / norms[0], target[3:] / norms[1])))
    orientation = float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))))
    if not plan_success:
        reason = "traditional_planning_failed"
    elif position > position_tolerance_m:
        reason = "outside_position_tolerance"
    elif orientation > orientation_tolerance_deg:
        reason = "outside_orientation_tolerance"
    else:
        reason = "pregrasp_reached"
    return HandoffDecision(reason == "pregrasp_reached", reason, position, orientation)
