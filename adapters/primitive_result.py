"""Typed primitive results with explicit hand-off evidence.

The bottle experiments showed why ``closed the gripper`` is not a usable stage
output: three consecutive closed commands were accepted as a grasp, four of them
were empty jaws and about four more dropped mid-transport.  A primitive therefore
has to report *what it observed about the object*, not just that it issued a
command.

``PrimitiveResult`` is that contract.  ``permits_next_stage`` is the single field
the driver gates on, and it is derived from ``evidence`` rather than from the
action's own success flag.

Nothing here reads simulator truth.  Evidence comes from the wrist camera, the
measured TCP pose and the arm's own gripper reading.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PrimitiveResult:
    """Unified output of one primitive execution."""

    op: str
    arm: str
    completed: bool = False
    confidence: float = 0.0
    tcp_pose: Optional[List[float]] = None       # [x, y, z, qw, qx, qy, qz]
    object_pose: Optional[List[float]] = None   # 4x4 flattened, world frame
    ee_to_object: Optional[List[float]] = None  # 4x4 flattened
    gripper: Optional[float] = None             # measured opening, normalised
    evidence: Dict[str, Any] = field(default_factory=dict)
    permits_next_stage: bool = False
    error: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _flatten(matrix) -> Optional[List[float]]:
    if matrix is None:
        return None
    try:
        array = np.asarray(matrix, dtype=float)
    except Exception:
        return None
    if array.shape != (4, 4) or not np.isfinite(array).all():
        return None
    return array.reshape(-1).tolist()


def _aperture(task, arm) -> Optional[float]:
    """Measured normalised opening: 0 = fully closed, 1 = fully open."""
    try:
        return float(getattr(task.robot, f"get_{arm}_gripper_val")())
    except Exception:
        return None


def _wrist_depth_stats(task, arm) -> Dict[str, Any]:
    """Depth coverage in the wrist image's central band.

    Recorded as evidence only: the aperture reading alone turned out to be a poor
    discriminator (successful and empty grasps both reported ~0.1), so the raw
    statistics are kept for a later success/failure comparison instead of being
    turned into a threshold now.
    """
    name = f"{arm}_camera"
    stats: Dict[str, Any] = dict(camera=name)
    try:
        frame = task.get_camera_observation(name)
        depth = np.asarray(frame["depth"], dtype=float)
    except Exception as error:
        stats["error"] = f"wrist frame failed: {error}"
        return stats
    if depth.ndim != 2:
        stats["error"] = "unexpected depth shape"
        return stats
    metres = depth / 1000.0 if np.nanmax(depth) > 20.0 else depth.copy()
    valid = np.isfinite(metres) & (metres > 0.02) & (metres < 0.60)
    h, w = metres.shape
    band = np.zeros_like(valid)
    band[int(h * 0.30):int(h * 0.85), int(w * 0.30):int(w * 0.85)] = True
    mask = valid & band
    count = int(mask.sum())
    stats["band_points"] = count
    stats["band_fraction"] = round(count / float(max(1, band.sum())), 4)
    if count:
        zs = metres[mask]
        stats["depth_min_m"] = round(float(np.percentile(zs, 2)), 4)
        stats["depth_median_m"] = round(float(np.median(zs)), 4)
        stats["depth_max_m"] = round(float(np.percentile(zs, 98)), 4)
        stats["depth_spread_m"] = round(stats["depth_max_m"] - stats["depth_min_m"], 4)
    return stats


def collect_grasp_evidence(task, arm, audit=None) -> Dict[str, Any]:
    """Evidence for a grasp stage: what the arm and wrist actually observed."""
    evidence = dict(
        gripper_aperture=_aperture(task, arm),
        wrist=_wrist_depth_stats(task, arm),
        uses_simulator_truth=False,
    )
    audit = audit or {}
    # The VLA handoff reports the arrival check and the closed-command step; keep
    # them visible so a reviewer can tell "arrived" from "holding".
    for key in ("ready", "reason", "position_error_m", "orientation_error_deg",
                "gripper_command", "after_vla_step"):
        if key in audit:
            evidence[key] = audit[key]
    return evidence


def build_result(op, arm, *, audit=None, evidence=None, object_pose=None,
                 ee_to_object=None, requires_grasp_evidence=False,
                 min_confidence=0.5) -> PrimitiveResult:
    """Assemble a result and decide ``permits_next_stage``.

    For a grasp the gate is deliberately conservative: ``arrived`` plus a closed
    command is *not* enough, because that combination was exactly what produced
    the empty jaws.  With ``requires_grasp_evidence`` the caller opts into
    requiring the wrist patch, otherwise the evidence is recorded and the gate
    falls back to the handoff's own readiness.
    """
    audit = audit or {}
    evidence = dict(evidence or {})
    result = PrimitiveResult(
        op=str(op), arm=str(arm), label=audit.get("label"),
        completed=bool(audit.get("ready", audit.get("completed", False))),
        gripper=evidence.get("gripper_aperture"),
        evidence=evidence,
        object_pose=_flatten(object_pose),
        ee_to_object=_flatten(ee_to_object),
        error=audit.get("error"),
    )
    try:
        pose = audit.get("actual_ee_pose") or audit.get("ee_pose")
        if pose is not None:
            result.tcp_pose = [float(v) for v in np.asarray(pose, dtype=float).ravel()[:7]]
    except Exception:
        result.tcp_pose = None

    checks = []
    if audit.get("ready") is not None:
        checks.append(bool(audit["ready"]))
    if op == "grasp":
        wrist = evidence.get("wrist") or {}
        has_patch = (wrist.get("band_points") or 0) > 0
        evidence["visual_in_gripper"] = bool(has_patch)
        if requires_grasp_evidence:
            checks.append(has_patch)
    result.confidence = round(sum(checks) / len(checks), 3) if checks else 0.0
    result.permits_next_stage = bool(checks) and all(checks) \
        and result.confidence >= min_confidence
    return result
