"""Load the object-to-support relation extracted from a task's demo.

`functional_matrix` exists for cup, pillbottle and shoe, but not for fan, mouse or
stapler — and `placement.py` indexes it unconditionally, so those tasks fail with
`IndexError` before any motion.  The extracted relation (`tools/
extract_placement_relation.py`) is a static prior with the same role: it says
where the object's origin belongs relative to the support's origin at release.

Runtime use is one matrix multiply against the visually localized support; no
simulator pose is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xuhuixin/ula")
RELATIONS_DIR = ROOT / "rules" / "placement_relations"

# An extracted relation is only trusted when its samples agreed closely.
# Larger objects place less repeatably: the fan varies by ~12 mm across
# demos, still far inside RoboTwin's 40 mm success tolerance.
MAX_TRUSTED_SPREAD_M = 0.015
MIN_TRUSTED_SAMPLES = 1


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest proper rotation to a noisy/averaged 3x3 matrix."""
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=float))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _rotation_medoid(rotations: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Select an observed rotation, avoiding invalid element-wise SO(3) means."""
    rotations = [_project_rotation(value) for value in rotations]
    distances = np.asarray([
        [ _rotation_angle(left, right) for right in rotations]
        for left in rotations
    ])
    index = int(np.argmin(np.sum(distances, axis=1)))
    return rotations[index], float(np.mean(distances[index]))


def relation_path(task_name: str) -> Path:
    return RELATIONS_DIR / f"{task_name}.json"


def load_placement_relation(task_name: str) -> dict | None:
    """Return the stored relation payload, or None when absent/untrustworthy."""
    path = relation_path(task_name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    matrix = payload.get("object_to_support")
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        return None
    payload["_matrix"] = array
    payload["trusted"] = (
        len(payload.get("samples") or []) >= MIN_TRUSTED_SAMPLES
        and float(payload.get("translation_spread_m") or 0.0) <= MAX_TRUSTED_SPREAD_M
    )
    return payload


def functional_from_relation(task_name: str, scale) -> np.ndarray | None:
    """Build a `functional_matrix`-equivalent 4x4 from the stored relation.

    `placement.py` multiplies the functional translation by the asset scale, so
    the stored metric offset is divided by the same scale here.  The rotation is
    kept in the matrix for callers that need the placement orientation; the
    current EE-pose solver takes orientation from its own candidate.
    """
    payload = load_placement_relation(task_name)
    if payload is None or not payload.get("trusted"):
        return None
    matrix = payload["_matrix"]
    scale = np.asarray(scale, dtype=float).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    scale = np.where(np.abs(scale) < 1e-9, 1.0, scale)
    functional = np.eye(4)
    functional[:3, :3] = matrix[:3, :3]
    functional[:3, 3] = matrix[:3, 3] / scale[:3]
    return functional


def relation_orientation(task_name: str) -> np.ndarray | None:
    """The object's rotation relative to the support, if a relation exists."""
    payload = load_placement_relation(task_name)
    if payload is None or not payload.get("trusted"):
        return None
    return payload["_matrix"][:3, :3].copy()


def placement_orientation(task_name: str, support_rotation=None):
    """Return a demo-derived world-frame target rotation and its provenance.

    A support-relative orientation is only useful when the support CAD frame is
    observable.  Thin rectangular/circular pads are rotationally ambiguous to
    CAD registration, while their tasks often require a fixed world orientation
    (fan facing the robot, mouse/stapler lying in a prescribed direction).

    We therefore compare successful demonstrations in two frames and use the
    frame with the tighter rotation cluster.  Ties prefer the world frame; this
    prevents an arbitrary symmetric-pad CAD yaw from rotating the task target.
    A single demonstration cannot establish invariance, so it conservatively
    keeps the support-relative relation.
    """
    payload = load_placement_relation(task_name)
    if payload is None or not payload.get("trusted"):
        return None, "unavailable"

    samples = payload.get("samples") or []
    world_rotations = []
    relative_rotations = []
    for sample in samples:
        try:
            object_rotation = _project_rotation(
                np.asarray(sample["object_pose"], dtype=float)[:3, :3])
            sampled_support = _project_rotation(
                np.asarray(sample["support_pose"], dtype=float)[:3, :3])
        except (KeyError, ValueError, np.linalg.LinAlgError):
            continue
        world_rotations.append(object_rotation)
        relative_rotations.append(sampled_support.T @ object_rotation)

    explicit_frame = payload.get("orientation_frame")
    if world_rotations:
        world_rotation, world_spread = _rotation_medoid(world_rotations)
        relative_rotation, relative_spread = _rotation_medoid(relative_rotations)
    else:
        world_rotation = None
        world_spread = float("inf")
        relative_rotation = _project_rotation(payload["_matrix"][:3, :3])
        relative_spread = float("inf")

    if explicit_frame in ("world", "support_relative"):
        frame = explicit_frame
    elif len(world_rotations) < 2:
        frame = "support_relative"
    else:
        # One degree is only a tie margin, not a task tuning threshold.  It
        # absorbs demonstration/controller noise when both frames are equally
        # stable and deliberately selects world for symmetric planar supports.
        frame = ("world" if world_spread <= relative_spread + np.deg2rad(1.0)
                 else "support_relative")

    if frame == "world":
        return world_rotation, "demo_world_rotation_medoid"
    if support_rotation is None:
        return None, "support_rotation_required"
    return (_project_rotation(support_rotation) @ relative_rotation,
            "visual_support_rotation_times_demo_relative_medoid")
