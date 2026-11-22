"""Calibrate fixed rigid transforms between recorded EE, plan target, ee_link, and TCP.

In RoboTwin, demonstrations record ``task.get_arm_pose(arm)`` (``recorded_ee``),
while CuRobo planners may consume either ``robot.*_plan_path`` targets or raw
``ee_link`` poses, and contact/safety checks reference the gripper TCP.  This
module measures and audits those fixed transforms across multiple robot
configurations so runtime execution never relies on unverified frame assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from adapters.primitive_geometry import _transform, relative_pose

SCHEMA_VERSION = 1


def pose_wxyz_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """Convert ``[x, y, z, qw, qx, qy, qz]`` into a rigid 4x4 matrix."""
    arr = np.asarray(pose, dtype=float).reshape(-1)
    if arr.shape != (7,) or not np.isfinite(arr).all():
        raise ValueError(f"Expected 7-element [x,y,z,qw,qx,qy,qz] pose, got {arr.shape}")
    x, y, z, qw, qx, qy, qz = arr
    norm = float(np.linalg.norm([qw, qx, qy, qz]))
    if norm < 1e-8:
        raise ValueError("Zero-norm quaternion in pose")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    R = np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose_wxyz(matrix: np.ndarray) -> list[float]:
    """Convert a rigid 4x4 matrix into ``[x, y, z, qw, qx, qy, qz]``."""
    T = _transform(matrix, "matrix")
    R = T[:3, :3]
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
            float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle in degrees between two 3x3 rotation matrices."""
    cos_val = float(np.clip((np.trace(R_a.T @ R_b) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def average_rigid_transforms(transforms: Sequence[np.ndarray]) -> tuple[np.ndarray, float, float]:
    """Compute mean rigid transform and max translation (m) / rotation (deg) residuals."""
    mats = [_transform(t, f"transform[{i}]") for i, t in enumerate(transforms)]
    if not mats:
        raise ValueError("At least one transform sample is required")
    mean_t = np.mean([m[:3, 3] for m in mats], axis=0)
    R_sum = np.sum([m[:3, :3] for m in mats], axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0.0:
        U[:, -1] *= -1.0
        R_mean = U @ Vt
    mean = np.eye(4, dtype=float)
    mean[:3, :3] = R_mean
    mean[:3, 3] = mean_t
    max_trans = max(float(np.linalg.norm(m[:3, 3] - mean_t)) for m in mats)
    max_rot = max(rotation_error_deg(m[:3, :3], R_mean) for m in mats)
    return mean, max_trans, max_rot


@dataclass(frozen=True)
class ArmFrameCalibration:
    arm: str
    ee_link: str
    sample_count: int
    T_recorded_ee_to_plan_target: list[list[float]]
    T_recorded_ee_to_ee_link: list[list[float]]
    T_recorded_ee_to_tcp: list[list[float]]
    max_plan_target_trans_err_m: float
    max_plan_target_rot_err_deg: float
    max_ee_link_trans_err_m: float
    max_ee_link_rot_err_deg: float
    max_tcp_trans_err_m: float
    max_tcp_rot_err_deg: float


@dataclass(frozen=True)
class FrameCalibration:
    schema_version: int
    robot_name: str
    source_task: str
    arms: dict[str, ArmFrameCalibration]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "robot_name": self.robot_name,
            "source_task": self.source_task,
            "arms": {arm: asdict(cal) for arm, cal in self.arms.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrameCalibration":
        ver = data.get("schema_version")
        if int(ver if ver is not None else -1) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported calibration schema_version: {ver}")
        arms = {
            arm: ArmFrameCalibration(**arm_data)
            for arm, arm_data in data["arms"].items()
        }
        return cls(
            schema_version=SCHEMA_VERSION,
            robot_name=str(data.get("robot_name", "unknown")),
            source_task=str(data.get("source_task", "unknown")),
            arms=arms,
        )

    @classmethod
    def identity(cls, robot_name: str = "identity", source_task: str = "synthetic") -> "FrameCalibration":
        eye = np.eye(4, dtype=float).tolist()
        arms = {
            arm: ArmFrameCalibration(
                arm=arm,
                ee_link=f"{arm}_ee_link",
                sample_count=1,
                T_recorded_ee_to_plan_target=eye,
                T_recorded_ee_to_ee_link=eye,
                T_recorded_ee_to_tcp=eye,
                max_plan_target_trans_err_m=0.0,
                max_plan_target_rot_err_deg=0.0,
                max_ee_link_trans_err_m=0.0,
                max_ee_link_rot_err_deg=0.0,
                max_tcp_trans_err_m=0.0,
                max_tcp_rot_err_deg=0.0,
            )
            for arm in ("left", "right")
        }
        return cls(SCHEMA_VERSION, robot_name, source_task, arms)


def estimate_arm_calibration(
    arm: str,
    ee_link: str,
    samples: Sequence[Mapping[str, np.ndarray]],
    *,
    max_trans_tol_m: float = 5e-4,
    max_rot_tol_deg: float = 0.1,
) -> ArmFrameCalibration:
    """Estimate fixed transforms for one arm from multi-pose world-frame samples."""
    if not samples:
        raise ValueError(f"No samples provided for arm {arm}")
    rel_plan = [relative_pose(s["recorded_ee"], s["plan_target"]) for s in samples]
    rel_link = [relative_pose(s["recorded_ee"], s["ee_link"]) for s in samples]
    rel_tcp = [relative_pose(s["recorded_ee"], s["tcp"]) for s in samples]

    T_plan, plan_t_err, plan_r_err = average_rigid_transforms(rel_plan)
    T_link, link_t_err, link_r_err = average_rigid_transforms(rel_link)
    T_tcp, tcp_t_err, tcp_r_err = average_rigid_transforms(rel_tcp)

    for label, t_err, r_err in (
        ("plan_target", plan_t_err, plan_r_err),
        ("ee_link", link_t_err, link_r_err),
        ("tcp", tcp_t_err, tcp_r_err),
    ):
        if t_err > max_trans_tol_m or r_err > max_rot_tol_deg:
            raise ValueError(
                f"Arm {arm} frame transform {label} is not rigid across samples: "
                f"trans_err={t_err:.6f}m (tol {max_trans_tol_m}m), "
                f"rot_err={r_err:.4f}deg (tol {max_rot_tol_deg}deg)"
            )

    return ArmFrameCalibration(
        arm=arm,
        ee_link=ee_link,
        sample_count=len(samples),
        T_recorded_ee_to_plan_target=T_plan.tolist(),
        T_recorded_ee_to_ee_link=T_link.tolist(),
        T_recorded_ee_to_tcp=T_tcp.tolist(),
        max_plan_target_trans_err_m=plan_t_err,
        max_plan_target_rot_err_deg=plan_r_err,
        max_ee_link_trans_err_m=link_t_err,
        max_ee_link_rot_err_deg=link_r_err,
        max_tcp_trans_err_m=tcp_t_err,
        max_tcp_rot_err_deg=tcp_r_err,
    )


def _sapien_pose_to_matrix(pose_obj: Any) -> np.ndarray:
    if hasattr(pose_obj, "to_transformation_matrix"):
        return np.asarray(pose_obj.to_transformation_matrix(), dtype=float)
    p = np.asarray(pose_obj.p, dtype=float).reshape(3)
    q = np.asarray(pose_obj.q, dtype=float).reshape(4)
    return pose_wxyz_to_matrix(np.concatenate([p, q]))


def _sample_single_pose(task: Any, arm: str) -> dict[str, np.ndarray]:
    robot = task.robot
    recorded_ee_wxyz = task.get_arm_pose(arm)
    T_recorded_ee = pose_wxyz_to_matrix(recorded_ee_wxyz)
    tcp_wxyz = robot.get_left_tcp_pose() if arm == "left" else robot.get_right_tcp_pose()
    T_tcp = pose_wxyz_to_matrix(tcp_wxyz)

    link_obj = robot.left_ee if arm == "left" else robot.right_ee
    T_live_link = _sapien_pose_to_matrix(link_obj.global_pose)
    T_global = np.eye(4, dtype=float)
    gmat = getattr(robot, "left_global_trans_matrix" if arm == "left" else "right_global_trans_matrix", None)
    if gmat is not None:
        T_global[:3, :3] = np.asarray(gmat, dtype=float)
    T_live_planner_endlink = T_live_link @ T_global
    T_converted_link = _sapien_pose_to_matrix(
        robot._trans_from_gripper_to_endlink(recorded_ee_wxyz, arm_tag=arm)
    )
    T_plan_to_link = relative_pose(T_recorded_ee, T_converted_link)
    T_plan_target = T_live_planner_endlink @ np.linalg.inv(T_plan_to_link)
    return {
        "recorded_ee": T_recorded_ee,
        "plan_target": T_plan_target,
        "ee_link": T_live_link,
        "tcp": T_tcp,
    }


def calibrate_robot_frames(
    task: Any,
    *,
    arms: Sequence[str] = ("left", "right"),
    num_samples: int = 8,
    seed: int = 0,
    delta_rad: float = 0.15,
    source_task: str = "handover_block",
) -> FrameCalibration:
    """Measure fixed EE/plan-target/ee_link/TCP transforms inside a live RoboTwin task."""
    rng = np.random.default_rng(seed)
    robot = task.robot
    arm_results: dict[str, ArmFrameCalibration] = {}

    for arm in arms:
        entity = robot.left_entity if arm == "left" else robot.right_entity
        link_obj = robot.left_ee if arm == "left" else robot.right_ee
        ee_link_name = str(getattr(link_obj, "name", f"{arm}_ee_link"))
        orig_qpos = np.asarray(entity.get_qpos(), dtype=float).copy()
        arm_dof = len(robot.left_arm_joints if arm == "left" else robot.right_arm_joints)
        samples: list[dict[str, np.ndarray]] = []
        try:
            for idx in range(max(1, int(num_samples))):
                q = orig_qpos.copy()
                if idx > 0:
                    q[:arm_dof] = orig_qpos[:arm_dof] + rng.uniform(-delta_rad, delta_rad, size=arm_dof)
                entity.set_qpos(q)
                if hasattr(task, "scene") and task.scene is not None:
                    task.scene.update_render()
                samples.append(_sample_single_pose(task, arm))
        finally:
            entity.set_qpos(orig_qpos)
            if hasattr(task, "scene") and task.scene is not None:
                task.scene.update_render()

        arm_results[arm] = estimate_arm_calibration(arm, ee_link_name, samples)

    robot_name = str(getattr(robot, "name", getattr(robot, "robot_name", "aloha_agilex")))
    return FrameCalibration(
        schema_version=SCHEMA_VERSION,
        robot_name=robot_name,
        source_task=source_task,
        arms=arm_results,
    )


def reconcile_recorded_ee_to_plan_target(
    T_world_recorded_ee: np.ndarray,
    calibration: FrameCalibration,
    arm: str,
) -> np.ndarray:
    T_rec = _transform(T_world_recorded_ee, "T_world_recorded_ee")
    T_offset = _transform(calibration.arms[arm].T_recorded_ee_to_plan_target, "T_recorded_ee_to_plan_target")
    return T_rec @ T_offset


def reconcile_recorded_ee_to_ee_link(
    T_world_recorded_ee: np.ndarray,
    calibration: FrameCalibration,
    arm: str,
) -> np.ndarray:
    T_rec = _transform(T_world_recorded_ee, "T_world_recorded_ee")
    T_offset = _transform(calibration.arms[arm].T_recorded_ee_to_ee_link, "T_recorded_ee_to_ee_link")
    return T_rec @ T_offset


def reconcile_recorded_ee_to_tcp(
    T_world_recorded_ee: np.ndarray,
    calibration: FrameCalibration,
    arm: str,
) -> np.ndarray:
    T_rec = _transform(T_world_recorded_ee, "T_world_recorded_ee")
    T_offset = _transform(calibration.arms[arm].T_recorded_ee_to_tcp, "T_recorded_ee_to_tcp")
    return T_rec @ T_offset


def save_calibration(calibration: FrameCalibration, output_path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing calibration file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_calibration(path: str | Path) -> FrameCalibration:
    return FrameCalibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
