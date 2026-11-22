"""Shake primitive: grasp-relative lift and oscillation from robot state only.

No object pose or simulator truth is read; every move is expressed relative to
the measured TCP pose and composed with pure rotation matrices.
"""

import math
import numpy as np

from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz, pose_wxyz_to_matrix


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _quaternion_matrix(quaternion):
    w, x, y, z = (float(v) for v in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("Shake pre-quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _move(task, arm, matrix, stage, result):
    from envs.utils import Action, ArmTag

    target = matrix_to_pose_wxyz(matrix)
    moved = bool(task.move((ArmTag(arm), [Action(ArmTag(arm), "move", target_pose=target)])))
    success = bool(moved and task.plan_success)
    result.setdefault("actions", []).append(dict(
        action=stage, target_pose=target.tolist(), success=success))
    if not success:
        raise RuntimeError(f"Shake failed at {stage}: motion plan failed")
    return pose_wxyz_to_matrix(task.get_arm_pose(arm))


def shake(task, arm, result, *, axis="y", amplitude_rad=7 * math.pi / 8, cycles=3,
          lift_step_m=0.05, pre_quaternion_wxyz=(0.707, 0.0, 0.0, 0.707),
          horizontal=False, pre_lift_m=0.1):
    """Lift, optionally tilt horizontal, oscillate, and return to the pre-pose."""
    if axis != "y":
        raise ValueError("Shake primitive currently supports the y axis only")
    if not np.isfinite(amplitude_rad) or amplitude_rad <= 0:
        raise ValueError("Shake amplitude must be positive")
    if cycles < 1 or lift_step_m <= 0 or pre_lift_m < 0:
        raise ValueError("Shake cycle and lift values must be positive")
    result.setdefault("actions", [])
    result["shake"] = dict(
        axis=axis, amplitude_rad=float(amplitude_rad), cycles=int(cycles),
        lift_step_m=float(lift_step_m), horizontal=bool(horizontal),
        pre_lift_m=float(pre_lift_m), pre_quaternion_wxyz=[float(v) for v in pre_quaternion_wxyz],
        uses_simulator_truth=False,
    )
    current = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    prepare = current.copy()
    prepare[:3, 3] = current[:3, 3] + np.array([0.0, 0.0, pre_lift_m])
    prepare[:3, :3] = _quaternion_matrix(pre_quaternion_wxyz)
    current = _move(task, arm, prepare, "shake_lift_prepare", result)
    base = prepare.copy()
    base[:3, 3] = current[:3, 3]
    if horizontal:
        horizontal_matrix = base.copy()
        horizontal_matrix[:3, :3] = base[:3, :3] @ _rotation_y(math.pi / 2)
        current = _move(task, arm, horizontal_matrix, "shake_horizontal", result)
        base[:3, :3] = horizontal_matrix[:3, :3]
        base[:3, 3] = current[:3, 3]
    positive = _rotation_y(+amplitude_rad) @ base[:3, :3]
    negative = _rotation_y(-amplitude_rad) @ base[:3, :3]
    for index in range(cycles):
        current = pose_wxyz_to_matrix(task.get_arm_pose(arm))
        up = base.copy()
        up[:3, 3] = current[:3, 3] + np.array([0.0, 0.0, lift_step_m])
        up[:3, :3] = positive
        current = _move(task, arm, up, f"shake_up_{index}", result)
        down = base.copy()
        down[:3, 3] = current[:3, 3] - np.array([0.0, 0.0, lift_step_m])
        down[:3, :3] = negative
        current = _move(task, arm, down, f"shake_down_{index}", result)
    back = base.copy()
    back[:3, 3] = current[:3, 3]
    back[:3, :3] = _quaternion_matrix(pre_quaternion_wxyz)
    _move(task, arm, back, "shake_return", result)
    result["shake"]["final_height_m"] = float(
        pose_wxyz_to_matrix(task.get_arm_pose(arm))[2, 3])
    return result
