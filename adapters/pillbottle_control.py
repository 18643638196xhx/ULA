"""Pill-bottle-only measured robot state and bounded trajectory settling.

Never reads an object actor, object contact, identity mask or task success.
Patching is instance-local and restored on every exit, including exceptions.
"""
from contextlib import contextmanager

import numpy as np

from adapters.handoff import handoff_ready


def robot_state(robot):
    commanded, measured = [], []
    saved = getattr(robot, "_ula_command_state_getters", {})
    for arm in ("left", "right"):
        command_getter = saved.get(arm, getattr(robot, f"get_{arm}_arm_jointState"))
        commanded.extend(command_getter())
        measured.extend(getattr(robot, f"get_{arm}_arm_real_jointState")())
    command = np.asarray(commanded, dtype=float)
    actual = np.asarray(measured, dtype=float)
    if command.shape != (14,) or actual.shape != (14,) or not np.isfinite([command, actual]).all():
        raise ValueError("Expected finite commanded/measured dual-arm 14-D state")
    joint_indices = [*range(6), *range(7, 13)]
    return dict(commanded=command.tolist(), measured=actual.tolist(),
        max_joint_tracking_error_rad=float(np.max(np.abs(command[joint_indices]-actual[joint_indices]))),
        gripper_state_source="control_command_not_aperture_or_contact")


@contextmanager
def measured_joint_input(robot, enabled=True):
    """Synchronize policy observations AND TOPP starting states to robot qpos."""
    if not enabled:
        yield
        return
    changes = []
    if hasattr(robot, "_ula_command_state_getters"):
        raise RuntimeError("Measured joint-input context must not be nested")
    getters = {}
    try:
        for arm in ("left", "right"):
            name = f"get_{arm}_arm_jointState"
            original = getattr(robot, name)
            changes.append((name, name in vars(robot), original))
            getters[arm] = original
            setattr(robot, name, getattr(robot, f"get_{arm}_arm_real_jointState"))
        robot._ula_command_state_getters = getters
        yield
    finally:
        for name, owned, original in reversed(changes):
            if owned:
                setattr(robot, name, original)
            else:
                delattr(robot, name)
        if hasattr(robot, "_ula_command_state_getters"):
            del robot._ula_command_state_getters


@contextmanager
def audit_topp(robot, audit):
    """Expose exceptions/zero-waypoint results silently swallowed by RoboTwin."""
    changes = []
    try:
        for arm in ("left", "right"):
            planner = getattr(robot, f"{arm}_mplib_planner")
            original = planner.TOPP
            changes.append((planner, "TOPP" in vars(planner), original))
            def wrapper(*a, _original=original, _arm=arm, **kw):
                item = dict(arm=_arm)
                audit.append(item)
                try:
                    result = _original(*a, **kw)
                    item.update(waypoints=len(result[1]), success=len(result[1]) > 0)
                    return result
                except Exception as error:
                    item.update(success=False, error=f"{type(error).__name__}: {error}")
                    raise
            planner.TOPP = wrapper
        yield
    finally:
        for planner, owned, original in reversed(changes):
            if owned:
                planner.TOPP = original
            else:
                del planner.TOPP


def settle_at_target(task, arm, target_pose, target_joints, audit, *, max_steps=200,
                     check_every=20, position_tolerance=.02, joint_tolerance=.025):
    """Hold the already executed final joint target; never teleport or regrasp.

    Recompute passive-force compensation each step, clear trajectory-end velocity,
    and require three settled samples. Stop after a bounded hold if still off.
    """
    target = np.asarray(target_joints, dtype=float)
    if (target.shape != (6,) or not np.isfinite(target).all() or max_steps < 1
            or check_every < 1 or position_tolerance <= 0 or joint_tolerance <= 0):
        raise ValueError("Invalid settling target/budget")
    audit.update(samples=[], ready=False, max_steps=max_steps, velocity_target_zero=True,
        uses_object_truth=False, joint_tolerance_rad=joint_tolerance)
    consecutive = 0
    for index in range(max_steps+1):
        if index % check_every == 0 or index == max_steps:
            actual = np.asarray(getattr(task.robot, f"get_{arm}_arm_real_jointState")(), dtype=float)[:6]
            joint_error = float(np.max(np.abs(actual-target)))
            arrival = handoff_ready(task.get_arm_pose(arm), target_pose, task.plan_success,
                                    position_tolerance, 15.)
            good = arrival.ready and joint_error <= joint_tolerance
            consecutive = consecutive+1 if good else 0
            audit["samples"].append(dict(physics_steps=index, joint_error_rad=joint_error,
                arrival=arrival.to_dict(), consecutive=consecutive))
            if consecutive >= 3:
                audit.update(ready=True, physics_steps=index)
                return True
        if index < max_steps:
            task.robot.set_arm_joints(target, np.zeros(6), arm)
            task.scene.step()
    audit["physics_steps"] = max_steps
    return False


def settle_closed_gripper(task, arm, audit, *, max_steps=150):
    """Wait for measured finger motion to settle; not a grasp certificate."""
    if max_steps < 1:
        raise ValueError("Closure settling must have a finite positive budget")
    robot = task.robot
    entity = getattr(robot, f"{arm}_entity")
    joints = entity.get_active_joints()
    indices = [joints.index(pair[0]) for pair in getattr(robot, f"{arm}_gripper")]
    hold = np.asarray(getattr(robot, f"get_{arm}_arm_real_jointState")(), dtype=float)[:6]
    previous, stable = None, 0
    audit.update(samples=[], max_steps=max_steps, uses_object_truth=False,
        grasp_verified=False, arm_hold_source="measured_joint_angles", complete=False)
    for step in range(max_steps):
        robot.set_arm_joints(hold, np.zeros(6), arm)
        robot.set_gripper(0., arm)
        task.scene.step()
        if (step+1) % 10 == 0:
            actual = np.asarray(entity.get_qpos(), dtype=float)[indices]
            delta = None if previous is None else float(np.max(np.abs(actual-previous)))
            stable = stable+1 if delta is not None and delta < .0005 else 0
            audit["samples"].append(dict(physics_steps=step+1, finger_joint_positions=actual.tolist(),
                movement_since_previous_rad=delta, stable_samples=stable))
            previous = actual
            if stable >= 3:
                audit.update(complete=True, physics_steps=step+1)
                return True
    audit["physics_steps"] = max_steps
    return False
