"""Robot-only second policy with an optional donor-protection boundary.

The learned takeover policy predicts a 14-D dual-arm action.  During compiled
handover the runtime, rather than the policy, owns the donor release decision.
``protect_donor=True`` therefore freezes the demonstrated donor handover pose
and keeps its gripper closed, while the receiver executes the learned local
takeover.  A later visual/geometric gate decides whether the donor may release.
"""
import numpy as np


def measured_gripper_opening(robot, arm):
    """Return normalized physical finger opening from articulation qpos.

    RoboTwin's ``get_*_gripper_val`` is the last command, not measured finger
    motion.  The articulation qpos is the simulator counterpart of a real
    robot's gripper joint encoder and is therefore valid runtime feedback.
    """
    direct = getattr(robot, "get_actual_normalized_gripper_val", None)
    if direct is not None:
        return float(direct(arm))
    if arm not in ("left", "right"):
        raise ValueError(f"Unsupported arm: {arm!r}")
    entity = getattr(robot, f"{arm}_entity")
    gripper = getattr(robot, f"{arm}_gripper")
    scale = np.asarray(getattr(robot, f"{arm}_gripper_scale"), dtype=float)
    if not gripper or scale.shape != (2,):
        raise ValueError(f"Invalid {arm} gripper articulation metadata")
    base_joint = gripper[0][0]
    active_joints = list(entity.get_active_joints())
    try:
        joint_index = active_joints.index(base_joint)
    except ValueError as exc:
        raise ValueError(f"{arm} gripper base joint is not active") from exc
    qpos = float(np.asarray(entity.get_qpos(), dtype=float)[joint_index])
    # The configured drive scale may intentionally extend outside the URDF
    # joint range (ALOHA uses -0.01 while the physical lower limit is 0).  Use
    # the physical limits for encoder normalization or an empty closed gripper
    # is incorrectly reported as 18% open.
    limits = np.asarray(base_joint.get_limits(), dtype=float).reshape(-1, 2)[0]
    lower, upper = float(limits[0]), float(limits[1])
    if not np.isfinite([lower, upper]).all() or upper - lower < 1e-9:
        lower, upper = float(scale[0]), float(scale[1])
    if upper - lower < 1e-9:
        raise ValueError(f"Degenerate {arm} gripper opening range")
    return float(np.clip((qpos - lower) / (upper - lower), 0.0, 1.0))


def settle_receiver_gripper(
    task,
    result,
    *,
    max_steps=30,
    stable_steps=4,
    opening_delta_tol=0.002,
    minimum_blocked_opening=0.02,
):
    """Hold both arms while the receiver fingers physically settle.

    A successful close must leave a measurable non-zero aperture after the
    encoder has stabilized.  Empty fingers converge to the fully closed limit
    and are rejected before the donor is released.
    """
    if max_steps < stable_steps or stable_steps < 2:
        raise ValueError("Invalid receiver settling budget")
    if opening_delta_tol <= 0 or minimum_blocked_opening <= 0:
        raise ValueError("Invalid receiver aperture thresholds")
    left = np.asarray(task.robot.get_left_arm_jointState(), dtype=float)
    right = np.asarray(task.robot.get_right_arm_jointState(), dtype=float)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError("Expected 7-D arm/gripper command states")
    action = np.concatenate([left, right])
    action[13] = 0.0
    old_limit = task.step_lim
    task.step_lim = max(int(old_limit), int(task.take_action_cnt) + int(max_steps))
    samples = [measured_gripper_opening(task.robot, "right")]
    stable_run = 0
    try:
        for _ in range(max_steps):
            task.take_action(action)
            opening = measured_gripper_opening(task.robot, "right")
            stable_run = stable_run + 1 if abs(opening - samples[-1]) <= opening_delta_tol else 0
            samples.append(opening)
            if stable_run >= stable_steps:
                break
    finally:
        task.step_lim = old_limit
    final_opening = float(samples[-1])
    settled = bool(stable_run >= stable_steps)
    blocked = bool(settled and final_opening >= minimum_blocked_opening)
    audit = {
        "settled": settled,
        "steps": len(samples) - 1,
        "stable_steps": int(stable_run),
        "opening_delta_tol": float(opening_delta_tol),
        "minimum_blocked_opening": float(minimum_blocked_opening),
        "initial_measured_opening": float(samples[0]),
        "final_measured_opening": final_opening,
        "receiver_aperture_blocked": blocked,
        "samples": [float(v) for v in samples],
        "signal": "physical_gripper_joint_encoder",
        "uses_object_truth": False,
        "uses_contact_truth": False,
    }
    result["receiver_settle"] = audit
    result["receiver_aperture_confirmed"] = blocked
    return audit


def rollout_take(
    task,
    policy,
    result,
    max_steps=200,
    *,
    protect_donor=False,
    freeze_donor_arm=True,
    receiver_closed_threshold=.05,
    stable_receiver_close_steps=3,
    opening_delta_tol=0.002,
    minimum_blocked_opening=0.02,
    stable_blocked_steps=4,
):
    if max_steps < 1:
        raise ValueError("Positive second-stage action budget required")
    if stable_receiver_close_steps < 1 or not 0 <= receiver_closed_threshold <= 1:
        raise ValueError("Invalid receiver-closure configuration")
    if stable_blocked_steps < 2 or opening_delta_tol <= 0 or minimum_blocked_opening <= 0:
        raise ValueError("Invalid receiver blocked-opening configuration")
    start, old_limit = int(task.take_action_cnt), task.step_lim
    original = task.take_action
    task.step_lim = start+max_steps
    donor_arm_target = None
    donor_gripper_target = None
    if protect_donor:
        donor_state = np.asarray(task.robot.get_left_arm_jointState(), dtype=float)
        if donor_state.shape != (7,) or not np.isfinite(donor_state).all():
            raise ValueError("Expected finite 7-D donor joint/gripper state")
        donor_arm_target = donor_state[:6].copy()
        donor_gripper_target = float(donor_state[6])
    receiver_closed_run = 0
    blocked_stable_run = 0
    prev_blocked_opening = None
    result.update(start_step=start, max_steps=max_steps, commands=[],
        uses_object_truth=False, stop_reason="fixed_action_budget", trial_lift=False,
        chunk_steps=policy.chunk_steps, instruction=task.get_instruction(),
        donor_protected=bool(protect_donor), receiver_close_stable=False,
        receiver_close_command_stable=False, receiver_blocked_stable=False,
        receiver_closed_threshold=float(receiver_closed_threshold),
        stable_receiver_close_steps=int(stable_receiver_close_steps),
        stable_blocked_steps=int(stable_blocked_steps),
        opening_delta_tol=float(opening_delta_tol),
        minimum_blocked_opening=float(minimum_blocked_opening))
    def action(values, *a, **kw):
        nonlocal receiver_closed_run, blocked_stable_run, prev_blocked_opening
        values = np.asarray(values)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("Expected finite 14-D dual-arm action")
        executed = values.copy()
        if protect_donor:
            # The traditional controller has already reproduced the
            # demonstrated donor handover pose.  Keep that arm and its grasp
            # fixed; only receiver-side policy outputs may affect the scene.
            if freeze_donor_arm:
                executed[:6] = donor_arm_target
            executed[6] = donor_gripper_target
        original(executed, *a, **kw)
        right_value = float(task.robot.get_right_gripper_val())
        receiver_closed_run = receiver_closed_run + 1 if right_value < receiver_closed_threshold else 0
        right_opening = measured_gripper_opening(task.robot, "right") if protect_donor else None
        if protect_donor:
            if right_value < receiver_closed_threshold and right_opening >= minimum_blocked_opening:
                if prev_blocked_opening is not None and abs(right_opening - prev_blocked_opening) <= opening_delta_tol:
                    blocked_stable_run += 1
                else:
                    blocked_stable_run = 1
                prev_blocked_opening = right_opening
            else:
                blocked_stable_run = 0
                prev_blocked_opening = None
        cmd_entry = dict(step=int(task.take_action_cnt),
            phase_step=int(task.take_action_cnt)-start,
            left=float(task.robot.get_left_gripper_val()),
            right=right_value,
            policy_left_gripper=float(values[6]),
            executed_left_gripper=float(executed[6]),
            left_ee_pose=list(task.get_arm_pose("left")),
            right_ee_pose=list(task.get_arm_pose("right")))
        if right_opening is not None:
            cmd_entry["right_measured_opening"] = float(right_opening)
            cmd_entry["blocked_stable_run"] = int(blocked_stable_run)
        result["commands"].append(cmd_entry)
        if protect_donor and receiver_closed_run >= stable_receiver_close_steps:
            # Record consecutive close commands for audit, but do NOT early-stop
            # on command alone; an empty close must let VLA keep adjusting.
            if not result["receiver_close_command_stable"]:
                result["receiver_close_command_step"] = int(task.take_action_cnt)
            result["receiver_close_command_stable"] = True
        if protect_donor and right_value < receiver_closed_threshold and blocked_stable_run >= stable_blocked_steps:
            result.update(
                receiver_close_stable=True,
                receiver_blocked_stable=True,
                receiver_close_step=int(task.take_action_cnt),
                stop_reason="receiver_blocked_stable_donor_protected",
            )
            # NativePi05Pickup checks the step limit between actions in its
            # current chunk, so this terminates only once physical finger
            # encoders confirm the object is stably held.
            task.step_lim = int(task.take_action_cnt)
    task.take_action = action
    try:
        policy.reset()
        while task.take_action_cnt < task.step_lim:
            before = task.take_action_cnt
            policy.step(task, task.get_obs())
            print(f"[native-pi05-take] step={task.take_action_cnt-start}/{max_steps}",flush=True)
            if task.take_action_cnt == before:
                raise RuntimeError("Right-take policy executed no action")
    finally:
        task.take_action = original
        task.step_lim = old_limit
        result.update(end_step=int(task.take_action_cnt),
            vla_steps=int(task.take_action_cnt)-start,
            complete=bool(task.take_action_cnt == start+max_steps or result["receiver_close_stable"]))
