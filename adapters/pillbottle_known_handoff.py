"""Measured-state handoff inside an unchanged Cartesian pre-grasp region.

Only robot encoders/kinematics are read. This is NOT a grasp or lift check.
"""
import numpy as np

from adapters.handoff import handoff_ready


def settle_measured_handoff(task, arm, target_pose, audit, *, max_steps=200,
                           check_every=20, stable_delta_rad=.001, maximum_hold_drift_rad=.025):
    audit.update(ready=False, uses_object_truth=False, samples=[], trial_lift=False,
        cartesian_position_tolerance_m=.02, cartesian_orientation_tolerance_deg=15.,
        stable_delta_rad=stable_delta_rad, maximum_hold_drift_rad=maximum_hold_drift_rad,
        target_source="measured_joint_angles_within_original_pregrasp_region")
    decision = handoff_ready(task.get_arm_pose(arm), target_pose, task.plan_success, .02, 15.)
    if not decision.ready:
        audit.update(reason="outside_original_cartesian_region", arrival=decision.to_dict())
        return False
    initial = np.asarray(getattr(task.robot, f"get_{arm}_arm_real_jointState")(), dtype=float)[:6]
    if initial.shape != (6,) or not np.isfinite(initial).all():
        raise ValueError("Invalid measured handoff joints")
    previous, consecutive = initial.copy(), 0
    # Clear the unreachable drive target. Never set articulation qpos or move
    # the Cartesian target: hold the measured pose and recompute passive forces.
    for step in range(max_steps+1):
        if step % check_every == 0:
            current = np.asarray(getattr(task.robot, f"get_{arm}_arm_real_jointState")(), dtype=float)[:6]
            if current.shape != (6,) or not np.isfinite(current).all():
                raise ValueError("Invalid measured handoff joints")
            delta = float(np.max(np.abs(current-previous)))
            drift = float(np.max(np.abs(current-initial)))
            arrival = handoff_ready(task.get_arm_pose(arm), target_pose, task.plan_success, .02, 15.)
            good = arrival.ready and delta <= stable_delta_rad and drift <= maximum_hold_drift_rad
            consecutive = consecutive+1 if good else 0
            audit["samples"].append(dict(physics_steps=step, measured_joints=current.tolist(),
                delta_rad=delta, drift_rad=drift, arrival=arrival.to_dict(), consecutive=consecutive))
            if drift > maximum_hold_drift_rad or not arrival.ready:
                audit["reason"] = "measured_hold_left_allowed_region"
                return False
            if consecutive >= 3:
                task.robot.set_arm_joints(current, np.zeros(6), arm)
                audit.update(ready=True, physics_steps=step, final_measured_joints=current.tolist())
                return True
            previous = current.copy()
        if step < max_steps:
            task.robot.set_arm_joints(initial, np.zeros(6), arm)
            task.scene.step()
    audit["reason"] = "measured_state_not_stable_within_budget"
    return False
