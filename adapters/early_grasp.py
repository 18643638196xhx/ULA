"""Opt-in early handoff: bilateral contact, then a traditional trial lift.

Simulation truth is explicitly used for both contact and lift verification.
Object truth never generates the lift target; it is a fixed EE +world-Z move.
There is no release, regrasp, or placement on an unverified trial lift.
"""

import numpy as np

from adapters.grasp import SimulationGraspMonitor, active_gripper_contacts
from adapters.handoff import handoff_ready


def finger_contact_counts(task, arm):
    """Count cup contacts separately on moving finger links, not the palm."""
    names = {item[0].child_link.get_name() for item in getattr(task.robot, f"{arm}_gripper")}
    counts = dict.fromkeys(sorted(names), 0)
    cup_name = task.cup.get_name()
    for contact in task.scene.get_contacts():
        first, second = (body.entity.name for body in contact.bodies)
        finger = second if first == cup_name else first if second == cup_name else None
        if finger in counts:
            counts[finger] += len(contact.points)
    return counts


class SimulationContactMonitor:
    def __init__(self, initial_cup_z, *, stable_steps=3, gripper_threshold=0.25):
        if stable_steps < 1 or not np.isfinite(gripper_threshold) or not 0 < gripper_threshold <= 0.6:
            raise ValueError("Invalid early-contact thresholds")
        self.initial_cup_z = float(initial_cup_z)
        self.stable_steps = stable_steps
        self.gripper_threshold = gripper_threshold
        self.count = 0

    def update(self, cup_world, ee_world, gripper, contact_count, step, *, finger_contacts):
        bilateral = sum(count > 0 for count in finger_contacts.values()) >= 2
        eligible = bilateral and 0 <= gripper < self.gripper_threshold and contact_count > 0
        self.count = self.count+1 if eligible else 0
        return dict(ready=self.count >= self.stable_steps,
                    detector="simulation_bilateral_contact_before_trial_lift",
                    uses_simulator_truth=True, after_vla_step=int(step),
                    cup_lift_m=float(cup_world[2, 3]-self.initial_cup_z),
                    gripper=float(gripper), contact_count=int(contact_count),
                    finger_contacts=finger_contacts, bilateral_contact=bilateral,
                    stable_steps=self.count, required_stable_steps=self.stable_steps,
                    gripper_threshold=self.gripper_threshold,
                    gripper_signal_source="robot_control_state_not_measured_aperture_or_force",
                    lifted_grasp_verified=False)


def slower_trajectory(plan, factor=2):
    """At a fixed physics timestep, double the samples and halve velocities."""
    if not np.isfinite(factor) or factor < 1:
        raise ValueError("Trajectory slowdown must be finite and >=1")
    pos, vel = np.asarray(plan["position"]), np.asarray(plan["velocity"])
    if (pos.ndim != 2 or pos.shape != vel.shape or len(pos) < 2
            or not np.isfinite(pos).all() or not np.isfinite(vel).all()):
        raise ValueError("Invalid trial-lift trajectory")
    old = np.arange(len(pos))
    new = np.linspace(0, len(pos)-1, int(np.ceil((len(pos)-1)*factor))+1)
    effective_factor = (len(new)-1)/(len(pos)-1)
    result = dict(plan)
    for key, values, power in (("position", pos, 0), ("velocity", vel, 1)):
        result[key] = np.stack([np.interp(new, old, values[:, j])
                                for j in range(pos.shape[1])], axis=1)/effective_factor**power
    return result


def verify_trial_lift(task, arm, initial_cup_z, result, *, lift_m=0.05,
                      required_lift_m=0.03, stable_steps=3, slowdown=2, check_slip=True):
    from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix

    if (not np.isfinite(lift_m) or not 0.03 <= lift_m <= 0.10
            or not 0 < required_lift_m <= lift_m or stable_steps < 1):
        raise ValueError("Invalid trial-lift distance or verification thresholds")
    result.update(complete=False, uses_simulator_truth_for_verification=True,
                  uses_object_truth_for_target=False, lift_distance_m=lift_m,
                  slowdown_factor=slowdown, segments=[], stability_checks=[],
                  slip_check_enabled=check_slip,
                  relative_drift_role="gating" if check_slip else "diagnostic_only",
                  failure_policy="stop_without_release_or_placement")
    start = np.asarray(task.get_arm_pose(arm), dtype=float)
    relative_start = np.linalg.solve(pose_wxyz_to_matrix(start),
                                     task.cup.get_pose().to_transformation_matrix())[:3, 3]
    # Three small planned segments permit checks before continuing upward.
    for index in range(3):
        target = start.copy()
        target[2] += lift_m*(index+1)/3
        plan = getattr(task.robot, f"{arm}_plan_path")(target)
        audit = dict(target_ee_pose=target.tolist(), status=plan["status"],
                     planner_status=plan.get("planner_status"))
        result["segments"].append(audit)
        if plan["status"] != "Success":
            raise RuntimeError("Traditional trial lift planning failed")
        controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
        controls[f"{arm}_arm"] = slower_trajectory(plan, slowdown)
        moved = bool(task.take_dense_action(controls))
        arrival = handoff_ready(task.get_arm_pose(arm), target, moved and task.plan_success)
        audit["arrival"] = arrival.to_dict()
        cup = task.cup.get_pose().to_transformation_matrix()
        ee = pose_wxyz_to_matrix(task.get_arm_pose(arm))
        slip = float(np.linalg.norm(np.linalg.solve(ee, cup)[:3, 3]-relative_start))
        contacts = active_gripper_contacts(task, arm)
        gripper = getattr(task.robot, f"get_{arm}_gripper_val")()
        audit.update(relative_drift_from_contact_m=slip, contact_count=contacts,
                     gripper=float(gripper), cup_lift_m=float(cup[2, 3]-initial_cup_z))
        if not arrival.ready or (check_slip and slip > 0.01) or contacts <= 0 or not 0 <= gripper < 0.6:
            raise RuntimeError("Traditional trial lift failed arrival/contact/slip verification")
    monitor = SimulationGraspMonitor(initial_cup_z, lift_m=required_lift_m, stable_steps=stable_steps)
    decision = None
    contact_lift_count = 0
    for _ in range(stable_steps):
        # RoboTwin's delay() accepts integer command repetitions, not seconds.
        for _ in range(max(1, int(np.ceil(0.05/task.scene.get_timestep())))):
            task.scene.step()
        cup = task.cup.get_pose().to_transformation_matrix()
        ee = pose_wxyz_to_matrix(task.get_arm_pose(arm))
        drift = float(np.linalg.norm(np.linalg.solve(ee, cup)[:3, 3]-relative_start))
        gripper = getattr(task.robot, f"get_{arm}_gripper_val")()
        contacts = active_gripper_contacts(task, arm)
        if check_slip:
            decision = monitor.update(cup, ee, gripper, contacts, task.take_action_cnt)
        else:
            lift = float(cup[2, 3]-initial_cup_z)
            eligible = lift >= required_lift_m and contacts > 0 and 0 <= gripper < 0.6
            contact_lift_count = contact_lift_count+1 if eligible else 0
            decision = dict(ready=contact_lift_count >= stable_steps,
                detector="simulation_contact_lift_without_slip_check", uses_simulator_truth=True,
                after_vla_step=int(task.take_action_cnt), cup_lift_m=lift,
                gripper=float(gripper), contact_count=int(contacts), stable_steps=contact_lift_count,
                required_stable_steps=stable_steps, lift_threshold_m=required_lift_m)
        decision["slip_check_enabled"] = check_slip
        decision["relative_drift_from_contact_m"] = drift
        result["stability_checks"].append(dict(decision))
        if check_slip and drift > 0.01:
            raise RuntimeError("Traditional trial lift slipped during held stability checks")
    if not decision["ready"]:
        raise RuntimeError("Traditional trial lift did not verify a stable lifted grasp")
    decision.update(verification_controller="traditional_trial_lift", ee_pose=list(task.get_arm_pose(arm)))
    result.update(complete=True, lifted_grasp_verified=True)
    return decision
