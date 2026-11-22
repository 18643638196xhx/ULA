"""Simulation-only grasp completion monitor for the minimal RoboTwin experiment.

Object pose/contact truth is used to detect grasp completion, NOT as the
placement planner's geometry. This detector must be replaced for a real robot.
"""

import numpy as np


class GraspComplete(Exception):
    """Interrupt the current VLA chunk exactly after a completed action."""


class SimulationGraspMonitor:
    def __init__(self, initial_cup_z, *, lift_m=0.03, stable_steps=3,
                 relative_tolerance_m=0.01):
        if lift_m <= 0 or stable_steps < 1 or relative_tolerance_m <= 0:
            raise ValueError("Invalid grasp thresholds")
        self.initial_cup_z = float(initial_cup_z)
        self.lift_m = lift_m
        self.stable_steps = stable_steps
        self.relative_tolerance_m = relative_tolerance_m
        self.anchor = None
        self.count = 0

    def update(self, cup_world, ee_world, gripper, contact_count, step):
        cup = np.asarray(cup_world, dtype=float)
        ee = np.asarray(ee_world, dtype=float)
        if (cup.shape != (4, 4) or ee.shape != (4, 4)
                or not np.isfinite(cup).all() or not np.isfinite(ee).all()):
            raise ValueError("Expected finite 4x4 cup/EE transforms")
        lift = float(cup[2, 3] - self.initial_cup_z)
        relative = np.linalg.solve(ee, cup)[:3, 3]
        eligible = lift >= self.lift_m and 0 <= gripper < 0.6 and contact_count > 0
        deviation = 0.0
        if not eligible:
            self.anchor = None
            self.count = 0
        else:
            if self.anchor is not None:
                deviation = float(np.linalg.norm(relative - self.anchor))
            if self.anchor is None or deviation > self.relative_tolerance_m:
                self.anchor = relative.copy()
                self.count = 1
            else:
                self.count += 1
        return dict(
            ready=self.count >= self.stable_steps,
            detector="simulation_contact_lift_and_relative_position",
            uses_simulator_truth=True, after_vla_step=int(step), cup_lift_m=lift,
            gripper=float(gripper), contact_count=int(contact_count),
            stable_steps=self.count, required_stable_steps=self.stable_steps,
            relative_position_drift_m=deviation,
            lift_threshold_m=self.lift_m,
            relative_tolerance_m=self.relative_tolerance_m,
        )


def active_gripper_contacts(task, arm):
    """Count cup contacts with the selected arm, not the stationary other arm."""
    joints = getattr(task.robot, f"{arm}_gripper")
    names = {item[0].child_link.get_name() for item in joints}
    names.update(getattr(task.robot, f"{arm}_fix_gripper_name"))
    cup_name = task.cup.get_name()
    count = 0
    for contact in task.scene.get_contacts():
        first, second = (body.entity.name for body in contact.bodies)
        if ((first == cup_name and second in names)
                or (second == cup_name and first in names)):
            count += len(contact.points)
    return count
