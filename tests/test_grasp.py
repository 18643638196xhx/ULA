import unittest

import numpy as np

from adapters.grasp import SimulationGraspMonitor


class GraspTests(unittest.TestCase):
    def sample(self, monitor, z=0.78, contacts=2, gripper=0.1, step=1, x=0):
        cup = np.eye(4)
        cup[:3, 3] = [x, 0, z]
        return monitor.update(cup, np.eye(4), gripper, contacts, step)

    def test_contact_and_closed_gripper_alone_do_not_prove_lift(self):
        monitor = SimulationGraspMonitor(0.74, stable_steps=1)
        self.assertFalse(self.sample(monitor, z=0.74)["ready"])

    def test_requires_continuous_stable_lifted_contact(self):
        monitor = SimulationGraspMonitor(0.74, stable_steps=3)
        self.assertFalse(self.sample(monitor)["ready"])
        self.assertFalse(self.sample(monitor, contacts=0)["ready"])
        self.assertFalse(self.sample(monitor)["ready"])
        self.assertFalse(self.sample(monitor)["ready"])
        self.assertTrue(self.sample(monitor)["ready"])

    def test_open_gripper_and_relative_slip_reset_the_gate(self):
        monitor = SimulationGraspMonitor(0.74, stable_steps=2)
        self.sample(monitor)
        self.assertFalse(self.sample(monitor, x=0.03)["ready"])
        self.assertFalse(self.sample(monitor, x=0.03, gripper=1)["ready"])


if __name__ == "__main__":
    unittest.main()
