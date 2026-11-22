from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.handover.pickup_stage import PickupProgress, rollout_pickup
from adapters.handover.pickup_audit import lowest_surface_z, PickupAudit


def matrix(z):
    value = np.eye(4)
    value[2, 3] = z
    return value


class PickupTests(unittest.TestCase):
    def test_closing_alone_never_finishes(self):
        progress = PickupProgress()
        self.assertFalse(any(progress.update(0., 1.) for _ in range(20)))

    def test_hand_rise_finishes_but_reopening_resets_reference(self):
        progress = PickupProgress()
        self.assertFalse(progress.update(0., 1.))
        self.assertFalse(progress.update(0., 1.))
        self.assertFalse(progress.update(0., 1.))
        for _ in range(4):
            self.assertFalse(progress.update(0., 1.10))
        self.assertTrue(progress.update(0., 1.10))
        self.assertFalse(progress.update(1., 1.10))
        self.assertFalse(progress.update(0., 1.10))

    def test_lowest_surface_accounts_for_rotation(self):
        horizontal = matrix(.77)
        horizontal[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        self.assertAlmostEqual(lowest_surface_z(horizontal), .74)
        self.assertAlmostEqual(lowest_surface_z(matrix(.84)), .74)
        audit = PickupAudit(horizontal)
        for step in range(10):
            audit.observe(matrix(.84), matrix(.9), 0., step)
        self.assertFalse(audit.report()["pickup_success"])

    def test_success_requires_sustained_clearance_and_still_held_at_end(self):
        audit = PickupAudit(matrix(.84))
        for step in range(4):
            audit.observe(matrix(.89), matrix(.96), 0., step)
        self.assertFalse(audit.report()["pickup_success"])
        audit.observe(matrix(.89), matrix(.96), 0., 4)
        self.assertTrue(audit.report()["pickup_success"])
        audit.observe(matrix(.84), matrix(.96), 0., 5)
        self.assertTrue(audit.report()["ever_sustained_pickup"])
        self.assertFalse(audit.report()["pickup_success"])

    def test_open_hand_or_far_object_is_not_pickup(self):
        for command, hand_z in [(1., .96), (0., 1.3)]:
            audit = PickupAudit(matrix(.84))
            for step in range(10):
                audit.observe(matrix(.89), matrix(hand_z), command, step)
            self.assertFalse(audit.report()["pickup_success"])

    def test_controller_has_no_object_access_and_cannot_overrun_budget(self):
        class Forbidden:
            def __getattr__(self, name):
                raise AssertionError("Controller accessed object truth")
        class Task:
            box, scene = Forbidden(), Forbidden()
            take_action_cnt, step_lim = 0, 7
            robot = SimpleNamespace(get_left_gripper_val=lambda: 0., get_right_gripper_val=lambda: 1.)
            def get_arm_pose(self, arm):
                return [0, 0, 1., 1, 0, 0, 0]
            def get_obs(self):
                return {}
            def take_action(self, values):
                self.take_action_cnt += 1
        class Policy:
            def reset(self):
                pass
            def step(self, task, obs):
                for _ in range(10):
                    task.take_action(np.zeros(14))
        task, record = Task(), {}
        original = task.take_action
        rollout_pickup(task, Policy(), record)
        self.assertEqual(task.take_action_cnt, 7)
        self.assertFalse(record["robot_endpoint_reached"])
        self.assertEqual(task.take_action, original)


if __name__ == "__main__":
    unittest.main()
