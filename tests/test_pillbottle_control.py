import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.pillbottle_control import (audit_topp, measured_joint_input, robot_state,
    settle_at_target, settle_closed_gripper)


class FakeRobot:
    def get_left_arm_jointState(self): return [1.]*6+[.8]
    def get_right_arm_jointState(self): return [2.]*6+[.9]
    def get_left_arm_real_jointState(self): return [.5]*6+[.8]
    def get_right_arm_real_jointState(self): return [1.5]*6+[.9]


class ControlTests(unittest.TestCase):
    def test_measured_input_retains_command_diagnostics_and_restores_class_methods(self):
        robot = FakeRobot()
        with self.assertRaisesRegex(RuntimeError, "test"):
            with measured_joint_input(robot):
                self.assertEqual(robot.get_left_arm_jointState()[0], .5)
                value = robot_state(robot)
                self.assertEqual(value["commanded"][0], 1.)
                self.assertEqual(value["measured"][0], .5)
                self.assertEqual(value["max_joint_tracking_error_rad"], .5)
                raise RuntimeError("test")
        self.assertEqual(robot.get_left_arm_jointState()[0], 1.)
        self.assertNotIn("get_left_arm_jointState", vars(robot))
        self.assertFalse(hasattr(robot, "_ula_command_state_getters"))

    def test_original_instance_override_is_restored_and_disabled_is_noop(self):
        robot = FakeRobot()
        getter = lambda: [3.]*6+[.8]
        robot.get_left_arm_jointState = getter
        with measured_joint_input(robot, False):
            self.assertIs(robot.get_left_arm_jointState, getter)
        with measured_joint_input(robot):
            self.assertEqual(robot.get_left_arm_jointState()[0], .5)
        self.assertIs(robot.get_left_arm_jointState, getter)

    def test_topp_failures_are_visible_and_original_is_restored(self):
        class Planner:
            def TOPP(self, path): raise ValueError("bad path")
        robot = SimpleNamespace(left_mplib_planner=Planner(), right_mplib_planner=Planner())
        events = []
        with audit_topp(robot, events):
            with self.assertRaises(ValueError): robot.left_mplib_planner.TOPP(None)
        self.assertFalse(events[0]["success"])
        self.assertIn("bad path", events[0]["error"])
        self.assertNotIn("TOPP", vars(robot.left_mplib_planner))

    def test_settling_waits_for_actual_position_not_command_and_is_bounded(self):
        for follows in (True, False):
            qpos = np.zeros(6)
            command = np.ones(6)
            counter = [0]
            def set_joints(q, vel, arm):
                command[:] = q
                np.testing.assert_array_equal(vel, np.zeros(6))
            def step():
                counter[0] += 1
                if follows: qpos[:] += .5*(command-qpos)
            robot = SimpleNamespace(set_arm_joints=set_joints,
                get_left_arm_real_jointState=lambda: qpos.tolist()+[.8])
            task = SimpleNamespace(robot=robot, plan_success=True,
                get_arm_pose=lambda arm: [float(qpos[0]), 0, 1, 1, 0, 0, 0],
                scene=SimpleNamespace(step=step))
            audit = {}
            ready = settle_at_target(task, "left", [1, 0, 1, 1, 0, 0, 0], np.ones(6), audit,
                max_steps=100, check_every=10)
            self.assertEqual(ready, follows)
            self.assertEqual(audit["ready"], follows)
            self.assertLessEqual(counter[0], 100)
            self.assertGreaterEqual(counter[0], 20)

    def test_measured_finger_settling_is_bounded_and_does_not_claim_a_grasp(self):
        for follows in (True, False):
            fingers = np.array([.03, .03])
            joints = [object(), object()]
            count = [0]
            def step():
                count[0] += 1
                fingers[:] = .005 if follows else (.03 if count[0] % 20 == 0 else .01)
            entity = SimpleNamespace(get_active_joints=lambda: joints, get_qpos=lambda: fingers.copy())
            robot = SimpleNamespace(left_entity=entity, left_gripper=[(j, 1, 0) for j in joints],
                get_left_arm_real_jointState=lambda: [0.]*6+[0.],
                set_arm_joints=lambda q, vel, arm: None, set_gripper=lambda value, arm: None)
            audit = {}
            ready = settle_closed_gripper(SimpleNamespace(robot=robot, scene=SimpleNamespace(step=step)),
                "left", audit, max_steps=100)
            self.assertEqual(ready, follows)
            self.assertFalse(audit["grasp_verified"])
            self.assertLessEqual(count[0], 100)


if __name__ == "__main__": unittest.main()
