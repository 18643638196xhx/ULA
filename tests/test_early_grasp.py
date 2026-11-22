from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tra_pipe/RoboTwin"))
from adapters.early_grasp import SimulationContactMonitor, finger_contact_counts, slower_trajectory, verify_trial_lift
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
from test_geometry_approach import runner


class ContactTests(unittest.TestCase):
    def sample(self, monitor, fingers=None, g=.1):
        cup = np.eye(4)
        cup[2, 3] = .74  # no lift at all
        return monitor.update(cup, np.eye(4), g, 4, 1,
                              finger_contacts=fingers or {"finger1": 2, "finger2": 2})

    def test_can_handoff_before_lift_but_only_after_three_bilateral_contacts(self):
        m = SimulationContactMonitor(.74)
        self.assertFalse(self.sample(m)["ready"])
        self.assertFalse(self.sample(m)["ready"])
        decision = self.sample(m)
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["cup_lift_m"], 0)
        self.assertFalse(decision["lifted_grasp_verified"])
        self.assertTrue(decision["uses_simulator_truth"])

    def test_unilateral_contact_or_open_gripper_resets(self):
        m = SimulationContactMonitor(.74, stable_steps=2)
        self.sample(m)
        self.assertFalse(self.sample(m, {"finger1": 4, "finger2": 0})["ready"])
        self.assertFalse(self.sample(m)["ready"])
        self.assertFalse(self.sample(m, g=.3)["ready"])
        self.assertFalse(self.sample(m)["ready"])
        self.assertTrue(self.sample(m)["ready"])

    def test_other_arm_palm_and_non_cup_contacts_do_not_count_as_fingers(self):
        joint = lambda name: (SimpleNamespace(child_link=SimpleNamespace(get_name=lambda: name)),)
        body = lambda name: SimpleNamespace(entity=SimpleNamespace(name=name))
        contact = lambda x, y: SimpleNamespace(bodies=[body(x), body(y)], points=[0, 1])
        task = SimpleNamespace(cup=SimpleNamespace(get_name=lambda: "cup"),
            robot=SimpleNamespace(left_gripper=[joint("finger1"), joint("finger2")]),
            scene=SimpleNamespace(get_contacts=lambda: [contact("cup", "finger1"),
                  contact("palm", "cup"), contact("cup", "other_arm"), contact("table", "finger2")]))
        self.assertEqual(finger_contact_counts(task, "left"), {"finger1": 2, "finger2": 0})

    def test_default_factory_keeps_old_lift_monitor(self):
        from adapters.grasp import SimulationGraspMonitor
        self.assertIsInstance(runner.make_grasp_monitor(
            SimpleNamespace(grasp_lift=.03, grasp_stable_steps=3), .74), SimulationGraspMonitor)

    def test_contact_interrupt_is_not_marked_as_verified_grasp(self):
        cup = np.eye(4)
        cup[2, 3] = .74
        task = SimpleNamespace(take_action_cnt=0, step_lim=100, eval_success=False,
                               cup=SimpleNamespace(get_pose=lambda: SimpleNamespace(p=cup[:3, 3],
                                   to_transformation_matrix=lambda: cup)),
                               robot=SimpleNamespace(get_left_gripper_val=lambda: .1),
                               get_arm_pose=lambda arm: [0, 0, 1, 1, 0, 0, 0], get_obs=lambda: {})
        def action(value):
            task.take_action_cnt += 1
        task.take_action = action
        def chunk(task, observation):
            for _ in range(10):
                task.take_action(np.zeros(14))
        record = dict(evaluation_only=dict(max_cup_lift_m=0), grasp_handoff=None)
        with patch.object(runner, "active_gripper_contacts", return_value=4), \
                patch.object(runner, "finger_contact_counts", return_value={"a": 2, "b": 2}):
            runner.rollout(task, SimpleNamespace(reset=lambda: None, step=chunk), .74,
                           record, SimulationContactMonitor(.74), "left")
        self.assertEqual(task.take_action_cnt, 3)
        self.assertIsNone(record["grasp_handoff"])
        self.assertTrue(record["contact_handoff"]["ready"])
        self.assertIs(task.take_action, action)


class TrialLiftTests(unittest.TestCase):
    def setUp(self):
        self.ee = np.array([0., 0., 1., 1., 0., 0., 0.])
        self.cup = np.eye(4)
        self.cup[2, 3] = .74
        self.follow = True
        self.fail_plan = False
        self.miss_arrival = False
        self.held_steps = 0
        self.targets = []
        self.task = SimpleNamespace(plan_success=True, take_action_cnt=41,
            get_arm_pose=lambda arm: self.ee.copy(),
            cup=SimpleNamespace(get_pose=lambda: SimpleNamespace(to_transformation_matrix=lambda: self.cup.copy())),
            robot=SimpleNamespace(left_plan_path=self.plan, get_left_gripper_val=lambda: .1),
            take_dense_action=self.execute,
            scene=SimpleNamespace(get_timestep=lambda: .004, step=self.physics))

    def physics(self):
        self.held_steps += 1

    def plan(self, target):
        self.targets.append(np.array(target))
        return dict(status="Fail" if self.fail_plan else "Success", target=np.array(target),
                    position=np.zeros((3, 6)), velocity=np.zeros((3, 6)))

    def execute(self, controls):
        new = controls["left_arm"]["target"]
        if not self.miss_arrival:
            if self.follow:
                self.cup[:3, 3] += new[:3]-self.ee[:3]
            self.ee[:] = new
        return True

    def run_trial(self, **kwargs):
        self.result = {}
        with patch("adapters.early_grasp.active_gripper_contacts", return_value=4):
            return verify_trial_lift(self.task, "left", .74, self.result, **kwargs)

    def test_disabled_slip_gate_still_rejects_no_lift(self):
        self.follow = False
        with self.assertRaisesRegex(RuntimeError, "did not verify"):
            self.run_trial(check_slip=False)
        self.assertFalse(self.result["slip_check_enabled"])
        self.assertEqual(len(self.targets), 3)

    def test_disabled_slip_gate_does_not_check_absolute_or_held_window_drift(self):
        original_physics = self.task.scene.step
        def moving_cup():
            original_physics()
            self.cup[0, 3] += .001  # 13 mm between each held check
        self.task.scene.step = moving_cup
        decision = self.run_trial(check_slip=False)
        self.assertTrue(decision["ready"])
        self.assertGreater(decision["relative_drift_from_contact_m"], .03)
        self.assertFalse(decision["slip_check_enabled"])
        self.assertEqual(self.result["relative_drift_role"], "diagnostic_only")

    def test_disabled_slip_gate_still_rejects_missing_arrival_or_contact(self):
        self.miss_arrival = True
        with self.assertRaisesRegex(RuntimeError, "arrival/contact/slip"):
            self.run_trial(check_slip=False)
        self.miss_arrival = False
        self.result = {}
        with patch("adapters.early_grasp.active_gripper_contacts", return_value=0), \
                self.assertRaisesRegex(RuntimeError, "arrival/contact/slip"):
            verify_trial_lift(self.task, "left", .74, self.result, check_slip=False)

    def test_verified_trial_lifts_with_fixed_robot_target_and_preserves_vla_count(self):
        decision = self.run_trial()
        self.assertTrue(decision["ready"])
        self.assertTrue(self.result["complete"])
        self.assertFalse(self.result["uses_object_truth_for_target"])
        np.testing.assert_allclose(self.ee[:3], [0, 0, 1.05])
        self.assertEqual(len(self.targets), 3)
        self.assertEqual(self.task.take_action_cnt, 41)
        self.assertEqual(self.held_steps, 39)

    def test_plan_failure_never_executes_or_verifies(self):
        self.fail_plan = True
        with self.assertRaisesRegex(RuntimeError, "planning failed"):
            self.run_trial()
        self.assertFalse(self.result["complete"])
        self.assertAlmostEqual(self.ee[2], 1.)

    def test_empty_grasp_or_slip_rejected_before_placement(self):
        self.follow = False
        with self.assertRaisesRegex(RuntimeError, "slip verification"):
            self.run_trial()
        self.assertFalse(self.result["complete"])
        self.assertEqual(len(self.targets), 1)

    def test_missing_arrival_rejected(self):
        self.miss_arrival = True
        with self.assertRaisesRegex(RuntimeError, "arrival/contact/slip"):
            self.run_trial()
        self.assertFalse(self.result["complete"])

    def test_lost_contact_rejected(self):
        self.result = {}
        with patch("adapters.early_grasp.active_gripper_contacts", return_value=0), \
                self.assertRaisesRegex(RuntimeError, "arrival/contact/slip"):
            verify_trial_lift(self.task, "left", .74, self.result)
        self.assertFalse(self.result["complete"])

    def test_slowdown_preserves_endpoints_and_scales_velocity_without_mutation(self):
        plan = dict(position=np.array([[0., 1.], [1., 2.], [2., 3.]]), velocity=np.ones((3, 2)))
        slow = slower_trajectory(plan)
        self.assertEqual(len(slow["position"]), 5)
        np.testing.assert_allclose(slow["position"][[0, -1]], plan["position"][[0, -1]])
        np.testing.assert_allclose(slow["velocity"], .5)
        np.testing.assert_allclose(plan["velocity"], 1.)

    def test_invalid_slowdown_rejected(self):
        with self.assertRaises(ValueError):
            slower_trajectory({}, 0)


if __name__ == "__main__":
    unittest.main()
