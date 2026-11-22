import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from adapters.visual_grasp import (ClosedCommandMonitor, ClosedCommandRobotLiftMonitor,
                                   ObjectTruthTerminationGuard, verify_visual_trial_lift,
                                   visual_comotion)
from test_geometry_approach import runner


class ForbiddenTruth:
    def __getattr__(self, name):
        raise AssertionError(f"Controller attempted object/contact truth: {name}")


class VisualGraspTests(unittest.TestCase):
    def test_command_candidate_is_not_a_grasp_certificate_and_resets(self):
        monitor = ClosedCommandMonitor(stable_steps=2)
        self.assertFalse(monitor.update(.01, 1)["ready"])
        self.assertFalse(monitor.update(.2, 2)["ready"])
        self.assertFalse(monitor.update(.01, 3)["ready"])
        candidate = monitor.update(.01, 4)
        self.assertTrue(candidate["ready"])
        self.assertFalse(candidate["lifted_grasp_verified"])
        self.assertFalse(candidate["uses_simulator_truth"])

    def test_robot_lift_monitor_does_not_stop_at_closure_alone(self):
        pose = np.array([0., 0., .8, 1., 0., 0., 0.])
        monitor = ClosedCommandRobotLiftMonitor(
            lambda: pose, minimum_upward_m=.03, stable_steps=2,
            gripper_threshold=.05)
        self.assertFalse(monitor.update(.01, 1)["ready"])
        closed = monitor.update(.01, 2)
        self.assertFalse(closed["ready"])
        self.assertTrue(closed["closure_gate_reached"])
        pose[2] += .02
        self.assertFalse(monitor.update(.01, 3)["ready"])
        pose[2] += .011
        lifted = monitor.update(.01, 4)
        self.assertTrue(lifted["ready"])
        self.assertFalse(lifted["lifted_grasp_verified"])
        self.assertEqual(lifted["verification_required"],
                         "fresh_RGBD_object_hand_comotion")

    def test_guard_blocks_runtime_truth_and_restores_on_exception(self):
        class Task:
            eval_success = True
            def check_success(self):
                raise AssertionError("Final scorer must not run during control")
        task = Task()
        with self.assertRaisesRegex(RuntimeError, "controller failed"):
            with ObjectTruthTerminationGuard(task):
                self.assertFalse(task.check_success())
                self.assertFalse(task.eval_success)
                raise RuntimeError("controller failed")
        self.assertNotIn("check_success", vars(task))
        self.assertTrue(task.eval_success)
        with self.assertRaises(AssertionError):
            task.check_success()

    def test_guard_restores_instance_override_idempotently(self):
        scorer = lambda: True
        task = SimpleNamespace(check_success=scorer, eval_success=False)
        guard = ObjectTruthTerminationGuard(task)
        with guard:
            self.assertFalse(task.check_success())
            guard.close()
            self.assertIs(task.check_success, scorer)
        self.assertIs(task.check_success, scorer)

    def test_visual_rollout_never_reads_pose_contacts_or_lift_truth(self):
        task = SimpleNamespace(take_action_cnt=0, step_lim=100, eval_success=False,
            cup=ForbiddenTruth(), scene=ForbiddenTruth(),
            robot=SimpleNamespace(get_left_gripper_val=lambda: .01),
            get_arm_pose=lambda arm: [0, 0, 1, 1, 0, 0, 0], get_obs=lambda: {})
        def action(value):
            task.take_action_cnt += 1
        task.take_action = action
        def chunk(task, obs):
            for _ in range(10):
                task.take_action(np.zeros(14))
        record = dict(evaluation_only={}, grasp_handoff=None)
        with patch.object(runner, "active_gripper_contacts", side_effect=AssertionError("No contacts")), \
                patch.object(runner, "finger_contact_counts", side_effect=AssertionError("No contacts")):
            runner.rollout(task, SimpleNamespace(reset=lambda: None, step=chunk), None,
                           record, ClosedCommandMonitor(), "left")
        self.assertEqual(task.take_action_cnt, 3)
        self.assertTrue(record["candidate_handoff"]["ready"])
        self.assertIsNone(record["grasp_handoff"])
        self.assertEqual(record["evaluation_only"], {})
        self.assertIs(task.take_action, action)

    def test_surface_motion_not_transform_translation_and_stationary_rejected(self):
        points = np.random.default_rng(2).uniform(-.02, .02, (100, 3))+[1, 1, 1]
        ee = [0, 0, .05]
        motion = np.eye(4)
        self.assertFalse(visual_comotion(points, motion, ee)["ready"])
        motion[2, 3] = .05
        self.assertTrue(visual_comotion(points, motion, ee)["ready"])
        motion[2, 3] = -.05
        self.assertFalse(visual_comotion(points, motion, ee)["ready"])
        motion[2, 3] = .12
        self.assertFalse(visual_comotion(points, motion, ee)["ready"])
        motion[2, 3] = .05
        motion[:2, :2] = [[0, -1], [1, 0]]
        motion[0, 3] = 2  # rotate about [1,1], not about world origin
        self.assertTrue(visual_comotion(points, motion, ee)["ready"])

    def trial(self, *, follow=True, invalid_baseline=False, head_bad_after=False):
        ee = np.array([0., 0., 1., 1., 0., 0., 0.])
        initial = ee.copy()
        targets, calls = [], []
        points = np.random.default_rng(7).uniform(-.03, .03, (400, 3))+[0, 0, .8]
        def plan(target):
            targets.append(target.copy())
            return dict(status="Success", target=target.copy(),
                        position=np.zeros((3, 6)), velocity=np.zeros((3, 6)))
        def execute(controls):
            ee[:] = controls["left_arm"]["target"]
            return True
        def observe(frame, label, **kwargs):
            calls.append(label)
            if invalid_baseline or (head_bad_after and "after_head" in label):
                raise RuntimeError("Segmentation uncertain")
            cloud = points.copy()
            if "after" in label and follow:
                cloud += ee[:3]-initial[:3]
            return SimpleNamespace(cup_points_world=cloud,
                cup_bounds_world=np.array([cloud.min(0), cloud.max(0)]), observation_audit={})
        task = SimpleNamespace(cup=ForbiddenTruth(), scene=ForbiddenTruth(),
            robot=SimpleNamespace(left_plan_path=plan), get_arm_pose=lambda arm: ee.copy(),
            get_camera_observation=lambda name: {"camera": name},
            take_dense_action=execute, plan_success=True)
        result = {}
        try:
            decision = verify_visual_trial_lift(task, SimpleNamespace(supplementary=observe), "left", result)
        except RuntimeError:
            if not invalid_baseline and follow:
                raise
            decision = None
        return result, decision, targets, calls

    def test_complete_trial_without_any_object_or_contact_access(self):
        result, decision, targets, _ = self.trial()
        self.assertTrue(result["complete"])
        self.assertTrue(decision["ready"])
        self.assertFalse(result["slip_check_enabled"])
        self.assertEqual(len(targets), 3)

    def test_empty_grasp_stops_without_placement_permission(self):
        result, decision, _, _ = self.trial(follow=False)
        self.assertIsNone(decision)
        self.assertFalse(result["complete"])

    def test_untrusted_baseline_causes_no_motion(self):
        result, decision, targets, _ = self.trial(invalid_baseline=True)
        self.assertIsNone(decision)
        self.assertEqual(targets, [])
        self.assertFalse(result["complete"])

    def test_fallback_has_matching_wrist_before_after_pair(self):
        result, decision, _, calls = self.trial(head_bad_after=True)
        self.assertTrue(result["complete"])
        self.assertEqual(decision["matching_camera"], "left_camera")
        self.assertIn("trial_before_left_camera", calls)
        self.assertIn("trial_after_left_camera_0", calls)

    def test_batch_truth_only_at_final_scoring_and_guard_restored_on_failure(self):
        for fail_trial in (False, True):
            with self.subTest(fail_trial=fail_trial), tempfile.TemporaryDirectory() as tmp:
                scored = []
                def scorer():
                    scored.append(True)
                    return True
                def final_point(*args):
                    self.assertTrue(scored, "Functional points allowed only after final scoring")
                    return SimpleNamespace(p=np.zeros(3))
                task = SimpleNamespace(eval_success=False, cup=SimpleNamespace(
                    get_pose=lambda: self.fail("No initial or online cup pose"),
                    get_functional_point=final_point),
                    coaster=SimpleNamespace(get_functional_point=final_point),
                    plan_success=True, take_action_cnt=0, setup_demo=lambda **kw: None,
                    set_instruction=lambda s: None, check_success=scorer,
                    is_left_gripper_open=lambda: True, is_right_gripper_open=lambda: True,
                    close_env=lambda: None)
                def no_truth(*args, **kw):
                    self.assertFalse(task.check_success())
                    self.assertFalse(scored)
                def candidate(task, policy, z, record, monitor, arm, stall_check=None):
                    no_truth()
                    self.assertIsNone(z)
                    self.assertIsInstance(monitor, ClosedCommandMonitor)
                    record["candidate_handoff"] = dict(ready=True)
                def trial(*args, **kw):
                    no_truth()
                    if fail_trial:
                        raise RuntimeError("Uncertain visual trial")
                    return dict(ready=True)
                env = ModuleType("envs.place_empty_cup")
                env.place_empty_cup = lambda: task
                controller = ModuleType("traditional_place_empty_cup.controller")
                controller.TraditionalPlaceEmptyCupController = None
                policy = SimpleNamespace(load=lambda: None, model=object())
                old_cwd = Path.cwd()
                try:
                    with patch.object(sys, "argv", ["runner", "--output", tmp,
                        "--mode", "hybrid", "--approach-mode", "pointcloud",
                        "--grasp-handoff", "visual-trial-lift"]):
                        args = runner.parse_args()
                    with patch.dict(sys.modules, {"envs.place_empty_cup": env,
                        "traditional_place_empty_cup.controller": controller}), \
                        patch.object(runner, "task_config", return_value={}), \
                        patch.object(runner, "WorldLA", return_value=policy), \
                        patch.object(runner, "snapshot"), \
                        patch.object(runner, "traditional_timing", return_value=nullcontext()), \
                        patch.object(runner, "vla_timing", return_value=nullcontext()), \
                        patch.object(runner, "approach", side_effect=lambda *a: (no_truth() or dict(ready=True, arm="left"))), \
                        patch.object(runner, "rollout", side_effect=candidate), \
                        patch.object(runner, "verify_visual_trial_lift", side_effect=trial), \
                        patch.object(runner, "finish_pointcloud_placement", side_effect=no_truth), \
                        patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                        status = runner.run_batch(args, SimpleNamespace(fp_iterations=5, summary=lambda: {}))
                    self.assertEqual(status, int(fail_trial))
                    self.assertIs(task.check_success, scorer)
                    self.assertEqual(len(scored), 0 if fail_trial else 1)
                    report = runner.json.loads((Path(tmp)/"report.json").read_text())
                    self.assertFalse(report["control_uses_object_truth"])
                    self.assertEqual(report["successes"], 0 if fail_trial else 1)
                finally:
                    os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
