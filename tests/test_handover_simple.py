import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.handover.simple_control import (assumed_attachment, block_metadata,
    propagated_object, rollout_phase, RigidAttachmentPerception)


class ForbiddenObject:
    def __getattr__(self, name):
        raise AssertionError("Minimal command handoff must not inspect object truth or contacts")


class FakeTask:
    def __init__(self, limit=30):
        self.box, self.scene = ForbiddenObject(), ForbiddenObject()
        self.take_action_cnt, self.step_lim = 0, limit
        self.left_command, self.right_command = 1., 1.
        self.robot = SimpleNamespace(get_left_gripper_val=lambda: self.left_command,
                                     get_right_gripper_val=lambda: self.right_command)
    def get_arm_pose(self, arm):
        return ([0., 0., 1.07, 1., 0., 0., 0.] if arm == "left" else
                [.1, 0., .93, 1., 0., 0., 0.])
    def get_obs(self):
        return {}
    def take_action(self, action):
        self.take_action_cnt += 1
        self.left_command, self.right_command = float(action[6]), float(action[13])


class FakePolicy:
    def __init__(self, commands):
        self.commands = iter(commands)
        self.resets = 0
    def reset(self):
        self.resets += 1
    def step(self, task, observation):
        left, right = next(self.commands)
        action = np.zeros(14)
        action[6], action[13] = left, right
        task.take_action(action)


class HandoverSimpleTests(unittest.TestCase):
    def test_public_primitive_contacts_and_bottom_scale(self):
        metadata = block_metadata()
        self.assertEqual(len(metadata["contact_points_pose"]), 8)
        self.assertAlmostEqual(metadata["contact_points_pose"][0][2][3]*metadata["scale"][2], .07)
        self.assertEqual(metadata["functional_matrix"][0][2][3]*metadata["scale"][2], -.1)

    def test_attachment_propagates_robot_motion_not_object_truth(self):
        hand = np.eye(4)
        hand[:3, 3] = [.1, .2, .3]
        block = np.eye(4)
        block[:3, 3] = [.15, .2, .3]
        attachment = assumed_attachment(hand, block)
        task = SimpleNamespace(get_arm_pose=lambda arm: [.2, .2, .3, 1., 0., 0., 0.], box=ForbiddenObject())
        np.testing.assert_allclose(propagated_object(task, "left", attachment)[:3, 3], [.25, .2, .3])

    def test_left_grasp_uses_only_three_closed_commands(self):
        task, record = FakeTask(), {}
        original = task.take_action
        policy = FakePolicy([(0., 1.)]*3)
        rollout_phase(task, policy, "left_grasp", record)
        self.assertTrue(record["ready"])
        self.assertFalse(record["handoff_verified"])
        self.assertFalse(record["trial_lift"])
        self.assertEqual(task.take_action_cnt, 3)
        self.assertEqual(task.take_action, original)
        self.assertEqual(policy.resets, 1)

    def test_right_close_and_vla_left_release_are_not_grasp_confirmation(self):
        task, record = FakeTask(), {}
        # After three right-close commands VLA itself opens the left hand.
        policy = FakePolicy([(0., 0.)]*3+[(1., 0.)]*3)
        attachment = rollout_phase(task, policy, "handover", record, left_attachment=np.eye(4))
        self.assertTrue(record["ready"])
        self.assertFalse(record["handoff_verified"])
        self.assertFalse(record["uses_contacts"])
        self.assertEqual(task.take_action_cnt, 6)
        self.assertEqual(record["right_closed_candidate_step"], 3)
        self.assertEqual(attachment.shape, (4, 4))

    def test_closed_right_without_vla_left_release_does_not_end_phase(self):
        task, record = FakeTask(limit=4), {}
        with self.assertRaisesRegex(RuntimeError, "shared VLA budget"):
            rollout_phase(task, FakePolicy([(0., 0.)]*4), "handover", record, left_attachment=np.eye(4))
        self.assertFalse(record["ready"])

    def test_two_phases_share_counter_and_total_budget(self):
        task = FakeTask(limit=4)
        rollout_phase(task, FakePolicy([(0., 1.)]*3), "left_grasp", {})
        with self.assertRaisesRegex(RuntimeError, "shared VLA budget"):
            rollout_phase(task, FakePolicy([(1., 0.)]), "handover", {}, left_attachment=np.eye(4))
        self.assertEqual(task.take_action_cnt, 4)

    def test_bad_action_restores_original_callback(self):
        task = FakeTask()
        original = task.take_action
        policy = SimpleNamespace(reset=lambda: None, step=lambda task, obs: task.take_action(np.zeros(13)))
        with self.assertRaisesRegex(ValueError, "14-D"):
            rollout_phase(task, policy, "left_grasp", {})
        self.assertEqual(task.take_action, original)

    def test_state_estimate_is_explicitly_not_fresh_vision(self):
        estimate = RigidAttachmentPerception(FakeTask(), np.eye(4), [.2, .1, .75]).estimate()
        self.assertFalse(estimate.observation_audit["fresh_object_measurement"])
        np.testing.assert_allclose(estimate.coaster_world_center, [.2, .1, .75])

    def test_valid_denominator_keeps_initial_perception_failure(self):
        spec = importlib.util.spec_from_file_location("handover_simple_test_runner",
            ROOT/"scripts/run_handover_simple_randomized.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        records = [dict(seed=0, initialized=False, success=False, elapsed_s=1.),
            dict(seed=1, initialized=True, success=False, task_success=False, failure_stage="initial_perception", elapsed_s=1.),
            dict(seed=2, initialized=True, success=True, task_success=True, controller_complete=True, elapsed_s=8.)]
        result = runner.summarize(records)
        self.assertEqual(result["valid_scenes"], 2)
        self.assertEqual(result["success_rate"], .5)
        self.assertEqual(result["initialization_rejection_seeds"], [0])
        self.assertEqual(result["failure_stages"], {"initial_perception":1})


if __name__ == "__main__":
    unittest.main()
