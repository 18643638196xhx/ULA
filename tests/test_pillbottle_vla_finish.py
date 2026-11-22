from contextlib import nullcontext
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.pillbottle_vla_finish import visual_finish_decision, rollout_finish


class VLAFinishTests(unittest.TestCase):
    def value(self):
        pose = np.eye(4)
        pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        pose[2, 3] = .74
        return SimpleNamespace(cup_world_pose=pose, coaster_world_center=np.array([0., 0., .74]),
            cup_bounds_world=np.array([[-.035, -.035, .74], [.035, .035, .89]]))

    def metadata(self):
        return dict(scale=[1., 1., 1.], functional_matrix=[np.eye(4).tolist()])

    def test_visual_finish_needs_support_on_pad_upright_and_hand_clear(self):
        value, metadata = self.value(), self.metadata()
        self.assertTrue(visual_finish_decision(value, [0., 0., 1.1], metadata)["ready"])
        self.assertFalse(visual_finish_decision(value, [0., 0., .88], metadata)["ready"])
        value.cup_world_pose[0, 3] = .04
        self.assertFalse(visual_finish_decision(value, [0., 0., 1.1], metadata)["ready"])
        value.cup_world_pose[0, 3] = 0.
        value.cup_world_pose[2, 3] += .02
        self.assertFalse(visual_finish_decision(value, [0., 0., 1.1], metadata)["ready"])
        value.cup_world_pose = np.eye(4)
        value.cup_world_pose[2, 3] = .74
        self.assertFalse(visual_finish_decision(value, [0., 0., 1.1], metadata)["ready"])

    def test_nonrigid_or_nonfinite_visual_pose_is_rejected(self):
        value = self.value()
        value.cup_world_pose[0, 0] = 2.
        with self.assertRaises(ValueError):
            visual_finish_decision(value, [0., 0., 1.1], self.metadata())
        value = self.value()
        value.coaster_world_center[0] = np.nan
        with self.assertRaises(ValueError):
            visual_finish_decision(value, [0., 0., 1.1], self.metadata())

    def rollout(self, *, geometry_error=False, always_open=False, invalid_action=False):
        task = SimpleNamespace(take_action_cnt=0, step_lim=30)
        command = [1.]
        def action(value):
            task.take_action_cnt += 1
            command[0] = 1. if always_open or task.take_action_cnt > 3 else 0.
        task.take_action = action
        task.robot = SimpleNamespace(get_left_gripper_val=lambda: command[0],
            get_left_tcp_pose=lambda: [0., 0., 1.1, 1., 0., 0., 0.])
        task.get_obs = lambda: {}
        resets, calls = [], []
        def step(env, obs):
            for _ in range(min(10, env.step_lim-env.take_action_cnt)):
                env.take_action(np.zeros(13 if invalid_action else 14))
        policy = SimpleNamespace(reset=lambda: resets.append(1), step=step)
        def localize(*a):
            calls.append(1)
            if geometry_error:
                raise RuntimeError("untrusted segmentation")
            return self.value()
        perception = SimpleNamespace(localize_held=localize)
        record = {}
        with patch("adapters.pillbottle_vla_finish.robot_state", return_value={}), patch(
                "adapters.pillbottle_vla_finish.audit_topp", side_effect=lambda *a: nullcontext()):
            if geometry_error or always_open or invalid_action:
                with self.assertRaises((RuntimeError, ValueError)):
                    rollout_finish(task, policy, perception, "left", record, self.metadata(), synchronize=False)
            else:
                rollout_finish(task, policy, perception, "left", record, self.metadata(), synchronize=False)
        self.assertIs(task.take_action, action)
        self.assertEqual(resets, [1])
        return task, record["vla_finish"], calls

    def test_closed_grasp_does_not_interrupt_continuous_vla_or_reset_it(self):
        task, audit, calls = self.rollout()
        self.assertTrue(audit["complete"])
        self.assertEqual(task.take_action_cnt, 16)
        self.assertEqual(len(calls), 2)
        self.assertEqual(audit["policy_resets"], 1)
        self.assertFalse(audit["trial_lift_enabled"])

    def test_bad_visual_pose_never_declares_completion_and_restores_action(self):
        task, audit, calls = self.rollout(geometry_error=True)
        self.assertEqual(task.take_action_cnt, 30)
        self.assertFalse(audit["complete"])
        self.assertTrue(calls)

    def test_initial_open_gripper_is_not_placement_completion(self):
        task, audit, calls = self.rollout(always_open=True)
        self.assertEqual(task.take_action_cnt, 30)
        self.assertFalse(audit["complete"])
        self.assertFalse(calls)

    def test_invalid_action_is_rejected_before_robot_motion(self):
        task, audit, calls = self.rollout(invalid_action=True)
        self.assertEqual(task.take_action_cnt, 0)
        self.assertFalse(audit["complete"])
        self.assertFalse(calls)

    def test_episode_only_reads_official_scores_after_control_and_never_calls_traditional_place(self):
        import tempfile
        from adapters.pillbottle_vla_finish import run_episode
        from adapters.visual_grasp import ObjectTruthTerminationGuard
        from adapters.timing import EpisodeTiming
        for fail in (False, True):
            events = []
            task = SimpleNamespace(eval_success=False, take_action_cnt=0,
                setup_demo=lambda **kw: events.append("setup"), set_instruction=lambda text: None,
                close_env=lambda: events.append("close"))
            def score():
                self.assertIn("control_stopped", events)
                events.append("official_score")
                return True
            def actor_pose():
                self.assertIn("control_stopped", events)
                events.append("evaluation_actor_read")
                return SimpleNamespace(p=np.array([0., 0., .74]))
            task.check_success = score
            task.pillbottle = task.pad = SimpleNamespace(get_pose=actor_pose)
            task.pillbottle_id = 1
            task.robot = SimpleNamespace(is_left_gripper_open=lambda: True, is_right_gripper_open=lambda: True)
            policy = SimpleNamespace(load=lambda: None, model=None)
            base = SimpleNamespace(ROOT=ROOT, EpisodeTiming=EpisodeTiming,
                ObjectTruthTerminationGuard=ObjectTruthTerminationGuard,
                traditional_timing=lambda *a: nullcontext(), vla_timing=lambda *a: nullcontext(),
                snapshot=lambda *a: None, dump=lambda *a: None,
                PillbottlePerception=lambda *a, **kw: None,
                approach_bottle=lambda *a: dict(ready=True, arm="left"),
                policy_without_truth=lambda *a: nullcontext())
            def rollout(*a, **kw):
                self.assertFalse(task.check_success())
                events.append("control_stopped")
                if fail:
                    raise RuntimeError("budget exhausted")
            with tempfile.TemporaryDirectory() as output, patch(
                    "adapters.pillbottle_vla_finish.rollout_finish", rollout):
                args = SimpleNamespace(seed=0, output=Path(output), max_vla_steps=30,
                    instruction="place bottle", mode="hybrid", joint_state_source="commanded",
                    known_metadata=ROOT/"tra_pipe/RoboTwin/assets/objects/080_pillbottle/model_data1.json")
                record = run_episode(base, task, policy, None, args, {}, 0)
            self.assertEqual(events.count("official_score"), 1)
            self.assertEqual(record["success"], not fail)
            self.assertTrue(record["task_success"])
            self.assertIsNone(record["traditional_placement"])
            self.assertEqual(task.check_success, score)


if __name__ == "__main__":
    unittest.main()
