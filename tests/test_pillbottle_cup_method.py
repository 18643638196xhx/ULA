from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.pillbottle_focus.cup_method import (CupMethodBottleVision,
    BottleGeometryFrontend, place, task_class)
from adapters.visual_grasp import ClosedCommandMonitor

SPEC = importlib.util.spec_from_file_location("pill_cup_method_test",
    ROOT/"scripts/run_move_pillbottle_pad_cup_method.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def observation():
    return SimpleNamespace(cup_bounds_world=np.array([[0., 0., .75], [.05, .05, .85]]),
        cup_points_world=np.tile([.025, .025, .8], (80, 1)),
        coaster_world_center=np.array([.15, .1, .74]),
        observation_audit=dict(table_height_m=.74), timings_s=dict(perception_total=0.))


class PillbottleCupMethodTests(unittest.TestCase):
    def test_coarse_frontend_does_not_call_cad_and_static_target_is_cached(self):
        vision = CupMethodBottleVision.__new__(CupMethodBottleVision)
        vision.reset_episode()
        first, second = observation(), observation()
        second.coaster_world_center[:] = np.nan
        with patch("adapters.pillbottle_focus.cup_method.FocusedBottleFrontend.estimate",
                   side_effect=[first, second]) as frontend:
            vision.estimate(None, None, None, None)
            value = vision.estimate(None, None, None, None)
        self.assertFalse(frontend.call_args_list[0].kwargs["cup_only"])
        self.assertTrue(frontend.call_args_list[1].kwargs["cup_only"])
        np.testing.assert_allclose(value.coaster_world_center, [.15, .1, .74])
        self.assertIsNone(value.cup_world_pose)
        self.assertFalse(hasattr(vision, "foundationpose"))

    def test_mro_reuses_the_original_cup_cad_implementation(self):
        from adapters.cup_known_model import KnownCupVision
        order = CupMethodBottleVision.__mro__
        self.assertLess(order.index(KnownCupVision), order.index(BottleGeometryFrontend))
        vision = CupMethodBottleVision.__new__(CupMethodBottleVision)
        vision.reset_episode()
        vision.fp_iterations = 5
        vision.model_samples = np.zeros((80, 3))
        vision.torch = SimpleNamespace(inference_mode=nullcontext)
        vision.foundationpose = SimpleNamespace(register=lambda **kw: np.eye(4))
        with patch("adapters.pillbottle_focus.cup_method.FocusedBottleFrontend.estimate",
                   return_value=observation()), patch("adapters.cup_known_model.surface_mask",
                   return_value=np.ones((10, 10), bool)), patch("adapters.cup_known_model.pose_fit",
                   return_value=dict(accepted=True)) as fit, patch("adapters.cup_known_model.table_clearance",
                   return_value=dict(accepted=True)):
            value = vision.estimate(np.zeros((10, 10, 3), np.uint8), np.ones((10, 10)),
                np.eye(3), np.eye(4), require_cad=True)
        fit.assert_called_once()
        self.assertEqual(value.observation_audit["known_model"], "080_pillbottle/base1")
        self.assertFalse(value.observation_audit["robot_transport_pose_prior"])

    def test_episode_reset_clears_stationary_pad(self):
        vision = CupMethodBottleVision.__new__(CupMethodBottleVision)
        vision.reference_pad = np.ones(3)
        vision.reference_table_z = .8
        vision.reset_episode()
        self.assertIsNone(vision.reference_pad)
        self.assertIsNone(vision.reference_table_z)

    def test_placement_calls_exact_cup_function_with_bottle_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"reference.npz"
            np.savez(path, cup_world_pose=np.eye(4))
            result = {}
            with patch("adapters.placement.finish_placement") as finish:
                place(None, None, "right", path, result)
            self.assertIn("080_pillbottle/model_data1.json", str(finish.call_args.args[3]))
            np.testing.assert_array_equal(finish.call_args.kwargs["upright_reference"], np.eye(4))

    def test_closed_command_handoff_does_not_require_nearby_object_or_truth(self):
        class Forbidden:
            def __getattr__(self, name):
                raise AssertionError("Object truth must not be read during closed-command handoff")
        task = SimpleNamespace(cup=Forbidden(), take_action_cnt=0, step_lim=10, eval_success=False,
            robot=SimpleNamespace(get_right_gripper_val=lambda: 0.),
            get_arm_pose=lambda arm: [10., 10., 10., 1., 0., 0., 0.], get_obs=lambda: {})
        def take_action(action):
            task.take_action_cnt += 1
        task.take_action = take_action
        policy = SimpleNamespace(reset=lambda: None,
            step=lambda task, obs: task.take_action(np.zeros(14)))
        record = dict(evaluation_only={})
        runner.base.rollout(task, policy, None, record,
            ClosedCommandMonitor(stable_steps=3, gripper_threshold=.05), "right")
        self.assertEqual(task.take_action_cnt, 3)
        self.assertIn("candidate_handoff", record)
        self.assertNotIn("visual_proximity", record["candidate_handoff"])
        self.assertIs(task.take_action, take_action)

    def test_identity_is_fixed_before_spawn_and_factory_restored(self):
        seen = []
        module = SimpleNamespace(create_actor=lambda **kwargs: seen.append(kwargs) or object())
        factory = module.create_actor
        class Task:
            def load_actors(self):
                self.pillbottle = module.create_actor(modelname="080_pillbottle", model_id=5)
                self.pad = object()
        adapted = task_class(Task, module)()
        adapted.load_actors()
        self.assertEqual(seen[0]["model_id"], 1)
        self.assertIs(module.create_actor, factory)
        self.assertIs(adapted.cup, adapted.pillbottle)
        self.assertIs(adapted.coaster, adapted.pad)

    def test_factory_restored_after_spawn_failure(self):
        factory = lambda **kwargs: None
        module = SimpleNamespace(create_actor=factory)
        class Task:
            def load_actors(self):
                raise RuntimeError("setup failed")
        with self.assertRaisesRegex(RuntimeError, "setup failed"):
            task_class(Task, module)().load_actors()
        self.assertIs(module.create_actor, factory)

    def test_worker_and_config_are_task_specific_without_changing_sensors(self):
        worker = runner.BottleWorker(ROOT, ROOT/"results/unused_cup_bottle_test", backend="foundationpose")
        command, _, _ = worker._launch_spec()
        self.assertEqual(Path(command[1]).name, "perceive_pillbottle_cup_method.py")
        source = dict(task_name="place_empty_cup", data_type=dict(depth=True, actor_segmentation=True))
        config = runner.bottle_config(source)
        self.assertEqual(config["task_name"], "move_pillbottle_pad")
        self.assertTrue(config["data_type"]["depth"])
        self.assertFalse(config["data_type"]["actor_segmentation"])
        self.assertTrue(source["data_type"]["actor_segmentation"])
        self.assertFalse(runner.metadata()["handoff_requires_visual_proximity"])

    def test_task_import_working_directory_is_correct_and_restored(self):
        previous = Path.cwd()
        def run():
            self.assertEqual(Path.cwd(), ROOT/"tra_pipe/RoboTwin")
            raise RuntimeError("test restore")
        with patch.object(runner, "_run", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "test restore"):
                runner.main()
        self.assertEqual(Path.cwd(), previous)


if __name__ == "__main__":
    unittest.main()
