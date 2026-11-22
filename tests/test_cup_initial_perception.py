"""Initial-only RGB-D input isolation and no-hard-color regression tests."""
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
from adapters.cup_initial_perception import InitialCupPerception, blue_evidence, cup_geometry
from adapters.cup_initial_audit import mask_metrics, no_task_actions


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.value
    def __getitem__(self, item):
        return Tensor(self.value[item])


class InitialCupTests(unittest.TestCase):
    def fake_vision(self, *, pad_error=False, no_cup=False):
        vision = InitialCupPerception.__new__(InitialCupPerception)
        mask = np.ones((12, 12), bool)
        def pad(*args):
            if pad_error:
                raise RuntimeError("pad not seen")
            return np.array([.1, .1, .74]), mask, {}
        vision._coaster = pad
        vision._run_dino = lambda *a: (Tensor(np.empty((0, 4))) if no_cup else Tensor([[1, 1, 10, 10]]),
                                       Tensor([]) if no_cup else Tensor([.6]), None)
        vision._xyxy = lambda box, *a: box
        vision._refine_cup_mask = lambda *a, **kw: mask
        return vision

    def run_fake(self, vision):
        rgb = np.full((12, 12, 3), [60, 140, 55], np.uint8)
        bounds = np.array([[-.2, -.1, .745], [-.13, -.03, .835]])
        points = np.repeat(bounds.mean(0)[None], 80, axis=0)
        with patch("adapters.cup_initial_perception.world_points_image",
                   return_value=(np.zeros((12, 12, 3)), np.ones((12, 12), bool))), \
             patch("adapters.cup_initial_perception.visual_table_height", return_value=.74), \
             patch("adapters.cup_initial_perception.estimate_visible_bounds",
                   return_value=(bounds, {}, points)):
            return vision.estimate(rgb, np.ones((12, 12)), np.eye(3), np.eye(4))

    def test_green_illuminated_cup_does_not_need_any_blue_pixels(self):
        result = self.run_fake(self.fake_vision())
        self.assertTrue(result.observation_audit["cup_accepted"])
        self.assertEqual(result.observation_audit["selected_cup"]["blue_pixel_fraction"], 0.)
        self.assertFalse(result.observation_audit["color_is_hard_gate"])

    def test_pad_failure_does_not_erase_cup(self):
        result = self.run_fake(self.fake_vision(pad_error=True))
        self.assertTrue(result.observation_audit["cup_accepted"])
        self.assertFalse(result.observation_audit["pad_accepted"])

    def test_cup_failure_does_not_erase_pad(self):
        result = self.run_fake(self.fake_vision(no_cup=True))
        self.assertFalse(result.observation_audit["cup_accepted"])
        self.assertTrue(result.observation_audit["pad_accepted"])

    def test_bottle_like_height_does_not_pass_generic_cup_geometry(self):
        with self.assertRaises(RuntimeError):
            cup_geometry(np.array([[0, 0, .745], [.04, .04, .99]]), np.zeros((80, 3)), .74)

    def test_input_rejects_simulator_masks_before_inference(self):
        spec = importlib.util.spec_from_file_location("ula_initial_protocol_test",
                                                     ROOT/"scripts/perceive_cup_initial.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input.npz"
            np.savez(path, rgb=np.zeros((2, 2, 3)), depth=np.ones((2, 2)),
                     intrinsic=np.eye(3), world_to_camera=np.eye(4), simulator_mask=np.ones((2, 2)))
            with self.assertRaises(ValueError):
                module.estimate_file(object(), path, Path(directory)/"output.npz")

    def test_task_actions_and_task_success_scoring_are_forbidden_and_restored(self):
        task = SimpleNamespace(move=lambda: "original", check_success=lambda: True)
        with no_task_actions(task):
            with self.assertRaises(RuntimeError):
                task.move()
            with self.assertRaises(RuntimeError):
                task.check_success()
        self.assertEqual(task.move(), "original")
        self.assertTrue(task.check_success())

    def test_returned_wrong_object_is_not_a_correct_identity(self):
        actual = np.zeros((10, 10), bool)
        actual[1:4, 1:4] = True
        wrong = np.zeros_like(actual)
        wrong[6:9, 6:9] = True
        self.assertFalse(mask_metrics(wrong, actual)["identity_match"])
        self.assertTrue(mask_metrics(actual, actual)["identity_match"])
        self.assertFalse(mask_metrics(np.zeros_like(actual), actual)["identity_match"])


if __name__ == "__main__":
    unittest.main()
