from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.cup_known_model import KnownCupVision, KnownCupWorker, place_known_cup


class ArrayTensor:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def __getitem__(self, key): return ArrayTensor(self.value[key])


class CupKnownModelTests(unittest.TestCase):
    def test_trial_observation_does_not_require_or_touch_foundationpose(self):
        vision = KnownCupVision.__new__(KnownCupVision)
        value = SimpleNamespace(observation_audit={}, timings_s={})
        with patch("adapters.cup_known_model.FocusedCupPerception.estimate", return_value=value):
            result = vision.estimate(None, None, None, None, cup_only=True, require_cad=False)
        self.assertIsNone(result.cup_world_pose)
        self.assertFalse(result.observation_audit["cad_pose_required"])
        self.assertFalse(hasattr(vision, "foundationpose"))

    def test_fresh_pose_must_pass_table_check_and_masks_background_depth(self):
        vision = KnownCupVision.__new__(KnownCupVision)
        vision.reference_table_z = .74
        vision.model_samples = np.array([[x, y, z] for x in (-.03, .03) for y in (0., .1) for z in (-.03, .03)])
        vision.fp_iterations = 5
        vision.torch = SimpleNamespace(inference_mode=nullcontext)
        bad, good = np.eye(4), np.eye(4)
        bad[2, 3], good[2, 3] = .65, .8
        depths = []
        def register(**kw):
            depths.append(kw["depth"])
            fp.poses = ArrayTensor(np.stack([bad, good]))
            return bad
        fp = SimpleNamespace(register=register, get_tf_to_centered_mesh=lambda: ArrayTensor(np.eye(4)))
        vision.foundationpose = fp
        value = SimpleNamespace(cup_points_world=np.zeros((80, 3)), observation_audit={},
            timings_s=dict(perception_total=0.))
        mask = np.zeros((12, 12), bool)
        mask[2:10, 2:10] = True
        with patch("adapters.cup_known_model.FocusedCupPerception.estimate", return_value=value), patch(
                "adapters.cup_known_model.surface_mask", return_value=mask), patch(
                "adapters.cup_known_model.pose_fit", return_value=dict(accepted=True)):
            result = vision.estimate(np.zeros((12, 12, 3), np.uint8), np.ones((12, 12)),
                np.eye(3), np.eye(4), require_cad=True)
        np.testing.assert_allclose(result.cup_world_pose, good)
        self.assertTrue(np.all(depths[0][~mask] == 0))
        self.assertFalse(result.observation_audit["foundationpose_hypotheses"][0]["table_clearance"]["accepted"])

    def test_no_valid_model_hypothesis_never_returns_a_coarse_pose(self):
        vision = KnownCupVision.__new__(KnownCupVision)
        vision.reference_table_z = .74
        vision.model_samples = np.zeros((80, 3))
        vision.fp_iterations = 5
        vision.torch = SimpleNamespace(inference_mode=nullcontext)
        vision.foundationpose = SimpleNamespace(register=lambda **kw: np.eye(4))
        value = SimpleNamespace(cup_points_world=np.zeros((80, 3)), observation_audit={}, timings_s={})
        with patch("adapters.cup_known_model.FocusedCupPerception.estimate", return_value=value), patch(
                "adapters.cup_known_model.surface_mask", return_value=np.ones((12, 12), bool)), patch(
                "adapters.cup_known_model.pose_fit", return_value=dict(accepted=True)):
            with self.assertRaisesRegex(RuntimeError, "CAD/depth validation rejected"):
                vision.estimate(np.zeros((12, 12, 3), np.uint8), np.ones((12, 12)),
                    np.eye(3), np.eye(4), require_cad=True)
        self.assertIsNone(value.cup_world_pose)

    def test_model_placement_uses_saved_visual_upright_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"reference.npz"
            initial = np.eye(4)
            initial[2, 3] = .74
            np.savez(path, cup_world_pose=initial)
            result = {}
            with patch("adapters.placement.finish_placement") as finish:
                place_known_cup(SimpleNamespace(), SimpleNamespace(), "left", path, result)
            np.testing.assert_allclose(finish.call_args.kwargs["upright_reference"], initial)
            self.assertTrue(result["uses_prepared_object_model"])
            self.assertNotIn("uses_object_truth", result)

    def test_worker_uses_isolated_known_cup_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = KnownCupWorker(ROOT, directory, backend="foundationpose")
            command, cwd, environment = worker._launch_spec()
            self.assertEqual(Path(command[1]).name, "perceive_cup_known_model.py")
            self.assertIn("FoundationPose", environment["PYTHONPATH"])
            worker.close()


if __name__ == "__main__":
    unittest.main()
