"""CPU-only isolation and URDF coordinate checks for the opt-in CAD pilot."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from adapters.microwave.known_model import load_door_model, model_hinge, KnownModelWorker
from adapters.microwave.perception import MicrowaveWorker

ROOT = Path(__file__).resolve().parents[1]


class KnownModelTests(unittest.TestCase):
    def test_offline_door_mesh_is_in_hinge_frame_in_metres(self):
        for asset in ("7167", "7310"):
            model = load_door_model(ROOT/"tra_pipe/RoboTwin/assets/objects/044_microwave"/asset)
            self.assertAlmostEqual(model["scale"], .15)
            np.testing.assert_array_equal(model["hinge_local"], [0, 0, 0])
            np.testing.assert_array_equal(model["axis_local"], [0, -1, 0])
            self.assertTrue(.13 < model["handle_local"][0] < .18)
            self.assertTrue(.01 < model["handle_local"][2] < .04)
            self.assertLess(np.max(model["mesh"].extents), .20)

    def test_hinge_comes_from_estimated_cad_origin_not_visible_edge(self):
        panel = dict(normal=np.array([0., -1., 0.]),
            known_model_hinge_world=[-.3, .15, .9], known_model_axis_world=[0, 0, -1])
        hinge = model_hinge(panel, np.array([.09, .13, 1.02]))
        np.testing.assert_allclose(hinge["hinge_world"], [-.3, .15, 1.02])
        self.assertFalse(hinge["uses_object_truth"])
        self.assertFalse(hinge["hinge_at_panel_edge_assumed"])
        self.assertTrue(hinge["uses_prepared_object_model"])
        panel["known_model_axis_world"] = [1, 0, 0]
        with self.assertRaisesRegex(RuntimeError, "vertical"):
            model_hinge(panel, np.array([.09, .13, 1.02]))

    def test_known_worker_has_independent_entry_default_remains_pointcloud(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            worker = KnownModelWorker(ROOT, ROOT/"results/not_created_known", vision_gpu=0,
                model_directory=ROOT/"tra_pipe/RoboTwin/assets/objects/044_microwave/7167")
            command, _, env = worker._launch_spec()
            self.assertIn("perceive_microwave_known_model.py", command[1])
            self.assertIn("FoundationPose", env["PYTHONPATH"])
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
            self.assertTrue(worker.summary()["uses_prepared_object_model"])
            default = MicrowaveWorker(ROOT, ROOT/"results/not_created_default")
            self.assertFalse(default.summary()["uses_prepared_object_model"])
            self.assertIn("perceive_microwave_rgbd.py", default._launch_spec()[0][1])


if __name__ == "__main__":
    unittest.main()
