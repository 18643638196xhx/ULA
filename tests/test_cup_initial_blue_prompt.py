"""Caption ablation leaves coaster, geometry and no-hard-color behavior alone."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.cup_initial_blue_prompt import BlueInitialCupPerception, BlueInitialCupWorker
from adapters.cup_initial_perception import InitialCupPerception
import test_cup_initial_perception as initial_tests


class BluePromptTests(unittest.TestCase):
    def test_only_generic_cup_caption_is_replaced(self):
        value = BlueInitialCupPerception.__new__(BlueInitialCupPerception)
        with patch.object(InitialCupPerception, "_run_dino", return_value="result") as detect:
            self.assertEqual(value._run_dino("image", "cup . mug ."), "result")
            detect.assert_called_with("image", "blue cup .")
            value._run_dino("image", "round coaster .")
            detect.assert_called_with("image", "round coaster .")

    def test_zero_blue_pixels_still_pass_geometry(self):
        helper = initial_tests.InitialCupTests()
        value = helper.fake_vision()
        value.__class__ = BlueInitialCupPerception
        result = helper.run_fake(value)
        self.assertTrue(result.observation_audit["cup_accepted"])
        self.assertFalse(result.observation_audit["color_is_hard_gate"])
        self.assertEqual(result.observation_audit["cup_semantic_caption"], "blue cup .")

    def test_independent_worker_entry_imports(self):
        spec = importlib.util.spec_from_file_location("blue_initial_entry_test",
                                                     ROOT/"scripts/perceive_cup_initial_blue.py")
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        self.assertTrue(callable(value.initial.estimate_file))
        worker = BlueInitialCupWorker(ROOT, ROOT/"results/unused_test_worker")
        command, _, _ = worker._launch_spec()
        self.assertEqual(Path(command[1]).name, "perceive_cup_initial_blue.py")


if __name__ == "__main__":
    unittest.main()
