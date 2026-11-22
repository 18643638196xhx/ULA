import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
SPEC = importlib.util.spec_from_file_location("cup_randomized_eval_test", ROOT/"scripts/run_place_empty_cup_randomized.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RandomizedCupTests(unittest.TestCase):
    def test_randomization_is_effective_and_rgbd_runtime_settings_stay_enabled(self):
        clean = dict(domain_randomization=dict(random_light=False), eval_instruction="seen",
            camera=dict(collect_wrist_camera=True), data_type=dict(rgb=True, depth=True),
            force_arm="left", fast_eval_mode=True)
        result = runner.randomized_config(clean)
        for key in ("random_background", "random_light", "cluttered_table"):
            self.assertTrue(result["domain_randomization"][key])
        self.assertEqual(result["domain_randomization"]["random_table_height"], .03)
        self.assertTrue(result["data_type"]["depth"])
        self.assertTrue(result["camera"]["collect_wrist_camera"])
        self.assertEqual(result["force_arm"], "left")
        self.assertEqual(clean["domain_randomization"], dict(random_light=False))
        result["data_type"]["depth"] = False
        self.assertTrue(clean["data_type"]["depth"])

    def test_clean_controller_and_original_worker_are_protected(self):
        hashes = runner.fingerprints()
        self.assertIn("scripts/run_place_empty_cup.py", hashes)
        self.assertIn("scripts/perceive_rgbd.py", hashes)
        self.assertIn("adapters/pointcloud_placement.py", hashes)
        self.assertIn("tra_pipe/RoboTwin/envs/_base_task.py", hashes)


if __name__ == "__main__":
    unittest.main()
