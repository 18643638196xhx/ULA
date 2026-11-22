import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pill_random_test", ROOT/"scripts/run_move_pillbottle_pad_known_randomized.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RandomizedBottleTests(unittest.TestCase):
    def test_scene_override_preserves_sensors_and_does_not_mutate_input(self):
        source = dict(domain_randomization=dict(cluttered_table=False),
                      camera=dict(collect_wrist_camera=True), data_type=dict(depth=True), enable_mplib=True)
        result = runner.randomized_config(source)
        self.assertTrue(result["domain_randomization"]["cluttered_table"])
        self.assertTrue(result["domain_randomization"]["random_light"])
        self.assertEqual(result["domain_randomization"]["random_table_height"], .03)
        self.assertTrue(result["data_type"]["depth"])
        self.assertTrue(result["enable_mplib"])
        self.assertFalse(source["domain_randomization"]["cluttered_table"])

    def test_valid_denominator_and_warm_time(self):
        records = [dict(seed=0, success=False, task_success=None, failure_stage="setup", elapsed_s=1.),
                   dict(seed=1, success=True, task_success=True, failure_stage=None, elapsed_s=99.),
                   dict(seed=2, success=False, task_success=True, failure_stage="traditional_placement", elapsed_s=3.),
                   dict(seed=3, success=True, task_success=True, failure_stage=None, elapsed_s=8.)]
        result = runner.progress(records)
        self.assertEqual(result["completed_valid_episodes"], 3)
        self.assertEqual(result["initialization_rejection_seeds"], [0])
        self.assertAlmostEqual(result["success_rate"], 2/3)
        self.assertEqual(result["official_success_rate"], 1.)
        self.assertEqual(result["statistics"]["excluding_first_episode"]["mean_successful_completion_s"], 8.)

    def test_vision_gpu_override_changes_only_worker_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = runner.SplitKnownWorker(ROOT, directory, model_id=1)
            worker.placement_policy = "visual"
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES":"7", "PILLBOTTLE_RANDOM_VISION_GPU":"6"}):
                command, _, environment = worker._launch_spec()
                self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "6")
                self.assertIn("--geometry-guard", command)
                self.assertEqual(worker.summary()["known_model_id"], 1)


if __name__ == "__main__":
    unittest.main()
