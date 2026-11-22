import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "microwave_direct_test", ROOT/"scripts/run_open_microwave_direct_randomized.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class DirectMicrowaveTests(unittest.TestCase):
    def test_valid_denominator_and_first_valid_warmup(self):
        records = [dict(seed=0, success=False, task_success=None,
                        failure_stage="setup", elapsed_s=1.),
                   dict(seed=1, success=True, task_success=True,
                        failure_stage=None, elapsed_s=100.),
                   dict(seed=2, success=False, task_success=True,
                        failure_stage="traditional_opening", elapsed_s=4.),
                   dict(seed=3, success=True, task_success=True,
                        failure_stage=None, elapsed_s=8.)]
        result = module.progress(records)
        self.assertEqual(result["completed_valid_episodes"], 3)
        self.assertEqual(result["initialization_rejection_seeds"], [0])
        self.assertAlmostEqual(result["success_rate"], 2/3)
        self.assertEqual(result["official_success_rate"], 1.)
        self.assertEqual(result["statistics"]["excluding_first_episode"]["mean_successful_completion_s"], 8.)

    def test_original_controller_not_replaced_by_entry_import(self):
        from adapters.microwave import control
        self.assertIs(module.base.open_door, control.open_door)
        self.assertEqual(module.progress([])["success_rate"], None)


if __name__ == "__main__":
    unittest.main()
