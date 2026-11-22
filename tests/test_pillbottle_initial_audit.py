import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pill_initial_test", ROOT/"scripts/run_pillbottle_initial_randomized.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PillInitialTests(unittest.TestCase):
    def test_partial_correct_surface_not_mislabeled_wrong_identity(self):
        actual = np.ones((20, 20), bool)
        measured = np.zeros_like(actual)
        measured[5:10, 5:10] = True
        result = runner.identity_metrics(measured, actual)
        self.assertTrue(result["identity_match"])
        self.assertFalse(result["mask_iou_at_least_half"])

    def test_background_contamination_or_wrong_object_not_identity_match(self):
        actual, measured = np.zeros((20, 20), bool), np.zeros((20, 20), bool)
        actual[:10, :10] = True
        measured[:] = True
        self.assertFalse(runner.identity_metrics(measured, actual)["identity_match"])
        measured[:] = False
        measured[10:, 10:] = True
        self.assertFalse(runner.identity_metrics(measured, actual)["identity_match"])

    def test_setup_failures_excluded_but_perception_failures_retained(self):
        records = [dict(seed=0, initialized=False),
            dict(seed=1, initialized=True, bottle_output=True, pad_output=True, cad_output=True,
                 both_identity_match=True, localization_verified=True, evaluation_only={}),
            dict(seed=2, initialized=True, pad_output=True, failure_stage="bottle_recognition")]
        summary = runner.summarize(records)
        self.assertEqual(summary["valid_scenes"], 2)
        self.assertEqual(summary["initialization_rejections"], [0])
        self.assertEqual(summary["rates"]["cad_output"], .5)
        self.assertEqual(summary["failures"]["bottle_recognition"], [2])

    def test_worker_is_separate_diagnostic_process(self):
        worker = runner.InitialWorker(ROOT, ROOT/"results/unused_initial_audit_test", backend="foundationpose")
        command, _, _ = worker._launch_spec()
        self.assertEqual(Path(command[1]).name, "perceive_pillbottle_initial_audit.py")
        self.assertNotIn("--model-id", command)


if __name__ == "__main__":
    unittest.main()
