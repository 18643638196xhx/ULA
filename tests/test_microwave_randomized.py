"""CPU-only isolation and accounting checks for randomized microwave scenes."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ula_microwave_randomized_test", ROOT/"scripts/run_open_microwave_randomized.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RandomizedMicrowaveTests(unittest.TestCase):
    def test_scene_override_preserves_sensors_and_does_not_mutate_clean(self):
        clean = dict(domain_randomization=dict(cluttered_table=False),
                     camera=dict(collect_wrist_camera=True), data_type=dict(depth=True),
                     enable_mplib=True)
        result = runner.randomized_config(clean)
        self.assertTrue(result["domain_randomization"]["cluttered_table"])
        self.assertTrue(result["domain_randomization"]["random_light"])
        self.assertEqual(result["domain_randomization"]["random_table_height"], .03)
        self.assertTrue(result["data_type"]["depth"])
        self.assertTrue(result["enable_mplib"])
        result["camera"]["collect_wrist_camera"] = False
        self.assertTrue(clean["camera"]["collect_wrist_camera"])
        self.assertFalse(clean["domain_randomization"]["cluttered_table"])

    def test_setup_rejections_are_separate_from_initialized_and_official_score(self):
        records = [dict(seed=0, success=True, task_success=True, failure_stage=None),
                   dict(seed=1, success=False, task_success=None, failure_stage="setup"),
                   dict(seed=2, success=False, task_success=True,
                        failure_stage="traditional_opening")]
        result = runner.outcome_counts(records)
        self.assertEqual(result["setup_failure_seeds"], [1])
        self.assertEqual(result["complete_success_rate_all_attempts"], 1/3)
        self.assertEqual(result["complete_success_rate_initialized"], .5)
        self.assertEqual(result["official_success_rate_initialized"], 1.)
        self.assertIsNone(runner.outcome_counts([])["complete_success_rate_initialized"])

    def test_main_reports_real_config_and_restores_baseline(self):
        original_config, original_dump = runner.base.task_config, runner.base.dump
        clean = dict(domain_randomization={}, camera={}, data_type=dict(depth=True))
        def fake_run(args):
            self.assertTrue(runner.base.task_config()["domain_randomization"]["cluttered_table"])
            runner.base.dump(args.output/"report.json", dict(episodes=[
                dict(seed=0, success=False, task_success=True,
                     failure_stage="traditional_opening")]))
            return 1
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(runner.base, "task_config", return_value=clean) as config_mock:
                with patch.object(runner.base, "configure_runtime"), \
                     patch.object(runner.base, "run", side_effect=fake_run):
                    code = runner.main(["--episodes", "1", "--output", temporary])
                self.assertIs(runner.base.task_config, config_mock)
            report = json.loads((Path(temporary)/"report.json").read_text())
        self.assertEqual(code, 1)
        self.assertEqual(report["run_status"], "finished")
        self.assertEqual(report["task_config"], "demo_randomized")
        self.assertEqual(report["completed_episodes"], 1)
        self.assertFalse(report["seed_filtering"])
        self.assertFalse(report["uses_prepared_object_model"])
        self.assertTrue(report["randomized_sources_unchanged"])
        self.assertIs(runner.base.task_config, original_config)
        self.assertIs(runner.base.dump, original_dump)


if __name__ == "__main__":
    unittest.main()
