from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ula_microwave_clean_test", ROOT/"scripts/run_open_microwave_no_extent_clean.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class CleanMicrowaveTests(unittest.TestCase):
    def test_clean_config_preserves_rgbd_and_runtime_without_mutating_input(self):
        runtime = dict(domain_randomization=dict(random_light=True, cluttered_table=True),
                       camera=dict(collect_wrist_camera=True), data_type=dict(depth=True),
                       enable_mplib=True)
        before = deepcopy(runtime)
        clean = runner.clean_config(runtime)
        self.assertEqual(runtime, before)
        self.assertFalse(clean["domain_randomization"]["random_light"])
        self.assertFalse(clean["domain_randomization"]["cluttered_table"])
        self.assertFalse(clean["domain_randomization"]["random_background"])
        self.assertEqual(clean["domain_randomization"]["random_table_height"], 0)
        self.assertEqual(clean["domain_randomization"]["clean_background_rate"], 1)
        self.assertEqual(clean["eval_instruction"], "seen")
        self.assertTrue(clean["data_type"]["depth"])
        self.assertTrue(clean["enable_mplib"])

    def test_real_wrapper_chain_reports_clean_and_restores_state(self):
        base, randomized = runner.base, runner.randomized
        original_scene, original_config, original_dump = randomized.SCENE, randomized.randomized_config, base.dump
        runtime = dict(domain_randomization={}, camera={}, data_type=dict(depth=True))
        def fake_run(args):
            config = base.task_config()
            self.assertFalse(config["domain_randomization"]["random_light"])
            base.dump(args.output/"report.json", dict(episodes=[dict(
                seed=0, success=True, task_success=True, failure_stage=None)]))
            return 0
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(base, "task_config", return_value=runtime), \
                 patch.object(base, "configure_runtime"), patch.object(base, "run", side_effect=fake_run):
                self.assertEqual(runner.main(["--episodes", "1", "--output", directory]), 0)
            report = json.loads((Path(directory)/"report.json").read_text())
        self.assertEqual(report["run_status"], "finished")
        self.assertEqual(report["task_config"], "demo_clean")
        self.assertFalse(report["fixed_panel_extent_check_enabled"])
        self.assertFalse(report["controller_changed_vs_no_extent_randomized"])
        self.assertEqual(report["effective_domain_randomization"]["random_table_height"], 0)
        self.assertTrue(report["randomized_sources_unchanged"])
        self.assertIs(randomized.randomized_config, original_config)
        self.assertEqual(randomized.SCENE, original_scene)
        self.assertIs(base.dump, original_dump)


if __name__ == "__main__":
    unittest.main()
