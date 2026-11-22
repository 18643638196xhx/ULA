#!/usr/bin/env python3
"""Opt-in orange/white bottle, blue pad, randomized-scene ULA evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ula_focused_pill_random", ROOT/"scripts/run_move_pillbottle_pad_known_randomized.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


class FocusedWorker(baseline.SplitKnownWorker):
    def _launch_spec(self):
        if self.model_id != 1:
            raise ValueError("Focused orange/white appearance is currently validated only for configured base1")
        command, cwd, environment = super()._launch_spec()
        command[1] = str(ROOT/"scripts/perceive_pillbottle_focused.py")
        return command, cwd, environment

    def summary(self):
        return dict(super().summary(), frontend="pillbottle_color_depth_semantic",
                    appearance_prior="orange_cap_white_body_and_blue_square_pad")


def main():
    old_worker, old_hashes = baseline.SplitKnownWorker, baseline.source_hashes
    old_run = baseline.run_valid
    paths = [Path(__file__), ROOT/"scripts/perceive_pillbottle_focused.py",
             ROOT/"scripts/run_move_pillbottle_pad_focused_randomized.sh",
             ROOT/"scripts/pillbottle_focused_python.sh",
             *sorted((ROOT/"adapters/pillbottle_focus").glob("*.py"))]
    def hashes():
        return dict(old_hashes(), **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in paths})
    def run(args, resource):
        try:
            return old_run(args, resource)
        finally:
            path = args.output/"report.json"
            if path.exists():
                report = json.loads(path.read_text())
                report.update(perception_frontend="pillbottle_color_depth_semantic",
                    controller_changed_vs_latest_known_clean=False,
                    perception_changed_vs_latest_known_clean=True,
                    appearance_prior="orange_cap_white_body_and_blue_square_pad",
                    appearance_prior_is_task_specific=True,
                    frontend_uses_object_truth=False,
                    backend_cad_fit_thresholds_unchanged=True)
                baseline.base.dump(path, report)
    baseline.SplitKnownWorker, baseline.source_hashes, baseline.run_valid = FocusedWorker, hashes, run
    try:
        return baseline.main()
    finally:
        baseline.SplitKnownWorker, baseline.source_hashes, baseline.run_valid = old_worker, old_hashes, old_run


if __name__ == "__main__":
    raise SystemExit(main())
