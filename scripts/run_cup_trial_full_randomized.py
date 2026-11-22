#!/usr/bin/env python3
"""Full randomized task, changing only the trial perception/confirmation path."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tra_pipe/RoboTwin')]
spec = importlib.util.spec_from_file_location('cup_trial_full_known_base', ROOT/'scripts/run_place_empty_cup_known_randomized.py')
known = importlib.util.module_from_spec(spec)
spec.loader.exec_module(known)
from adapters.cup_trial_diagnostic import TrialCupWorker, verify_segmented_trial


class FullTrialCupWorker(TrialCupWorker):
    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(self.root/'scripts/perceive_cup_trial_full.py')
        return command, cwd, environment


def main():
    base = known.randomized.base
    original_worker = known.KnownCupWorker
    original_verify, original_dump = base.verify_visual_trial_lift, base.dump_json
    files = ['scripts/run_cup_trial_full_randomized.py', 'scripts/perceive_cup_trial_full.py',
             'adapters/cup_trial_diagnostic.py']
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    def dump(path, value):
        if Path(path).name == 'report.json':
            value.update(experiment='known_CAD_full_task_with_semantic_segmented_grasp_verification',
                success_scope='complete_place_empty_cup_not_grasp_only',
                grasp_verification_geometry='episode_local_semantic_RGBD_segmented_surface_comotion',
                grasp_stage_truth_audit_enabled=False,
                grasp_trial_implementation_hashes=hashes,
                grasp_trial_files_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
                                                for p,h in hashes.items()),
                placement_implementation='unchanged_original_known_CAD_placement',
                initial_perception_implementation='unchanged_original_known_CAD_initial_perception')
        return original_dump(path, value)
    known.KnownCupWorker = FullTrialCupWorker
    base.verify_visual_trial_lift = verify_segmented_trial
    base.dump_json = dump
    try:
        return known.main()
    finally:
        known.KnownCupWorker = original_worker
        base.verify_visual_trial_lift, base.dump_json = original_verify, original_dump


if __name__ == '__main__':
    raise SystemExit(main())
