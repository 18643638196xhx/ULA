"""Full integration must preserve CAD placement and omit mid-task truth audits."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CupTrialFullTests(unittest.TestCase):
    def test_full_protocol_requests_cad_only_for_initial_and_placement(self):
        worker = load('trial_full_protocol_test', 'perceive_cup_trial_full.py')
        calls = []
        def estimate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(cup_bounds_world=np.zeros((2,3)), cup_points_world=np.zeros((80,3)),
                cup_world_pose=np.eye(4) if kwargs['require_cad'] else None,
                coaster_world_center=np.zeros(3), timings_s={}, observation_audit={})
        vision = SimpleNamespace(reset_episode=lambda: None, estimate=estimate)
        with tempfile.TemporaryDirectory() as tmp:
            for name, expected in [('perception', True), ('trial_segment_1_head_camera_0', False),
                                   ('placement_perception', True)]:
                directory = Path(tmp)/name
                directory.mkdir()
                np.savez(directory/'rgbd.npz', rgb=np.zeros((2,2,3)), depth=np.ones((2,2)),
                    intrinsic=np.eye(3), world_to_camera=np.eye(4))
                worker.estimate_file(vision, directory/'rgbd.npz', directory/'estimate.npz')
                self.assertIs(calls[-1]['require_cad'], expected)
                with np.load(directory/'estimate.npz') as values:
                    self.assertEqual('cup_world_pose' in values.files, expected)

    def test_full_wrapper_uses_no_terminal_truth_hook_and_restores(self):
        wrapper = load('trial_full_runner_test', 'run_cup_trial_full_randomized.py')
        base = wrapper.known.randomized.base
        original_worker = wrapper.known.KnownCupWorker
        original_verify, original_dump = base.verify_visual_trial_lift, base.dump_json
        def main():
            self.assertIs(base.verify_visual_trial_lift, wrapper.verify_segmented_trial)
            self.assertIs(wrapper.known.KnownCupWorker, wrapper.FullTrialCupWorker)
            return 0
        with patch.object(wrapper.known, 'main', side_effect=main):
            self.assertEqual(wrapper.main(), 0)
        self.assertIs(base.verify_visual_trial_lift, original_verify)
        self.assertIs(base.dump_json, original_dump)
        self.assertIs(wrapper.known.KnownCupWorker, original_worker)

    def test_full_protocol_rejects_simulator_masks(self):
        worker = load('trial_full_allowlist_test', 'perceive_cup_trial_full.py')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'input.npz'
            np.savez(source, rgb=np.zeros((2,2,3)), depth=np.ones((2,2)),
                intrinsic=np.eye(3), world_to_camera=np.eye(4), simulator_mask=np.ones((2,2)))
            with self.assertRaises(ValueError):
                worker.estimate_file(object(), source, Path(tmp)/'output.npz')


if __name__ == '__main__':
    unittest.main()
