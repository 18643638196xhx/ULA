"""Isolated diagnostic: no hard color gate, no truth in control, no early pass."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tra_pipe/RoboTwin')]
from adapters.cup_trial_diagnostic import TrialCupVision, local_identity_geometry, verify_segmented_trial
import test_cup_initial_perception as initial_helpers


class Forbidden:
    def __getattr__(self, name):
        raise AssertionError(f'No object/contact truth: {name}')


class CupTrialTests(unittest.TestCase):
    def test_partial_cup_need_not_be_upright(self):
        bounds = np.array([[0, 0, .75], [.08, .07, .77]])
        self.assertLess(local_identity_geometry(bounds, np.zeros((80, 3)), bounds.mean(0)), .001)

    def test_distant_distractor_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'identity envelope'):
            local_identity_geometry(np.array([[.4, .4, .75], [.47, .48, .84]]),
                                    np.zeros((80, 3)), np.array([0, 0, .8]))

    def test_green_cup_accepted_without_blue_pixels(self):
        vision = TrialCupVision.__new__(TrialCupVision)
        bounds = np.array([[-.2, -.1, .75], [-.13, -.03, .83]])
        vision.trial_anchor = bounds.mean(0)
        tensor = initial_helpers.Tensor
        vision._run_dino = lambda *a: (tensor([[1, 1, 10, 10]]), tensor([.7]), None)
        vision._xyxy = lambda box, *a: box
        vision._refine_cup_mask = lambda *a, **kw: np.ones((12, 12), bool)
        with patch('adapters.cup_trial_diagnostic.estimate_visible_bounds',
                   return_value=(bounds, {'table_height_m': .74}, np.zeros((80, 3)))):
            value = vision.estimate(np.full((12,12,3), [60,140,55], np.uint8),
                np.ones((12,12)), np.eye(3), np.eye(4), cup_only=True)
        self.assertFalse(value.observation_audit['color_is_hard_gate'])
        self.assertEqual(value.observation_audit['selected']['caption'], 'blue cup .')

    def run_trial(self, directory, *, follows=True, bad=False):
        ee = np.array([0., 0., 1., 1., 0., 0., 0.])
        start = ee.copy()
        cloud = np.random.default_rng(4).uniform(-.025, .025, (400,3)) + [0,0,.8]
        calls = []
        def camera(name):
            return dict(rgb=np.zeros((12,12,3), np.uint8), depth=np.ones((12,12)),
                intrinsic_cv=np.eye(3), extrinsic_cv=np.eye(4),
                measured_displacement=ee[:3]-start[:3])
        def supplementary(frame, label, **kwargs):
            calls.append(label)
            if bad:
                raise RuntimeError('no visible cup')
            points = cloud + (frame['measured_displacement'] if follows else 0)
            return SimpleNamespace(cup_bounds_world=np.array([points.min(0),points.max(0)]),
                cup_points_world=points, observation_audit={'table_height_m': .74})
        def execute(controls):
            ee[:] = controls['left_arm']['target']
            return True
        task = SimpleNamespace(cup=Forbidden(), scene=Forbidden(), plan_success=True,
            get_arm_pose=lambda arm: ee.copy(), get_camera_observation=camera,
            robot=SimpleNamespace(left_plan_path=lambda target: dict(status='Success',target=target.copy())),
            take_dense_action=execute)
        perception = SimpleNamespace(output=Path(directory), supplementary=supplementary)
        result = {}
        with patch('adapters.cup_trial_diagnostic.slower_trajectory', side_effect=lambda plan, n: plan):
            try:
                decision = verify_segmented_trial(task, perception, 'left', result)
            except RuntimeError:
                decision = None
        return result, decision, calls

    def test_all_segment_views_saved_and_early_small_lift_not_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, decision, calls = self.run_trial(tmp)
            self.assertTrue(decision['ready'])
            self.assertEqual(len(result['segments']), 3)
            self.assertEqual(len(list((Path(tmp)/'raw_frames').glob('*/*/rgbd.npz'))), 8)
            early = [o['comotion'] for o in result['observations'] if o.get('segment') == 1]
            self.assertTrue(early)
            self.assertFalse(any(d['ready'] for d in early))
            self.assertEqual(decision['minimum_visual_lift_m'], .03)
            self.assertFalse(result['slip_check_enabled'])

    def test_stationary_cup_not_promoted_to_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, decision, _ = self.run_trial(tmp, follows=False)
            self.assertIsNone(decision)
            self.assertEqual(result['failure_category'], 'registered_surface_not_following')

    def test_missing_baseline_no_motion_and_both_raw_frames_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, decision, _ = self.run_trial(tmp, bad=True)
            self.assertIsNone(decision)
            self.assertEqual(result['segments'], [])
            self.assertEqual(len(list((Path(tmp)/'raw_frames').glob('*/*/rgbd.npz'))), 2)

    def test_worker_rejects_identity_masks_before_inference(self):
        spec = importlib.util.spec_from_file_location('trial_protocol_test', ROOT/'scripts/perceive_cup_trial_diagnostic.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'input.npz'
            np.savez(source, rgb=np.zeros((2,2,3)), depth=np.ones((2,2)),
                intrinsic=np.eye(3), world_to_camera=np.eye(4), simulator_mask=np.ones((2,2)))
            with self.assertRaises(ValueError):
                module.estimate_file(object(), source, Path(tmp)/'output.npz')

    def test_terminal_truth_only_after_controller_has_returned_or_raised(self):
        spec = importlib.util.spec_from_file_location('trial_terminal_test', ROOT/'scripts/run_cup_trial_diagnostic.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                stopped = []
                def measured_pose():
                    self.assertTrue(stopped, 'No truth read before control terminates')
                    return SimpleNamespace(to_transformation_matrix=lambda: np.eye(4))
                def measured_bottom(*args):
                    self.assertTrue(stopped)
                    return SimpleNamespace(p=np.zeros(3))
                def controller(*args, **kwargs):
                    stopped.append(True)
                    if fail:
                        raise RuntimeError('visual reject')
                    return {'ready': True}
                task = SimpleNamespace(cup=SimpleNamespace(get_pose=measured_pose,
                    get_functional_point=measured_bottom), get_arm_pose=lambda arm: [0]*7)
                with patch.object(module, 'verify_segmented_trial', side_effect=controller):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'visual reject'):
                            module.terminal_audit(task, SimpleNamespace(output=Path(tmp)), 'left', {'observations': []})
                    else:
                        self.assertTrue(module.terminal_audit(task, SimpleNamespace(output=Path(tmp)),
                            'left', {'observations': []})['ready'])
                self.assertTrue((Path(tmp)/'evaluation_after_control.json').exists())


if __name__ == '__main__':
    unittest.main()
