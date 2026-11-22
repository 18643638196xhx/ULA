"""Fresh-depth evidence, occlusion and isolated dispatch tests; CPU only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]

from adapters.microwave import control
from adapters.microwave.geometry import fit_panel, hinge_motion, panel_corners
from adapters.microwave.panel_tracking import (
    ApplianceFirstVision, attach_reference, observe_panel, track_panel, verify_panel_step)

def camera_for(reference=None, degrees=0., occlude=False, foreground=False):
    height, width, focal = 120, 160, 160.
    intrinsic = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
    world = np.eye(4)
    world[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    world[:3, 3] = [0, -.6, 1.03]
    y, x = np.indices((height, width))
    rays = np.stack(((x-width/2)/focal, np.ones_like(x), -(y-height/2)/focal), axis=-1)
    motion = np.eye(4) if reference is None else hinge_motion(
        reference["tracking_root"]["hinge"]["hinge_world"],
        np.radians(degrees)*reference["tracking_root"]["hinge"]["opening_sign"])
    normal = motion[:3, :3]@np.array([0., -1., 0.])
    centre = motion[:3, :3]@np.array([0., .15, 1.02])+motion[:3, 3]
    depth = ((centre-world[:3, 3])@normal)/(rays@normal)
    points = world[:3, 3]+rays*depth[..., None]
    initial = (points-motion[:3, 3])@motion[:3, :3]
    valid = ((initial[..., 0] >= -.27) & (initial[..., 0] <= .11)
             & (initial[..., 2] >= .9) & (initial[..., 2] <= 1.14))
    depth = np.where(valid, depth, 0.)
    if occlude:
        # An opaque patch can hide the entire handle region without moving the
        # remaining visible door. Not a simulator identity mask.
        blocked = valid & (initial[..., 0] > -.015)
        depth[blocked] = 0.
    if foreground:
        depth[valid] -= .06
    camera = dict(rgb=np.zeros((height, width, 3), np.uint8), depth=depth,
                  intrinsic_cv=intrinsic, extrinsic_cv=np.linalg.inv(world))
    return camera, points[valid]


def reference_frame():
    camera, points = camera_for()
    panel = fit_panel(points, [0, -1, 0])
    result = dict(door_points=points, handle_world=np.array([.09, .132, 1.02]),
                  camera_world=np.array([0, -.6, 1.03]), panel=panel,
                  tracking_corners=panel_corners(panel), vision_audit={})
    return attach_reference(result)


class PanelTrackingTests(unittest.TestCase):
    def test_appliance_box_can_initialize_without_door_or_handle_text_detection(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.value
            def __getitem__(self, index):
                return Tensor(self.value[index])
        reference = reference_frame()
        vision = ApplianceFirstVision.__new__(ApplianceFirstVision)
        prompts = []
        def detect(rgb, prompt):
            prompts.append(prompt)
            return Tensor([[12., 12., 130., 100.]]), Tensor([.8]), None
        vision.models = SimpleNamespace(_run_dino=detect, _xyxy=lambda value, *args: value)
        frame = dict(door_points=reference["door_points"], handle_world=reference["handle_world"],
                     camera_world=reference["camera_world"], handle_audit={})
        camera, _ = camera_for()
        with patch("adapters.microwave.panel_tracking.initial_depth_frame", return_value=frame), \
             patch("adapters.microwave.perception.MicrowaveVision.estimate",
                   side_effect=AssertionError("semantic-door fallback must not be needed")):
            result = vision.estimate(camera["rgb"], camera["depth"],
                                     camera["intrinsic_cv"], camera["extrinsic_cv"])
        self.assertEqual(prompts, ["microwave ."])
        self.assertTrue(result["audit"]["handle_measured"])
        self.assertFalse(result["audit"]["uses_prepared_object_model"])

    def test_worker_entry_imports_by_path_despite_robotwin_scripts_package(self):
        spec = importlib.util.spec_from_file_location(
            "ula_panel_worker_import_test", ROOT/"scripts/perceive_microwave_panel.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.protocol.serve))

    def test_actual_plane_rotation_passes_without_visible_handle(self):
        reference = reference_frame()
        camera, _ = camera_for(reference, 3., occlude=True)
        motion = hinge_motion(reference["tracking_root"]["hinge"]["hinge_world"],
                              np.radians(3)*reference["tracking_root"]["hinge"]["opening_sign"])
        result = track_panel(camera, reference, motion)
        decision = verify_panel_step(reference["tracking_root"]["hinge"], reference, result, motion, 3.)
        self.assertTrue(decision["ready"])
        self.assertAlmostEqual(decision["opening_angle_deg"], 3., places=4)
        self.assertFalse(result["vision_audit"]["handle_measured"])
        self.assertTrue(result["vision_audit"]["hidden_handle_anchor_inferred"])

    def test_commanded_rotation_cannot_make_static_door_pass(self):
        reference = reference_frame()
        camera, _ = camera_for(reference, 0., occlude=True)
        motion = hinge_motion(reference["tracking_root"]["hinge"]["hinge_world"],
                              np.radians(3)*reference["tracking_root"]["hinge"]["opening_sign"])
        result = track_panel(camera, reference, motion)
        decision = verify_panel_step(reference["tracking_root"]["hinge"], reference, result, motion, 3.)
        self.assertFalse(decision["ready"])
        self.assertAlmostEqual(decision["opening_angle_deg"], 0., places=4)

    def test_missing_or_foreground_only_depth_rejected(self):
        reference = reference_frame()
        camera, _ = camera_for(reference, 0., foreground=True)
        with self.assertRaises(RuntimeError):
            track_panel(camera, reference)
        camera["depth"][:] = 0
        with self.assertRaises(RuntimeError):
            track_panel(camera, reference)

    def test_wrong_direction_does_not_pass_and_footprint_does_not_grow(self):
        reference = reference_frame()
        camera, _ = camera_for(reference, -2.)
        result = track_panel(camera, reference)
        decision = verify_panel_step(reference["tracking_root"]["hinge"], reference,
                                     result, np.eye(4), 3.)
        self.assertFalse(decision["ready"])
        self.assertIs(result["tracking_root"], reference["tracking_root"])
        self.assertEqual(result["tracking_root"]["corners"].shape, (4, 3))

    def test_foreign_legacy_result_is_not_accepted_as_panel_evidence(self):
        reference = reference_frame()
        with self.assertRaises(RuntimeError):
            verify_panel_step(reference["tracking_root"]["hinge"], reference, reference, np.eye(4), 3.)

    def test_postgrasp_wrist_can_rescue_head_without_detector_or_truth(self):
        reference = reference_frame()
        camera, _ = camera_for()
        task = SimpleNamespace(get_camera_observation=lambda name: camera)
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            with patch("adapters.microwave.panel_tracking.track_panel",
                       side_effect=[RuntimeError("head blocked"), dict(reference)]) as tracker:
                result = observe_panel(task, object(), directory, "closed", [], reference=reference)
        self.assertEqual(result["camera"], "left_camera")
        self.assertEqual(tracker.call_count, 2)

    def test_new_worker_uses_separate_initial_perception_entry(self):
        from adapters.microwave.panel_tracking import PanelWorker
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            worker = PanelWorker(ROOT, directory)
            command, _, _ = worker._launch_spec()
        self.assertEqual(Path(command[1]).name, "perceive_microwave_panel.py")

    def test_isolated_runner_restores_all_overrides_on_error(self):
        spec = importlib.util.spec_from_file_location(
            "ula_panel_dispatch_test", ROOT/"scripts/run_open_microwave_panel_randomized.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        base = runner.randomized.base
        original = base.observe, base.MicrowaveWorker, control.verify_step, base.dump, base.open_door
        def fail(_):
            self.assertIs(base.observe, observe_panel)
            self.assertIs(control.verify_step, verify_panel_step)
            raise RuntimeError("intentional stop")
        with patch.object(runner.randomized, "main", side_effect=fail):
            with self.assertRaises(RuntimeError):
                runner.main([])
        self.assertEqual(original, (base.observe, base.MicrowaveWorker, control.verify_step, base.dump, base.open_door))


if __name__ == "__main__":
    unittest.main()
