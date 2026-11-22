from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.cup_target_focus import blue_cup_candidates, visual_table_height, FocusedCupPerception
from adapters.cup_vla_focus import soften_background, projected_rectangle, focused_observation


def scene():
    y, x = np.indices((100, 160))
    world = np.stack((x*.0025, y*.002, np.full(x.shape, .74)), axis=2)
    rgb = np.full((100, 160, 3), 210, np.uint8)
    rgb[20:61, 20:56] = [80, 130, 170]
    world[20:61, 20:56, 2] = np.linspace(.84, .75, 41)[:, None]
    return rgb, world, np.ones((100, 160), bool)


class CupTargetFocusTests(unittest.TestCase):
    def test_blue_flat_distractor_does_not_replace_upright_cup(self):
        rgb, world, valid = scene()
        rgb[20:61, 100:146] = [0, 80, 220]
        world[20:61, 100:146, 2] = np.linspace(.78, .76, 41)[:, None]
        good, bad = blue_cup_candidates(rgb, world, valid, .74)
        self.assertEqual(len(good), 1)
        self.assertLess(good[0]["bbox"][2], 60)
        self.assertTrue(any(b["reason"] == "not_initial_upright_cup_shape" for b in bad))

    def test_blue_table_removed_before_connected_component_grouping(self):
        rgb, world, valid = scene()
        rgb[:] = [80, 130, 170]
        good, bad = blue_cup_candidates(rgb, world, valid, .74)
        self.assertEqual(len(good), 1)
        self.assertLess(good[0]["pixels"], 2000)

    def test_far_identity_jump_rejected_even_with_blue_cup_shape(self):
        rgb, world, valid = scene()
        good, bad = blue_cup_candidates(rgb, world, valid, .74,
            previous_center=np.array([1., 1., .8]))
        self.assertFalse(good)
        self.assertTrue(any(b["reason"] == "too_far_from_previous_visual_target" for b in bad))

    def test_visual_table_estimate_follows_depth_not_fixed_height(self):
        rgb, world, valid = scene()
        world[..., 2] -= .025
        self.assertAlmostEqual(visual_table_height(world, valid), .715)

    def test_episode_reset_does_not_carry_old_targets(self):
        vision = FocusedCupPerception.__new__(FocusedCupPerception)
        vision.previous_cup_center = np.ones(3)
        vision.reference_pad = np.ones(3)
        vision.reference_table_z = 100.
        vision.reset_episode()
        self.assertIsNone(vision.previous_cup_center)
        self.assertIsNone(vision.reference_pad)
        self.assertIsNone(vision.reference_table_z)

    def test_soft_focus_preserves_roi_pixels_shape_and_original_input(self):
        rgb = np.random.default_rng(1).integers(0, 256, (60, 80, 3), dtype=np.uint8)
        saved = rgb.copy()
        focused = soften_background(rgb, [[20, 20, 40, 40]])
        np.testing.assert_array_equal(rgb, saved)
        np.testing.assert_array_equal(focused[20:41, 20:41], rgb[20:41, 20:41])
        self.assertEqual(focused.shape, rgb.shape)
        self.assertFalse(np.array_equal(focused[:5, :5], rgb[:5, :5]))

    def test_projection_rejects_target_behind_camera(self):
        frame = dict(rgb=np.zeros((50, 50, 3), np.uint8), extrinsic_cv=np.eye(4),
            intrinsic_cv=np.array([[20., 0, 25], [0, 20., 25], [0, 0, 1.]]))
        self.assertIsNone(projected_rectangle(np.array([[0., 0., -1.]]), frame))

    def test_focus_does_not_invent_roi_without_accepted_target(self):
        original = dict(observation={})
        value, audit = focused_observation(SimpleNamespace(), original, SimpleNamespace())
        self.assertIs(value, original)
        self.assertFalse(audit["applied"])


if __name__ == "__main__":
    unittest.main()
