import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.handover.initial_geometry import (estimate_upright_block, estimate_platform,
    cuboid_surface_distance, yaw_error_modulo_square_degrees,estimate_supported_upright_block)
from adapters.handover.initial_vision import color_masks


def visible_block(yaw=.3, center=(-.1, .1, .84)):
    x, y = np.meshgrid(np.linspace(-.03, .03, 20), np.linspace(-.03, .03, 20))
    top = np.c_[x.ravel(), y.ravel(), np.full(x.size, .1)]
    x, z = np.meshgrid(np.linspace(-.03, .03, 20), np.linspace(-.1, .1, 50))
    front = np.c_[x.ravel(), np.full(x.size, -.03), z.ravel()]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return np.concatenate((top, front))@rotation.T+center


class HandoverInitialTests(unittest.TestCase):
    def test_visible_top_recovers_hidden_center(self):
        pose, audit = estimate_upright_block(visible_block(), .74)
        np.testing.assert_allclose(pose[:3, 3], [-.1, .1, .84], atol=1e-5)
        self.assertTrue(audit["hidden_center_inferred_not_directly_observed"])
        self.assertFalse(audit["uses_object_truth"])

    def test_table_height_and_initial_location_are_not_fixed(self):
        pose, _ = estimate_upright_block(visible_block(center=(.2, -.1, .81)), .71)
        np.testing.assert_allclose(pose[:3, 3], [.2, -.1, .81], atol=1e-5)

    def test_not_a_cube_wrong_shape_rejected(self):
        points = visible_block()
        points[:, 2] = .74+(points[:, 2]-.74)*.25
        with self.assertRaisesRegex(RuntimeError, "extent"):
            estimate_upright_block(points, .74)

    def test_missing_top_or_mismatched_known_height_rejected(self):
        with self.assertRaises(RuntimeError):
            estimate_upright_block(visible_block(center=(0., 0., 1.)), .74)
        with self.assertRaises(ValueError):
            estimate_upright_block(np.zeros((10, 3)), .74)

    def test_cuboid_surface_distance_not_volume_membership(self):
        points = np.array([[.03, 0, 0], [0, 0, 0], [.05, 0, 0]])
        np.testing.assert_allclose(cuboid_surface_distance(points, np.eye(4)), [0, .03, .02])

    def test_platform_measures_top_not_actor_origin(self):
        x, y = np.meshgrid(np.linspace(.1, .2, 20), np.linspace(.05, .15, 20))
        top = np.c_[x.ravel(), y.ravel(), np.full(x.size, .745)]
        center, _ = estimate_platform(top, .74)
        np.testing.assert_allclose(center, [.15, .10, .745], atol=1e-5)
        top[:, 2] = .82
        with self.assertRaisesRegex(RuntimeError, "horizontal"):
            estimate_platform(top, .74)

    def test_color_masks_separate_red_blue_neutral(self):
        red, blue = color_masks(np.array([[[255, 0, 0], [0, 0, 255], [200, 200, 200]]]))
        self.assertEqual(red.tolist(), [[True, False, False]])
        self.assertEqual(blue.tolist(), [[False, True, False]])

    def test_square_yaw_symmetry_is_not_counted_as_error(self):
        pose = np.eye(4)
        pose[:2, :2] = [[0, -1], [1, 0]]
        self.assertAlmostEqual(yaw_error_modulo_square_degrees(pose, np.eye(4)), 0.)

    def test_cropped_top_uses_measured_side_and_table_not_fake_top(self):
        points=visible_block(yaw=0.)
        points=points[points[:,2]<.90]
        pose,audit=estimate_supported_upright_block(points,.74,[0.,-.5,1.5])
        np.testing.assert_allclose(pose[:3,3],[-.1,.1,.84],atol=.001)
        self.assertFalse(audit['measured_top_required'])
        self.assertFalse(audit['uses_object_truth'])

    def test_partial_side_without_enough_width_is_rejected(self):
        points=visible_block(yaw=0.)
        points=points[(points[:,2]<.90)&(abs(points[:,0]+.1)<.01)]
        with self.assertRaises(RuntimeError):
            estimate_supported_upright_block(points,.74,[0.,-.5,1.5])

    def test_side_fallback_respects_changed_table_and_object_location(self):
        points=visible_block(yaw=.3,center=(.2,-.1,.81))
        points=points[points[:,2]<.87]
        # Camera is outside the visible local -Y face.
        camera=np.array([.2,-.1,1.5])+np.array([np.sin(.3),-np.cos(.3),0.])
        pose,_=estimate_supported_upright_block(points,.71,camera)
        np.testing.assert_allclose(pose[:3,3],[.2,-.1,.81],atol=.002)

    def test_depth_components_split_image_connected_different_surfaces(self):
        from adapters.handover.initial_depth import split_depth_masks
        yy,xx=np.indices((20,20))
        world=np.stack([xx*.002,yy*.002,np.full(xx.shape,.8)],axis=2)
        world[:,10:,0]+=.2
        masks=split_depth_masks(np.ones((20,20),bool),world)
        self.assertEqual(len(masks),2)
        self.assertEqual(sorted(int(m.sum()) for m in masks),[200,200])

    def test_curved_red_bottle_patch_does_not_become_cuboid(self):
        angle,z=np.meshgrid(np.linspace(-1.1,1.1,55),np.linspace(.75,.925,60))
        points=np.c_[.03*np.sin(angle.ravel()),-.03*np.cos(angle.ravel()),z.ravel()]
        with self.assertRaises(RuntimeError):
            estimate_supported_upright_block(points,.74,[0.,-.5,1.5])

    def test_sensor_plane_tolerance_is_explicit_and_validated(self):
        with self.assertRaises(ValueError):
            estimate_supported_upright_block(visible_block(),.74,[0,-1,1],plane_tolerance_m=0)

    def test_summary_keeps_no_output_in_valid_denominator(self):
        spec = importlib.util.spec_from_file_location("handover_initial_test_runner",
            ROOT/"scripts/run_handover_initial_randomized.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        records = [dict(seed=0, initialized=False),
            dict(seed=1, initialized=True, task_actions_executed=0, block_output=True,
                 platform_output=True, both_output=True, both_identity_match=True, localization_verified=True),
            dict(seed=2, initialized=True, task_actions_executed=0, block_output=False,
                 platform_output=True, both_output=False)]
        result = runner.summarize(records)
        self.assertEqual(result["initialization_rejections"], [0])
        self.assertEqual(result["rates"]["localization_verified"], .5)
        self.assertEqual(result["block_no_output_seeds"], [2])
        self.assertEqual(result["task_actions_executed"], 0)


if __name__ == "__main__":
    unittest.main()
