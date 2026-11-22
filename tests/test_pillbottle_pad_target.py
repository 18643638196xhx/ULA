from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.pillbottle_pad_target import pad_height_check, recover_blue_pad, square_bounds


def frame(mask):
    rgb = np.full((200, 200, 3), 230, np.uint8)
    rgb[mask] = [0, 0, 255]
    return dict(rgb=rgb, depth=np.full((200, 200), .742, np.float32),
        intrinsic_cv=np.array([[742., 0, 100.], [0, 742., 100.], [0, 0, 1.]]),
        extrinsic_cv=np.eye(4))


class PadTargetTests(unittest.TestCase):
    def test_elevated_or_nonfinite_semantic_target_is_rejected_not_clamped(self):
        self.assertFalse(pad_height_check([.1, .2, .954], .741)["accepted"])
        self.assertFalse(pad_height_check([.1, np.nan, .741], .741)["accepted"])
        self.assertTrue(pad_height_check([.1, .2, .742], .741)["accepted"])

    def test_occlusion_split_blue_fragments_keep_full_observed_extents(self):
        mask = np.zeros((200, 200), bool)
        mask[60:141, 60:73] = True
        mask[60:141, 128:141] = True
        center, audit = recover_blue_pad({"head": frame(mask)}, .74)
        np.testing.assert_allclose(center, [0., 0., .742], atol=1e-6)
        self.assertFalse(audit["uses_pad_cad"])
        self.assertFalse(audit["shape_completion"])

    def test_one_strip_is_not_mistaken_for_complete_pad(self):
        mask = np.zeros((200, 200), bool)
        mask[60:141, 60:80] = True
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            recover_blue_pad({"head": frame(mask)}, .74)

    def test_other_camera_can_supply_missing_extent_without_moving_robot(self):
        first, second = np.zeros((200, 200), bool), np.zeros((200, 200), bool)
        first[60:141, 60:80] = True
        second[60:141, 121:141] = True
        center, audit = recover_blue_pad({"head": frame(first), "wrist": frame(second)}, .74)
        np.testing.assert_allclose(center, [0., 0., .742], atol=1e-6)
        self.assertEqual(len(audit["cameras"]), 2)

    def test_blue_object_above_table_is_not_a_pad(self):
        mask = np.zeros((200, 200), bool)
        mask[60:141, 60:141] = True
        camera = frame(mask)
        camera["depth"][:] = .9
        with self.assertRaisesRegex(RuntimeError, "No credible blue pad"):
            recover_blue_pad({"head": camera}, .74)

    def test_multiple_far_blue_regions_are_not_merged_to_large_target(self):
        xy = np.array([[x, y, .742] for x in (-.2, .2) for y in (-.04, .04)])
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            square_bounds(xy)

    def test_initial_recovery_replaces_cached_bad_target_only_after_validation(self):
        from adapters.pillbottle_known_model import KnownPerception
        perception = KnownPerception.__new__(KnownPerception)
        perception.worker = SimpleNamespace()
        selected = []
        perception.select_reference = lambda value: selected.append(value.coaster_world_center.copy())
        value = SimpleNamespace(coaster_world_center=np.array([.1, .2, .954]),
            observation_audit=dict(table_height_m=.741))
        cameras = []
        task = SimpleNamespace(get_camera_observation=lambda name: cameras.append(name) or {})
        with patch("adapters.pillbottle_pad_target.recover_blue_pad",
                side_effect=[RuntimeError("partial"), RuntimeError("partial"),
                    (np.array([.1, .2, .742]), dict(accepted=True))]):
            audit = perception.resolve_initial_pad(task, value)
        self.assertEqual(cameras, ["head_camera", "left_camera", "right_camera"])
        np.testing.assert_allclose(selected[0], [.1, .2, .742])
        self.assertFalse(audit["initial_check"]["accepted"])

    def test_valid_initial_target_is_not_replaced_or_extra_cameras_requested(self):
        from adapters.pillbottle_known_model import KnownPerception
        perception = KnownPerception.__new__(KnownPerception)
        perception.worker = SimpleNamespace()
        perception.select_reference = lambda value: None
        value = SimpleNamespace(coaster_world_center=np.array([.1, .2, .742]),
            observation_audit=dict(table_height_m=.741))
        perception.resolve_initial_pad(SimpleNamespace(), value)
        np.testing.assert_allclose(value.coaster_world_center, [.1, .2, .742])


if __name__ == "__main__":
    unittest.main()
