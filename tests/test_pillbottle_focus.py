import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]

import numpy as np

from adapters.pillbottle_focus.frontend import (appearance_masks, bottle_proposals,
    bottle_metrics, pad_candidates, choose_unambiguous)


def scene():
    y, x = np.indices((100, 180))
    world = np.stack((x*.002, y*.002, np.full(x.shape, .74)), axis=2)
    rgb = np.full((100, 180, 3), 160, np.uint8)
    mask = np.zeros((100, 180), bool)
    mask[15:65, 20:50] = True
    rgb[15:25, 20:50] = [220, 85, 15]
    rgb[25:65, 20:50] = [210, 205, 180]
    world[15:65, 20:50, 2] = np.linspace(.86, .75, 50)[:, None]
    return rgb, world, np.ones(mask.shape, bool), mask


class PillbottleFocusTests(unittest.TestCase):
    def test_color_priors_are_distinct(self):
        orange, white, blue = appearance_masks(np.array([[[230, 90, 10], [220, 220, 200], [0, 0, 255]]]))
        self.assertEqual(orange.tolist(), [[True, False, False]])
        self.assertEqual(white.tolist(), [[False, True, False]])
        self.assertEqual(blue.tolist(), [[False, False, True]])

    def test_white_table_does_not_connect_bottle_to_background(self):
        rgb, world, valid, mask = scene()
        candidates = bottle_proposals(rgb, world, valid, .74, [])
        self.assertTrue(any(p["source"] == "color_depth" for p in candidates))
        self.assertTrue(all(p["box"][2] < 60 for p in candidates))

    def test_actual_color_depth_semantic_bottle_accepted(self):
        rgb, world, valid, mask = scene()
        audit, measured = bottle_metrics(mask, rgb, world, valid, .74, semantic=.5)
        self.assertTrue(audit["accepted"])
        np.testing.assert_array_equal(mask, measured)

    def test_wrong_color_semantic_false_positive_rejected(self):
        rgb, world, valid, mask = scene()
        rgb[mask] = [20, 60, 230]
        audit, _ = bottle_metrics(mask, rgb, world, valid, .74, semantic=.9)
        self.assertFalse(audit["accepted"])

    def test_color_alone_does_not_override_absent_semantics(self):
        rgb, world, valid, mask = scene()
        audit, _ = bottle_metrics(mask, rgb, world, valid, .74, semantic=0.)
        self.assertIn("no_matching_semantic_evidence", audit["reasons"])

    def test_occluded_cap_only_allowed_for_held_semantic_or_reference_support(self):
        rgb, world, valid, mask = scene()
        rgb[mask] = [210, 205, 180]
        before, _ = bottle_metrics(mask, rgb, world, valid, .74, semantic=.5)
        after, _ = bottle_metrics(mask, rgb, world, valid, .74, semantic=.5, held=True)
        self.assertFalse(before["accepted"])
        self.assertTrue(after["accepted"])
        self.assertTrue(after["cap_occlusion_allowed"])

    def test_no_depth_cannot_be_rescued_by_color(self):
        rgb, world, valid, mask = scene()
        valid[:] = False
        audit, _ = bottle_metrics(mask, rgb, world, valid, .74, semantic=.9)
        self.assertFalse(audit["accepted"])

    def test_elevated_blue_distractor_not_pad(self):
        rgb, world, valid, _ = scene()
        rgb[70:95, 20:45] = [0, 0, 255]
        rgb[70:95, 100:125] = [0, 0, 255]
        world[70:95, 100:125, 2] = .80
        detections = [(np.array([15, 65, 50, 98]), .4), (np.array([95, 65, 130, 98]), .9)]
        pads = pad_candidates(rgb, world, valid, .74, detections)
        self.assertEqual(len(pads), 1)
        self.assertLess(pads[0]["center"][0], .1)

    def test_blue_background_without_semantics_not_pad(self):
        rgb, world, valid, _ = scene()
        rgb[70:95, 20:45] = [0, 0, 255]
        self.assertFalse(pad_candidates(rgb, world, valid, .74, []))

    def test_ambiguous_two_objects_stop_but_duplicate_views_do_not(self):
        a = dict(score=1., center=np.array([0., 0., .8]))
        b = dict(score=.95, center=np.array([.2, 0., .8]))
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            choose_unambiguous([a, b])
        b["center"] = np.array([.01, 0., .8])
        self.assertIs(choose_unambiguous([a, b]), a)

    def test_worker_replaces_only_script_not_motion_or_cad_flags(self):
        spec = importlib.util.spec_from_file_location("focus_runner_test",
            ROOT/"scripts/run_move_pillbottle_pad_focused_randomized.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        worker = module.FocusedWorker(ROOT, ROOT/"results/unused_focus_test", model_id=1)
        worker.placement_policy = "visual"
        command, _, _ = worker._launch_spec()
        self.assertEqual(Path(command[1]).name, "perceive_pillbottle_focused.py")
        self.assertIn("--geometry-guard", command)
        self.assertEqual(command[command.index("--model-id")+1], "1")
        worker.model_id = 2
        with self.assertRaisesRegex(ValueError, "base1"):
            worker._launch_spec()


if __name__ == "__main__":
    unittest.main()
