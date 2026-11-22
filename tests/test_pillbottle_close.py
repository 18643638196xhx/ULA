import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"tra_pipe/RoboTwin")]
from adapters.pillbottle_close import bounds_distance, observe_close_candidate, registered_proximity


class CloseTests(unittest.TestCase):
    def data(self):
        points = np.random.default_rng(4).uniform([-.03, -.03, .75], [.03, .03, .85], (800, 3))
        return SimpleNamespace(cup_points_world=points, observation_audit={},
            cup_bounds_world=np.array([points.min(0), points.max(0)]))

    def test_bounds_distance_does_not_need_skin_contact(self):
        self.assertEqual(bounds_distance([[0, 0, 0], [1, 1, 1]], [.5, .5, .5]), 0.)
        self.assertAlmostEqual(bounds_distance([[0, 0, 0], [1, 1, 1]], [1.02, .5, .5]), .02)

    def test_propagated_observed_bounds_require_measured_registration(self):
        ref, current = self.data(), self.data()
        shift = np.array([.01, .01, .04])
        current.cup_points_world += shift
        current.cup_bounds_world += shift
        result = registered_proximity(ref, current, [0.01, .01, .84])
        self.assertTrue(result["accepted"])
        self.assertFalse(result["uses_object_truth"])
        far = registered_proximity(ref, current, [.3, .3, .84])
        self.assertFalse(far["accepted"])
        with patch("adapters.pillbottle_close.estimate_pointcloud_motion", return_value=(None, {"accepted": False})):
            # Direct current bounds still permit a provisional trial. A
            # propagated, currently occluded flank needs successful ICP.
            self.assertTrue(registered_proximity(ref, current, [.01, .01, .84])["accepted"])
            self.assertFalse(registered_proximity(ref, current, [.06, .01, .84])["accepted"])

    def test_clipped_raw_mask_gets_changed_prompt_before_authorizing_lift(self):
        reference, clipped, correct = self.data(), self.data(), self.data()
        clipped.observation_audit["cup_mask_touches_border"] = True
        calls = []
        def hinted(*a, **kw): calls.append("hinted");return correct
        perception = SimpleNamespace(supplementary=lambda *a, **kw: clipped,
            hinted=hinted, select_reference=lambda value: calls.append("selected"),
            reference_path=Path("/tmp/pillbottle_current_surface.npz"))
        task = SimpleNamespace(get_camera_observation=lambda name: {},
            robot=SimpleNamespace(get_left_tcp_pose=lambda: [0, 0, .8, 1, 0, 0, 0]))
        audit = []
        result = observe_close_candidate(task, perception, "left", 3, reference, audit)
        self.assertTrue(result["accepted"])
        self.assertEqual(calls, ["hinted", "selected"])
        self.assertEqual(len(audit), 2)
        self.assertFalse(audit[0]["quality"]["geometry_ok"])


if __name__ == "__main__": unittest.main()
