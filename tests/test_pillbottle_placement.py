import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.pillbottle_placement import VerifiedSurfacePerception, lift_consistency
from adapters.pointcloud_registration import RegistrationConfig


class PlacementTests(unittest.TestCase):
    def test_global_lift_rejects_stationary_or_dropped_surface_despite_local_gate(self):
        points = np.random.default_rng(2).uniform([-.03, -.03, .75], [.03, .03, .87], (800, 3))
        initial = SimpleNamespace(cup_points_world=points, cup_bounds_world=np.array([points.min(0), points.max(0)]),
            observation_audit=dict(table_height_m=.74))
        for lift, expected in [(.05, True), (0., False), (-.01, False)]:
            current = SimpleNamespace(cup_points_world=points+[0, 0, lift])
            self.assertEqual(lift_consistency(initial, current, RegistrationConfig())["ready"], expected)

    def test_cache_requires_same_robot_pose_and_refreshes_pad_only(self):
        points = np.random.default_rng(1).uniform(0, .1, (100, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"surface.npz"
            np.savez(path, cup_points_world=points, cup_bounds_world=[points.min(0), points.max(0)],
                observation_audit_json=np.asarray(json.dumps({})), timings_json=np.asarray(json.dumps({})))
            ee = np.array([0., 0., 1., 1., 0., 0., 0.])
            task = SimpleNamespace(get_arm_pose=lambda arm: ee)
            calls = []
            def target(frame):
                calls.append(frame)
                return SimpleNamespace(coaster_world_center=np.array([.2, .1, .74]), timings_s={})
            pad = SimpleNamespace(output=Path(tmp), pad_only=target)
            verification = dict(complete=True, verified_surface_path=path, verified_surface_ee_pose=ee.copy())
            perception = VerifiedSurfacePerception(task, "left", pad, verification)
            estimate = perception.estimate(None, None, None, None)
            np.testing.assert_array_equal(estimate.cup_points_world, points)
            self.assertEqual(len(calls), 1)
            ee[0] = .1
            with self.assertRaisesRegex(RuntimeError, "invalidated"):
                perception.estimate(None, None, None, None)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__": unittest.main()
