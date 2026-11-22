import unittest

import numpy as np

from adapters.adaptive_interaction import (
    ManipulationGeometryState, combined_uncertainty, contact_feedback,
    interaction_targets, measured_resolution, minimum_safe_escape,
)


def grid(z, x0=-.1, x1=.1, y0=-.1, y1=.1, n=12):
    x, y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]


class AdaptiveInteractionTests(unittest.TestCase):
    def state(self, obstacle=None):
        tcp = np.eye(4); tcp[:3, 3] = [0, 0, .8]
        hand_to_object = np.eye(4)
        object_points = grid(.76, -.02, .02, -.02, .02, 6)
        target_points = grid(.78, .18, .22, -.02, .02, 6)
        scene = grid(.75, -.3, .3, -.3, .3, 20)
        if obstacle is not None:
            scene = np.vstack((scene, grid(obstacle, .08, .12, -.03, .03, 6)))
        return ManipulationGeometryState(tcp, hand_to_object, np.array([0, 0, -.04]),
            np.array([.2, 0, .78]), np.array([0, 0, 1]), np.array([0, 0, 1]),
            object_points, target_points, scene, .004)

    def test_uncertainty_comes_from_cloud_spacing_and_arrival_residual(self):
        a, b = grid(0, n=8), grid(.1, n=10)
        value = combined_uncertainty(a, b, [.001, .003])
        self.assertGreater(value, measured_resolution(a))
        self.assertGreater(value, measured_resolution(b))

    def test_escape_is_zero_without_obstacle_above_support_uncertainty(self):
        target, audit = minimum_safe_escape(self.state())
        self.assertGreaterEqual(audit["escape_distance_m"], 0)
        self.assertAlmostEqual(target[2, 3] - .8, audit["escape_distance_m"])
        self.assertFalse(audit["task_specific_clearance_constant"])

    def test_obstacle_height_increases_escape_from_same_geometry(self):
        low = minimum_safe_escape(self.state())[1]["escape_distance_m"]
        # Obstacles inside the held object's swept height must raise escape;
        # points above its top are deliberately treated as robot/gripper self.
        high = minimum_safe_escape(self.state(.762))[1]["escape_distance_m"]
        self.assertGreater(high, low)

    def test_precontact_and_contact_bound_equal_measured_uncertainty(self):
        state = self.state()
        targets = interaction_targets(state)
        pre = targets["precontact_tcp"] @ state.hand_to_object
        limit = targets["contact_limit_tcp"] @ state.hand_to_object
        pre_point = pre[:3, :3] @ state.functional_point_object + pre[:3, 3]
        limit_point = limit[:3, :3] @ state.functional_point_object + limit[:3, 3]
        self.assertAlmostEqual(pre_point[2] - state.target_point_world[2], state.uncertainty_m)
        self.assertAlmostEqual(limit_point[2] - state.target_point_world[2], -state.uncertainty_m)
        gaps = targets["audit"]["planner_fallback_gaps_m"]
        self.assertEqual(gaps, sorted(gaps))
        self.assertAlmostEqual(gaps[0], state.uncertainty_m)
        self.assertEqual(len(gaps), 6)
        self.assertEqual(len(gaps), len(targets["precontact_tcp_candidates"]))
        self.assertFalse(targets["audit"]["task_specific_distance_constant"])

    def test_contact_feedback_uses_geometry_and_tracking_not_contact_truth(self):
        state = self.state(); targets = interaction_targets(state)
        result = contact_feedback(state, targets["surface_tcp"], targets["surface_tcp"], .002)
        self.assertTrue(result["relation_observed"])
        self.assertFalse(result["uses_simulator_contact_truth"])

    def test_being_near_but_above_surface_is_not_called_contact(self):
        state = self.state(); targets = interaction_targets(state)
        result = contact_feedback(
            state, targets["precontact_tcp"], targets["precontact_tcp"], .002
        )
        self.assertTrue(result["geometry_within_uncertainty"])
        self.assertFalse(result["estimated_surface_crossed"])
        self.assertFalse(result["relation_observed"])

    def test_desired_object_rotation_and_candidate_budget_are_generic_inputs(self):
        state = self.state()
        rotation = np.diag([-1., -1., 1.])
        targets = interaction_targets(
            state, desired_object_rotation_world=rotation, candidate_budget=3
        )
        desired = targets["surface_tcp"] @ state.hand_to_object
        np.testing.assert_allclose(desired[:3, :3], rotation)
        self.assertEqual(len(targets["precontact_tcp_candidates"]), 3)

    def test_invalid_geometry_is_rejected(self):
        state = self.state()
        with self.assertRaises(ValueError):
            ManipulationGeometryState(np.zeros((4, 4)), state.hand_to_object,
                state.functional_point_object, state.target_point_world,
                state.target_normal_world, state.up_world, state.object_points_world,
                state.target_points_world, state.scene_points_world, state.uncertainty_m)


if __name__ == "__main__":
    unittest.main()
