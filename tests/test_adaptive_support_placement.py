import unittest

import numpy as np

from adapters.adaptive_interaction import ManipulationGeometryState, interaction_targets
from adapters.adaptive_support_placement import fuse_support_clouds, release_decision, support_fit


def plane(z, half, n=12):
    x, y = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]


class AdaptiveSupportPlacementTests(unittest.TestCase):
    def state(self):
        tcp = np.eye(4); tcp[:3, 3] = [0., 0., .8]
        return ManipulationGeometryState(
            tcp_world=tcp,
            hand_to_object=np.eye(4),
            functional_point_object=np.array([0., 0., -.05]),
            target_point_world=np.array([0., 0., .75]),
            target_normal_world=np.array([0., 0., 1.]),
            up_world=np.array([0., 0., 1.]),
            object_points_world=plane(.75, .02),
            target_points_world=plane(.75, .06),
            scene_points_world=plane(.70, .30),
            uncertainty_m=.003,
        )

    def test_support_fit_uses_measured_footprints(self):
        fit = support_fit(plane(.05, .02), np.zeros(3), plane(0., .06), .003)
        self.assertTrue(fit["geometry_can_contain_object"])
        self.assertTrue(np.all(fit["containment_margin_m"] > 0))

    def test_support_fit_rejects_smaller_target(self):
        fit = support_fit(plane(.05, .05), np.zeros(3), plane(0., .03), .003)
        self.assertFalse(fit["geometry_can_contain_object"])

    def test_support_fusion_recovers_complementary_visible_edges(self):
        full = plane(0., .06, n=20)
        left = full[full[:, 0] <= .015]
        right = full[full[:, 0] >= -.015]
        fused, audit = fuse_support_clouds([left, right])
        self.assertEqual(audit["accepted_views"], 2)
        self.assertGreater(np.ptp(fused[:, 0]), np.ptp(left[:, 0]))
        self.assertGreater(np.ptp(fused[:, 0]), np.ptp(right[:, 0]))
        self.assertFalse(audit["uses_simulator_truth"])

    def test_support_fusion_deduplicates_repeated_static_view(self):
        points = plane(0., .06, n=20)
        fused, audit = fuse_support_clouds([points, points.copy()])
        self.assertLessEqual(len(fused), len(points))
        self.assertGreater(audit["raw_points"], audit["points"])
        fit = support_fit(plane(.05, .02), np.zeros(3), fused, .003)
        self.assertTrue(fit["geometry_can_contain_object"])

    def test_release_requires_contact_and_containment(self):
        state = self.state()
        fit = support_fit(plane(.05, .02), np.zeros(3), plane(0., .06), .003)
        targets = interaction_targets(state, candidate_budget=6)
        no_contact = release_decision(
            state, targets["surface_tcp"], fit, {"relation_observed": False}
        )
        contact = release_decision(
            state, targets["surface_tcp"], fit, {"relation_observed": True}
        )
        self.assertFalse(no_contact["ready"])
        self.assertTrue(contact["ready"])
        self.assertFalse(contact["uses_simulator_truth"])


if __name__ == "__main__":
    unittest.main()
