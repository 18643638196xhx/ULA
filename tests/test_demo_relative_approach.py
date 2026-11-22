import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from adapters.approach_geometry import GeometryApproachConfig, generate_bounds_candidates
from adapters.demo_relative_approach import (
    generate_demo_relative_candidates, load_demo_approach_prior,
    prepend_demo_candidates,
)


def anchor_prior(arm="left"):
    return dict(
        schema="ula_demo_relative_approach_v1", task="example",
        mode="visual_bounds_top_anchor", arm=arm,
        interaction_to_approach_world_m=[-.02, -.01, .08],
        approach_ee_rotation_world=np.eye(3).tolist(),
    )


class DemoRelativeApproachTests(unittest.TestCase):
    def test_exact_demo_transfer_is_first_and_uses_current_visual_anchor(self):
        bounds = np.array([[-.2, -.1, .74], [-.1, .1, .84]])
        candidates, audit = generate_demo_relative_candidates(
            anchor_prior(), bounds, "left", np.eye(4), uncertainty_m=.003,
            safe_tcp_rotation_world=np.eye(3))
        np.testing.assert_allclose(candidates[0]["ee_world"][:3, 3],
                                   [-.17, -.01, .92])
        self.assertEqual(candidates[0]["source"], "demo_exact")
        self.assertTrue(all(item["source"].startswith("demo_") for item in candidates))
        self.assertFalse(audit["demonstration_world_position_replayed"])

    def test_visual_error_neighbours_use_measured_uncertainty(self):
        bounds = np.array([[-.2, -.1, .74], [-.1, .1, .84]])
        candidates, audit = generate_demo_relative_candidates(
            anchor_prior(), bounds, "left", np.eye(4), uncertainty_m=.004,
            safe_tcp_rotation_world=np.eye(3))
        nominal = candidates[0]["tcp_world"][:3, 3]
        distances = [np.linalg.norm(item["tcp_world"][:3, 3]-nominal)
                     for item in candidates[1:]]
        np.testing.assert_allclose(distances, [.004]*4)
        self.assertEqual(audit["neighbourhood_radius_source"],
                         "current_RGBD_point_spacing")

    def test_geometry_bounds_candidates_are_after_all_demo_candidates(self):
        bounds = np.array([[-.2, -.1, .74], [-.1, .1, .84]])
        fallback, audit = generate_bounds_candidates(
            bounds, "left", np.eye(4), GeometryApproachConfig())
        combined, merged = prepend_demo_candidates(
            fallback, audit, anchor_prior(), bounds, "left", np.eye(4),
            uncertainty_m=.003)
        self.assertEqual([item["source"] for item in combined[:5]],
                         ["demo_exact"]+["demo_visual_uncertainty_neighbour"]*4)
        self.assertEqual([item["source"] for item in combined[5:7]],
                         ["demo_safe_orientation_variant"]*2)
        self.assertTrue(all(item["source"] == "geometry_bounds_fallback"
                            for item in combined[7:]))
        self.assertEqual(merged["candidate_order"], [
            "demo_exact", "demo_visual_uncertainty_neighbours",
            "demo_safe_orientation_variants",
            "geometry_bounds_fallback"])

    def test_opposite_arm_mirrors_demonstrated_direction_and_orientation(self):
        bounds = np.array([[.1, -.1, .74], [.2, .1, .84]])
        candidates, audit = generate_demo_relative_candidates(
            anchor_prior("left"), bounds, "right", np.eye(4), uncertainty_m=.003,
            safe_tcp_rotation_world=np.eye(3))
        np.testing.assert_allclose(candidates[0]["ee_world"][:3, 3],
                                   [.17, -.01, .92])
        self.assertEqual(audit["runtime_arm"], "right")

    def test_bounds_anchor_is_tcp_not_ee_and_uses_safe_geometry_rotation(self):
        bounds = np.array([[-.2, -.1, .74], [-.1, .1, .84]])
        ee_to_tcp = np.eye(4); ee_to_tcp[0, 3] = .12
        safe = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
        prior = anchor_prior()
        # This valid but unrelated demonstration rotation must not be applied.
        prior["approach_ee_rotation_world"] = np.diag([-1., -1., 1.]).tolist()
        candidates, audit = generate_demo_relative_candidates(
            prior, bounds, "left", ee_to_tcp, uncertainty_m=.003,
            safe_tcp_rotation_world=safe)
        np.testing.assert_allclose(candidates[0]["tcp_world"][:3, 3],
                                   [-.17, -.01, .92])
        np.testing.assert_allclose(candidates[0]["tcp_world"][:3, :3], safe)
        np.testing.assert_allclose(candidates[0]["ee_world"] @ ee_to_tcp,
                                   candidates[0]["tcp_world"])
        self.assertFalse(audit["stored_demo_absolute_rotation_applied"])
        self.assertTrue(audit["ee_tcp_offset_applied_exactly_once"])

    def test_full_object_pose_prior_is_composed_exactly(self):
        prior = dict(schema="ula_demo_relative_approach_v1", task="example",
                     mode="object_6d", arm="left", object_to_ee=np.eye(4).tolist())
        prior["object_to_ee"][0][3] = .1
        object_pose = np.eye(4); object_pose[1, 3] = .2
        candidates, audit = generate_demo_relative_candidates(
            prior, np.array([[0., 0., 0.], [1., 1., 1.]]), "left", np.eye(4),
            object_pose_world=object_pose, uncertainty_m=.003)
        np.testing.assert_allclose(candidates[0]["ee_world"][:3, 3], [.1, .2, 0.])
        self.assertEqual(audit["prior_mode"], "object_6d")

    def test_loader_rejects_an_unlabelled_or_nonrigid_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"prior.json"
            path.write_text(json.dumps(dict(anchor_prior(), schema="wrong")))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_demo_approach_prior(path)
            bad = anchor_prior(); bad["approach_ee_rotation_world"][0][0] = 2.
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "rigid"):
                load_demo_approach_prior(path)


if __name__ == "__main__":
    unittest.main()
