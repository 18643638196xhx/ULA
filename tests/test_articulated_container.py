import unittest
from pathlib import Path

import numpy as np

from adapters.articulated_container import (
    load_articulated_container_model, required_opening_distance,
    resolve_upright_container_pose,
)
from adapters.task_rules import load_articulated_container_rule


ROOT = Path(__file__).resolve().parents[1]


class ArticulatedContainerTests(unittest.TestCase):
    def test_rule_declares_dual_grasp_and_no_control_truth(self):
        rule = load_articulated_container_rule(ROOT / "rules/put_object_cabinet.yaml")
        self.assertEqual(rule.task, "put_object_cabinet")
        self.assertEqual(rule.prismatic_joint, "joint_1")
        self.assertEqual(rule.contact_point_ids, (0, 1))
        self.assertFalse(rule.simulator_truth_for_control)

    def test_cabinet_annotations_follow_visual_root_and_prismatic_axis(self):
        model = load_articulated_container_model(
            ROOT, "036_cabinet", 46653, joint_name="joint_1", link_name="link_1")
        self.assertAlmostEqual(model.maximum_travel_m, .66 * .27)
        root = np.eye(4)
        axis = model.pull_axis_world(root)
        before = model.functional_world(root, 0, 0.0)[:3, 3]
        after = model.functional_world(root, 0, .10)[:3, 3]
        np.testing.assert_allclose(after - before, axis * .10, atol=1e-9)

    def test_opening_is_measured_extent_clamped_by_urdf(self):
        bounds = np.array([[-.05, -.02, .70], [.05, .02, .76]])
        distance = required_opening_distance(bounds, [1, 0, 0], .1782,
                                             position_uncertainty_m=.015,
                                             gripper_clearance_m=.04)
        self.assertAlmostEqual(distance, .13)
        huge = required_opening_distance(
            np.array([[-1, -1, 0], [1, 1, 1]]), [1, 0, 0], .1782)
        self.assertAlmostEqual(huge, .1782)

    def test_upright_pose_uses_rgbd_surface_and_recovers_horizontal_pull(self):
        model = load_articulated_container_model(
            ROOT, "036_cabinet", 46653, joint_name="joint_1", link_name="link_1")
        # Synthetic visible front in the same convention as the real cabinet:
        # mesh X is world X, mesh Y is gravity-up, mesh Z faces -world Y.
        mesh_pose = np.eye(4)
        mesh_pose[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        mesh_pose[:3, 3] = [.02, .16, .959]
        xs = np.linspace(model.mesh_bounds_local[0, 0], model.mesh_bounds_local[1, 0], 40)
        ys = np.linspace(model.mesh_bounds_local[0, 1], model.mesh_bounds_local[1, 1], 30)
        local = np.array([[x, y, model.mesh_bounds_local[1, 2]] for x in xs for y in ys])
        points = local @ mesh_pose[:3, :3].T + mesh_pose[:3, 3]
        root, audit = resolve_upright_container_pose(model, points, [0, -.5, 1.3])
        np.testing.assert_allclose(model.pull_axis_world(root), [0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(
            root @ model.root_to_link_zero, mesh_pose, atol=2e-3)
        self.assertFalse(audit["uses_simulator_truth"])


if __name__ == "__main__":
    unittest.main()
