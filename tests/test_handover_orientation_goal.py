import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from adapters.handover.orientation_goal import SupportConstraint,project_orientation


class OrientationGoalTests(unittest.TestCase):
    def test_already_acceptable_keeps_arbitrary_yaw_exactly(self):
        value=Rotation.from_euler("zy",[47,5],degrees=True).as_matrix()
        goal,audit=project_orientation(value)
        np.testing.assert_array_equal(goal,value)
        self.assertEqual(audit["decision"],"preserve")
        self.assertEqual(audit["rotation_needed_deg"],0.)

    def test_large_tilt_gets_only_minimum_required_correction(self):
        value=Rotation.from_euler("y",30,degrees=True).as_matrix()
        goal,audit=project_orientation(value)
        self.assertAlmostEqual(np.degrees(np.arccos(goal[2,2])),10.)
        self.assertAlmostEqual(audit["rotation_needed_deg"],20.)
        self.assertEqual(audit["yaw_constraint"],"free")

    def test_uncertainty_reserves_margin_inside_task_cone(self):
        value=Rotation.from_euler("x",20,degrees=True).as_matrix()
        goal,audit=project_orientation(value,uncertainty_deg=3.)
        self.assertAlmostEqual(np.degrees(np.arccos(goal[2,2])),7.)
        self.assertAlmostEqual(audit["rotation_needed_deg"],13.)

    def test_other_task_can_require_exact_upright(self):
        value=Rotation.from_euler("x",5,degrees=True).as_matrix()
        goal,audit=project_orientation(value,SupportConstraint(max_tilt_deg=0.))
        np.testing.assert_allclose(goal[:,2],[0,0,1],atol=1e-8)
        self.assertEqual(audit["decision"],"correct_required_tilt")

    def test_antiparallel_and_bad_input(self):
        value=Rotation.from_euler("x",180,degrees=True).as_matrix()
        goal,_=project_orientation(value)
        self.assertAlmostEqual(np.degrees(np.arccos(goal[2,2])),10.)
        with self.assertRaises(ValueError):
            project_orientation(np.zeros((3,3)))


if __name__=="__main__":
    unittest.main()
