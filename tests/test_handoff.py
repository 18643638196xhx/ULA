import unittest

import numpy as np

from adapters.handoff import handoff_ready


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.array([0.1, -0.2, 0.8, 1, 0, 0, 0.])

    def test_quaternion_sign_does_not_change_orientation(self):
        equivalent = self.pose.copy()
        equivalent[3:] *= -1
        self.assertTrue(handoff_ready(equivalent, self.pose, True).ready)

    def test_near_target_cannot_override_failed_planning(self):
        self.assertFalse(handoff_ready(self.pose, self.pose, False).ready)

    def test_position_and_orientation_both_gate_the_switch(self):
        far = self.pose.copy()
        far[0] += 0.03
        self.assertEqual(handoff_ready(far, self.pose, True).reason, "outside_position_tolerance")
        rotated = self.pose.copy()
        rotated[3:] = [0, 1, 0, 0]
        self.assertEqual(handoff_ready(rotated, self.pose, True).reason, "outside_orientation_tolerance")

    def test_nonfinite_or_invalid_pose_rejected(self):
        for invalid in ([0] * 7, [np.nan] * 7, [0] * 6):
            with self.assertRaises(ValueError):
                handoff_ready(invalid, self.pose, True)


if __name__ == "__main__":
    unittest.main()
