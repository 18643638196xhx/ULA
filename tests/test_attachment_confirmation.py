import unittest
from unittest.mock import patch

import numpy as np

from adapters.attachment_confirmation import attachment_consistency, expected_rigid_cloud


def cloud():
    rng = np.random.default_rng(7)
    return rng.uniform([-.03, -.02, .74], [.03, .02, .84], (400, 3))


class AttachmentConfirmationTests(unittest.TestCase):
    def test_expected_cloud_uses_measured_hand_transform(self):
        points = cloud()
        before = np.eye(4)
        after = np.eye(4); after[:3, 3] = [.08, -.02, .04]
        expected, motion = expected_rigid_cloud(points, before, after)
        np.testing.assert_allclose(expected, points + after[:3, 3])
        np.testing.assert_allclose(motion, after)

    def test_stationary_object_cannot_pass_after_hand_moves(self):
        points = cloud()
        before = np.eye(4)
        after = np.eye(4); after[0, 3] = .10
        with patch("adapters.attachment_confirmation.estimate_pointcloud_motion",
                   return_value=(None, {"accepted": False})):
            audit = attachment_consistency(points, points, before, after)
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["reason"],
                         "observed_object_did_not_match_hand_propagated_reference")
        self.assertFalse(audit["uses_simulator_truth"])
        self.assertFalse(audit["extra_trial_motion"])

    def test_rigid_comotion_passes_with_sensor_scaled_residual(self):
        points = cloud()
        before = np.eye(4)
        after = np.eye(4); after[:3, 3] = [.07, .01, .03]
        observed = points + after[:3, 3]
        registration = dict(selected_attempt=0,
            attempts=[dict(trimmed_rmse_m=.001)], accepted=True)
        with patch("adapters.attachment_confirmation.estimate_pointcloud_motion",
                   return_value=(np.eye(4), registration)):
            audit = attachment_consistency(points, observed, before, after)
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["reason"], "rigid_comotion_confirmed")
        self.assertLessEqual(audit["rigid_residual_m"], audit["acceptance_bound_m"])


if __name__ == "__main__":
    unittest.main()
