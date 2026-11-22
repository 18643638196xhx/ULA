import unittest
from unittest.mock import patch

import numpy as np

from adapters.microwave import geometry, panel_tracking
from adapters.microwave.no_extent import fit_panel_without_extent, tracking_without_extent


def surface(width=.15, height=.13):
    x, z = np.meshgrid(np.linspace(-width/2, width/2, 32),
                       np.linspace(.9-height/2, .9+height/2, 32))
    return np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))


class NoExtentTests(unittest.TestCase):
    def test_valid_fit_is_numerically_identical(self):
        old = geometry.fit_panel(surface(), [0, -1, 0])
        new = fit_panel_without_extent(surface(), [0, -1, 0])
        self.assertEqual(old.keys(), new.keys())
        for key in old:
            np.testing.assert_equal(old[key], new[key])

    def test_small_patch_passes_only_new_extent_policy(self):
        points = surface(height=.06)
        with self.assertRaisesRegex(RuntimeError, "extent"):
            geometry.fit_panel(points, [0, -1, 0])
        fitted = fit_panel_without_extent(points, [0, -1, 0])
        self.assertLess(fitted["height_m"], .08)
        self.assertGreater(fitted["height_m"], .04)

    def test_degenerate_line_and_missing_depth_still_rejected(self):
        with self.assertRaises(ValueError):
            fit_panel_without_extent(surface()[:10], [0, -1, 0])
        with self.assertRaises(RuntimeError):
            fit_panel_without_extent(surface(height=.001), [0, -1, 0])
        points = surface()
        points[0] = np.nan
        with self.assertRaises(ValueError):
            fit_panel_without_extent(points, [0, -1, 0])

    def test_only_tracking_call_temporarily_replaces_fit(self):
        original_fit = panel_tracking.fit_panel
        def tracker():
            self.assertIs(panel_tracking.fit_panel, fit_panel_without_extent)
            self.assertIs(geometry.fit_panel, original_fit)
            return dict(vision_audit={})
        with patch.object(panel_tracking, "track_panel", tracker):
            with tracking_without_extent():
                self.assertIs(panel_tracking.fit_panel, original_fit)
                result = panel_tracking.track_panel()
                self.assertFalse(result["vision_audit"]["fixed_panel_extent_check_enabled"])
                self.assertIs(panel_tracking.fit_panel, original_fit)
            self.assertIs(panel_tracking.track_panel, tracker)

    def test_restore_after_tracking_exception(self):
        original_fit = panel_tracking.fit_panel
        def tracker():
            raise RuntimeError("missing depth")
        with patch.object(panel_tracking, "track_panel", tracker):
            with self.assertRaisesRegex(RuntimeError, "missing depth"):
                with tracking_without_extent():
                    panel_tracking.track_panel()
            self.assertIs(panel_tracking.fit_panel, original_fit)
            self.assertIs(panel_tracking.track_panel, tracker)


if __name__ == "__main__":
    unittest.main()
