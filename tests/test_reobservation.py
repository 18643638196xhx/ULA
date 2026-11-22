from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tra_pipe/RoboTwin"))
from adapters.perception import ExternalPerception
from adapters.pointcloud_registration import RegistrationConfig
from adapters.reobservation import (ReobservationRejected, observation_quality,
                                     observe_before_approach, observe_for_placement)
from adapters.rgbd_perception import GroundedPointCloudPerception, depth_foreground_seed
from test_geometry_approach import runner


def estimate(*, bad=False, clipped=False, marker=0.):
    bounds = np.array([[-.04, -.04, .745], [.04, .04, .83]])
    # A genuine anomaly is an unreadable surface, not merely a tall object:
    # vertical aspect is only a diagnostic warning now.
    count = 20 if bad else 100
    points = np.random.default_rng(7).uniform(bounds[0], bounds[1], (count, 3))+marker
    return SimpleNamespace(cup_bounds_world=bounds, cup_points_world=points,
        coaster_world_center=np.array([.1, .2, .745]), timings_s={},
        observation_audit=dict(table_height_m=.74, cup_mask_touches_border=clipped))


def camera():
    return dict(rgb=np.zeros((20, 30, 3)), depth=np.ones((20, 30)),
                intrinsic_cv=np.eye(3), extrinsic_cv=np.eye(4))


class BeforeApproachTests(unittest.TestCase):
    def task(self):
        # No actor truth, mesh, robot region or motion APIs are available.
        return SimpleNamespace(get_camera_observation=Mock(side_effect=lambda _: camera()))

    def test_clean_head_fast_path_never_resegments(self):
        p = SimpleNamespace(estimate=Mock(return_value=estimate()), supplementary=Mock())
        audit = {}
        out = observe_before_approach(self.task(), p, "left", audit)
        p.supplementary.assert_not_called()
        self.assertEqual(out.observation_audit["registration_reference_camera"], "head_camera")
        self.assertEqual(len(audit["attempts"]), 1)

    def test_anomalous_geometry_resegments_before_any_motion(self):
        p = SimpleNamespace(estimate=Mock(return_value=estimate(bad=True)),
                            supplementary=Mock(return_value=estimate()))
        audit = {}
        out = observe_before_approach(self.task(), p, "left", audit)
        self.assertTrue(audit["resolved"])
        self.assertEqual(audit["selected_attempt"], 1)
        self.assertTrue(p.supplementary.call_args.kwargs["refine_cup"])
        self.assertFalse(audit["observation_motion"])
        self.assertTrue(observation_quality(out)["geometry_ok"])

    def test_clipped_refined_head_is_only_for_coarse_approach(self):
        p = SimpleNamespace(estimate=Mock(return_value=estimate(clipped=True)),
                            supplementary=Mock(return_value=estimate(clipped=True)))
        audit = {}
        observe_before_approach(self.task(), p, "left", audit)
        self.assertTrue(audit["requires_postapproach_wrist"])
        self.assertEqual(len(audit["attempts"]), 2)

    def test_wrist_fallback_preserves_head_coaster_and_marks_wrist_reference(self):
        wrist = estimate()
        wrist.coaster_world_center[:] = np.nan
        p = SimpleNamespace(estimate=Mock(return_value=estimate(bad=True)),
                            supplementary=Mock(side_effect=[estimate(bad=True), wrist]))
        audit = {}
        out = observe_before_approach(self.task(), p, "left", audit)
        np.testing.assert_allclose(out.coaster_world_center, [.1, .2, .745])
        self.assertEqual(out.observation_audit["registration_reference_camera"], "left_camera")

    def test_exhaustion_is_bounded_and_audit_survives(self):
        p = SimpleNamespace(estimate=Mock(return_value=estimate(bad=True)),
                            supplementary=Mock(return_value=estimate(bad=True)))
        audit = {}
        with self.assertRaises(ReobservationRejected) as caught:
            observe_before_approach(self.task(), p, "left", audit)
        self.assertIs(caught.exception.reobservation_audit, audit)
        self.assertEqual(len(audit["attempts"]), 3)
        self.assertEqual(p.supplementary.call_count, 2)
        self.assertFalse(audit["resolved"])

    def test_auto_arm_does_not_guess_from_invalid_geometry(self):
        p = SimpleNamespace(estimate=Mock(return_value=estimate(bad=True)),
                            supplementary=Mock(return_value=estimate(bad=True)))
        audit = {}
        with self.assertRaises(ReobservationRejected):
            observe_before_approach(self.task(), p, "auto", audit)
        self.assertEqual(len(audit["attempts"]), 2)

    def test_quality_growth_uses_episode_observation_not_cad(self):
        tall = estimate()
        tall.cup_bounds_world = np.array([[-.04, -.04, .745], [.04, .04, 1.0]])
        tall.cup_points_world = np.random.default_rng(7).uniform(
            tall.cup_bounds_world[0], tall.cup_bounds_world[1], (100, 3))
        q = observation_quality(tall, estimate().cup_bounds_world)
        self.assertIn("extent_growth_from_episode_reference", q["reasons"])
        # Vertical aspect is a warning only: bowls/containers/shoes are tall.
        self.assertIn("elongated_vertical_extent", q["warnings"])
        self.assertNotIn("elongated_vertical_extent", q["reasons"])
        e = estimate()
        e.cup_points_world[0] = np.nan
        self.assertFalse(observation_quality(e)["geometry_ok"])

    def test_runner_cannot_prepare_gripper_or_plan_when_reobservation_rejects(self):
        task = SimpleNamespace(move=Mock(), get_camera_observation=Mock())
        args = SimpleNamespace(approach_mode="pointcloud", pointcloud_reobserve=True, arm="left")
        utils = ModuleType("envs.utils")
        utils.ArmTag = str
        with patch.dict(sys.modules, {"envs.utils": utils}), \
                patch.object(runner, "observe_before_approach", side_effect=ReobservationRejected("bad", {})), \
                self.assertRaises(ReobservationRejected):
            runner.approach(task, SimpleNamespace(), args)
        task.move.assert_not_called()


class PlacementReobservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"selected_reference.npz"
        ref = estimate()
        self.ref = ref
        np.savez(self.path, cup_points_world=ref.cup_points_world,
                 cup_bounds_world=ref.cup_bounds_world,
                 observation_audit_json=np.asarray(json.dumps(ref.observation_audit)))
        self.task = SimpleNamespace(get_camera_observation=Mock(side_effect=lambda _: camera()))

    def run_observation(self, p, registration=None):
        self.audit = {}
        with patch("adapters.pointcloud_registration.estimate_pointcloud_motion",
                   side_effect=registration or (lambda *a: (np.eye(4), {"accepted": True}))) as call:
            output = observe_for_placement(self.task, p, "left", self.path, RegistrationConfig(), self.audit)
        return output, call

    def test_raw_geometry_anomaly_does_not_reach_icp_until_refined(self):
        p = SimpleNamespace(estimate=Mock(return_value=estimate(bad=True)),
                            supplementary=Mock(return_value=estimate()))
        output, call = self.run_observation(p)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(self.audit["selected_attempt"], 1)

    def test_icp_rejection_also_triggers_changed_prompt(self):
        p = SimpleNamespace(estimate=Mock(return_value=estimate()), supplementary=Mock(return_value=estimate()))
        output, call = self.run_observation(p, registration=Mock(side_effect=[
            (None, {"accepted": False}), (np.eye(4), {"accepted": True})]))
        self.assertEqual(call.call_count, 2)
        self.assertTrue(self.audit["resolved"])

    def test_clipped_head_retries_then_uses_unchanged_icp_gates(self):
        p = SimpleNamespace(estimate=Mock(return_value=estimate(clipped=True)),
                            supplementary=Mock(return_value=estimate(clipped=True)))
        output, call = self.run_observation(p)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(self.audit["selected_attempt"], 1)
        self.assertEqual(call.call_args.args[2], RegistrationConfig())

    def test_cross_camera_fallback_requires_matching_saved_wrist_reference(self):
        wrist_ref = estimate(marker=1.)
        cam = camera()
        np.savez(self.path.parent/"wrist_reference_frame.npz", rgb=cam["rgb"], depth=cam["depth"],
                 intrinsic=cam["intrinsic_cv"], world_to_camera=cam["extrinsic_cv"], camera_name="left_camera")
        wrist = estimate()
        wrist.coaster_world_center[:] = np.nan
        def supplementary(cam, label, **kw):
            if label == "reobserve_saved_wrist_reference":
                return wrist_ref
            return estimate(bad=True) if label == "reobserve_place_1" else wrist
        p = SimpleNamespace(estimate=Mock(return_value=estimate(bad=True)), supplementary=Mock(side_effect=supplementary))
        output, call = self.run_observation(p)
        np.testing.assert_allclose(call.call_args.args[0], wrist_ref.cup_points_world)
        np.testing.assert_allclose(output[0].coaster_world_center, [.1, .2, .745])
        self.assertEqual(self.audit["matching_reference_camera"], "left_camera")

    def test_no_saved_wrist_reference_rejects_instead_of_mixing_views(self):
        p = SimpleNamespace(estimate=Mock(return_value=estimate(bad=True)),
                            supplementary=Mock(return_value=estimate(bad=True)))
        audit = {}
        with self.assertRaises(ReobservationRejected), \
                patch("adapters.pointcloud_registration.estimate_pointcloud_motion") as call:
            observe_for_placement(self.task, p, "left", self.path, RegistrationConfig(), audit)
        call.assert_not_called()
        self.assertEqual(audit["attempts"][-1]["rejected"], "no_matching_pregrasp_wrist_reference")
        self.assertEqual(p.supplementary.call_count, 1)

    def test_primary_wrist_reference_gets_current_wrist_not_head_surface(self):
        np.savez(self.path, cup_points_world=self.ref.cup_points_world,
                 cup_bounds_world=self.ref.cup_bounds_world,
                 observation_audit_json=np.asarray(json.dumps(dict(self.ref.observation_audit,
                     registration_reference_camera="left_camera"))))
        p = SimpleNamespace(estimate=Mock(return_value=estimate(bad=True)),
                            supplementary=Mock(return_value=estimate()))
        output, call = self.run_observation(p)
        self.assertEqual(self.audit["selected_camera"], "left_camera")
        self.assertEqual([c.args[0] for c in self.task.get_camera_observation.call_args_list],
                         ["head_camera", "left_camera"])
        np.testing.assert_allclose(call.call_args.args[0], self.ref.cup_points_world)


class RequestAndSegmentationTests(unittest.TestCase):
    def test_depth_prompt_uses_lower_foreground_without_color_or_robot_identity(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:15, 10:30] = True
        mask[20:30, 10:30] = True
        depth = np.full((40, 40), .76)
        depth[5:15, 10:30] = .50  # higher structure
        depth[20:30, 10:30] = .67  # lower foreground, above table
        extrinsic = np.diag([1., -1., -1., 1.])
        extrinsic[2, 3] = 1.5
        seed = depth_foreground_seed(mask, depth, np.eye(3), extrinsic, .74, [0, 0, 39, 39])
        self.assertGreaterEqual(seed[1], 20)
        self.assertLess(seed[1], 30)

    def test_saved_reference_capture_is_lazy_and_refinement_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = SimpleNamespace(fp_iterations=5, estimate_files=Mock(return_value=estimate()))
            p = ExternalPerception(Path(tmp), Path(tmp)/"frames", worker=worker)
            saved = p.capture_reference_view(camera(), "left_camera")
            worker.estimate_files.assert_not_called()
            with np.load(saved, allow_pickle=False) as data:
                self.assertEqual(str(data["camera_name"]), "left_camera")
            p.supplementary(camera(), "refined", refine_cup=True)
            with np.load(p.output/"refined/rgbd.npz", allow_pickle=False) as data:
                self.assertTrue(bool(data["refine_cup"]))
                self.assertTrue(bool(data["cup_only"]))

    def test_resegmentation_changes_sam_prompt_and_embeds_crop_back_in_full_frame(self):
        obj = GroundedPointCloudPerception.__new__(GroundedPointCloudPerception)
        obj.torch = SimpleNamespace(inference_mode=nullcontext, autocast=lambda *a, **kw: nullcontext(), bfloat16=0)
        shape = []
        def set_image(image):
            shape[:] = image.shape[:2]
        def predict(**kwargs):
            self.assertTrue(kwargs["multimask_output"])
            self.assertEqual(kwargs["point_coords"].shape, (1, 2))
            masks = np.zeros((3, *shape), dtype=bool)
            masks[1, 3:8, 3:8] = True
            return masks, np.array([.1, .9, .2]), None
        obj.sam = SimpleNamespace(set_image=set_image, predict=predict)
        result = obj._refine_cup_mask(np.zeros((50, 60, 3), dtype=np.uint8), [15, 15, 35, 35])
        self.assertEqual(result.shape, (50, 60))
        self.assertEqual(result.sum(), 25)
        self.assertFalse(result[:10].any())
        with self.assertRaisesRegex(RuntimeError, "no credible foreground" ):
            obj._refine_cup_mask(np.zeros((50, 60, 3), dtype=np.uint8),
                                 [15, 15, 35, 35], mask_validator=lambda _: False)

    def test_cli_opt_in_and_nonpointcloud_rejected(self):
        with patch.object(sys, "argv", ["runner", "--output", "/tmp/ula_args"]):
            self.assertFalse(runner.parse_args().pointcloud_reobserve)
        with patch.object(sys, "argv", ["runner", "--output", "/tmp/ula_args", "--pointcloud-reobserve"]), \
                patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args()


if __name__ == "__main__":
    unittest.main()
