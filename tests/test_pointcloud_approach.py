import io
import os
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adapters.approach_geometry import GeometryApproachConfig, generate_bounds_candidates
from adapters.pointcloud_geometry import estimate_visible_bounds
from test_geometry_approach import FakeTask, runner


def scene():
    cup = np.zeros((64,64), dtype=bool)
    cup[20:44,20:44] = True
    coaster = np.zeros_like(cup)
    coaster[3:12,3:12] = True
    depth = np.full(cup.shape, 0.76)
    depth[cup] = 0.67
    intrinsic = np.array([[100,0,32],[0,100,32],[0,0,1]])
    extrinsic = np.diag([1.,-1.,-1.,1.])
    extrinsic[2,3] = 1.5
    return cup, coaster, depth, intrinsic, extrinsic


class PointcloudTests(unittest.TestCase):
    def test_bounds_are_observed_surface_not_completed_shape_or_6d_pose(self):
        bounds, audit = estimate_visible_bounds(*scene())
        np.testing.assert_allclose(bounds[:,2], [0.83,0.83])
        self.assertAlmostEqual(audit["table_height_m"], 0.74)
        self.assertTrue(audit["visible_surface_only"])
        self.assertFalse(audit["shape_completion"])
        self.assertFalse(audit["uses_prepared_object_model"])
        targets, _ = generate_bounds_candidates(bounds,"left",np.eye(4),GeometryApproachConfig())
        self.assertAlmostEqual(targets[0]["tcp_world"][2,3],0.93)

    def test_millimetre_depth_produces_same_bounds(self):
        inputs = list(scene())
        expected, _ = estimate_visible_bounds(*inputs)
        inputs[2] = (inputs[2]*1000).astype(np.uint16)
        actual, _ = estimate_visible_bounds(*inputs)
        np.testing.assert_allclose(actual,expected,atol=1e-7)

    def test_missing_depth_table_or_segmentation_fails_without_model_fallback(self):
        for change in ("depth", "table", "mask"):
            inputs = list(scene())
            if change == "depth":
                inputs[2][inputs[0]] = 0
            elif change == "table":
                inputs[2][~inputs[0]] = 0
            else:
                inputs[0][:] = False
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                estimate_visible_bounds(*inputs)

    def test_table_pixels_inside_cup_mask_are_removed(self):
        inputs = list(scene())
        inputs[2][25:30,25:30] = 0.76
        bounds, audit = estimate_visible_bounds(*inputs)
        self.assertLess(audit["cup_points_above_table"], audit["raw_valid_cup_points"])
        np.testing.assert_allclose(bounds[:,2],[0.83,0.83])

    def test_runner_never_reads_cup_metadata_or_object_truth(self):
        task = FakeTask()
        args = SimpleNamespace(arm="left",approach_mode="pointcloud",geometry_top_clearance=0.1,
                               geometry_edge_fraction=1,geometry_yaw_offsets=[0,-0.15,0.15],
                               pregrasp_distance=0.1,position_tolerance=0.02,orientation_tolerance=15)
        estimate = SimpleNamespace(cup_bounds_world=np.array([[-0.2,0.1,0.74],[-0.1,0.2,0.83]]),
                                   observation_audit={},coaster_world_center=np.zeros(3),timings_s={})
        utils = ModuleType("envs.utils")
        utils.ArmTag = str
        with patch.dict(sys.modules,{"envs.utils":utils}), patch.object(
                Path,"read_text",side_effect=AssertionError("No cup model reads allowed")):
            result = runner.approach(task,SimpleNamespace(estimate=lambda *a: estimate),args)
        self.assertTrue(result["ready"])
        self.assertIsNone(result["estimated_cup_world_pose"])
        self.assertFalse(result["geometry_approach"]["requires_known_model_geometry"])
        self.assertEqual(task.calls,[("prepare",0.6),"plan","execute"])

    def test_cli_pointcloud_hybrid_is_explicit_and_pure_traditional_is_unchanged(self):
        for mode in ("traditional","vla"):
            with self.subTest(mode=mode), patch.object(sys,"argv",[
                    "runner","--output","/tmp/ula_args","--mode",mode,"--approach-mode","pointcloud"
            ]), patch("sys.stderr",new=io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args()
        for mode in ("hybrid","grasp-only","approach-only"):
            with patch.object(sys,"argv",["runner","--output","/tmp/ula_args",
                                          "--mode",mode,"--approach-mode","pointcloud"]):
                self.assertEqual(runner.parse_args().approach_mode,"pointcloud")

    def test_cli_default_quality_limits_preserved_and_relaxation_explicit(self):
        with patch.object(sys,"argv",["runner","--output","/tmp/ula_args"]):
            args = runner.parse_args()
            self.assertEqual(args.registration_max_rmse,0.008)
            self.assertEqual(args.registration_max_tilt,25.)
        with patch.object(sys,"argv",["runner","--output","/tmp/ula_args","--approach-mode","pointcloud",
                                      "--registration-max-rmse","0.012","--registration-max-tilt","35"]):
            args = runner.parse_args()
            self.assertEqual(args.registration_max_rmse,0.012)
            self.assertEqual(args.registration_max_tilt,35.)
        for flags in (["--registration-max-rmse","nan"],["--registration-max-tilt","90"],
                      ["--registration-max-rmse","-0.001"]):
            with patch.object(sys,"argv",["runner","--output","/tmp/ula_args"]+flags), \
                    patch("sys.stderr",new=io.StringIO()),self.assertRaises(SystemExit):
                runner.parse_args()

    def run_fake_batch(self, mode):
        pose = SimpleNamespace(p=np.zeros(3))
        actor = SimpleNamespace(get_pose=lambda:pose,get_functional_point=lambda *a:pose)
        task = SimpleNamespace(cup=actor,coaster=actor,plan_success=True,take_action_cnt=0,
                               setup_demo=lambda **k:None,set_instruction=lambda s:None,
                               check_success=lambda:False,is_left_gripper_open=lambda:False,
                               is_right_gripper_open=lambda:True,close_env=lambda:None)
        env_module = ModuleType("envs.place_empty_cup")
        env_module.place_empty_cup = lambda:task
        controller_module = ModuleType("traditional_place_empty_cup.controller")
        controller_module.TraditionalPlaceEmptyCupController = None
        policy = SimpleNamespace(load=lambda:None,model=object())
        worker = SimpleNamespace(fp_iterations=5,summary=lambda:dict(backend="pointcloud"))

        def lifted_grasp(task, policy, initial_z, record, monitor, arm, stall_check=None):
            self.assertIsNotNone(monitor)
            self.assertEqual(arm,"left")
            record["grasp_handoff"] = dict(ready=True)

        original_cwd = Path.cwd()
        self.addCleanup(os.chdir,original_cwd)
        with tempfile.TemporaryDirectory() as tmp, patch.object(sys,"argv",[
                "runner","--output",tmp,"--mode",mode,"--approach-mode","pointcloud"]):
            args = runner.parse_args()
            with patch.dict(sys.modules,{"envs.place_empty_cup":env_module,
                                         "traditional_place_empty_cup.controller":controller_module}), \
                    patch.object(runner,"task_config",return_value={}), \
                    patch.object(runner,"WorldLA",return_value=policy), \
                    patch.object(runner,"snapshot"), \
                    patch.object(runner,"traditional_timing",return_value=nullcontext()), \
                    patch.object(runner,"vla_timing",return_value=nullcontext()), \
                    patch.object(runner,"approach",return_value=dict(ready=True,arm="left")), \
                    patch.object(runner,"rollout",side_effect=lifted_grasp), \
                    patch.object(runner,"finish_placement") as placement, \
                    patch.object(runner,"finish_pointcloud_placement") as pointcloud_placement, \
                    patch("sys.stdout",new=io.StringIO()):
                self.assertEqual(runner.run_batch(args,worker),0)
                placement.assert_not_called()
                self.assertEqual(pointcloud_placement.call_count,1 if mode == "hybrid" else 0)
            report = runner.json.loads((Path(tmp) / "report.json").read_text())
            return report

    def test_grasp_only_batch_counts_grasp_not_placement_and_never_calls_placement(self):
        report = self.run_fake_batch("grasp-only")
        self.assertEqual(report["success_definition"],"stable_lifted_grasp_not_task_placement")
        self.assertTrue(report["episodes"][0]["success"])
        self.assertFalse(report["episodes"][0]["task_success"])

    def test_pointcloud_hybrid_dispatches_no_cad_placement_and_counts_task_success(self):
        report = self.run_fake_batch("hybrid")
        self.assertEqual(report["placement_mode"],"pointcloud_registration")
        self.assertEqual(report["success_definition"],"task_placement")
        self.assertFalse(report["episodes"][0]["success"])


if __name__ == "__main__":
    unittest.main()
