import copy
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from adapters.approach_geometry import (
    GeometryApproachConfig, execute_geometry_approach, generate_bounds_candidates,
    generate_geometry_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tra_pipe/RoboTwin"))
from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz, pose_wxyz_to_matrix

SPEC = importlib.util.spec_from_file_location("ula_geometry_runner", ROOT / "scripts/run_place_empty_cup.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def metadata():
    # Deliberately no contact_points_pose, functional points or target annotations.
    return dict(center=[0.0, 0.05, 0.0], extents=[0.08, 0.10, 0.06], scale=[1., 1., 1.])


class GeometryTests(unittest.TestCase):
    def test_target_uses_bounding_top_and_robot_tcp_reference(self):
        cup = np.eye(4)
        cup[:3, 3] = [-0.2, 0.1, 0.74]
        ee_to_tcp = np.eye(4)
        ee_to_tcp[0, 3] = 0.12
        targets, audit = generate_geometry_candidates(cup, metadata(), "left", ee_to_tcp,
                                                      GeometryApproachConfig())
        np.testing.assert_allclose(targets[0]["tcp_world"][:3, 3], [-0.24, 0.15, 0.87])
        np.testing.assert_allclose(targets[0]["ee_world"][:3, 3], [-0.24, 0.15, 0.99])
        self.assertFalse(audit["uses_contact_annotations"])
        for target in targets:
            np.testing.assert_allclose(target["ee_world"] @ ee_to_tcp, target["tcp_world"], atol=1e-12)

    def test_no_contact_field_is_read_even_if_present_and_invalid(self):
        class GeometryOnly(dict):
            def __getitem__(self, key):
                if "contact" in key:
                    raise AssertionError("Contact annotations must not be accessed")
                return super().__getitem__(key)

        values = GeometryOnly(metadata(), contact_points_pose="invalid")
        generate_geometry_candidates(np.eye(4), values, "left", np.eye(4), GeometryApproachConfig())

    def test_object_rotation_and_nonuniform_scale_change_the_bounds(self):
        cup = np.eye(4)
        cup[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        values = dict(center=[0, 0, 0], extents=[2, 4, 6], scale=[0.1, 0.2, 0.3])
        targets, audit = generate_geometry_candidates(cup, values, "left", np.eye(4),
                                                      GeometryApproachConfig())
        np.testing.assert_allclose(audit["bbox_world_min"], [-0.4, -0.1, -0.9])
        np.testing.assert_allclose(targets[0]["tcp_world"][:3, 3], [-0.4, 0, 1.0])

    def test_model_transform_is_applied_and_inputs_are_not_mutated(self):
        values = metadata()
        values["transform_matrix"] = np.eye(4).tolist()
        values["transform_matrix"][2][3] = 0.1
        before = copy.deepcopy(values)
        targets, audit = generate_geometry_candidates(np.eye(4), values, "left", np.eye(4),
                                                      GeometryApproachConfig())
        self.assertAlmostEqual(audit["bbox_world_max"][2], 0.13)
        self.assertAlmostEqual(targets[0]["tcp_world"][2, 3], 0.23)
        self.assertEqual(values, before)

    def test_right_left_mirror_and_yaw_keep_a_downward_approach(self):
        left, _ = generate_geometry_candidates(np.eye(4), metadata(), "left", np.eye(4),
                                               GeometryApproachConfig())
        right, _ = generate_geometry_candidates(np.eye(4), metadata(), "right", np.eye(4),
                                                GeometryApproachConfig())
        self.assertAlmostEqual(left[0]["tcp_world"][0, 3], -right[0]["tcp_world"][0, 3])
        for target in left + right:
            np.testing.assert_allclose(target["tcp_world"][:3, 0], [0, 0, -1], atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(target["tcp_world"][:3, :3]), 1)

    def test_zero_fraction_is_above_centre_and_clearance_is_configurable(self):
        targets, _ = generate_geometry_candidates(np.eye(4), metadata(), "left", np.eye(4),
                                                  GeometryApproachConfig(0.2, 0, (0,)))
        np.testing.assert_allclose(targets[0]["tcp_world"][:3, 3], [0, 0.05, 0.23])

    def test_general_rigid_tcp_offset_is_inverted_in_the_correct_order(self):
        ee_to_tcp = np.eye(4)
        ee_to_tcp[:3, :3] = [[0,-1,0],[1,0,0],[0,0,1]]
        ee_to_tcp[:3, 3] = [0.12, 0.02, -0.03]
        targets, _ = generate_geometry_candidates(np.eye(4), metadata(), "right", ee_to_tcp,
                                                  GeometryApproachConfig())
        for target in targets:
            np.testing.assert_allclose(target["ee_world"] @ ee_to_tcp, target["tcp_world"], atol=1e-12)

    def test_current_tcp_adds_scale_adaptive_reach_facing_position_shell(self):
        bounds = np.array([[-.13, .25, .72], [-.08, .28, .85]])
        current_tcp = np.eye(4)
        current_tcp[:3, 3] = [-.18, -.31, .94]
        targets, audit = generate_bounds_candidates(
            bounds, "left", np.eye(4), GeometryApproachConfig(.02, 1., (0.,)),
            reference_tcp_world=current_tcp)
        self.assertEqual([item["position_source"] for item in targets], [
            "arm_side_bounds", "reach_facing_surface", "reach_facing_observed_radius"])
        self.assertLess(targets[1]["tcp_world"][1, 3], bounds[:, 1].mean())
        self.assertLess(targets[2]["tcp_world"][1, 3], targets[1]["tcp_world"][1, 3])
        self.assertEqual(audit["position_search_source"],
                         "observed_bounds_and_current_tcp_reach_direction")
        for target in targets:
            np.testing.assert_allclose(target["ee_world"], target["tcp_world"])

    def test_invalid_config_bounds_and_transforms_are_rejected(self):
        for kwargs in (dict(top_clearance_m=-1), dict(top_clearance_m=float("nan")),
                       dict(edge_fraction=1.1), dict(yaw_offsets_rad=()),
                       dict(yaw_offsets_rad=(float("inf"),))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GeometryApproachConfig(**kwargs)
        for key,value in (("extents",[0,1,1]), ("scale",[-1,1,1]), ("center",[float("nan"),0,0])):
            values = metadata()
            values[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                generate_geometry_candidates(np.eye(4), values, "left", np.eye(4), GeometryApproachConfig())
        invalid = np.eye(4)
        invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            generate_geometry_candidates(invalid, metadata(), "left", np.eye(4), GeometryApproachConfig())


class FakeTask:
    def __init__(self, failures=0, always_fail=False, invalid_start=False, miss_target=False):
        self.plan_success = True
        self.ee = np.eye(4)
        self.ee_to_tcp = np.eye(4)
        self.ee_to_tcp[0, 3] = 0.12
        self.calls = []
        self.failures = failures
        self.always_fail = always_fail
        self.invalid_start = invalid_start
        self.miss_target = miss_target
        self.robot = SimpleNamespace(left_plan_path=self.plan,
                                     get_left_tcp_pose=lambda: matrix_to_pose_wxyz(self.ee @ self.ee_to_tcp))

    def get_arm_pose(self, arm):
        return matrix_to_pose_wxyz(self.ee)

    def plan(self, target):
        self.calls.append("plan")
        if self.always_fail or self.failures > 0:
            self.failures -= 1
            return dict(status="Fail", planner_status="INVALID_START_STATE" if self.invalid_start else "IK_FAIL")
        return dict(status="Success", position=np.zeros((1, 6)), target=target)

    def take_dense_action(self, controls):
        self.calls.append("execute")
        if not self.miss_target:
            self.ee = pose_wxyz_to_matrix(controls["left_arm"]["target"])
        return True

    def get_camera_observation(self, _camera):
        return dict(rgb=0, depth=0, intrinsic_cv=0, extrinsic_cv=0)

    def close_gripper(self, arm, pos):
        self.calls.append(("prepare", pos))
        return arm, []

    def move(self, _bundle):
        return True

    def grasp_actor(self, *_args, **_kwargs):
        raise AssertionError("Geometry mode must not call annotated grasp_actor")


class GeometryExecutionTests(unittest.TestCase):
    def test_failed_candidates_never_execute_then_success_executes_once(self):
        task = FakeTask(failures=1)
        target, moved, audit = execute_geometry_approach(task, np.eye(4), metadata(), "left",
                                                       GeometryApproachConfig())
        self.assertTrue(moved)
        self.assertEqual(task.calls, ["plan", "plan", "execute"])
        self.assertEqual(audit["selected_candidate"], 1)
        self.assertEqual(audit["planning_attempts"][0]["status"], "Fail")

    def test_all_failed_or_invalid_start_never_execute(self):
        for invalid in (False, True):
            task = FakeTask(always_fail=True, invalid_start=invalid)
            _, moved, audit = execute_geometry_approach(task, np.eye(4), metadata(), "left",
                                                       GeometryApproachConfig())
            self.assertFalse(moved)
            self.assertNotIn("execute", task.calls)
            self.assertEqual(len(task.calls), 1 if invalid else 3)
            self.assertIsNone(audit["selected_candidate"])
            self.assertTrue(task.plan_success)

    def test_execution_failure_never_starts_a_new_candidate(self):
        task = FakeTask()
        task.take_dense_action = lambda controls: task.calls.append("execute_failed") or False
        _, moved, audit = execute_geometry_approach(task, np.eye(4), metadata(), "left",
                                                   GeometryApproachConfig())
        self.assertFalse(moved)
        self.assertEqual(task.calls, ["plan", "execute_failed"])
        self.assertEqual(audit["selected_candidate"], 0)

    def run_approach(self, task):
        args = SimpleNamespace(arm="left", approach_mode="geometry", geometry_top_clearance=0.1,
                               geometry_edge_fraction=1, geometry_yaw_offsets=[0,-0.15,0.15],
                               pregrasp_distance=0.1, position_tolerance=0.02, orientation_tolerance=15)
        utils = ModuleType("envs.utils")
        utils.ArmTag = str
        estimate = SimpleNamespace(cup_world_pose=np.eye(4), coaster_world_center=np.zeros(3), timings_s={})
        perception = SimpleNamespace(estimate=lambda *args: estimate)
        with patch.dict(sys.modules, {"envs.utils":utils}), patch.object(
                runner.json, "loads", return_value=metadata()):
            return runner.approach(task, perception, args)

    def test_runner_without_any_contact_annotations_or_object_truth(self):
        task = FakeTask()
        result = self.run_approach(task)
        self.assertTrue(result["ready"])
        self.assertFalse(result["uses_contact_annotations"])
        self.assertEqual(result["approach_mode"], "geometry")
        self.assertIsNone(result["pregrasp_distance_m"])
        self.assertEqual(task.calls, [("prepare",0.6), "plan", "execute"])

    def test_success_return_without_actual_arrival_cannot_trigger_handoff(self):
        result = self.run_approach(FakeTask(miss_target=True))
        self.assertFalse(result["ready"])

    def test_runner_preserves_candidate_diagnostics_when_no_plan_is_feasible(self):
        result = self.run_approach(FakeTask(always_fail=True))
        self.assertFalse(result["ready"])
        self.assertEqual(result["traditional_actions"], ["prepare_gripper"])
        self.assertEqual(result["geometry_approach"]["failure_reason"], "no_feasible_candidate")


class GeometryArgumentTests(unittest.TestCase):
    def parse(self, flags):
        with patch.object(sys, "argv", ["runner", "--output", "/tmp/ula_geometry_test_args"] + flags):
            return runner.parse_args()

    def test_old_default_is_preserved_and_geometry_is_explicit(self):
        self.assertEqual(self.parse([]).approach_mode, "annotated")
        args = self.parse(["--approach-mode", "geometry", "--geometry-yaw-offsets", "0"])
        self.assertEqual(args.approach_mode, "geometry")
        self.assertEqual(args.geometry_yaw_offsets, [0.0])

    def test_invalid_or_ambiguous_geometry_options_fail(self):
        import io
        for flags in (["--geometry-top-clearance", "nan"],
                      ["--geometry-edge-fraction", "-0.1"],
                      ["--approach-mode", "geometry", "--mode", "traditional"],
                      ["--approach-mode", "geometry", "--pregrasp-distance", "0.2"]):
            with self.subTest(flags=flags), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                self.parse(flags)


if __name__ == "__main__":
    unittest.main()
