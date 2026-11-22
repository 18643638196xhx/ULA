"""CPU-only tests; no simulator, model load, CUDA context or external writes."""

from contextlib import nullcontext
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tra_pipe/RoboTwin"))
from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz, pose_wxyz_to_matrix
from adapters.microwave.geometry import (approach_target, estimate_hinge, fit_panel,
    hinge_motion, opening_angle, verify_step, protruding_handle, isolate_door_surface,
    panel_corners, update_panel_corners)
from adapters.microwave.control import (approach_handle, open_door, vla_close_only,
    traditional_close_only, without_object_truth, record_topp)
from adapters.microwave.perception import (MicrowaveWorker, observe, select_handle_box,
    validate_masks, reference_depth_frame, initial_depth_frame, MicrowaveVision)

SPEC = importlib.util.spec_from_file_location("ula_microwave_runner", ROOT/"scripts/run_open_microwave.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def sample():
    rng = np.random.default_rng(4)
    points = np.column_stack((rng.uniform(-.27, .11, 800),
        np.full(800, .15), rng.uniform(.9, 1.14, 800)))
    points[:, 1] += rng.normal(0, .0005, len(points))
    handle = np.array([.09, .11, 1.02])
    panel = fit_panel(points, [0, -1, 0])
    hinge = estimate_hinge(panel, handle)
    return dict(door_points=points, panel=panel, handle_world=handle), hinge


def moved(observation, hinge, degrees):
    transform = hinge_motion(hinge["hinge_world"], np.radians(degrees)*hinge["opening_sign"])
    points = observation["door_points"]@transform[:3, :3].T+transform[:3, 3]
    handle = transform[:3, :3]@observation["handle_world"]+transform[:3, 3]
    return dict(door_points=points, panel=fit_panel(points, hinge["initial_normal"]), handle_world=handle)


class Forbidden:
    def __getattr__(self, name):
        raise AssertionError(f"Online object/contact truth accessed: {name}")


class FakeTask:
    def __init__(self, *, follow=True):
        self.reference, self.hinge = sample()
        self.angle, self.command = 0., .01
        self.follow = follow
        self.ee = np.array([.09, -.05, 1.02, 1., 0., 0., 0.])
        self.plan_success, self.eval_success = True, False
        self.take_action_cnt, self.step_lim, self.scores = 0, 500, 0
        self.executions, self.plans, self.prepares = 0, 0, 0
        self.scene, self.microwave = Forbidden(), Forbidden()
        self.robot = SimpleNamespace(left_plan_path=self.plan,
            get_left_gripper_val=lambda: self.command,
            get_left_tcp_pose=self.tcp)

    def tcp(self):
        matrix = pose_wxyz_to_matrix(self.ee)
        offset = np.eye(4)
        offset[0, 3] = .12
        return matrix_to_pose_wxyz(matrix@offset)

    def get_arm_pose(self, arm):
        return self.ee.copy()

    def plan(self, target):
        self.plans += 1
        return dict(status="Success", target=np.asarray(target),
                    position=np.zeros((3, 6)), velocity=np.zeros((3, 6)))

    def take_dense_action(self, controls):
        old = pose_wxyz_to_matrix(self.ee)
        target = controls["left_arm"]["target"]
        new = pose_wxyz_to_matrix(target)
        delta = new[:3, :3]@old[:3, :3].T
        if self.follow and self.command < .05:
            self.angle += np.degrees(np.arctan2(delta[1, 0], delta[0, 0]))*self.hinge["opening_sign"]
        self.ee = target.copy()
        self.executions += 1
        return True

    def open_gripper(self, arm, pos):
        self.command = pos
        return "prepare"

    def close_gripper(self, arm, pos=0.):
        self.command = pos
        return "close"

    def move(self, command):
        self.prepares += 1
        return True

    def take_action(self, action):
        self.take_action_cnt += 1
        self.command = .01
        # Real Base_Task checks success inside dense VLA actions. Suppress it.
        self.check_success(target=.7)

    def check_success(self, target=.6):
        raise AssertionError("Original scorer invoked during control")

    def get_obs(self):
        return {}

    def observation(self, *args, **kwargs):
        return moved(self.reference, self.hinge, self.angle)


class GeometryTests(unittest.TestCase):
    def test_static_control_strip_beyond_handle_is_not_used_as_door(self):
        observation, _ = sample()
        points, audit = isolate_door_surface(observation["door_points"], observation["panel"],
                                             observation["handle_world"])
        self.assertLessEqual(points[:, 0].max(), .094)
        self.assertLess(len(points), len(observation["door_points"]))
        self.assertFalse(audit["uses_object_truth"])
        # Mirrored geometry must retain the other side, not hard-code left.
        mirrored = observation["door_points"].copy()
        mirrored[:, 0] *= -1
        handle = observation["handle_world"].copy()
        handle[0] *= -1
        panel = fit_panel(mirrored, [0, -1, 0])
        retained, _ = isolate_door_surface(mirrored, panel, handle)
        self.assertGreaterEqual(retained[:, 0].min(), -.094)

    def depth_scene(self):
        x, z = np.meshgrid(np.linspace(-.27, .11, 120), np.linspace(.9, 1.14, 100))
        grid = np.stack((x, np.full_like(x, .15), z), axis=-1)
        panel = fit_panel(grid.reshape(-1, 3), [0, -1, 0])
        grid[20:80, 109:112, 1] -= .018
        # A knob is wider and much shorter than the vertical bar.
        grid[10:14, 85:95, 1] -= .012
        return grid, panel

    def test_depth_bar_selected_without_text_or_object_truth(self):
        grid, panel = self.depth_scene()
        point, mask, audit = protruding_handle(grid, np.ones(grid.shape[:2], bool), panel)
        self.assertEqual(audit["source"], "depth_protrusion")
        self.assertAlmostEqual(point[1], .132)
        self.assertEqual(int(mask.sum()), 180)
        self.assertFalse(audit["uses_object_truth"])

    def test_depth_no_bar_and_outside_roi_rejected(self):
        grid, panel = self.depth_scene()
        roi = np.ones(grid.shape[:2], bool)
        roi[:, 105:] = False
        with self.assertRaisesRegex(RuntimeError, "No credible"):
            protruding_handle(grid, roi, panel)
        grid[20:80, 109:112, 1] = .15
        with self.assertRaisesRegex(RuntimeError, "No credible"):
            protruding_handle(grid, np.ones_like(roi), panel)

    def test_competing_vertical_bars_are_rejected(self):
        grid, panel = self.depth_scene()
        grid[20:80, 8:11, 1] -= .018
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            protruding_handle(grid, np.ones(grid.shape[:2], bool), panel)

    def test_visual_hinge_and_direction_are_not_actor_inputs(self):
        observation, hinge = sample()
        self.assertFalse(hinge["uses_object_truth"])
        self.assertFalse(hinge["hidden_hinge_measured"])
        self.assertLess(hinge["hinge_world"][0], -.24)
        self.assertEqual(hinge["opening_sign"], -1.)
        self.assertLess(observation["panel"]["plane_rmse_m"], .002)

    def test_plane_handles_bounded_outliers(self):
        observation, _ = sample()
        noise = np.random.default_rng(3).uniform(-1, 1, (120, 3))
        panel = fit_panel(np.vstack((observation["door_points"], noise)), [0, -1, 0])
        self.assertGreater(panel["inlier_fraction"], .8)

    def test_horizontal_nonplanar_and_missing_depth_rejected(self):
        observation, _ = sample()
        horizontal = observation["door_points"][:, [0, 2, 1]]
        for cloud in (horizontal, np.random.default_rng(2).normal(size=(300, 3))):
            with self.assertRaises(RuntimeError):
                fit_panel(cloud, [0, -1, 0])

    def test_vertical_plane_fit_does_not_accept_a_line_or_wrong_tracking_direction(self):
        points = np.column_stack((np.linspace(-.2, .2, 200), np.full(200, .15), np.linspace(.8, 1.2, 200)))
        with self.assertRaisesRegex(RuntimeError, "line"):
            fit_panel(points, [0, -1, 0])
        observation, _ = sample()
        with self.assertRaisesRegex(RuntimeError, "No credible"):
            fit_panel(observation["door_points"], [1, 0, 0], normal_cone_deg=12)
        for cloud in (np.zeros((5, 3)), np.full((100, 3), np.nan)):
            with self.assertRaises(ValueError):
                fit_panel(cloud, [0, -1, 0])

    def test_central_or_far_handle_cannot_identify_hinge(self):
        observation, _ = sample()
        for point in (observation["panel"]["centre"], [2, 2, 2]):
            with self.assertRaises(RuntimeError):
                estimate_hinge(observation["panel"], point)

    def test_hinge_motion_preserves_radius_height_and_axis(self):
        _, hinge = sample()
        transform = hinge_motion(hinge["hinge_world"], -.3)
        np.testing.assert_allclose(transform[:3, :3]@hinge["hinge_world"]+transform[:3, 3], hinge["hinge_world"])
        np.testing.assert_allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1)
        self.assertEqual(transform[2, 3], 0)

    def test_pitched_approach_keeps_horizontal_pinch_and_valid_robot_offset(self):
        observation, hinge = sample()
        offset = np.eye(4)
        offset[0, 3] = .12
        target = approach_target(hinge, observation["handle_world"], offset, .01, 20.)
        tcp = target@offset
        self.assertLess(tcp[2, 0], 0)
        self.assertAlmostEqual(tcp[2, 1], 0)
        self.assertGreater(tcp[2, 3], observation["handle_world"][2])
        np.testing.assert_allclose(tcp[:3, :3].T@tcp[:3, :3], np.eye(3), atol=1e-12)

    def test_observed_opening_angle_and_front_approach_tcp_offset(self):
        observation, hinge = sample()
        after = moved(observation, hinge, 35)
        self.assertAlmostEqual(opening_angle(hinge, after["panel"]), 35, places=1)
        offset = np.eye(4)
        offset[0, 3] = .12
        target = approach_target(hinge, observation["handle_world"], offset)
        tcp = target@offset
        np.testing.assert_allclose(tcp[:3, 3], observation["handle_world"]+hinge["initial_normal"]*.08)
        np.testing.assert_allclose(tcp[:3, 0], -hinge["initial_normal"])
        np.testing.assert_allclose(tcp[:3, :3].T@tcp[:3, :3], np.eye(3), atol=1e-12)

    def test_visual_gate_needs_both_panel_rotation_and_handle_motion(self):
        before, hinge = sample()
        transform = hinge_motion(hinge["hinge_world"], np.radians(3)*hinge["opening_sign"])
        after = moved(before, hinge, 3)
        self.assertTrue(verify_step(hinge, before, after, transform, 3)["ready"])
        self.assertFalse(verify_step(hinge, before, before, transform, 3)["ready"])
        after["handle_world"] = before["handle_world"]
        self.assertFalse(verify_step(hinge, before, after, transform, 3)["ready"])

    def test_wrong_direction_or_large_visual_jump_rejected(self):
        before, hinge = sample()
        transform = hinge_motion(hinge["hinge_world"], np.radians(3)*hinge["opening_sign"])
        for angle in (-3, 15):
            self.assertFalse(verify_step(hinge, before, moved(before, hinge, angle), transform, 3)["ready"])


class ControlTests(unittest.TestCase):
    def test_traditional_close_is_one_gripper_trajectory_not_an_extra_pull(self):
        task = FakeTask()
        with without_object_truth(task):
            approach_handle(task, task.hinge, task.reference["handle_world"], {}, clearance=.025,
                            gripper_opening=.45)
            before = task.executions
            pose = task.get_arm_pose("left")
            result = {}
            traditional_close_only(task, result, handle_world=task.reference["handle_world"])
        self.assertTrue(result["complete"])
        self.assertFalse(result["uses_vla"])
        self.assertFalse(result["last_decision"]["physical_grasp_certificate"])
        self.assertEqual(task.executions, before)
        np.testing.assert_array_equal(task.get_arm_pose("left"), pose)
        self.assertEqual(task.take_action_cnt, 0)

    def test_traditional_close_rejects_distant_or_failed_candidates(self):
        task = FakeTask()
        with self.assertRaisesRegex(RuntimeError, "outside visual handle range"):
            traditional_close_only(task, {}, handle_world=task.reference["handle_world"])
        self.assertEqual(task.prepares, 0)
        task.ee[:3] += task.reference["handle_world"]-np.asarray(task.tcp())[:3]
        task.move = lambda *args: False
        with self.assertRaisesRegex(RuntimeError, "candidate failed"):
            traditional_close_only(task, {}, handle_world=task.reference["handle_world"])
        self.assertEqual(task.executions, 0)
        with self.assertRaises(ValueError):
            traditional_close_only(task, {}, handle_world=[np.nan, 0, 0])

    def test_topp_diagnostics_preserve_results_errors_and_restore(self):
        def topp(path, *args, **kwargs):
            if len(path) < 2:
                raise RuntimeError("bad TOPP")
            return ([], np.zeros((3, 6)), [], [], 1.)
        planner = SimpleNamespace(TOPP=topp)
        robot, records = SimpleNamespace(left_mplib_planner=planner), []
        with record_topp(robot, records):
            self.assertEqual(len(planner.TOPP(np.zeros((2, 6)))[1]), 3)
            with self.assertRaisesRegex(RuntimeError, "bad TOPP"):
                planner.TOPP(np.zeros((1, 6)))
        self.assertIs(planner.TOPP, topp)
        self.assertTrue(records[0]["success"])
        self.assertFalse(records[1]["success"])

    def test_guard_blocks_truth_diagnostics_and_restores_even_on_error(self):
        task = FakeTask()
        old = lambda task: self.fail("Diagnostic qpos/contact read")
        policy = SimpleNamespace(module=SimpleNamespace(_task_metrics=old), model=SimpleNamespace(oracle_route_state_mode="off"))
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with without_object_truth(task, policy):
                self.assertFalse(task.check_success(target=.7))
                self.assertEqual(policy.module._task_metrics(task), {})
                raise RuntimeError("stop")
        self.assertNotIn("check_success", vars(task))
        self.assertIs(policy.module._task_metrics, old)
        with self.assertRaises(AssertionError):
            task.check_success()

    def test_oracle_route_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            with without_object_truth(FakeTask(), SimpleNamespace(model=SimpleNamespace(oracle_route_state_mode="correct"))):
                pass

    def test_approach_uses_robot_transform_without_object_access(self):
        task = FakeTask()
        result = {}
        with without_object_truth(task):
            approach_handle(task, task.hinge, task.reference["handle_world"], result)
        self.assertTrue(result["complete"])
        self.assertEqual(task.executions, 1)
        self.assertEqual(task.prepares, 1)

    def test_candidate_interrupts_vla_chunk_without_door_truth(self):
        task = FakeTask()
        original = task.take_action
        def step(task, observation):
            for _ in range(10):
                task.take_action(np.zeros(14))
        with without_object_truth(task):
            result = {}
            vla_close_only(task, SimpleNamespace(reset=lambda: None, step=step), result)
        self.assertEqual(task.take_action_cnt, 3)
        self.assertNotIn("take_action", vars(task))
        self.assertEqual(task.take_action, original)
        self.assertFalse(result["candidate"]["lifted_grasp_verified"])

    def test_near_approach_reaches_standoff_then_uses_bounded_segments(self):
        task = FakeTask()
        result = {}
        with without_object_truth(task):
            approach_handle(task, task.hinge, task.reference["handle_world"], result,
                            clearance=.01, gripper_opening=.45)
        self.assertTrue(result["complete"])
        self.assertEqual(task.executions, 5)
        distances = [.08]+[s["clearance_m"] for s in result["near_approach_segments"]]
        self.assertTrue(np.all(np.abs(np.diff(distances)) <= .020001))
        self.assertEqual(task.command, .45)

    def test_invalid_vla_action_restores_wrapper(self):
        task = FakeTask()
        def step(task, observation):
            task.take_action(np.zeros(7))
        with self.assertRaises(ValueError), without_object_truth(task):
            vla_close_only(task, SimpleNamespace(reset=lambda: None, step=step), {})
        self.assertNotIn("take_action", vars(task))

    def test_close_far_from_visual_handle_is_not_a_handoff(self):
        task = FakeTask()
        task.step_lim = 4
        result = {}
        policy = SimpleNamespace(reset=lambda: None, step=lambda task, obs: task.take_action(np.zeros(14)))
        with self.assertRaisesRegex(RuntimeError, "failed close-command"), without_object_truth(task):
            vla_close_only(task, policy, result, handle_world=task.reference["handle_world"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["last_decision"]["within_visual_handle_range"])
        self.assertEqual(result["last_decision"]["stable_steps"], 0)

    def test_first_pull_and_all_opening_are_traditional_without_actor_reads(self):
        task = FakeTask()
        result = {}
        with without_object_truth(task):
            open_door(task, task.observation, task.hinge, task.reference, result, target_deg=10)
        self.assertTrue(result["complete"])
        self.assertEqual(result["steps"][0]["stage"], "trial_pull")
        self.assertGreaterEqual(result["final_visual_angle_deg"], 9)
        self.assertEqual(task.take_action_cnt, 0)
        self.assertEqual(task.prepares, 0)  # No release, including at completion.

    def test_empty_grasp_is_rejected_after_only_one_bounded_pull(self):
        task = FakeTask(follow=False)
        result = {}
        with self.assertRaisesRegex(RuntimeError, "No credible"), without_object_truth(task):
            open_door(task, task.observation, task.hinge, task.reference, result, target_deg=10)
        self.assertFalse(result["complete"])
        self.assertEqual(task.executions, 1)
        self.assertEqual(len(result["steps"][0]["visual_checks"]), 2)
        self.assertEqual(task.prepares, 0)

    def test_observation_retry_does_not_repeat_physical_pull(self):
        task = FakeTask()
        calls = []
        def observer(label, refined=False, **kwargs):
            calls.append(refined)
            if not refined:
                raise RuntimeError("Bad segmentation")
            return task.observation()
        with without_object_truth(task):
            result = {}
            open_door(task, observer, task.hinge, task.reference, result, target_deg=10)
        self.assertTrue(result["complete"])
        self.assertEqual(len(calls), task.executions*2)

    def test_vla_opening_role_leakage_rejected_before_any_traditional_pull(self):
        task = FakeTask()
        baseline = moved(task.reference, task.hinge, 5)
        with self.assertRaisesRegex(RuntimeError, "role leakage"):
            open_door(task, task.observation, task.hinge, baseline, {})
        self.assertEqual(task.executions, 0)

    def test_failed_plan_does_not_execute_or_release(self):
        task = FakeTask()
        task.robot.left_plan_path = lambda target: dict(status="Fail")
        with self.assertRaisesRegex(RuntimeError, "planning failed"):
            open_door(task, task.observation, task.hinge, task.reference, {})
        self.assertEqual(task.executions, 0)
        self.assertEqual(task.prepares, 0)

    def test_planning_retry_happens_before_any_execution(self):
        task = FakeTask()
        first_plan = task.robot.left_plan_path
        count = []
        def plan(target):
            self.assertEqual(task.executions, 0)
            count.append(1)
            return dict(status="Fail") if len(count) == 1 else first_plan(target)
        task.robot.left_plan_path = plan
        with without_object_truth(task):
            result = {}
            approach_handle(task, task.hinge, task.reference["handle_world"], result)
        self.assertEqual(task.executions, 1)
        self.assertEqual(len(result["planning_attempts"]), 2)

    def test_same_endpoint_settles_without_loosening_arrival_check(self):
        task = FakeTask()
        original = task.take_dense_action
        def lag(controls):
            result = original(controls)
            if task.executions == 1:
                task.ee[1] -= .02
            return result
        task.take_dense_action = lag
        with without_object_truth(task):
            result = {}
            approach_handle(task, task.hinge, task.reference["handle_world"], result)
        self.assertFalse(result["arrival_before_settling"]["ready"])
        self.assertTrue(result["arrival"]["ready"])
        self.assertEqual(len(result["settling_attempts"]), 1)
        self.assertEqual(task.executions, 2)

    def test_alternative_standoff_only_planned_after_failed_unexecuted_goal(self):
        task = FakeTask()
        original = task.robot.left_plan_path
        calls = []
        def planner(target):
            if len(calls) < 4:
                self.assertEqual(task.executions, 0)
            calls.append(target)
            return dict(status="Fail") if len(calls) <= 3 else original(target)
        task.robot.left_plan_path = planner
        with without_object_truth(task):
            result = {}
            approach_handle(task, task.hinge, task.reference["handle_world"], result)
        self.assertEqual(task.executions, 1+len(result["near_approach_segments"]))
        self.assertEqual(result["approach_pitch_deg"], 0)
        self.assertEqual(result["standoff_m"], .10)
        self.assertEqual(len(result["standoff_candidates"]), 2)

    def test_invalid_limits_do_not_move(self):
        task = FakeTask()
        for values in (dict(target_deg=100), dict(step_deg=10), dict(step_deg=float("nan"))):
            with self.assertRaises(ValueError):
                open_door(task, task.observation, task.hinge, task.reference, {}, **values)
        self.assertEqual(task.executions, 0)


class IntegrationTests(unittest.TestCase):
    def reference_frame(self):
        height, width = 120, 160
        intrinsic = np.array([[200., 0, 80], [0, 200, 60], [0, 0, 1]])
        extrinsic = np.array([[1., 0, 0, 0], [0, 0, -1, 1.02], [0, 1, 0, .65]])
        yy, xx = np.indices((height, width))
        surface = (xx >= 15) & (xx <= 145) & (yy >= 20) & (yy <= 100)
        points = np.stack(((xx-80)*.8/200, np.full_like(xx, .15, dtype=float),
                           1.02-(yy-60)*.8/200), axis=-1)[surface]
        panel = fit_panel(points, [0, -1, 0])
        reference = dict(door_points=points, panel=panel, handle_world=np.array([.20, .13, 1.02]))
        depth = np.full((height, width), 2.)
        depth[surface] = .8
        depth[35:85, 130:133] = .78
        camera = dict(depth=depth, intrinsic_cv=intrinsic, extrinsic_cv=extrinsic,
                      rgb=np.zeros((height, width, 3), dtype=np.uint8))
        return camera, reference

    def test_reference_roi_requires_current_depth_not_old_geometry(self):
        camera, reference = self.reference_frame()
        result = reference_depth_frame(camera, reference)
        self.assertEqual(result["vision_audit"]["source"], "reference_roi_fresh_depth")
        self.assertTrue(result["vision_audit"]["current_depth_required"])
        self.assertFalse(result["vision_audit"]["uses_object_truth"])
        camera["depth"][:] = 0
        with self.assertRaises((RuntimeError, ValueError)):
            reference_depth_frame(camera, reference)

    def test_search_footprint_does_not_expand_with_accidentally_added_plane_points(self):
        camera, reference = self.reference_frame()
        reference["tracking_corners"] = panel_corners(reference["panel"])
        first = reference_depth_frame(camera, reference)
        reference["door_points"] = reference["door_points"].copy()
        reference["door_points"][:, 0] *= 3
        second = reference_depth_frame(camera, reference)
        self.assertEqual(first["vision_audit"]["projected_roi_xyxy"], second["vision_audit"]["projected_roi_xyxy"])
        corners = reference["tracking_corners"].copy()
        for _ in range(50):
            corners = update_panel_corners(corners, reference["panel"], reference["panel"],
                                           reference["handle_world"], reference["handle_world"])
        np.testing.assert_allclose(corners, reference["tracking_corners"])

    def test_recessed_window_is_separated_before_fitting_front_panel(self):
        camera, reference = self.reference_frame()
        camera["depth"][30:90, 25:110] = .84
        result = initial_depth_frame(camera, [15, 20, 145, 100])
        self.assertEqual(result["handle_audit"]["source"], "segmented_depth_planes")
        self.assertLess(result["handle_audit"]["selected_raw_roi_fraction"], .60)
        self.assertAlmostEqual(result["handle_world"][1], .13, places=3)
        # Neither the recessed window nor the measured frame alone is a grasp.
        camera["depth"][35:85, 130:133] = .8
        camera["depth"][30:90, 3:6] = .78  # robot-like bar outside the detected box
        with self.assertRaisesRegex(RuntimeError, "No depth-layer"):
            initial_depth_frame(camera, [15, 20, 145, 100])

    def test_failed_hinge_side_refit_uses_depth_layers_before_text_handle(self):
        camera, _ = self.reference_frame()
        door = np.zeros(camera["depth"].shape, bool)
        door[20:101, 15:146] = True
        class Tensor(np.ndarray):
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.asarray(self)
        detections = []
        def detector(view, prompt):
            detections.append(prompt)
            self.assertEqual(len(detections), 1, "Must retry depth before text handle detection")
            return np.array([[0., 0., 1., 1.]]).view(Tensor), np.array([.9]).view(Tensor), None
        vision = MicrowaveVision.__new__(MicrowaveVision)
        vision.prompts = dict(door="microwave door", handle="microwave handle")
        vision.models = SimpleNamespace(_run_dino=detector, _xyxy=lambda *args: np.array([15, 20, 145, 100]),
            sam=SimpleNamespace(set_image=lambda *a: None, predict=lambda **k: (door, None, None)),
            torch=SimpleNamespace(inference_mode=nullcontext, autocast=lambda *a, **k: nullcontext(),
                bfloat16=None, cuda=SimpleNamespace(synchronize=lambda: None)))
        fits = []
        def fit(points, hint, **kwargs):
            fits.append(len(points))
            if len(fits) == 2:
                raise RuntimeError("Hinge-side surface still mixes depth layers")
            return fit_panel(points, hint, **kwargs)
        with patch("adapters.microwave.perception.fit_panel", side_effect=fit):
            result = vision.estimate(camera["rgb"], camera["depth"], camera["intrinsic_cv"],
                                     camera["extrinsic_cv"])
        audit = result["audit"]["handle_detection"]
        self.assertEqual(audit["source"], "segmented_depth_planes")
        self.assertIn("Hinge-side", audit["single_plane_rejection"])
        self.assertEqual(len(detections), 1)

    def test_local_handle_fragment_uses_new_xy_and_explicit_reference_height(self):
        camera, reference = self.reference_frame()
        camera["depth"][35:85, 130:133] = .8
        camera["depth"][40:47, 130:133] = .78
        result = reference_depth_frame(camera, reference)
        audit = result["vision_audit"]["handle_detection"]
        self.assertEqual(audit["source"], "reference_vertical_bar_partial_depth")
        self.assertFalse(audit["hidden_grasp_point_measured"])
        self.assertAlmostEqual(result["handle_world"][2], reference["handle_world"][2])
        self.assertNotAlmostEqual(audit["measured_visible_centroid"][2], result["handle_world"][2])

    def test_matching_axis_fragments_merge_but_inconsistent_colours_do_not(self):
        camera, reference = self.reference_frame()
        camera["depth"][35:85, 130:133] = .8
        camera["depth"][40:47, 130:133] = .78
        camera["depth"][60:67, 130:133] = .78
        result = reference_depth_frame(camera, reference)
        audit = result["vision_audit"]["handle_detection"]
        self.assertEqual(audit["merged_visible_fragments"], 2)
        camera["rgb"][60:67, 130:133] = 200
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            reference_depth_frame(camera, reference)
        camera["rgb"][40:47, 130:133] = 100
        camera["rgb"][60:67, 130:133] = 200
        self.assertEqual(reference_depth_frame(camera, reference)["vision_audit"]["handle_detection"]
                         ["merged_visible_fragments"], 2)
        camera["rgb"][60:67, 130:133] = [0, 0, 200]
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            reference_depth_frame(camera, reference)

    def test_expected_motion_does_not_invent_observed_door_motion(self):
        camera, reference = self.reference_frame()
        hinge = estimate_hinge(reference["panel"], reference["handle_world"])
        motion = hinge_motion(hinge["hinge_world"], np.radians(3)*hinge["opening_sign"])
        result = reference_depth_frame(camera, reference, motion)
        result["panel"] = fit_panel(result["door_points"], [0, -1, 0])
        self.assertAlmostEqual(opening_angle(hinge, result["panel"]), 0, places=5)
        self.assertFalse(verify_step(hinge, reference, result, motion, 3)["ready"])
        camera["depth"][35:85, 130:133] = .8
        with self.assertRaisesRegex(RuntimeError, "No credible"):
            reference_depth_frame(camera, reference, motion)

    def test_reference_depth_can_observe_without_restarting_detector(self):
        camera, reference = self.reference_frame()
        task = SimpleNamespace(get_camera_observation=lambda name: camera)
        worker = SimpleNamespace(estimate_frame=lambda *a, **k: self.fail("No model call needed"))
        with tempfile.TemporaryDirectory() as output:
            result = observe(task, worker, output, "closed", [], reference=reference)
            self.assertEqual(result["camera"], "head_camera")
            self.assertTrue(Path(output, "closed_head_camera/rgbd.npz").exists())

    def test_failed_broad_plane_search_can_retry_narrower_without_lowering_support(self):
        camera, reference = self.reference_frame()
        calls = []
        def fit(points, hint, **kwargs):
            calls.append(len(points))
            if len(calls) == 1:
                raise RuntimeError("Foreground contaminated broad band")
            return fit_panel(points, hint, **kwargs)
        with patch("adapters.microwave.perception.fit_panel", side_effect=fit):
            result = reference_depth_frame(camera, reference)
        self.assertEqual(result["vision_audit"]["depth_band_m"], .012)
        self.assertLess(calls[1], calls[0])
        self.assertGreaterEqual(result["vision_audit"]["filtered_plane_support"], .60)

    def test_reference_depth_measures_actual_rotation_not_only_static_planes(self):
        camera, reference = self.reference_frame()
        hinge = estimate_hinge(reference["panel"], reference["handle_world"])
        motion = hinge_motion(hinge["hinge_world"], np.radians(3)*hinge["opening_sign"])
        yy, xx = np.indices(camera["depth"].shape)
        origin = np.array([0., -.65, 1.02])
        rays = np.stack(((xx-80)/200, np.ones_like(xx), -(yy-60)/200), axis=-1)
        normal = motion[:3, :3]@np.array([0., -1, 0])
        depth = np.full(camera["depth"].shape, 2.)
        # Ray-intersect a finite rotated panel and its protruding vertical bar.
        for point, xlimits, zlimits in (([0., .15, 1.02], (-.26, .26), (.86, 1.18)),
                                        ([.20, .13, 1.02], (.195, .207), (.93, 1.11))):
            transformed = motion[:3, :3]@point+motion[:3, 3]
            t = ((transformed-origin)@normal)/(rays@normal)
            world = origin+rays*t[:, :, None]
            local = (world-motion[:3, 3])@motion[:3, :3]
            visible = ((local[:, :, 0] >= xlimits[0]) & (local[:, :, 0] <= xlimits[1])
                       & (local[:, :, 2] >= zlimits[0]) & (local[:, :, 2] <= zlimits[1]))
            depth[visible] = t[visible]
        camera["depth"] = depth
        result = reference_depth_frame(camera, reference, motion)
        result["panel"] = fit_panel(result["door_points"], normal)
        self.assertAlmostEqual(opening_angle(hinge, result["panel"]), 3, places=2)
        self.assertTrue(verify_step(hinge, reference, result, motion, 3)["ready"])

    def test_whole_door_box_is_not_selected_as_handle_even_with_higher_confidence(self):
        door = np.array([20, 20, 80, 80])
        handle = np.array([70, 30, 76, 70])
        np.testing.assert_array_equal(select_handle_box([door, handle], [.9, .4], door), handle)
        with self.assertRaisesRegex(RuntimeError, "whole-door"):
            select_handle_box([door], [.9], door)

    def test_full_body_misidentified_as_handle_and_clipped_door_are_rejected(self):
        door = np.zeros((100, 100), dtype=bool)
        door[20:80, 20:80] = True
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            validate_masks(door, door)
        handle = np.zeros_like(door)
        handle[30:70, 72:76] = True
        self.assertLess(validate_masks(door, handle)["mask_iou"], .1)
        clipped = door.copy()
        clipped[:, :30] = True
        with self.assertRaisesRegex(RuntimeError, "clipped"):
            validate_masks(clipped, handle)
        top_clipped = door.copy()
        top_clipped[:30, 20:80] = True
        self.assertTrue(validate_masks(top_clipped, handle)["door_vertical_clipped"])

    def test_runtime_changes_directory_before_environment_import_and_restores_on_error(self):
        previous = Path.cwd()
        def enter(args):
            self.assertEqual(Path.cwd(), runner.ROBOTWIN)
            raise RuntimeError("Import failed")
        with patch.object(runner, "_run_in_robotwin", side_effect=enter), self.assertRaises(RuntimeError):
            runner.run(None)
        self.assertEqual(Path.cwd(), previous)

    def test_separate_vision_device_and_budget_do_not_change_parent_environment(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            worker = MicrowaveWorker(ROOT, ROOT/"results/not_created", vision_gpu=0, torch_fraction=.08)
            _, _, environment = worker._launch_spec()
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(environment["MICROWAVE_VISION_MEMORY_FRACTION"], "0.08")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")

    def test_shared_resource_policy_is_explicit_and_not_a_total_cap(self):
        policy = runner.resource_policy({"CUDA_VISIBLE_DEVICES": "7"})
        self.assertEqual(policy["jax_allocator_fraction"], .45)
        self.assertFalse(policy["hard_total_memory_limit"])
        self.assertFalse(policy["compute_isolation"])
        for environment in ({}, {"CUDA_VISIBLE_DEVICES": "0,1"},
                {"CUDA_VISIBLE_DEVICES": "7", "XLA_CLIENT_MEM_FRACTION": "nan"}):
            with self.assertRaises(ValueError):
                runner.resource_policy(environment)

    def test_worker_reuses_transport_but_has_separate_process_entry(self):
        worker = MicrowaveWorker(ROOT, ROOT/"results/not_created_by_this_test")
        command, _, _ = worker._launch_spec()
        self.assertIn(str(ROOT/"scripts/perceive_microwave_rgbd.py"), command)
        self.assertNotIn(str(ROOT/"scripts/perceive_rgbd.py"), command)
        self.assertEqual(worker.summary()["starts"], 0)

    def test_bad_head_frame_can_use_wrist_without_motion_or_truth(self):
        reference, _ = sample()
        calls = []
        def estimate(camera, *args, **kwargs):
            calls.append(camera)
            if camera == "head_camera":
                raise RuntimeError("No door")
            return dict(reference, camera_world=np.array([0, -.5, 1.3]), vision_audit={})
        task = SimpleNamespace(get_camera_observation=lambda name: name, microwave=Forbidden(), scene=Forbidden())
        result = observe(task, SimpleNamespace(estimate_frame=estimate), "/unused", "test", [])
        self.assertEqual(result["camera"], "left_camera")
        self.assertEqual(calls, ["head_camera", "left_camera"])

    def test_existing_placement_sources_are_unchanged(self):
        self.assertEqual(runner.protected_hashes(), runner.PROTECTED)

    def test_cli_is_independent_and_validates_before_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse_args(["--output", tmp])
            self.assertEqual(args.mode, "hybrid")
            self.assertEqual(args.target_angle_deg, 70)
            self.assertEqual(args.approach_clearance, .025)
            self.assertEqual(args.approach_gripper, .45)
            for flags in (["--opening-step-deg", "30"], ["--max-opening-steps", "1"],
                          ["--close-threshold", "nan"]):
                with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                    runner.parse_args(["--output", tmp]+flags)
            Path(tmp, "previous_result").touch()
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(["--output", tmp])

    def test_whole_episode_separates_control_from_final_truth_scoring(self):
        for mode in ("hybrid", "traditional", "approach-only", "close-only"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                task = FakeTask()
                def setup(**kw):
                    task.angle = 0.
                task.setup_demo, task.set_instruction = setup, lambda text: None
                task.close_env = lambda: None
                def scorer(*args, **kwargs):
                    task.scores += 1
                    return task.angle >= 9
                task.check_success = scorer
                def joint():
                    self.assertGreater(task.scores, 0, "Joint truth only after final scorer")
                    return [np.radians(task.angle)]
                task.microwave = SimpleNamespace(get_qpos=joint, get_qlimits=lambda: [[0, 1.5]])
                module = SimpleNamespace(_task_metrics=lambda task: self.fail("No oracle diagnostics"))
                policy = SimpleNamespace(load=lambda: None, reset=lambda: None,
                    model=SimpleNamespace(oracle_route_state_mode="off"), module=module)
                def policy_step(task, observation):
                    self.assertEqual(module._task_metrics(task), {})
                    # This fake VLA reaches the visually observed handle before
                    # closing. Closing while still at pre-grasp is not enough.
                    task.ee[:3] += task.reference["handle_world"]-np.asarray(task.tcp())[:3]
                    for _ in range(10):
                        task.take_action(np.zeros(14))
                policy.step = policy_step
                if mode == "traditional":
                    policy = Forbidden()
                args = runner.parse_args(["--output", tmp, "--target-angle-deg", "10", "--mode", mode])
                def observer(task, *args, **kwargs):
                    self.assertEqual(task.scores, 0)
                    return task.observation()
                with patch.object(runner, "observe", side_effect=observer), \
                     patch.object(runner, "snapshot"), \
                     patch.object(runner, "vla_timing", return_value=nullcontext()):
                    result = runner.run_episode(task, policy, None, args, {}, 0)
                self.assertIsNone(result["error"], result)
                self.assertTrue(result["success"])
                self.assertEqual(task.scores, 1)
                self.assertIs(task.check_success, scorer)
                self.assertEqual(task.take_action_cnt, 0 if mode in ("approach-only", "traditional") else 3)
                if mode == "traditional":
                    self.assertFalse(result["vla_used"])
                    self.assertIn("traditional_close", result["timing_profile"]["stages"])
                    self.assertNotIn("vla_load", result["timing_profile"]["stages"])
                self.assertTrue(Path(tmp, "episode_000_seed0/result.json").exists())

    def test_traditional_batch_never_constructs_worldla(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse_args(["--mode", "traditional", "--output", tmp])
            worker = SimpleNamespace(summary=lambda: {}, close=lambda: None)
            def episode(*args):
                return dict(seed=0, success=True, task_success=True, error=None, elapsed_s=1.)
            with patch.dict(sys.modules, {"envs.open_microwave": SimpleNamespace(open_microwave=lambda: None)}), \
                 patch.object(runner, "task_config", return_value={}), \
                 patch.object(runner, "resource_policy", return_value={}), \
                 patch.object(runner, "MicrowaveWorker", return_value=worker), \
                 patch.object(runner, "run_episode", side_effect=episode), \
                 patch.object(runner, "WorldLA", side_effect=AssertionError("Must not construct VLA")):
                self.assertEqual(runner._run_in_robotwin(args), 0)
            report = json.loads(Path(tmp, "report.json").read_text())
            self.assertFalse(report["vla_used"])
            self.assertEqual(report["success_definition"], "task_opening")


if __name__ == "__main__":
    unittest.main()
