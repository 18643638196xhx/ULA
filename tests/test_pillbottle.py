import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.pillbottle import (PillbottleWorker, PillbottleVision,
    approach_bottle, observation_quality, projected_object_prompt, rollout_bottle, verify_bottle_lift)
from adapters.pointcloud_registration import RegistrationConfig
SPEC = importlib.util.spec_from_file_location("ula_pillbottle_runner", ROOT/"scripts/run_move_pillbottle_pad.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ForbiddenTruth:
    def __getattr__(self, key):
        raise AssertionError(f"Controller read object/contact truth: {key}")


def estimate():
    points = np.random.default_rng(3).uniform([-.02, -.02, .75], [.02, .02, .86], (400, 3))
    return SimpleNamespace(cup_points_world=points,
        cup_bounds_world=np.array([points.min(0), points.max(0)]), observation_audit={},
        coaster_world_center=np.array([.15, 0, .741]), timings_s={})


class PillbottleTests(unittest.TestCase):
    def test_projected_prompt_is_only_a_search_region_from_measured_geometry(self):
        points = np.random.default_rng(19).uniform([-.05, -.05, 1.], [.05, .05, 1.01], (100, 3))
        frame = dict(rgb=np.zeros((100, 100, 3)), intrinsic_cv=np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]),
                     extrinsic_cv=np.eye(4))
        before = points.copy()
        box, seed = projected_object_prompt(points, frame)
        shifted, seed_shifted = projected_object_prompt(points, frame, displacement=[.02, 0, 0])
        np.testing.assert_array_equal(points, before)
        self.assertTrue(np.all(box[:2] < seed))
        self.assertTrue(np.all(box[2:] > seed))
        self.assertAlmostEqual(seed_shifted[0]-seed[0], 2., delta=.03)
        self.assertGreater(shifted[0], box[0])

    def test_bottle_quality_accepts_taller_shapes_but_rejects_growth_and_clipping(self):
        value = estimate()
        self.assertTrue(observation_quality(value)["geometry_ok"])
        ref = value.cup_bounds_world.copy()
        value.cup_bounds_world[1, 2] += .5
        self.assertFalse(observation_quality(value, ref)["geometry_ok"])
        value = estimate()
        value.observation_audit["cup_mask_touches_border"] = True
        self.assertFalse(observation_quality(value)["geometry_ok"])

    def test_cli_validates_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parse_args(["--output", tmp])
            self.assertEqual(args.arm, "auto")
            self.assertEqual(args.approach_mode, "pointcloud")
            (Path(tmp)/"prior.json").touch()
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(["--output", tmp])
        for option in (["--release-gap", "nan"], ["--trial-lift-distance", ".02"], ["--episodes", "0"]):
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(["--output", "/tmp/pillbottle_unused_arguments"]+option)

    def test_worker_uses_independent_entry_and_bottle_prompts(self):
        worker = PillbottleWorker(ROOT, "/tmp/pillbottle_worker_test")
        command, _, env = worker._launch_spec()
        self.assertEqual(Path(command[1]).name, "perceive_pillbottle_rgbd.py")
        self.assertIn("pill bottle", command)
        self.assertIn("blue square pad", command)
        self.assertIn("PILLBOTTLE_VISION_TORCH_FRACTION", env)
        self.assertIsNone(worker.process)

    def test_detector_queries_semantic_classes_without_cup_phrase_matching(self):
        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
            def __len__(self): return len(self.value)
            def __getitem__(self, index): return Tensor(self.value[index])
        vision = object.__new__(PillbottleVision)
        vision.prompts = dict(cup="pill bottle", coaster="blue square pad")
        captions = []
        def detection(rgb, caption):
            captions.append(caption)
            return Tensor([[.5, .5, .2, .2]]), Tensor([.9]), ["bottle" if len(captions) == 1 else "pad"]
        vision._run_dino = detection
        boxes = vision._detect_both(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(captions, ["pill bottle .", "blue square pad ."])
        self.assertEqual(set(boxes), {"cup", "coaster"})

    def test_policy_diagnostics_disabled_and_restored_even_on_error(self):
        metric = lambda *a: (_ for _ in ()).throw(AssertionError("No truth"))
        policy = SimpleNamespace(module=SimpleNamespace(_task_metrics=metric),
            model=SimpleNamespace(oracle_route_state_mode="off"))
        with self.assertRaisesRegex(RuntimeError, "test"):
            with runner.policy_without_truth(policy):
                self.assertEqual(policy.module._task_metrics(ForbiddenTruth()), {})
                raise RuntimeError("test")
        self.assertIs(policy.module._task_metrics, metric)
        policy.model.oracle_route_state_mode = "on"
        with self.assertRaises(ValueError):
            with runner.policy_without_truth(policy): pass

    def test_approach_searches_reach_facing_position_before_pitch_without_intermediate_motion(self):
        from test_geometry_approach import FakeTask
        task = FakeTask(failures=3)
        value = estimate()
        value.cup_bounds_world[:, 0] -= .2
        task.pillbottle = task.pad = ForbiddenTruth()
        args = SimpleNamespace(geometry_top_clearance=.04, geometry_edge_fraction=1.,
            geometry_yaw_offsets=[0., -.15, .15], approach_pitch_deg=[0., 20., 35.])
        utils = ModuleType("envs.utils")
        utils.ArmTag = str
        with patch.dict(sys.modules, {"envs.utils": utils}):
            result = approach_bottle(task, SimpleNamespace(estimate=lambda *a: value), args)
        self.assertTrue(result["ready"])
        self.assertEqual(task.calls, [("prepare", .6), "plan", "plan", "plan", "plan", "execute"])
        self.assertEqual(result["geometry_approach"]["selected_candidate"], 3)
        attempt = result["geometry_approach"]["planning_attempts"][-1]
        self.assertEqual(attempt["approach_pitch_deg"], 0.)
        self.assertEqual(result["geometry_approach"]["position_hypotheses"][1]["source"],
                         "reach_facing_surface")
        self.assertGreater(len(result["geometry_approach"]["pitch_candidates_deg"]), 3)
        self.assertEqual(result["geometry_approach"]["pitch_candidates_source"],
            "compatibility_seeds_plus_measured_object_aspect_ratio")

    def test_approach_tries_peer_arm_only_after_preferred_arm_has_no_plan(self):
        from test_geometry_approach import FakeTask
        task = FakeTask(failures=100)
        value = estimate()
        value.cup_bounds_world[:, 0] -= .2
        value.observation_audit["table_height_m"] = .74
        task.robot.right_plan_path = lambda target: task.plan(target)
        task.robot.get_right_tcp_pose = task.robot.get_left_tcp_pose
        original_dense = task.take_dense_action
        task.take_dense_action = lambda controls: original_dense(
            dict(controls, left_arm=controls.get("right_arm")))
        # The preferred arm exhausts one measured candidate.  The peer arm is
        # then made feasible without any preferred-arm execution.
        calls = {"left": 0, "right": 0}
        def left_plan(target):
            calls["left"] += 1
            return {"status": "IK_FAIL"}
        def right_plan(target):
            calls["right"] += 1
            return {"status": "Success", "position": np.zeros((2, 6)),
                    "velocity": np.zeros((2, 6)), "target": target}
        task.robot.left_plan_path = left_plan
        task.robot.right_plan_path = right_plan
        args = SimpleNamespace(geometry_top_clearance=.04, geometry_edge_fraction=1.,
            geometry_yaw_offsets=[0.], approach_pitch_deg=[0.], approach_style="diagonal",
            approach_gripper=1.)
        utils = ModuleType("envs.utils"); utils.ArmTag = str
        with patch.dict(sys.modules, {"envs.utils": utils}):
            result = approach_bottle(task, SimpleNamespace(estimate=lambda *a: value), args)
        self.assertTrue(result["ready"])
        self.assertEqual(result["arm"], "right")
        self.assertGreater(calls["left"], 0)
        self.assertEqual(calls["right"], 1)
        self.assertEqual([x["arm"] for x in result["geometry_approach"]["arm_reachability_attempts"]],
                         ["left", "right"])

    def test_closed_command_far_above_bottle_does_not_stop_vla(self):
        task = SimpleNamespace(take_action_cnt=0, step_lim=20, pillbottle=ForbiddenTruth(),
            pad=ForbiddenTruth(), scene=ForbiddenTruth(), get_obs=lambda: {},
            get_arm_pose=lambda arm: [0, 0, 1, 1, 0, 0, 0],
            get_camera_observation=lambda name: {})
        task.robot = SimpleNamespace(get_left_gripper_val=lambda: .01,
            get_left_tcp_pose=lambda: [0, 0, 1.1 if task.take_action_cnt < 6 else .8, 1, 0, 0, 0])
        def execute(value):
            task.take_action_cnt += 1
        task.take_action = execute
        def step(task, obs):
            for _ in range(10): task.take_action(np.zeros(14))
        perception = SimpleNamespace(supplementary=lambda *a, **kw: estimate())
        record = {}
        rollout_bottle(task, SimpleNamespace(reset=lambda: None, step=step), perception, "left", record)
        self.assertEqual(task.take_action_cnt, 6)
        self.assertFalse(record["close_proximity_checks"][0]["accepted"])
        self.assertTrue(record["candidate_handoff"]["visual_proximity"]["accepted"])
        self.assertFalse(record["candidate_handoff"]["lifted_grasp_verified"])
        self.assertIs(task.take_action, execute)

    def test_rule_driven_closed_command_handoff_uses_yaml_threshold_without_vision(self):
        task = SimpleNamespace(take_action_cnt=0, step_lim=20, get_obs=lambda: {},
            get_arm_pose=lambda arm: [0, 0, .8, 1, 0, 0, 0],
            get_camera_observation=lambda name: (_ for _ in ()).throw(
                AssertionError("rule-driven handoff must not query RGB-D")))
        task.robot = SimpleNamespace(
            get_left_gripper_val=lambda: .2,
            get_left_tcp_pose=lambda: [0, 0, .8, 1, 0, 0, 0])
        original = lambda value: setattr(task, "take_action_cnt", task.take_action_cnt+1)
        task.take_action = original

        def step(task, obs):
            for _ in range(10):
                task.take_action(np.zeros(14))

        record = {}
        perception = SimpleNamespace(supplementary=lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("rule-driven handoff must not run perception")))
        rollout_bottle(task, SimpleNamespace(reset=lambda: None, step=step), perception,
            "left", record, stable_steps=2, gripper_threshold=.25,
            require_visual_proximity=False)
        self.assertEqual(task.take_action_cnt, 2)
        self.assertEqual(record["close_proximity_checks"], [])
        self.assertEqual(record["candidate_handoff"]["detector"],
            "stable_closed_gripper_command")
        self.assertEqual(record["candidate_handoff"]["gripper_threshold"], .25)
        self.assertEqual(record["candidate_handoff"]["required_stable_steps"], 2)
        self.assertFalse(record["candidate_handoff"]["visual_proximity_required"])
        self.assertIs(task.take_action, original)

    def test_side_approach_is_in_front_of_body_with_open_gripper(self):
        from test_geometry_approach import FakeTask
        from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
        task, value = FakeTask(), estimate()
        value.cup_bounds_world[:, 0] -= .2
        value.observation_audit["table_height_m"] = .74
        args = SimpleNamespace(geometry_top_clearance=.04, geometry_edge_fraction=1.,
            geometry_yaw_offsets=[0.], approach_pitch_deg=[0.], approach_style="side",
            side_standoff=.06, approach_gripper=1.)
        utils = ModuleType("envs.utils")
        utils.ArmTag = str
        with patch.dict(sys.modules, {"envs.utils": utils}):
            result = approach_bottle(task, SimpleNamespace(estimate=lambda *a: value), args)
        target = pose_wxyz_to_matrix(result["geometry_approach"]["planning_attempts"][0]["target_tcp_pose"])
        self.assertAlmostEqual(target[1, 3], value.cup_bounds_world[0, 1]-.06)
        self.assertAlmostEqual(target[2, 3], .74+.5*(value.cup_bounds_world[1, 2]-.74))
        np.testing.assert_allclose(target[:3, 0], [0, 1, 0], atol=1e-12)
        self.assertEqual(task.calls[0], ("prepare", 1.))

    def test_diagonal_approach_matches_robot_grasp_convention_and_mirrors(self):
        from test_geometry_approach import FakeTask
        from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
        for arm, shift in [("left", -.2), ("right", .2)]:
            task, value = FakeTask(), estimate()
            value.cup_bounds_world[:, 0] += shift
            if arm == "right":
                task.robot.right_plan_path = task.plan
                task.robot.get_right_tcp_pose = task.robot.get_left_tcp_pose
                execute_left_fixture = task.take_dense_action
                task.take_dense_action = lambda controls: execute_left_fixture(
                    dict(controls, left_arm=controls["right_arm"]))
            args = SimpleNamespace(geometry_top_clearance=.02, geometry_edge_fraction=1.,
                geometry_yaw_offsets=[0.], approach_pitch_deg=[0.], approach_style="diagonal",
                approach_gripper=1.)
            utils = ModuleType("envs.utils")
            utils.ArmTag = str
            with patch.dict(sys.modules, {"envs.utils": utils}):
                result = approach_bottle(task, SimpleNamespace(estimate=lambda *a: value), args)
            target = pose_wxyz_to_matrix(result["geometry_approach"]["planning_attempts"][0]["target_tcp_pose"])
            self.assertTrue(result["ready"])
            self.assertEqual(result["arm"], arm)
            self.assertAlmostEqual(target[2, 3], value.cup_bounds_world[1, 2]+.02)
            self.assertAlmostEqual(-target[2, 0], np.cos(np.radians(15.)))
            self.assertAlmostEqual(np.linalg.det(target[:3, :3]), 1.)
            self.assertGreater(target[1, 0], 0.)
            self.assertEqual(np.sign(target[0, 0]), 1 if arm == "left" else -1)

    def test_pinching_center_inside_visible_bounds_does_not_need_to_touch_skin(self):
        from itertools import product
        value = estimate()
        corners = np.asarray(list(product([-.02, .02], [-.02, .02], [.75, .86])))
        value.cup_points_world = np.repeat(corners, 10, axis=0)
        value.cup_bounds_world = np.array([corners.min(0), corners.max(0)])
        task = SimpleNamespace(take_action_cnt=0, step_lim=10, get_obs=lambda: {},
            pillbottle=ForbiddenTruth(), get_camera_observation=lambda name: {},
            get_arm_pose=lambda arm: [0, 0, .9, 1, 0, 0, 0], robot=SimpleNamespace(
                get_left_gripper_val=lambda: .01, get_left_tcp_pose=lambda: [0, 0, .8, 1, 0, 0, 0]))
        def execute(action): task.take_action_cnt += 1
        task.take_action = execute
        def step(task, obs):
            for _ in range(10): task.take_action(np.zeros(14))
        record = {}
        rollout_bottle(task, SimpleNamespace(reset=lambda: None, step=step),
            SimpleNamespace(supplementary=lambda *a, **kw: value), "left", record)
        audit = record["candidate_handoff"]["visual_proximity"]
        self.assertGreater(audit["nearest_surface_distance_m"], .015)
        self.assertEqual(audit["distance_to_visible_bounds_m"], 0.)
        self.assertTrue(audit["accepted"])

    def test_trial_lift_no_object_reads_and_requires_visual_motion(self):
        for follows in (True, False):
            with self.subTest(follows=follows):
                initial = np.array([-.2, 0., 1., 1., 0., 0., 0.])
                ee = initial.copy()
                def plan(target):
                    return dict(status="Success", target=target.copy(),
                        position=np.zeros((2, 6)), velocity=np.zeros((2, 6)))
                def execute(controls):
                    ee[:] = controls["left_arm"]["target"]
                    return True
                def observe(frame, label, **kw):
                    value = estimate()
                    if "after" in label and follows:
                        delta = ee[:3]-initial[:3]
                        value.cup_points_world += delta
                        value.cup_bounds_world += delta
                    return value
                task = SimpleNamespace(pillbottle=ForbiddenTruth(), pad=ForbiddenTruth(), scene=ForbiddenTruth(),
                    robot=SimpleNamespace(left_plan_path=plan), plan_success=True,
                    get_arm_pose=lambda arm: ee.copy(), get_camera_observation=lambda name: {},
                    take_dense_action=execute)
                result = {}
                if follows:
                    decision = verify_bottle_lift(task, SimpleNamespace(supplementary=observe), "left",
                        result, RegistrationConfig())
                    self.assertTrue(decision["ready"])
                    self.assertFalse(decision["uses_simulator_truth"])
                else:
                    with self.assertRaisesRegex(RuntimeError, "co-motion"):
                        verify_bottle_lift(task, SimpleNamespace(supplementary=observe), "left",
                            result, RegistrationConfig())
                    self.assertFalse(result["complete"])

    def test_existing_cup_sources_are_unchanged(self):
        # The shared cup runner deliberately accepts the new task-independent
        # demonstration-relative approach prior.  Continue guarding unrelated
        # legacy placement and grasp sources against accidental edits.
        expected = {
            "adapters/pointcloud_placement.py": "896ae5ca311a2e1615fc1939e0b4e5e3af0200e6b205665fc7ae4addba5e9aa8",
            "adapters/visual_grasp.py": "932d7f3409b7df39e0cb89026831a1e21cceca5e13c7ca0dd82e62d7d2dd6ef3"}
        for path, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest)

    def test_invalid_or_missing_visual_baseline_never_moves_the_robot(self):
        def unavailable(*a, **kw):
            raise RuntimeError("No visible bottle")
        task = SimpleNamespace(get_camera_observation=lambda name: {},
            robot=ForbiddenTruth(), pillbottle=ForbiddenTruth(), pad=ForbiddenTruth())
        result = {}
        with self.assertRaisesRegex(RuntimeError, "no trial lift executed"):
            verify_bottle_lift(task, SimpleNamespace(supplementary=unavailable), "left", result,
                RegistrationConfig())
        self.assertEqual(result["segments"], [])
        for value in (float("nan"), -.1, .2):
            with self.assertRaises(ValueError):
                verify_bottle_lift(ForbiddenTruth(), None, "left", {}, RegistrationConfig(), lift_m=value)


if __name__ == "__main__":
    unittest.main()
