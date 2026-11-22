from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.pillbottle_pregrasp_search import (
    geometric_candidates, select_ranked_pregrasp, quaternion_distance, known_traditional_timing)


def candidate(index):
    return dict(contact_id=index//10, rotation_id=index % 10,
        target_ee_pose=[index/100., 0., 1., 1., 0., 0., 0.],
        category=0, orientation_score=0., translation_from_current_m=0.)


def trajectory(value=0.):
    return dict(status="Success", position=np.full((3, 6), value), velocity=np.zeros((3, 6)))


class PregraspSearchTests(unittest.TestCase):
    def setUp(self):
        # Load only the data-only config; importing envs itself initializes
        # unrelated assets relative to RoboTwin's cwd, unsuitable for unit tests.
        import importlib.util
        spec = importlib.util.spec_from_file_location("ula_test_grasp_directions",
            ROOT/"tra_pipe/RoboTwin/envs/_GLOBAL_CONFIGS.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        scoped = patch.dict(sys.modules, {"envs._GLOBAL_CONFIGS": config})
        scoped.start()
        self.addCleanup(scoped.stop)

    def make_task(self):
        names = [f"j{i}" for i in range(6)]
        entity = SimpleNamespace(get_active_joints=lambda: [SimpleNamespace(get_name=lambda n=n: n) for n in names],
            get_qlimits=lambda: np.tile([-1., 1.], (6, 1)))
        robot = SimpleNamespace(left_entity=entity, left_arm_joints_name=names,
            left_plan_path=Mock(return_value=trajectory()), left_plan_multi_path=Mock(),
            get_grasp_perfect_direction=lambda arm: "left_arm_perf",
            create_target_pose_list=Mock(side_effect=lambda origin, center, arm: [origin.copy() for _ in range(10)]))
        task = SimpleNamespace(robot=robot, get_arm_pose=lambda arm: [0., 0., 1., 1., 0., 0., 0.])
        return task

    def test_geometric_generation_keeps_all_contacts_and_rotations_without_planning(self):
        task = self.make_task()
        actor = SimpleNamespace(config=dict(contact_points_pose=[np.eye(4)]*16),
            get_contact_point=lambda i, ret: np.eye(4))
        values = geometric_candidates(task, actor, "left")
        self.assertEqual(len(values), 160)
        self.assertEqual({(c["contact_id"], c["rotation_id"]) for c in values},
                         {(i, r) for i in range(16) for r in range(10)})
        task.robot.left_plan_path.assert_not_called()
        task.robot.left_plan_multi_path.assert_not_called()
        # Same native convention: contact frame X offset maps into grasp +X.
        origin, center, arm = task.robot.create_target_pose_list.call_args.args
        np.testing.assert_allclose(origin[:3], [0., .18, 0.])
        np.testing.assert_allclose(center[:3], [0., 0., 0.])
        self.assertEqual(arm, "left")

    def test_selected_contact_group_remains_bounded(self):
        task = self.make_task()
        actor = SimpleNamespace(config=dict(contact_points_pose=[np.eye(4)]*3),
            get_contact_point=lambda i, ret: np.eye(4))
        values = geometric_candidates(task, actor, "left", [1])
        self.assertEqual(len(values), 10)
        self.assertTrue(all(c["contact_id"] == 1 for c in values))
        with self.assertRaises(ValueError):
            geometric_candidates(task, actor, "left", [3])

    def test_quaternion_score_is_sign_invariant_and_normalized(self):
        self.assertAlmostEqual(quaternion_distance([1, 0, 0, 0], [-2, 0, 0, 0]), 0.)
        self.assertAlmostEqual(quaternion_distance([1, 0, 0, 0], [0, 1, 0, 0]), 1.)
        with self.assertRaises(ValueError):
            quaternion_distance([0]*4, [1, 0, 0, 0])

    def test_first_success_stops_search_and_reuses_the_exact_trajectory(self):
        task, audit = self.make_task(), {}
        original = trajectory()
        task.robot.left_plan_path.return_value = original
        with patch("adapters.pillbottle_pregrasp_search.geometric_candidates", return_value=[candidate(i) for i in range(160)]):
            target, plan = select_ranked_pregrasp(task, None, "left", audit)
        self.assertIs(plan, original)
        self.assertEqual(target, candidate(0)["target_ee_pose"])
        self.assertEqual(task.robot.left_plan_path.call_count, 1)
        task.robot.left_plan_multi_path.assert_not_called()
        self.assertTrue(audit["trajectory_reused"])
        self.assertFalse(audit["expanded_search"])

    def test_failed_and_out_of_limit_candidates_are_not_executed(self):
        task, audit = self.make_task(), {}
        task.robot.left_plan_path.side_effect = [dict(status="Fail"), trajectory(2.), trajectory(.1)]
        with patch("adapters.pillbottle_pregrasp_search.geometric_candidates", return_value=[candidate(i) for i in range(20)]):
            target, plan = select_ranked_pregrasp(task, None, "left", audit)
        self.assertEqual(target, candidate(2)["target_ee_pose"])
        self.assertEqual(task.robot.left_plan_path.call_count, 3)
        self.assertFalse(audit["planning_attempts"][1]["physical_joint_limit_check"]["accepted"])

    def test_failed_shortlist_expands_and_pads_native_batches_but_does_not_select_padding(self):
        task, audit = self.make_task(), {}
        task.robot.left_plan_path.return_value = dict(status="Fail")
        positions = np.zeros((10, 3, 6))
        velocities = np.ones((10, 3, 6))*.1
        statuses = ["Fail"]*10
        statuses[2] = "Success"
        task.robot.left_plan_multi_path.side_effect = [dict(status=["Fail"]*10),
            dict(status=statuses, position=positions, velocity=velocities)]
        with patch("adapters.pillbottle_pregrasp_search.geometric_candidates", return_value=[candidate(i) for i in range(22)]):
            target, plan = select_ranked_pregrasp(task, None, "left", audit)
        self.assertEqual(target, candidate(20)["target_ee_pose"])
        self.assertEqual(audit["planning_calls"], dict(single=8, batch=2))
        self.assertTrue(audit["expanded_search"])
        self.assertEqual(len(audit["planning_attempts"]), 21)
        calls = task.robot.left_plan_multi_path.call_args_list
        self.assertTrue(all(len(c.args[0]) == 10 for c in calls))
        self.assertEqual(calls[-1].args[0][4:], [candidate(21)["target_ee_pose"]]*6)
        np.testing.assert_allclose(plan["velocity"], velocities[2])
        self.assertFalse(np.shares_memory(plan["position"], positions))

    def test_exhausted_candidates_stop_with_finite_search_and_no_motion(self):
        task, audit = self.make_task(), {}
        task.robot.left_plan_path.return_value = dict(status="Fail")
        task.robot.left_plan_multi_path.return_value = dict(status=["Fail"]*10)
        with patch("adapters.pillbottle_pregrasp_search.geometric_candidates", return_value=[candidate(i) for i in range(22)]):
            with self.assertRaisesRegex(RuntimeError, "bounded full candidate"):
                select_ranked_pregrasp(task, None, "left", audit)
        self.assertEqual(len(audit["planning_attempts"]), 22)
        self.assertEqual(audit["planning_calls"], dict(single=8, batch=2))
        self.assertIn("selection_wall_s", audit)

    def test_invalid_start_does_not_trigger_more_planning(self):
        task, audit = self.make_task(), {}
        task.robot.left_plan_path.return_value = dict(status="Fail", planner_status="INVALID_START_STATE")
        with patch("adapters.pillbottle_pregrasp_search.geometric_candidates", return_value=[candidate(i) for i in range(20)]):
            with self.assertRaisesRegex(RuntimeError, "start state"):
                select_ranked_pregrasp(task, None, "left", audit)
        self.assertEqual(task.robot.left_plan_path.call_count, 1)
        task.robot.left_plan_multi_path.assert_not_called()

    def test_missing_velocity_is_not_an_executable_success(self):
        task = self.make_task()
        task.robot.left_plan_path.return_value = dict(status="Success", position=np.zeros((2, 6)))
        with patch("adapters.pillbottle_pregrasp_search.geometric_candidates", return_value=[candidate(0)]):
            with self.assertRaisesRegex(ValueError, "trajectory"):
                select_ranked_pregrasp(task, None, "left", {})

    def test_scoped_batch_timing_counts_and_restores_on_exception(self):
        from adapters.timing import EpisodeTiming
        task = self.make_task()
        original_batch = task.robot.left_plan_multi_path
        task.choose_grasp_pose = lambda: task.robot.left_plan_multi_path([])
        original_choose = task.choose_grasp_pose
        timer = EpisodeTiming()
        with self.assertRaisesRegex(RuntimeError, "test"):
            with timer.phase("approach"), known_traditional_timing(lambda *a: nullcontext(), timer, task, None):
                task.choose_grasp_pose()
                raise RuntimeError("test")
        self.assertIs(task.robot.left_plan_multi_path, original_batch)
        self.assertIs(task.choose_grasp_pose, original_choose)
        self.assertEqual(timer.stages["approach"]["calls"]["batch_motion_planning"], 1)
        self.assertEqual(timer.stages["approach"]["calls"]["grasp_candidate_search"], 1)
        self.assertAlmostEqual(sum(timer.stages["approach"]["components_s"].values()),
                               timer.stages["approach"]["wall_s"])

    def test_ranked_approach_executes_selected_plan_without_native_selector_or_replanning(self):
        from adapters.pillbottle_known_model import approach
        task = self.make_task()
        task.plan_success = True
        task.open_gripper = lambda *a, **kw: "open"
        task.move = lambda action: True
        task.choose_grasp_pose = Mock(side_effect=AssertionError("native exhaustive search called"))
        target = task.get_arm_pose("left")
        plan = trajectory()
        value = SimpleNamespace(cup_world_pose=np.eye(4), timings_s={})
        perception = SimpleNamespace(estimate=lambda *a: value)
        task.get_camera_observation = lambda name: dict(rgb=None, depth=None, intrinsic_cv=None, extrinsic_cv=None)
        args = SimpleNamespace(known_approach="annotated", known_contact_region="all", known_approach_search="ranked",
            known_metadata=ROOT/"tra_pipe/RoboTwin/assets/objects/080_pillbottle/model_data1.json",
            approach_slowdown=2., settle_steps=200)
        stub_utils = SimpleNamespace(ArmTag=lambda name: name)
        stub_controller = SimpleNamespace(EstimatedActor=lambda *a: None)
        with patch.dict(sys.modules, {"envs.utils": stub_utils,
                "traditional_place_empty_cup.controller": stub_controller}), patch(
                "adapters.pillbottle_pregrasp_search.select_ranked_pregrasp", return_value=(target, plan)), patch(
                "adapters.pillbottle_known_model.execute_checked") as execute:
            result = approach(task, perception, args)
        self.assertTrue(result["ready"])
        self.assertIs(execute.call_args.args[3], plan)
        self.assertEqual(execute.call_count, 1)
        task.choose_grasp_pose.assert_not_called()
        task.robot.left_plan_path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
