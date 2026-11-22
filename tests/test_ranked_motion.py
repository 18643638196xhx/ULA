import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tra_pipe/RoboTwin"))

from adapters.ranked_motion import joint_score, rank_candidates, search_candidates
from adapters.microwave.ranked_approach import capture_query, geometric_candidates, replaced


class RankedMotionTests(unittest.TestCase):
    def test_ranking_changes_with_current_robot_state_not_angle(self):
        limits = [[-2., -2.], [2., 2.]]
        candidates = [dict(id=0, pitch_deg=10., ik_feasible=True),
                      dict(id=1, pitch_deg=-10., ik_feasible=True)]
        for current, expected in (([-1., 0.], 1), ([1., 0.], 0)):
            for candidate, q in zip(candidates, ([1., 0.], [-1., 0.])):
                candidate["joint_rank"] = joint_score(q, current, limits)
            self.assertEqual(rank_candidates(candidates)[0]["id"], expected)

    def test_joint_limits_are_hard_not_wraparound(self):
        with self.assertRaises(ValueError):
            joint_score([3.], [0.], [[-2.], [2.]])
        self.assertGreater(joint_score([1.9], [-1.9], [[-2.], [2.]])["normalized_joint_motion"], .9)

    def test_quick_pass_exhausts_candidates_before_full_and_revisits_ik_failure(self):
        candidates = [dict(id=0, ik_feasible=True, joint_rank=dict(score=0.)),
                      dict(id=1, ik_feasible=True, joint_rank=dict(score=1.)),
                      dict(id=2, ik_feasible=False)]
        calls, audit = [], {}
        def plan(candidate, budget):
            calls.append((candidate["id"], budget.name))
            return dict(status="Success" if candidate["id"] == 2 else "Fail")
        selected, _ = search_candidates(candidates, plan, audit)
        self.assertEqual(selected["id"], 2)
        self.assertEqual(calls, [(0, "quick"), (1, "quick"), (0, "full_fallback"),
                                 (1, "full_fallback"), (2, "full_fallback")])

    def test_quick_success_does_not_trigger_full_or_repeat(self):
        calls = []
        candidate = dict(id=0, ik_feasible=True, joint_rank=dict(score=0.))
        search_candidates([candidate], lambda c,b: calls.append(b.name) or dict(status="Success"), {})
        self.assertEqual(calls, ["quick"])

    def test_geometric_candidates_follow_measured_normal(self):
        hinge = dict(initial_normal=np.array([0., -1., 0.]))
        candidates = geometric_candidates(hinge, [0., 0., 1.], np.eye(4), np.eye(4))
        self.assertTrue({-20., -10., 0., 10., 20.}.issubset({c["pitch_deg"] for c in candidates}))
        for candidate in candidates:
            self.assertAlmostEqual(np.linalg.det(candidate["target_matrix"][:3,:3]), 1.)
            self.assertGreater(candidate["target_matrix"][1,3]*-1, 0.)

    def test_capture_reuses_native_transform_and_restores_solver(self):
        class MG:
            def plan_single(self, start, goal, config):
                raise AssertionError("No full planning during query capture")
        mg = MG()
        class Robot:
            communication_flag = False
            left_planner = SimpleNamespace(motion_gen=mg)
            def left_plan_path(self, pose):
                return mg.plan_single("native_start", ("native_goal", tuple(pose)), None)
        robot = Robot()
        returned_mg, start, goal = capture_query(robot, "left", np.eye(4))
        self.assertIs(returned_mg, mg)
        self.assertEqual(start, "native_start")
        self.assertEqual(goal[0], "native_goal")
        self.assertNotIn("plan_single", vars(mg))

    def test_replaced_restores_on_exception(self):
        class Example:
            value = 1
        obj = Example()
        with self.assertRaises(RuntimeError):
            with replaced(obj, "value", 2):
                self.assertEqual(obj.value, 2)
                raise RuntimeError("test")
        self.assertEqual(obj.value, 1)
        self.assertNotIn("value", vars(obj))


if __name__ == "__main__":
    unittest.main()
