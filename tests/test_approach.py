"""Verify the control boundary without a simulator, GPU, or object ground truth."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np


RUNNER = Path(__file__).resolve().parents[1] / "scripts/run_place_empty_cup.py"
SPEC = importlib.util.spec_from_file_location("ula_runner", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeTask:
    """Deliberately has no cup/coaster actor attributes."""

    def __init__(self, fail_move=False):
        self.plan_success = True
        self.fail_move = fail_move
        self.executed = []
        self.target = [0.1, -0.2, 0.8, 1, 0, 0, 0]

    def get_camera_observation(self, _camera):
        return dict(rgb=np.zeros((2, 2, 3)), depth=np.ones((2, 2)),
                    intrinsic_cv=np.eye(3), extrinsic_cv=np.eye(4))

    def close_gripper(self, arm, pos):
        return arm, [SimpleNamespace(action="prepare", target_gripper_pos=pos)]

    def grasp_actor(self, actor, arm, **kwargs):
        return arm, [SimpleNamespace(action="move", target_pose=self.target),
                     SimpleNamespace(action="move", target_pose=self.target),
                     SimpleNamespace(action="gripper", target_gripper_pos=0)]

    def move(self, bundle):
        self.executed.extend(action.action for action in bundle[1])
        return not (self.fail_move and bundle[1][0].action == "move")

    def get_arm_pose(self, _arm):
        return self.target


class ApproachTests(unittest.TestCase):
    def run_approach(self, task):
        utils = ModuleType("envs.utils")
        utils.ArmTag = str
        controller = ModuleType("traditional_place_empty_cup.controller")
        controller.EstimatedActor = lambda pose, metadata: (pose, metadata)
        estimate = SimpleNamespace(cup_world_pose=np.eye(4),
                                   coaster_world_center=np.zeros(3), timings_s={})
        perception = SimpleNamespace(estimate=lambda *args: estimate)
        args = SimpleNamespace(arm="left", pregrasp_distance=0.1,
                               position_tolerance=0.02, orientation_tolerance=15)
        with patch.dict(sys.modules, {"envs.utils": utils,
                                     "traditional_place_empty_cup.controller": controller}):
            return runner.approach(task, perception, args)

    def test_only_pregrasp_executes_without_object_truth(self):
        task = FakeTask()
        result = self.run_approach(task)
        self.assertTrue(result["ready"])
        self.assertEqual(task.executed, ["prepare", "move"])

    def test_failed_move_cannot_trigger_handoff(self):
        result = self.run_approach(FakeTask(fail_move=True))
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "traditional_planning_failed")

    def test_return_handoff_interrupts_chunk_before_any_transport_action(self):
        from adapters.grasp import SimulationGraspMonitor

        task = SimpleNamespace(take_action_cnt=0, step_lim=100, eval_success=False)
        cup = np.eye(4)
        cup[2, 3] = 0.78
        task.cup = SimpleNamespace(get_pose=lambda: SimpleNamespace(
            p=cup[:3, 3], to_transformation_matrix=lambda: cup))
        task.robot = SimpleNamespace(get_left_gripper_val=lambda: 0.1)
        task.get_arm_pose = lambda arm: [0, 0, 0, 1, 0, 0, 0]
        task.get_obs = lambda: {}
        executed = []

        def original_action(action):
            executed.append(action)
            task.take_action_cnt += 1

        def ten_action_chunk(task, observation):
            for _ in range(10):
                task.take_action(np.zeros(14))

        task.take_action = original_action
        policy = SimpleNamespace(reset=lambda: None, step=ten_action_chunk)
        record = dict(evaluation_only=dict(max_cup_lift_m=0), grasp_handoff=None)
        with patch.object(runner, "active_gripper_contacts", return_value=2):
            runner.rollout(task, policy, 0.74, record,
                           SimulationGraspMonitor(0.74, stable_steps=3), "left")
        self.assertEqual(len(executed), 3)
        self.assertIs(task.take_action, original_action)
        self.assertEqual(record["grasp_handoff"]["after_vla_step"], 3)

    def test_grasp_timeout_does_not_run_placement(self):
        from adapters.grasp import SimulationGraspMonitor

        task = SimpleNamespace(take_action_cnt=0, step_lim=0, eval_success=False)
        task.take_action = lambda action: None
        policy = SimpleNamespace(reset=lambda: None)
        record = dict(evaluation_only=dict(max_cup_lift_m=0), grasp_handoff=None)
        with self.assertRaisesRegex(RuntimeError, "stable lifted grasp"):
            runner.rollout(task, policy, 0.74, record, SimulationGraspMonitor(0.74), "left")


if __name__ == "__main__":
    unittest.main()
