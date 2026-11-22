import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from adapters.handover.take_rollout import rollout_take

ROOT=Path(__file__).resolve().parents[1]


class Forbidden:
    def __getattr__(self,key):
        raise AssertionError("Controller inspected object truth")


class Task:
    def __init__(self):
        self.take_action_cnt,self.step_lim=200,200
        self.box,self.scene=Forbidden(),Forbidden()
        self.robot=SimpleNamespace(get_left_gripper_val=lambda:1.,get_right_gripper_val=lambda:0.)
    def get_instruction(self):
        return "take in right hand"
    def get_arm_pose(self,arm):
        return [0,0,1,1,0,0,0]
    def get_obs(self):
        return {}
    def take_action(self,values):
        self.take_action_cnt+=1


class Policy:
    chunk_steps=10
    def reset(self):
        pass
    def step(self,task,observation):
        for _ in range(10):
            if task.take_action_cnt>=task.step_lim:
                break
            task.take_action(np.zeros(14))


class TakeEvalTests(unittest.TestCase):
    def test_second_stage_gets_own_budget_after_200_pickup_actions(self):
        task=Task()
        original=task.take_action
        result={}
        rollout_take(task,Policy(),result,13)
        self.assertEqual(task.take_action_cnt,213)
        self.assertEqual(result["vla_steps"],13)
        self.assertTrue(result["complete"])
        self.assertEqual(task.step_lim,200)
        self.assertEqual(task.take_action,original)
        self.assertEqual(result["commands"][-1]["phase_step"],13)

    def test_invalid_action_restores_hook_and_limit(self):
        task=Task()
        original=task.take_action
        class Bad(Policy):
            def step(self,task,observation):
                task.take_action(np.zeros(13))
        with self.assertRaises(ValueError):
            rollout_take(task,Bad(),{},10)
        self.assertEqual(task.take_action,original)
        self.assertEqual(task.step_lim,200)
        self.assertEqual(task.take_action_cnt,200)

    def test_summary_has_separate_overall_and_conditional_denominators(self):
        spec=importlib.util.spec_from_file_location("take_eval_summary_test",ROOT/"scripts/eval_pi05_handover_take.py")
        runner=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        rows=[dict(seed=1,initialized=True,success=True,prefix_success=True),
              dict(seed=2,initialized=True,success=False,prefix_success=True),
              dict(seed=3,initialized=True,success=False,prefix_success=False),
              dict(seed=4,initialized=True,success=False,prefix_success=False),
              dict(seed=5,initialized=False,success=False)]
        summary=runner.summarize(rows)
        self.assertEqual(summary["right_take_success_rate_given_verified_middle"],.5)
        self.assertEqual(summary["overall_success_rate_through_handover"],.25)
        self.assertEqual(summary["valid_scenes"],4)
        self.assertEqual(summary["initialization_rejection_seeds"],[5])


if __name__=="__main__":
    unittest.main()
