from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.handover.native_pi05_pickup import NativePi05Pickup,rollout_fixed_budget


class Forbidden:
    def __getattr__(self,key):
        raise AssertionError("Controller inspected object truth")


class Task:
    def __init__(self,limit=13):
        self.take_action_cnt,self.step_lim=0,limit
        self.box,self.scene=Forbidden(),Forbidden()
        self.robot=SimpleNamespace(get_left_gripper_val=lambda:0.,get_right_gripper_val=lambda:1.)
    def get_instruction(self):
        return "pick up and hold"
    def get_obs(self):
        return dict(joint_action=dict(vector=np.arange(14,dtype=np.float32)),
            observation={camera:dict(rgb=np.zeros((8,12,3),np.uint8))
                         for camera in ("head_camera","left_camera","right_camera")})
    def get_arm_pose(self,arm):
        # Huge upward hand motion must NOT stop this protocol early.
        return [0,0,float(self.take_action_cnt),1,0,0,0]
    def take_action(self,values):
        self.take_action_cnt+=1


class NativePi05Tests(unittest.TestCase):
    def test_policy_reads_only_images_command_state_and_language(self):
        policy=NativePi05Pickup("unused")
        inputs=[]
        def infer(sample):
            inputs.append(sample)
            return dict(actions=np.zeros((50,14)))
        policy.policy=SimpleNamespace(infer=infer)
        task=Task()
        policy.step(task,task.get_obs())
        self.assertEqual(task.take_action_cnt,10)
        self.assertEqual(set(inputs[0]),{"state","prompt","images"})
        self.assertEqual(inputs[0]["images"]["cam_high"].shape,(3,8,12))
        policy.step(task,task.get_obs())
        self.assertEqual(task.take_action_cnt,13)

    def test_fixed_budget_ignores_closed_hand_rise_and_restores_hook(self):
        task=Task(limit=7)
        original=task.take_action
        class Policy:
            chunk_steps=10
            def reset(self):
                pass
            def step(self,task,obs):
                task.take_action(np.zeros(14))
        result={}
        rollout_fixed_budget(task,Policy(),result)
        self.assertEqual(task.take_action_cnt,7)
        self.assertFalse(result["robot_endpoint_reached"])
        self.assertIsNone(result["robot_rise_stop_m"])
        self.assertEqual(task.take_action,original)

    def test_invalid_native_output_is_not_executed(self):
        policy=NativePi05Pickup("unused")
        policy.policy=SimpleNamespace(infer=lambda sample:dict(actions=np.zeros((50,13))))
        task=Task()
        with self.assertRaises(ValueError):
            policy.step(task,task.get_obs())
        self.assertEqual(task.take_action_cnt,0)


if __name__=="__main__":
    unittest.main()
