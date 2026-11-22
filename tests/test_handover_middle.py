import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tra_pipe/RoboTwin")]
from adapters.handover.middle_control import closure_attachment,carry_to_middle
from adapters.handover.middle_audit import evaluate_middle
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix


def pose(x=0.,z=1.):
    value=np.eye(4)
    value[:3,3]=[x,0,z]
    return value


def samples(block=None,left=0.):
    block=pose() if block is None else block
    return [dict(block_world=block.tolist(),left_ee_world=pose(x=block[0,3]-.12,z=1.07).tolist(),
                 left_command=left) for _ in range(5)]


class MiddleTests(unittest.TestCase):
    def test_attachment_uses_closure_hand_not_already_lifted_hand(self):
        commands=[dict(step=i,left=0.,left_ee_pose=[-.12,0,.91,1,0,0,0]) for i in range(1,4)]
        commands.append(dict(step=4,left=0.,left_ee_pose=[-.12,0,1.01,1,0,0,0]))
        initial=pose(z=.84)
        attachment,audit=closure_attachment(initial,commands)
        predicted=pose_wxyz_to_matrix(commands[-1]["left_ee_pose"])@attachment
        self.assertAlmostEqual(predicted[2,3],.94)
        self.assertEqual(audit["closure_step"],1)
        self.assertFalse(audit["uses_object_truth"])

    def test_reopening_resets_closure_anchor(self):
        commands=[dict(step=i,left=left,left_ee_pose=[-.12,0,.91,1,0,0,0])
                  for i,left in enumerate([0.,0.,0.,1.,0.,0.,0.])]
        _,audit=closure_attachment(pose(z=.84),commands)
        self.assertEqual(audit["closure_step"],4)
        with self.assertRaises(RuntimeError):
            closure_attachment(pose(z=.84),commands[:-1])

    def test_object_region_not_just_robot_arrival(self):
        result=evaluate_middle(samples(),.74)
        self.assertTrue(result["success"])
        self.assertTrue(result["precise_3cm_success"])
        self.assertFalse(evaluate_middle(samples(pose(x=-.2)),.74)["success"])
        result=evaluate_middle(samples(pose(x=.04)),.74)
        self.assertTrue(result["success"])
        self.assertFalse(result["precise_3cm_success"])

    def test_open_hand_tilt_or_relative_slip_cannot_pass(self):
        self.assertFalse(evaluate_middle(samples(left=1.),.74)["success"])
        tilted=pose()
        tilted[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
        result=evaluate_middle(samples(tilted),.74)
        self.assertEqual(result["reason"],"block_tilt_exceeds_tolerance")
        values=samples()
        values[-1]["block_world"]=pose(x=.03).tolist()
        self.assertFalse(evaluate_middle(values,.74)["success"])

    def test_controller_never_inspects_actor_or_contacts_and_does_not_trial_lift(self):
        class Forbidden:
            def __getattr__(self,key):
                raise AssertionError("Carry planner read simulation truth")
        class Task:
            box,scene=Forbidden(),Forbidden()
            plan_success=True
            robot=SimpleNamespace(get_left_gripper_val=lambda:0.)
            hand=[-.12,0,1.01,1,0,0,0]
            def get_arm_pose(self,arm):
                return self.hand
            def move(self,command):
                self.hand=list(command[1][0].target_pose)
                return True
        task=Task()
        result={}
        fake=SimpleNamespace(ArmTag=lambda name:name,
                             Action=lambda arm,kind,**kwargs:SimpleNamespace(**kwargs))
        with patch.dict(sys.modules,{"envs.utils":fake}):
            carry_to_middle(task,np.eye(4),result)
        self.assertTrue(result["complete"])
        self.assertFalse(result["additional_lift_before_transport"])

    def test_conditional_denominator_is_verified_pickup_not_just_command(self):
        spec=importlib.util.spec_from_file_location("middle_summary_tests",ROOT/"scripts/eval_pi05_handover_middle.py")
        runner=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        rows=[dict(seed=1,initialized=True,success=True,pickup_success=True,carry_entered=True),
              dict(seed=2,initialized=True,success=False,pickup_success=True,carry_entered=True),
              dict(seed=3,initialized=True,success=False,pickup_success=False,carry_entered=True),
              dict(seed=4,initialized=True,success=False,pickup_success=False,carry_entered=False),
              dict(seed=5,initialized=False,success=False)]
        summary=runner.summarize(rows)
        self.assertEqual(summary["overall_success_rate"],.25)
        self.assertEqual(summary["carry_success_rate_given_verified_pickup"],.5)
        self.assertEqual(summary["carry_attempts"],3)


if __name__=="__main__":
    unittest.main()
