import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tra_pipe/RoboTwin")]
from adapters.handover.place_control import right_attachment,place_on_platform,slower_trajectory
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix


def pose(x=0.,z=1.):
    return [x,0,z,1,0,0,0]


def command(step,left=0.,right=0.,right_z=.93):
    return dict(step=step,phase_step=step,left=left,right=right,
        left_ee_pose=pose(x=-.12,z=1.07),right_ee_pose=pose(x=.12,z=right_z))


class Forbidden:
    def __getattr__(self,key):
        raise AssertionError("Placement controller inspected simulation object truth")


class PlannerTask:
    def __init__(self,never_plan=False,never_arrive=False):
        self.box,self.target_box,self.scene=Forbidden(),Forbidden(),Forbidden()
        self.calls=[]
        self.hand=pose()
        self.plan_success=True
        self.never_plan,self.never_arrive=never_plan,never_arrive
        self.robot=SimpleNamespace(get_right_gripper_val=lambda:0.,get_left_gripper_val=lambda:1.,
            get_right_arm_real_jointState=lambda:[*np.zeros(6),0.],
            right_plan_path=self.plan,right_arm_joints_name=[f"joint{i}" for i in range(6)],
            right_entity=SimpleNamespace(get_active_joints=lambda:[SimpleNamespace(get_name=lambda i=i:f"joint{i}") for i in range(6)],
                                         get_qpos=lambda:np.zeros(6)))
    def plan(self,target,**kwargs):
        self.calls.append("plan_down" if "last_qpos" in kwargs else "plan_above")
        assert kwargs["constraint_pose"]==[1,1,1,0,0,0]
        if self.never_plan:
            return dict(status="Fail")
        return dict(status="Success",position=np.zeros((1,6)),velocity=np.zeros((1,6)),target=target)
    def get_arm_pose(self,arm):
        return self.hand
    def take_dense_action(self,controls,**kwargs):
        target=controls["right_arm"].get("target")
        self.calls.append("execute" if target is not None else "hold")
        if not self.never_arrive and target is not None:
            self.hand=list(target)
        return True
    def open_gripper(self,arm,pos):
        return "release"
    def move_by_displacement(self,arm,**kwargs):
        assert kwargs["move_axis"]=="arm"
        return "withdraw"
    def move(self,command):
        self.calls.append(command)
        return True


class HandoverPlaceTests(unittest.TestCase):
    def test_time_scale_preserves_path_endpoints_and_halves_speed(self):
        position=np.arange(18).reshape(3,6)
        result=slower_trajectory(dict(position=position,velocity=np.ones((3,6))),2.)
        self.assertEqual(result["position"].shape,(5,6))
        np.testing.assert_allclose(result["position"][[0,-1]],position[[0,-1]])
        np.testing.assert_allclose(result["velocity"],.5)

    def test_binding_transferred_before_left_releases_not_after_hand_moves(self):
        block=np.eye(4)
        block[2,3]=1.
        left_attachment=np.linalg.solve(pose_wxyz_to_matrix(pose(x=-.12,z=1.07)),block)
        commands=[command(1),command(2,left=1.,right_z=1.),command(3,left=1.,right_z=1.)]
        binding,audit=right_attachment(left_attachment,commands)
        final=pose_wxyz_to_matrix(commands[-1]["right_ee_pose"])@binding
        self.assertAlmostEqual(final[2,3],1.07)
        self.assertEqual(audit["anchor_phase_step"],1)
        self.assertFalse(audit["uses_object_truth"])

    def test_reopening_discards_old_right_attachment(self):
        commands=[command(1),command(2,right=1.),command(3,left=1.),
                  command(4,left=1.),command(5,left=1.)]
        with self.assertRaisesRegex(RuntimeError,"No dual-closed"):
            right_attachment(np.eye(4),commands)

    @patch.dict(sys.modules,{"envs.utils":SimpleNamespace(ArmTag=lambda value:value)})
    def test_both_plans_before_execution_and_release_after_actual_arrival(self):
        task=PlannerTask()
        result={}
        place_on_platform(task,np.eye(4),np.array([.2,.1,.75]),result)
        self.assertEqual([call for call in task.calls if call!="hold"],
            ["plan_above","plan_down","execute","execute","release","withdraw"])
        self.assertTrue(all(action["settling"]["ready"] for action in result["actions"][:2]))
        self.assertTrue(result["complete"])
        self.assertFalse(result["uses_object_truth_for_planning"])
        self.assertAlmostEqual(result["desired_block_pose"][2][3],.853)

    @patch.dict(sys.modules,{"envs.utils":SimpleNamespace(ArmTag=lambda value:value)})
    def test_plan_failure_never_releases_or_executes(self):
        task=PlannerTask(never_plan=True)
        with self.assertRaisesRegex(RuntimeError,"No feasible"):
            place_on_platform(task,np.eye(4),np.zeros(3),{})
        self.assertTrue(all(call=="plan_above" for call in task.calls))

    @patch.dict(sys.modules,{"envs.utils":SimpleNamespace(ArmTag=lambda value:value)})
    def test_dense_execution_return_alone_does_not_authorize_release(self):
        task=PlannerTask(never_arrive=True)
        with self.assertRaisesRegex(RuntimeError,"not releasing"):
            place_on_platform(task,np.eye(4),np.array([.2,.1,.75]),{})
        self.assertNotIn("release",task.calls)

    @patch.dict(sys.modules,{"envs.utils":SimpleNamespace(ArmTag=lambda value:value)})
    def test_tracking_lag_gets_bounded_same_target_hold_before_release(self):
        class Lagged(PlannerTask):
            def __init__(self):
                super().__init__(never_arrive=True)
                self.pending=None
                self.waited=0
            def take_dense_action(self,controls,**kwargs):
                target=controls["right_arm"].get("target")
                if target is not None:
                    self.pending=list(target)
                    self.waited=0
                else:
                    self.waited+=20
                    if self.waited>=40:
                        self.hand=self.pending
                return super().take_dense_action(controls,**kwargs)
        result={}
        place_on_platform(Lagged(),np.eye(4),np.array([.2,.1,.75]),result)
        self.assertTrue(result["complete"])
        for action in result["actions"][:2]:
            self.assertFalse(action["settling"]["samples"][0]["arrival"]["ready"])
            self.assertTrue(action["settling"]["ready"])
            self.assertLessEqual(action["settling"]["physics_steps"],200)

    def test_full_summary_keeps_handover_and_placement_denominators_distinct(self):
        spec=importlib.util.spec_from_file_location("handover_full_summary_test",ROOT/"scripts/eval_pi05_handover_full.py")
        runner=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        rows=[dict(seed=1,initialized=True,prefix_success=True,handover_success=True,success=True,task_success=True),
              dict(seed=2,initialized=True,prefix_success=True,handover_success=True,success=False,task_success=False),
              dict(seed=3,initialized=True,prefix_success=True,handover_success=False,success=False,task_success=False),
              dict(seed=4,initialized=True,prefix_success=False,handover_success=False,success=False,task_success=False)]
        result=runner.summarize(rows)
        self.assertEqual(result["handover_successes"],2)
        self.assertEqual(result["right_take_success_rate_given_verified_middle"],2/3)
        self.assertEqual(result["placement_success_rate_given_verified_handover"],.5)
        self.assertEqual(result["overall_full_task_success_rate"],.25)


if __name__=="__main__":
    unittest.main()
