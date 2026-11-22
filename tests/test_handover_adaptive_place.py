from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tra_pipe/RoboTwin")]
from adapters.handover.adaptive_place import adaptive_place
from adapters.handover.held_vision import fit_cuboid,HALF,canonical_rotation
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix,matrix_to_pose_wxyz


def measured_cloud(pose,partial=False):
    rng=np.random.default_rng(21)
    points=[]
    for axis,sign in ((0,1),(1,-1),(2,1)):
        if partial and axis==2:
            continue
        surface=rng.uniform(-HALF,HALF,size=(250,3))
        surface[:,axis]=sign*HALF[axis]
        if partial:
            surface[:,2]=rng.uniform(-.04,.04,len(surface))
        points.append(surface)
    points=np.concatenate(points)
    return points@pose[:3,:3].T+pose[:3,3]+rng.normal(0,.0001,points.shape)


class Forbidden:
    def __getattr__(self,name):
        raise AssertionError("Controller inspected simulation truth")


class Task:
    def __init__(self,tilt=2.):
        self.box,self.scene,self.target_box=Forbidden(),Forbidden(),Forbidden()
        self.hand=np.eye(4)
        self.hand[:3,3]=[0.,0.,1.]
        self.binding=np.eye(4)
        self.binding[:3,:3]=Rotation.from_euler("zy",[37,tilt],degrees=True).as_matrix()
        self.commands=[]
        self.plan_success=True
        self.robot=SimpleNamespace(get_left_gripper_val=lambda:1.,get_right_gripper_val=lambda:0.)
    def get_arm_pose(self,name):
        return matrix_to_pose_wxyz(self.hand).tolist()
    def open_gripper(self,arm,pos):
        return "release"
    def move_by_displacement(self,arm,**kw):
        return "withdraw"
    def move(self,command):
        self.commands.append(command)
        return True


class Observer:
    def __init__(self,task,partial=False):
        self.task,self.partial=task,partial
    def observe(self,prior):
        return self.task.hand@self.task.binding,dict(accepted=True,
            position_observable=not self.partial,axis_uncertainty_deg=.5)


def fake_move(task,target,audit,*,rotation=False):
    task.hand=target.copy()
    audit.update(status="Success",rotation=rotation)


@patch.dict(sys.modules,{"envs.utils":SimpleNamespace(ArmTag=lambda name:name)})
class AdaptivePlaceTests(unittest.TestCase):
    @patch("adapters.handover.adaptive_place.move_segment",fake_move)
    def test_same_controller_preserves_acceptable_orientation(self):
        task=Task(tilt=2.)
        initial=task.hand[:3,:3].copy()
        result={}
        adaptive_place(task,task.binding,[.1,.1,.75],result,observer=Observer(task))
        self.assertTrue(result["complete"])
        self.assertIn("release",task.commands)
        self.assertFalse(any(action.get("rotation") for action in result["actions"]))
        np.testing.assert_allclose(task.hand[:3,:3],initial,atol=1e-9)

    @patch("adapters.handover.adaptive_place.move_segment",fake_move)
    def test_same_controller_corrects_only_required_tilt_then_places(self):
        task=Task(tilt=25.)
        result={}
        adaptive_place(task,task.binding,[.1,.1,.75],result,observer=Observer(task))
        self.assertTrue(result["complete"])
        self.assertTrue(any(action.get("rotation") for action in result["actions"]))
        self.assertLessEqual(result["final_orientation_decision"]["measured_tilt_deg"],5.)
        self.assertEqual(result["final_orientation_decision"]["yaw_constraint"],"free")

    @patch("adapters.handover.adaptive_place.move_segment",fake_move)
    def test_hidden_bottom_prior_never_authorizes_release(self):
        task=Task()
        with self.assertRaisesRegex(RuntimeError,"hidden bottom"):
            adaptive_place(task,task.binding,[.1,.1,.75],{},observer=Observer(task,partial=True))
        self.assertNotIn("release",task.commands)

    def test_unreliable_observations_stop_without_trial_lift_or_release(self):
        task=Task()
        class Missing:
            calls=0
            def observe(self,prior):
                self.calls+=1
                raise RuntimeError("occluded")
        observer=Missing()
        with self.assertRaisesRegex(RuntimeError,"bounded re-observation"):
            adaptive_place(task,task.binding,[.1,.1,.75],{},observer=observer)
        self.assertEqual(observer.calls,2)
        self.assertEqual(task.commands,[])


class HeldGeometryTests(unittest.TestCase):
    def test_fresh_tilted_surfaces_recover_axis_and_center(self):
        pose=np.eye(4)
        pose[:3,:3]=Rotation.from_euler("zy",[18,25],degrees=True).as_matrix()
        pose[:3,3]=[.1,.1,1.]
        prior=np.eye(4)
        prior[:3,3]=pose[:3,3]+[.01,-.01,.005]
        estimated,audit=fit_cuboid(measured_cloud(pose),prior)
        self.assertTrue(audit["position_observable"])
        self.assertLess(np.linalg.norm(estimated[:3,3]-pose[:3,3]),.003)
        self.assertLess(np.degrees(np.arccos(np.clip(estimated[:3,2]@pose[:3,2],-1,1))),1.)
        self.assertFalse(audit["uses_object_truth"])

    def test_side_surfaces_do_not_invent_hidden_end_positions(self):
        pose=np.eye(4)
        pose[:3,:3]=Rotation.from_euler("x",20,degrees=True).as_matrix()
        pose[:3,3]=[.1,.1,1.]
        prior=pose.copy()
        prior[:3,3]+=.02*pose[:3,2]
        _,audit=fit_cuboid(measured_cloud(pose,partial=True),prior)
        self.assertFalse(audit["position_observable"])
        self.assertIn("prior",audit["longitudinal_position_source"])

    def test_equivalent_end_and_square_frames_do_not_request_wrist_flip(self):
        equivalent=Rotation.from_euler("x",180,degrees=True).as_matrix()
        output=canonical_rotation(equivalent,np.eye(3))
        np.testing.assert_allclose(output,np.eye(3),atol=1e-8)


if __name__=="__main__":
    unittest.main()
