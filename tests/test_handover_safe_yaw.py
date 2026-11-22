import unittest
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

from adapters.handover.pose_search import support_yaw_target,SupportYawExecutor,YawSearchBudget
from adapters.handover.mesh_path_guard import (GuardMargins,VisualMeshPathGuard,
    bounded_joint_path,lever_bounds)
from adapters.handover.safe_pose_planning import restore_collision_checks


class YawTests(unittest.TestCase):
    def test_anchor_and_tilt_preserved_for_any_support_normal(self):
        binding=np.eye(4);binding[:3,3]=[.04,.09,-.03]
        binding[:3,:3]=Rotation.from_euler('xyz',[.1,-.2,.8]).as_matrix()
        target=np.eye(4);target[:3,3]=[.2,.1,.9]
        target[:3,:3]=Rotation.from_euler('xyz',[.3,-.1,.5]).as_matrix()
        anchor=np.array([.01,0,-.1])
        for normal in ([0,0,1],[.3,.5,1]):
            normal=np.asarray(normal);normal=normal/np.linalg.norm(normal)
            old=target@binding
            for yaw in (-3.,-.3,0.,1.7):
                new=support_yaw_target(target,binding,yaw,anchor=anchor,normal=normal)@binding
                np.testing.assert_allclose(new[:3,3]+new[:3,:3]@anchor,
                    old[:3,3]+old[:3,:3]@anchor,atol=1e-12)
                self.assertAlmostEqual(new[:3,2]@normal,old[:3,2]@normal)
        new=support_yaw_target(target,binding,.5,anchor=anchor,normal=[0,0,1])
        self.assertGreater(np.linalg.norm(new[:3,3]-target[:3,3]),.001)

    def test_symmetric_generic_budget_not_seed_specific(self):
        values=YawSearchBudget().angles()
        self.assertEqual(values[:5],(0.,15.,-15.,30.,-30.))
        self.assertEqual(len(set(values)),len(values))
        self.assertEqual(values[-1],180.)
        for step in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):YawSearchBudget(step_deg=step).angles()

    def execute(self,results,*,usable=True):
        calls=[]
        def plan(target,**kwargs):
            calls.append((target,kwargs));return results[min(len(calls)-1,len(results)-1)]
        task=SimpleNamespace(robot=SimpleNamespace(right_plan_path=plan),take_dense_action=lambda controls,save_freq:True)
        audit={}
        def settle(task,plan,target,result):result['ready']=True
        with patch('adapters.handover.pose_search.settle_endpoint',side_effect=settle):
            SupportYawExecutor(YawSearchBudget(max_plans=4))(task,np.eye(4),audit,
                binding=np.eye(4),visual=np.eye(4),normal=[0,0,1],anchor=[0,0,-.1],position_usable=usable)
        return calls,audit

    def test_old_endpoint_then_unlock_then_yaw_and_check_actual_arrival(self):
        success=dict(status='Success',position=np.zeros((1,6)),velocity=np.zeros((1,6)),
                     independent_guard=dict(accepted=True))
        calls,audit=self.execute([dict(status='Fail'),dict(status='Fail'),success])
        self.assertIn('constraint_pose',calls[0][1]);self.assertEqual(calls[1][1],{})
        np.testing.assert_array_equal(calls[0][0],calls[1][0])
        self.assertEqual(audit['chosen_yaw_deg'],15.)
        self.assertTrue(audit['settling']['ready'])

    def test_unguarded_success_never_executed(self):
        with self.assertRaisesRegex(RuntimeError,'requires the independent'):
            self.execute([dict(status='Success')])

    def test_invalid_start_stops_search_without_trying_yaw(self):
        count=[]
        def fail(*args,**kwargs):count.append(1);return dict(status='Fail',start_rejected=True)
        task=SimpleNamespace(robot=SimpleNamespace(right_plan_path=fail))
        with self.assertRaises(RuntimeError):
            SupportYawExecutor()(task,np.eye(4),{},binding=np.eye(4),visual=np.eye(4),
                normal=[0,0,1],anchor=[0,0,-.1],position_usable=True)
        self.assertEqual(len(count),1)

    def test_hidden_position_cannot_authorize_a_different_object_yaw(self):
        count=[]
        def fail(*args,**kwargs):count.append(1);return dict(status='Fail')
        task=SimpleNamespace(robot=SimpleNamespace(right_plan_path=fail))
        with self.assertRaises(RuntimeError):
            SupportYawExecutor()(task,np.eye(4),{},binding=np.eye(4),visual=np.eye(4),
                normal=[0,0,1],anchor=[0,0,-.1],position_usable=False)
        self.assertEqual(len(count),2)


class SweepTests(unittest.TestCase):
    def test_intermediate_motion_bound_and_sample_budget(self):
        bounds=np.array([.8,.3])
        path=bounded_joint_path([0,0],[[.1,-.1]],bounds,.002,1000)
        self.assertTrue(np.all(np.abs(np.diff(path,axis=0))@bounds<=.00200000001))
        np.testing.assert_allclose(path[-1],[.1,-.1])
        with self.assertRaises(RuntimeError):bounded_joint_path([0,0],[[1,1]],bounds,.002,10)

    def test_link_lengths_bound_carried_object_not_just_robot(self):
        joints=[ET.fromstring('<joint name="q" type="revolute"><parent link="base"/><child link="a"/><origin xyz="0 0 0"/></joint>'),
                ET.fromstring('<joint name="fixed" type="fixed"><parent link="a"/><child link="fr_link6"/><origin xyz=".2 0 0"/></joint>')]
        geometry=SimpleNamespace(joints=joints,hulls=[('fr_link6',np.array([[.1,0,0]]),None)])
        small=lever_bounds(geometry,['q'],{})
        large=lever_bounds(geometry,['q'],{},dict(fr_link6=.4))
        np.testing.assert_allclose(small,[.3]);np.testing.assert_allclose(large,[.6])

    def test_collision_costs_restored_independently(self):
        calls=[]
        motion=SimpleNamespace(rollout_fn=SimpleNamespace(
            robot_self_collision_constraint=SimpleNamespace(enable_cost=lambda:calls.append('self')),
            primitive_collision_constraint=SimpleNamespace(enable_cost=lambda:calls.append('world'))))
        restore_collision_checks(motion)
        self.assertEqual(calls,['self','world'])


class MeshGuardTests(unittest.TestCase):
    def scene(self,keys,*,object_offset=(0,0,.15)):
        from scipy.spatial import ConvexHull
        vertices=np.array([[x,y,z] for x in (-.01,.01) for y in (-.01,.01) for z in (-.01,.01)])
        values={}
        geometry=SimpleNamespace(hulls=[('fr_link6',vertices,ConvexHull(vertices).equations)],joints=[],
            frames=lambda q,root:{'fr_link6':np.eye(4)})
        scene=SimpleNamespace(geometry=geometry,sequence=1,keys=np.asarray(keys),pitch=.01,
            table_z=-.5,table_bounds=np.array([[-1,-1],[1,1]]),root_pose=np.eye(4))
        config=dict(robot_cfg=dict(kinematics=dict(self_collision_ignore={})))
        guard=VisualMeshPathGuard(scene,[],config,half=(.01,.01,.01))
        pose=np.eye(4);pose[:3,3]=object_offset
        guard.bind(pose,values);guard.refresh()
        return guard,values

    def test_full_hull_catches_obstacle_outside_tiny_search_sphere(self):
        guard,values=self.scene([[0,0,0]])
        hit=guard.check_state([],values)
        self.assertEqual(hit['kind'],'robot_environment')
        self.assertEqual(hit['link'],'fr_link6')

    def test_held_primitive_hits_obstacle_while_robot_is_clear(self):
        guard,values=self.scene([[0,0,15]])
        self.assertEqual(guard.check_state([],values)['kind'],'carried_object_environment')

    def test_empty_looking_space_not_confused_with_missing_object_binding(self):
        guard,values=self.scene([[30,30,30]])
        self.assertIsNone(guard.check_state([],values))
        guard.link_to_object=None
        self.assertEqual(guard.check_state([],values)['kind'],'missing_visual_object_binding')

    def test_release_freezes_object_instead_of_following_withdrawing_hand(self):
        guard,values=self.scene([[30,30,30]])
        before=guard.link_to_object.copy()
        guard.release(values)
        self.assertIsNone(guard.link_to_object)
        np.testing.assert_array_equal(guard.released_pose,before)
        self.assertIsNone(guard.check_state([],values))

    def test_invalid_margin_is_not_silently_used(self):
        with self.assertRaises(ValueError):GuardMargins(sweep_m=0).validate()

    def test_rigid_camera_object_pair_does_not_get_duplicate_motion_margin(self):
        from scipy.spatial import ConvexHull
        guard,values=self.scene([[30,30,30]])
        scene=guard.scene
        vertices=np.array([[x,y,z] for x in (-.005,.005) for y in (-.005,.005) for z in (-.005,.005)])
        vertices+=[.0225,0,.15]  # padded object right face .015, camera left .0175
        scene.geometry.hulls.append(('right_camera',vertices,ConvexHull(vertices).equations))
        scene.geometry.frames=lambda values,root:{'fr_link6':np.eye(4),'right_camera':np.eye(4)}
        config=dict(robot_cfg=dict(kinematics=dict(self_collision_ignore={})))
        guard=VisualMeshPathGuard(scene,[],config,half=(.01,.01,.01))
        pose=np.eye(4);pose[2,3]=.15
        guard.bind(pose,values);guard.refresh()
        self.assertEqual(guard.relative_margin('fr_link6','right_camera'),0.)
        self.assertIsNone(guard.check_state([],values))
        # A genuinely moving-relative pair still gets relative sweep padding.
        guard.ancestors['right_camera']=frozenset(('different_active_joint',))
        self.assertEqual(guard.check_state([],values)['kind'],'object_robot')


if __name__=='__main__':unittest.main()
