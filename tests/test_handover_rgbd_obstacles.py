import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.handover.rgbd_obstacles import (RobotGeometry,transform_points,inside_box,
    voxel_surface,observed_free,world_to_planner,ObstaclePlanningSession,measured_locked_joints,dense_joint_path,
    replace_collision_world,MeasuredObstacleObserver)


class ObstacleTests(unittest.TestCase):
    def test_urdf_fk_requires_actual_joints_and_filters_robot_not_nearby_object(self):
        geometry=RobotGeometry(ROOT/'tra_pipe/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf')
        values={j.get('name'):0. for j in geometry.joints if j.get('type')!='fixed'}
        frames=geometry.frames(values,np.eye(4))
        link,vertices,_=next(h for h in geometry.hulls if h[0]=='fr_link3')
        center=transform_points(vertices.mean(0)[None],frames[link])[0]
        points=np.array([center,center+[0,0,.25]])
        mask=geometry.self_mask(points,frames)
        self.assertTrue(mask[0])
        # An external point beyond all robot geometry must survive filtering.
        self.assertFalse(geometry.self_mask(np.array([[2,2,2.]]),frames)[0])
        values.pop('fr_joint2')
        with self.assertRaisesRegex(ValueError,'Missing measured'):
            geometry.frames(values,np.eye(4))

    def test_target_exclusion_is_oriented_and_bounded(self):
        pose=np.eye(4)
        pose[:3,:3]=Rotation.from_euler('y',90,degrees=True).as_matrix()
        self.assertEqual(inside_box([[.09,0,0],[0,0,.09],[.12,0,0]],pose,[.03,.03,.1]).tolist(),[True,False,False])

    def test_mesh_is_closed_and_internal_faces_removed(self):
        import trimesh
        vertices,faces=voxel_surface([[0,0,0],[1,0,0]],.01)
        self.assertEqual(len(faces),20)
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(mesh.volume,0)
        self.assertAlmostEqual(mesh.volume,2e-6)

    def test_aloha_map_and_goal_use_identical_calibration(self):
        root=np.eye(4)
        root[:3,:3]=Rotation.from_euler('z',.7).as_matrix()
        root[:3,3]=[.2,-.6,0]
        planner=SimpleNamespace(robot_origion_pose=SimpleNamespace(to_transformation_matrix=lambda:root),
            frame_bias=np.array([-.2,.3,-.78]),yml_path='/aloha-agilex/curobo_right.yml')
        point=np.array([[.15,.1,.95]])
        local=transform_points(point,np.linalg.inv(root))+planner.frame_bias
        expected=local@Rotation.from_euler('z',-.01).as_matrix().T
        np.testing.assert_allclose(transform_points(point,world_to_planner(planner)),expected)

    def test_free_space_clearing_preserves_occluded_or_invalid_depth(self):
        frame=dict(depth=np.full((5,5),2.,dtype=float),intrinsic_cv=np.eye(3),extrinsic_cv=np.eye(4))
        points=np.array([[2,2,1.],[4,4,2.]])
        self.assertEqual(observed_free(points,frame,.02).tolist(),[True,False])
        frame['depth'][1,1]=.5
        self.assertFalse(observed_free(points,frame,.02)[0])
        frame['depth'][1,1]=np.nan
        self.assertFalse(observed_free(points,frame,.02)[0])

    def test_session_restores_original_planner_on_failure(self):
        robot=SimpleNamespace(right_plan_path='new')
        session=ObstaclePlanningSession(SimpleNamespace(robot=robot),None,{})
        session.planner=SimpleNamespace(motion_gen='new')
        session.original_plan='original'
        session.original_motion='original'
        session.__exit__(RuntimeError,RuntimeError(),None)
        self.assertEqual(robot.right_plan_path,'original')
        self.assertEqual(session.planner.motion_gen,'original')

    def test_gripper_geometry_uses_actual_joints_not_binary_command(self):
        entity=SimpleNamespace(get_active_joints=lambda:[SimpleNamespace(get_name=lambda:'finger')],
                               get_qpos=lambda:np.array([.013]))
        robot=SimpleNamespace(right_entity=entity,get_right_gripper_val=lambda:0.)
        self.assertEqual(measured_locked_joints(robot,['finger']),{'finger':.013})
        with self.assertRaises(ValueError):measured_locked_joints(robot,['missing'])

    def test_path_includes_measured_start_and_intermediate_swept_states(self):
        start=np.zeros(6)
        endpoint=np.ones((1,6))*.1
        path=dense_joint_path(start,endpoint)
        np.testing.assert_array_equal(path[0],start)
        np.testing.assert_array_equal(path[-1],endpoint[0])
        self.assertLessEqual(np.max(np.abs(np.diff(path,axis=0))),.010000001)
        # Endpoint-only testing misses an obstacle halfway along the motion.
        self.assertFalse(.04<endpoint[0,0]<.06)
        self.assertTrue(np.any((path[:,0]>.04)&(path[:,0]<.06)))

    def test_invalid_or_empty_path_cannot_be_certified(self):
        for path in [np.empty((0,6)),np.full((1,6),np.nan),np.zeros((1,5))]:
            with self.assertRaises(ValueError):dense_joint_path(np.zeros(6),path)

    def test_same_name_mesh_geometry_replaced_not_silently_reused(self):
        calls=[]
        cache={'scene':'old_vertices'}
        def clear():
            calls.append('clear');cache.clear()
        def update(world):
            calls.append('update')
            for name,value in world.items():cache.setdefault(name,value)
        motion=SimpleNamespace(clear_world_cache=clear,update_world=update)
        # This is the local CuRobo Warp-cache behaviour on same-name updates.
        update({'scene':'new_vertices'})
        self.assertEqual(cache['scene'],'old_vertices')
        calls.clear()
        replace_collision_world(motion,{'scene':'new_vertices'},synchronize=lambda:calls.append('sync'))
        self.assertEqual(calls,['sync','clear','update'])
        self.assertEqual(cache['scene'],'new_vertices')
        replace_collision_world(motion,{'scene':'next_episode'},synchronize=lambda:None)
        self.assertEqual(cache['scene'],'next_episode')

    def test_map_receives_measured_pose_not_transfer_prior(self):
        calls=[];prior=np.eye(4);measured=np.eye(4);measured[0,3]=.02
        def observe(value):
            self.assertIs(value,prior);calls.append('observe')
            return measured,dict(accepted=True)
        def update(value):
            self.assertIs(value,measured);calls.append('map')
            return dict(sequence=1)
        observer=MeasuredObstacleObserver(SimpleNamespace(observe=observe),SimpleNamespace(update=update))
        pose,quality=observer.observe(prior)
        self.assertIs(pose,measured)
        self.assertEqual(calls,['observe','map'])
        self.assertEqual(quality['obstacle_map']['sequence'],1)

    def test_rejected_observation_does_not_insert_held_object_ghosts(self):
        def update(value):raise AssertionError('No map update allowed')
        observer=MeasuredObstacleObserver(
            SimpleNamespace(observe=lambda prior:(prior,dict(accepted=False))),SimpleNamespace(update=update))
        with self.assertRaises(RuntimeError):observer.observe(np.eye(4))


if __name__=='__main__':unittest.main()
