import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from diagnostics.handover_obstacle_audit import yaw_target,fresh_observation_world,PhysicalAudit,make_world
from diagnostics.handover_obstacle_snapshot import shape_mesh
from diagnostics.handover_collision_attribution import analyze_witness


class AuditTests(unittest.TestCase):
    def test_alternate_yaw_preserves_support_anchor_and_tilt(self):
        hand=np.eye(4);hand[:3,:3]=Rotation.from_euler('xyz',[.1,.3,-.5]).as_matrix()
        hand[:3,3]=[.1,-.1,.95]
        obj=np.eye(4);obj[:3,:3]=Rotation.from_euler('xyz',[.025,-.03,.4]).as_matrix()
        obj[:3,3]=[.1,.03,1.03]
        target=hand.copy();target[:3,3]+=[.01,.02,-.025]
        binding=np.linalg.solve(hand,obj)
        desired=target@binding
        anchor=np.array([0,0,-.1,1.])
        for yaw in [-np.pi,-.5,.25,np.pi/2]:
            new_hand=yaw_target(target,obj,hand,yaw)
            new_object=new_hand@binding
            np.testing.assert_allclose((new_object@anchor)[:3],(desired@anchor)[:3],atol=1e-12)
            self.assertAlmostEqual(new_object[2,2],desired[2,2])
            np.testing.assert_allclose(new_object[:3,:3]@new_object[:3,:3].T,np.eye(3),atol=1e-12)

    def test_zero_yaw_does_not_change_any_goal(self):
        ee=np.eye(4);ee[:3,3]=[.2,.1,.9]
        obj=np.eye(4);obj[:3,3]=[0,0,1.]
        np.testing.assert_allclose(yaw_target(ee,obj,np.eye(4),0),ee)

    def test_physical_box_uses_half_size_without_inventing_padding(self):
        Shape=type('PhysxCollisionShapeBox',(),{})
        shape=Shape();shape.half_size=np.array([.1,.2,.3])
        vertices,faces=shape_mesh(shape)
        np.testing.assert_allclose(vertices.min(0),[-.1,-.2,-.3])
        np.testing.assert_allclose(vertices.max(0),[.1,.2,.3])
        self.assertEqual(faces.shape,(12,3))

    def test_mesh_scale_is_explicit(self):
        shape=SimpleNamespace(vertices=np.eye(3),triangles=[[0,1,2]],scale=[2,3,4])
        vertices,_=shape_mesh(shape)
        np.testing.assert_array_equal(vertices,np.diag([2,3,4]))

    def test_unknown_physical_geometry_not_silently_ignored(self):
        with self.assertRaises(ValueError):shape_mesh(SimpleNamespace())

    def test_full_replay_keeps_original_combined_mesh_representation(self):
        types=SimpleNamespace(Mesh=lambda **kw:SimpleNamespace(**kw),Cuboid=lambda **kw:SimpleNamespace(**kw),
                              WorldConfig=lambda **kw:SimpleNamespace(**kw))
        data=dict(mesh_vertices=np.arange(18).reshape(6,3),mesh_faces=np.array([[0,1,2],[3,4,5]]),
                  voxel_vertex_count=3,voxel_face_count=1,mesh_pose=np.array([0,0,0,1,0,0,0]),
                  table_pose=np.array([0,0,0,1,0,0,0]),table_dims=np.ones(3))
        with patch.dict(sys.modules,{'curobo.geom.types':types}):
            full=make_world(data,'full');rgbd=make_world(data,'rgbd');left=make_world(data,'left')
        self.assertEqual(len(full.mesh),1)
        np.testing.assert_array_equal(full.mesh[0].vertices,data['mesh_vertices'])
        np.testing.assert_array_equal(full.mesh[0].faces,data['mesh_faces'])
        np.testing.assert_array_equal(rgbd.mesh[0].vertices,data['mesh_vertices'][:3])
        np.testing.assert_array_equal(left.mesh[0].faces,[[0,1,2]])

    def test_current_rgbd_ablation_removes_history_but_preserves_left_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene=Path(temporary);(scene/'held_vision').mkdir();(scene/'failure_snapshot').mkdir()
            frame=dict(rgb=np.zeros((20,20,3),np.uint8),depth=np.full((20,20),.9),
                intrinsic=np.array([[100,0,10],[0,100,10],[0,0,1]]),world_to_camera=np.eye(4))
            for name in ('head_camera','left_camera','right_camera'):
                np.savez_compressed(scene/'held_vision'/f'observation_000_{name}.npz',**frame)
            old_vertices=np.array([[0,0,0],[0,1,0],[1,0,0],[2,0,0],[2,1,0],[3,0,0]],float)
            visual=np.eye(4);visual[:3,3]=[.4,.4,.9]
            data=dict(joints_actual=np.zeros(6),world_to_planner=np.eye(4),
                table_pose=np.array([0,0,.715,1,0,0,0]),table_dims=np.array([1,1,.05]),
                map_pitch=.01,visual_object_pose=visual,map_keys=np.array([[999,999,999]]),
                voxel_vertex_count=3,voxel_face_count=1,mesh_vertices=old_vertices,
                mesh_faces=np.array([[0,1,2],[3,4,5]]))
            metadata=dict(right_gripper_command=0.,latest_visual_quality=dict(observation_index=0),
                          map_path=str(scene/'map_001.npz'))
            audit=SimpleNamespace(frames=lambda q:{},robot_geometry=SimpleNamespace(
                self_mask=lambda points,frames:np.zeros(len(points),bool)))
            fresh,info=fresh_observation_world(scene/'failure_snapshot',metadata,data,audit)
            self.assertTrue(info['available']);self.assertFalse(info['uses_physical_truth'])
            self.assertFalse(info['safe_for_controller'])
            self.assertFalse(np.any(np.all(fresh['map_keys']==[999,999,999],axis=1)))
            np.testing.assert_array_equal(fresh['mesh_vertices'][-3:],old_vertices[-3:])
            np.testing.assert_array_equal(data['mesh_vertices'],old_vertices)

    def test_released_hand_does_not_invent_matching_capture_proprioception(self):
        value,info=fresh_observation_world(Path('/not/read'),dict(right_gripper_command=1.),{},None)
        self.assertIsNone(value);self.assertFalse(info['available'])

    def test_independent_physics_excludes_manipulated_target_not_clutter(self):
        import trimesh
        model=trimesh.creation.box([.02,.02,.02])
        for clutter_x,expected in [(.2,[]),(.015,[('fr_link1','clutter')])]:
            with self.subTest(clutter_x=clutter_x),tempfile.TemporaryDirectory() as temporary:
                shapes=[];arrays={}
                for i,(name,role,x) in enumerate([('fr_link1','robot',0.),('box','object',0.),('clutter','object',clutter_x)]):
                    pose=np.eye(4);pose[0,3]=x
                    arrays.update({f'vertices_{i}':model.vertices,f'faces_{i}':model.faces,f'pose_{i}':pose})
                    shapes.append(dict(index=i,name=name,role=role,shape='PhysxCollisionShapeBox'))
                path=Path(temporary)/'physical.npz';np.savez_compressed(path,**arrays)
                metadata=dict(robot_urdf='unused',joint_names=['fr_joint1'],arm_joint_names=['fr_joint1'],
                    independent_physical_audit=dict(path=str(path),shapes=shapes))
                data=dict(full_qpos=np.zeros(1),root_pose=np.eye(4))
                geometry=SimpleNamespace(frames=lambda values,root:dict(fr_link1=np.eye(4)))
                with patch('diagnostics.handover_obstacle_audit.RobotGeometry',return_value=geometry):
                    audit=PhysicalAudit(metadata,data)
                self.assertEqual(audit.collisions([0.]),expected)
                self.assertEqual(audit.nearest([0.,0.,0.],background_only=True)[0]['name'],'clutter')

    def test_small_collision_sphere_missing_visible_real_link_is_robot_model_error(self):
        import trimesh
        model=trimesh.creation.box([.02,.02,.02])
        with tempfile.TemporaryDirectory() as temporary:
            arrays={};shapes=[]
            for i,(name,role,x) in enumerate([('fr_link1','robot',0.),('clutter','object',.015)]):
                pose=np.eye(4);pose[0,3]=x
                arrays.update({f'vertices_{i}':model.vertices,f'faces_{i}':model.faces,f'pose_{i}':pose})
                shapes.append(dict(index=i,name=name,role=role,shape='PhysxCollisionShapeBox'))
            path=Path(temporary)/'physical.npz';np.savez_compressed(path,**arrays)
            metadata=dict(robot_urdf='unused',joint_names=['fr_joint1'],arm_joint_names=['fr_joint1'],
                independent_physical_audit=dict(path=str(path),shapes=shapes),
                robot_config=dict(robot_cfg=dict(kinematics=dict(collision_sphere_buffer=0.,
                    collision_spheres=dict(fr_link1=[dict(center=[0,0,0],radius=.003)])))))
            data=dict(full_qpos=np.zeros(1),root_pose=np.eye(4),map_pitch=.002,map_keys=np.array([[4,0,0]]),
                world_to_planner=np.eye(4),table_pose=np.array([0,0,-1,1,0,0,0]),table_dims=np.array([2,2,.1]))
            geometry=SimpleNamespace(frames=lambda values,root:dict(fr_link1=np.eye(4)))
            with patch('diagnostics.handover_obstacle_audit.RobotGeometry',return_value=geometry):
                audit=PhysicalAudit(metadata,data)
            result=analyze_witness(audit,metadata,data,np.array([0.]),'fr_link1','clutter')
            self.assertTrue(result['physical_link_exceeds_sphere_collision_envelope_at_partner'])
            self.assertTrue(result['physical_link_intersects_recorded_visual_voxels'])
            self.assertTrue(result['visible_occupied_space_missed_by_link_sphere_model'])


if __name__=='__main__':unittest.main()
