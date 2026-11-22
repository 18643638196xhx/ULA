"""Offline counterfactual IK/path audit. No simulation, servo or policy execution.

Planning cases use only the captured VISUAL world. Physical truth is used by
an independent FCL checker after solutions are computed, never to choose a
controller action. A failed finite search is never labelled a proof of no path.
"""
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from adapters.handover.rgbd_obstacles import (RobotGeometry,transform_points,
    replace_collision_world,dense_joint_path)
from diagnostics.handover_obstacle_snapshot import dump
from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz,pose_wxyz_to_matrix


def yaw_target(ee,object_pose,hand,angle,anchor=(0,0,-.1)):
    """Same requested object support anchor and tilt; only world-Z yaw changes."""
    binding=np.linalg.solve(hand,object_pose)
    desired=ee@binding
    support=desired[:3,:3]@np.asarray(anchor)+desired[:3,3]
    desired[:3,:3]=Rotation.from_euler('z',angle).as_matrix()@desired[:3,:3]
    desired[:3,3]=support-desired[:3,:3]@np.asarray(anchor)
    return desired@np.linalg.inv(binding)


def make_world(data,mode):
    from curobo.geom.types import Mesh,Cuboid,WorldConfig
    meshes=[]
    vertices,faces=data['mesh_vertices'],data['mesh_faces']
    nv,nf=int(data['voxel_vertex_count']),int(data['voxel_face_count'])
    if mode=='full':
        meshes.append(Mesh(name='combined_exact',pose=data['mesh_pose'].tolist(),vertices=vertices,faces=faces))
    if mode=='rgbd':
        meshes.append(Mesh(name='observed_voxels',pose=data['mesh_pose'].tolist(),
                           vertices=vertices[:nv],faces=faces[:nf]))
    if mode=='left':
        meshes.append(Mesh(name='static_left',pose=data['mesh_pose'].tolist(),
                           vertices=vertices[nv:],faces=faces[nf:]-nv))
    tables=[] if mode=='empty' else [Cuboid(name='visual_table',pose=data['table_pose'].tolist(),
                                         dims=data['table_dims'].tolist())]
    return WorldConfig(mesh=meshes,cuboid=tables)


class PhysicalAudit:
    def __init__(self,metadata,data):
        import mplib
        from mplib.collision_detection import fcl
        self.fcl,self.Pose=fcl,mplib.Pose
        self.robot_geometry=RobotGeometry(metadata['robot_urdf'])
        self.data=data
        self.metadata=metadata
        audit=metadata['independent_physical_audit']
        raw=np.load(audit['path'])
        self.static,self.robot,self.all_shapes=[],[],[]
        for record in audit['shapes']:
            i=record['index'];v=raw[f'vertices_{i}'];f=raw[f'faces_{i}'];pose=raw[f'pose_{i}']
            if 'TriangleMesh' in record['shape']:
                geometry=fcl.BVHModel();geometry.begin_model(len(f),len(v))
                geometry.add_sub_model(v,f);geometry.end_model()
            else:
                geometry=fcl.Convex(np.asarray(v,dtype=np.float64),np.asarray(f,dtype=np.int32),False)
            obj=fcl.CollisionObject(geometry,self.Pose(np.asarray(pose,dtype=np.float64)))
            value=dict(record=record,vertices=v,geometry=geometry,object=obj,pose=pose)
            self.all_shapes.append(value)
            name=record['name']
            if record['role']=='robot' and (name.startswith('fr_') or name=='right_camera'):
                self.robot.append(value)
            elif ((record['role']=='object' and name!='box') or
                  (record['role']=='robot' and (name.startswith('fl_') or name=='left_camera'))):
                world=transform_points(v,pose)
                value.update(lower=world.min(0),upper=world.max(0))
                self.static.append(value)
        self.full_q=dict(zip(metadata['joint_names'],data['full_qpos']))

    def frames(self,q):
        values=dict(self.full_q)
        values.update(zip(self.metadata['arm_joint_names'],np.asarray(q)))
        return self.robot_geometry.frames(values,self.data['root_pose'])

    def collisions(self,q):
        frames=self.frames(q)
        pairs=[]
        for value in self.robot:
            name=value['record']['name'];pose=frames[name]
            world=transform_points(value['vertices'],pose)
            lo,hi=world.min(0),world.max(0)
            value['object'].set_pose(self.Pose(np.asarray(pose,dtype=np.float64)))
            for other in self.static:
                if np.any(lo>other['upper']+.0001) or np.any(hi<other['lower']-.0001):continue
                if self.fcl.collide(value['object'],other['object']).is_collision():
                    pairs.append([name,other['record']['name']])
        return sorted({tuple(pair) for pair in pairs})

    def nearest(self,point,*,background_only=False):
        query=self.fcl.CollisionObject(self.fcl.Sphere(1e-5),self.Pose(np.asarray(point,dtype=float)))
        distances=[]
        for other in self.static if background_only else self.all_shapes:
            # The object may have been repositioned by a hypothetical q query.
            other['object'].set_pose(self.Pose(np.asarray(other['pose'],dtype=float)))
            distance=self.fcl.distance(query,other['object']).min_distance
            distances.append(dict(distance_m=float(distance),name=other['record']['name'],
                                  role=other['record']['role']))
        return sorted(distances,key=lambda x:x['distance_m'])[:5]

    def calibration(self):
        frames=self.frames(self.data['joints_actual'])
        errors=[]
        for value in self.robot:
            pose=frames[value['record']['name']];actual=value['pose']
            errors.append(dict(name=value['record']['name'],
                translation_m=float(np.linalg.norm(pose[:3,3]-actual[:3,3])),
                rotation_deg=float(np.degrees(Rotation.from_matrix(pose[:3,:3]@actual[:3,:3].T).magnitude()))))
        return dict(max_translation_m=max(x['translation_m'] for x in errors),
                    max_rotation_deg=max(x['rotation_deg'] for x in errors),links=errors)


class Solver:
    def __init__(self,metadata,data,*,wide=False):
        from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig
        config=copy.deepcopy(metadata['robot_config'])
        config['robot_cfg']['kinematics']['lock_joints']=metadata['locked_joints']
        self.motion=MotionGen(MotionGenConfig.load_from_robot_config(config,make_world(data,'full'),
            interpolation_dt=1/250,use_cuda_graph=False,num_ik_seeds=128 if wide else 32,
            num_trajopt_seeds=16 if wide else 4,num_graph_seeds=16 if wide else 4,
            collision_cache={'mesh':4,'obb':2},evaluate_interpolated_trajectory=True))
        self.data,self.metadata,self.wide=data,metadata,wide

    def world(self,mode):
        replace_collision_world(self.motion,make_world(self.data,mode))
        self.motion.reset(reset_seed=False)

    def normalize(self):
        # Local CuRobo check_start_state can return with self-cost disabled on
        # WORLD_COLLISION. Do not let one diagnostic contaminate the next.
        self.motion.rollout_fn.robot_self_collision_constraint.enable_cost()
        self.motion.rollout_fn.primitive_collision_constraint.enable_cost()
        self.motion.update_pose_cost_metric(self.metric(None))

    def metric(self,constraint):
        from curobo.wrap.reacher.motion_gen import PoseCostMetric
        return PoseCostMetric() if constraint is None else PoseCostMetric(hold_partial_pose=True,
            hold_vec_weight=self.motion.tensor_args.to_device(constraint))

    def state(self,q):
        from curobo.types.robot import JointState
        return JointState.from_position(self.motion.tensor_args.to_device(np.atleast_2d(q)),
                                        joint_names=self.metadata['arm_joint_names'])

    def feasible(self,q):
        self.normalize()
        return self.motion.check_constraints(self.state(q)).feasible.detach().cpu().numpy().reshape(-1)

    def ik(self,target):
        from curobo.types.math import Pose
        self.normalize()
        t=time.perf_counter()
        result=self.motion.ik_solver.solve_single(Pose.from_list(matrix_to_pose_wxyz(target).tolist()),
            retract_config=self.state(self.data['joints_planner_rounded']).position,
            return_seeds=128 if self.wide else 32)
        success=result.success.detach().cpu().numpy().reshape(-1)
        q=result.solution.detach().cpu().numpy().reshape(-1,len(self.metadata['arm_joint_names']))
        q=q[success]
        if len(q):q=q[np.unique(np.round(q,3),axis=0,return_index=True)[1]]
        return q,dict(ik_successes=int(success.sum()),distinct_solutions=len(q),
                       elapsed_s=time.perf_counter()-t)

    def plan(self,target,constraint,output,label):
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        from curobo.types.math import Pose
        self.normalize()
        start=self.state(self.data['joints_planner_rounded'])
        valid,status=self.motion.check_start_state(start)
        # Restore state after the vendor validity helper as well.
        self.normalize()
        if not valid:return dict(success=False,status=str(status),start_valid=False)
        before=time.perf_counter()
        result=self.motion.plan_single(start,Pose.from_list(matrix_to_pose_wxyz(target).tolist()),
            MotionGenPlanConfig(max_attempts=30 if self.wide else 10,timeout=10.,
                pose_cost_metric=self.metric(constraint)))
        entry=dict(success=bool(result.success.item()),status=str(result.status),start_valid=True,
                   elapsed_s=time.perf_counter()-before)
        if entry['success']:
            positions=result.interpolated_plan.position.detach().cpu().numpy()
            path=dense_joint_path(self.data['joints_planner_rounded'],positions)
            masks=[]
            for offset in range(0,len(path),128):masks.extend(self.feasible(path[offset:offset+128]).tolist())
            entry.update(all_path_samples_feasible=bool(all(masks)),path_samples=len(path))
            np.savez_compressed(output/(label+'.npz'),positions=positions)
            entry['path_file']=str(output/(label+'.npz'))
        return entry


def fresh_observation_world(snapshot,metadata,data,audit):
    """RGB-D-only history ablation, NOT a safe production replacement for occlusion.

    The released-gripper case lacks matching latest finger proprioception, so
    refuse to fabricate an exact fresh map for it.
    """
    if metadata['right_gripper_command']>=.05:
        return None,dict(available=False,reason='released_hand_no_matching_frame_joint_snapshot')
    from adapters.cup_target_focus import world_points_image
    from adapters.handover.rgbd_obstacles import inside_box,voxel_surface
    scene=Path(snapshot).parent
    index=metadata['latest_visual_quality']['observation_index']
    sequence=int(Path(metadata['map_path']).stem.split('_')[-1])
    frames=[]
    for name in ('head_camera','left_camera','right_camera'):
        raw=dict(np.load(scene/'held_vision'/f'observation_{index:03d}_{name}.npz'))
        frames.append((name,raw['depth'],raw['intrinsic'],raw['world_to_camera']))
    path=scene/f'frame_{sequence:03d}_front_camera.npz'
    if path.exists():
        raw=dict(np.load(path));frames.append(('front_camera',raw['depth'],raw['intrinsic_cv'],raw['extrinsic_cv']))
    geometry=audit.robot_geometry;robot_frames=audit.frames(data['joints_actual'])
    table_world=np.linalg.inv(data['world_to_planner'])@pose_wxyz_to_matrix(data['table_pose'])
    table_z=table_world[2,3]+float(data['table_dims'][2])/2
    clouds=[];counts={}
    for name,depth,intrinsic,extrinsic in frames:
        world,valid=world_points_image(depth,intrinsic,extrinsic)
        valid &= ((np.abs(world[...,0])<.8)&(world[...,1]>-.55)&(world[...,1]<.7)
                  &(world[...,2]>table_z+.006)&(world[...,2]<table_z+.85))
        points=world[valid]
        _,selected=np.unique(np.floor(points/(float(data['map_pitch'])/2)).astype(np.int64),axis=0,return_index=True)
        points=points[selected]
        remove=geometry.self_mask(points,robot_frames)|inside_box(points,data['visual_object_pose'],[.03,.03,.1],.004)
        clouds.append(points[~remove]);counts[name]=int((~remove).sum())
    keys=np.unique(np.floor(np.concatenate(clouds)/float(data['map_pitch'])).astype(np.int64),axis=0)
    v,f=voxel_surface(keys,float(data['map_pitch']));v=transform_points(v,data['world_to_planner'])
    nv,nf=int(data['voxel_vertex_count']),int(data['voxel_face_count'])
    result=dict(data,mesh_vertices=np.concatenate((v,data['mesh_vertices'][nv:])),
                mesh_faces=np.concatenate((f,data['mesh_faces'][nf:]-nv+len(v))),
                voxel_vertex_count=len(v),voxel_face_count=len(f),map_keys=keys)
    return result,dict(available=True,current_voxels=len(keys),history_voxels=len(data['map_keys']),
                       views=counts,uses_physical_truth=False,safe_for_controller=False)


def physical_path_check(audit,current,path_file):
    positions=np.load(path_file)['positions']
    path=dense_joint_path(current,positions)
    bad=[]
    pair_counts={}
    for index,q in enumerate(path):
        pairs=audit.collisions(q)
        if pairs:
            bad.append(index)
            for pair in pairs:
                key=' / '.join(pair);pair_counts[key]=pair_counts.get(key,0)+1
    return dict(samples=len(path),geometric_collision_free=not bad,colliding_samples=len(bad),
                first_colliding_indices=bad[:10],collision_pairs=pair_counts,
                physical_world_frozen_at_failure=True,held_object_sweep_certified=False,
                actual_actions_executed=False)


def analyze_snapshot(path,output):
    path,output=Path(path),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((path/'metadata.json').read_text())
    data=dict(np.load(path/'planning_inputs.npz'))
    audit=PhysicalAudit(metadata,data)
    target=data['target_end_planner']
    result=dict(snapshot=str(path),candidate_actions_executed=False,
        physical_truth_used_for_planning=False,geometry_and_arrival_thresholds_changed=False,
        captured_failure=metadata['failed_result'],cost_flags_at_capture=dict(
            before=metadata['pre_plan_cost_flags'],after=metadata['post_plan_cost_flags']),
        calibration=audit.calibration(),current_physical_intersections=audit.collisions(data['joints_actual']),
        finite_search_is_not_infeasibility_proof=True,cases={},
        cuda_graph_disabled_for_diagnostic_shape_flexibility=True,
        diagnostic_start_check_costs_reenabled_each_case=True)
    dump(output/'audit.json',result)
    low=Solver(metadata,data)
    for mode in ('empty','table','rgbd','left','full'):
        low.world(mode)
        q,entry=low.ik(target)
        entry['current_state_feasible']=bool(low.feasible(data['joints_planner_rounded'])[0])
        entry['solutions']=q.tolist()
        result['cases'][mode]=entry
        if mode=='empty':empty_q=q
        if mode=='full':full_q=q
        dump(output/'audit.json',result)
    low.world('full')
    mask=low.feasible(empty_q) if len(empty_q) else np.zeros(0,bool)
    result['empty_world_ik_solutions_valid_in_full_world']=int(mask.sum())
    result['empty_world_solution_physical_collisions']=[dict(q=q.tolist(),pairs=audit.collisions(q)) for q in empty_q]
    result['fixed_goal_baseline_replay']=low.plan(target,metadata['constraint_pose'],output,'fixed_baseline')
    # Changing ONLY trajectory orientation holding, NOT endpoint pose.
    result['fixed_goal_without_path_hold']=low.plan(target,None,output,'fixed_no_path_hold')
    fresh,fresh_info=fresh_observation_world(path,metadata,data,audit)
    result['fresh_observation_only']=fresh_info
    if fresh is not None:
        # Preserve goal, left arm, table and path constraint; replace ONLY the
        # observed mesh by current RGB-D points without historical occupancy.
        low.data=fresh;low.world('full')
        fresh_q,fresh_ik=low.ik(target)
        fresh_info.update(ik=fresh_ik,physical_collisions=[audit.collisions(q) for q in fresh_q],
                          path=low.plan(target,metadata['constraint_pose'],output,'fresh_only_fixed_goal'))
        low.data=data
    dump(output/'audit.json',result)
    del low
    import gc
    gc.collect()
    wide=Solver(metadata,data,wide=True)
    wide.world('full')
    q,entry=wide.ik(target)
    result['wide_fixed_goal_ik']=dict(entry,solutions=q.tolist())
    result['wide_fixed_goal_trials']=[]
    for index in range(3):
        trial=wide.plan(target,metadata['constraint_pose'],output,f'wide_fixed_{index}')
        result['wide_fixed_goal_trials'].append(trial)
        dump(output/'audit.json',result)
        if trial['success']:break
        if trial.get('start_valid') is False:break
    result['yaw_cases']=[]
    inverse=np.linalg.inv(data['world_to_planner'])
    ee_target=pose_wxyz_to_matrix(data['target_ee_world'])
    current_frames=audit.frames(data['joints_actual'])
    hand=current_frames['fr_link6'].copy()
    # This embodiment's delta is identity and gripper_bias is .12; enforce it
    # rather than silently treating an arbitrary end-link as the TCP.
    if not np.allclose(metadata['delta_matrix'],np.eye(3)) or abs(metadata['gripper_bias']-.12)>1e-8:
        raise ValueError('Explicit TCP conversion needed for this embodiment')
    held=metadata['right_gripper_command']<.05 and 'visual_object_pose' in data
    for degrees in (15,-15,30,-30,60,-60,90,-90,135,-135,180):
        if held:
            alternate=yaw_target(ee_target,data['visual_object_pose'],hand,np.radians(degrees))
        else:
            alternate=ee_target.copy()
            alternate[:3,:3]=Rotation.from_euler('z',degrees,degrees=True).as_matrix()@alternate[:3,:3]
        planner_target=data['world_to_planner']@alternate
        q,entry=wide.ik(planner_target)
        entry.update(yaw_change_deg=degrees,held_object_anchor_preserved=bool(held),
            terminal_pose_world=alternate.tolist(),physical_collisions=[audit.collisions(v) for v in q],
            solutions=q.tolist())
        if len(q) and not any(x.get('path',{}).get('success') for x in result['yaw_cases']):
            # Both fixed_goal_without_path_hold and this trial use NO extra
            # orientation-holding constraint. Never conflate the two changes.
            entry['path']=wide.plan(planner_target,None,output,f'yaw_{degrees}')
        result['yaw_cases'].append(entry)
        dump(output/'audit.json',result)
    # Locate start-state sphere/voxel overlap without simulator labels.
    spheres=wide.motion.kinematics.get_state(wide.state(data['joints_actual']).position).get_link_spheres()
    spheres=spheres.detach().cpu().numpy().reshape(-1,4)
    centers=transform_points(spheres[:,:3],inverse)
    voxels=(data['map_keys']+.5)*float(data['map_pitch'])
    overlaps=[]
    sphere_names=[name for name,values in metadata['robot_config']['robot_cfg']['kinematics']['collision_spheres'].items()
                  for _ in values] if isinstance(metadata['robot_config']['robot_cfg']['kinematics']['collision_spheres'],dict) else None
    if sphere_names is None:
        import yaml
        sphere_names=[name for name,values in yaml.safe_load(Path(metadata['robot_config']['robot_cfg']['kinematics']['collision_spheres']).read_text())['collision_spheres'].items() for _ in values]
    for i,center in enumerate(centers):
        distance=np.linalg.norm(np.maximum(np.abs(voxels-center)-float(data['map_pitch'])/2,0),axis=1)-spheres[i,3]
        closest=int(np.argmin(distance))
        if distance[closest]<0:
            overlaps.append(dict(link=sphere_names[i],sphere_center_world=center.tolist(),
                sphere_radius_m=float(spheres[i,3]),voxel_center_world=voxels[closest].tolist(),
                overlap_m=float(-distance[closest]),nearest_physical_shapes=audit.nearest(voxels[closest])))
            overlaps[-1]['nearest_background_shapes']=audit.nearest(voxels[closest],background_only=True)
    result['current_sphere_voxel_overlaps']=overlaps
    result['independent_path_geometry']={str(p):physical_path_check(audit,data['joints_planner_rounded'],p)
        for p in sorted(output.glob('*.npz'))}
    result['status']='finished'
    dump(output/'audit.json',result)
    return result
