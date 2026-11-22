"""Read-only failure capture. Never used by normal controllers or candidate execution."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from adapters.handover.rgbd_obstacles import transform_points, voxel_surface, world_to_planner


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def shape_mesh(shape):
    """Physical collision shape, not render mesh. SAPIEN 3 compatibility API."""
    import trimesh
    kind=type(shape).__name__
    if 'Plane' in kind:
        return None
    if hasattr(shape,'vertices'):
        return np.asarray(shape.vertices)*np.asarray(shape.scale),np.asarray(shape.triangles,dtype=int)
    if 'Box' in kind:
        mesh=trimesh.creation.box(2*np.asarray(shape.half_size))
    elif 'Sphere' in kind:
        # The engine primitive is round; this tessellation is audit-only.
        mesh=trimesh.creation.icosphere(subdivisions=3,radius=shape.radius)
    elif 'Capsule' in kind:
        mesh=trimesh.creation.capsule(radius=shape.radius,height=2*shape.half_length)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
    elif 'Cylinder' in kind:
        mesh=trimesh.creation.cylinder(radius=shape.radius,height=2*shape.half_length,sections=64)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
    else:
        raise ValueError('Unsupported physical collision shape: '+kind)
    return np.asarray(mesh.vertices),np.asarray(mesh.faces)


def physical_snapshot(task,directory):
    """Called ONLY after placement control has raised and no further actions run."""
    arrays,records,skipped={ },[],[]
    def append(body,role):
        entity=getattr(body,'entity',body)
        component=body if hasattr(body,'collision_shapes') else next(
            (c for c in entity.components if hasattr(c,'collision_shapes')),None)
        if component is None:return
        name=entity.name
        body_pose=entity.pose.to_transformation_matrix()
        for shape in component.collision_shapes:
            mesh=shape_mesh(shape)
            if mesh is None:
                skipped.append(dict(name=name,shape=type(shape).__name__))
                continue
            vertices,faces=mesh
            vertices=transform_points(vertices,shape.local_pose.to_transformation_matrix())
            index=len(records)
            arrays[f'vertices_{index}']=vertices
            arrays[f'faces_{index}']=faces
            arrays[f'pose_{index}']=body_pose
            records.append(dict(index=index,name=name,role=role,shape=type(shape).__name__,
                contact_offset=float(shape.contact_offset),rest_offset=float(shape.rest_offset),
                approximate_primitive=not hasattr(shape,'vertices') and 'Box' not in type(shape).__name__))
    for actor in task.scene.get_all_actors():append(actor,'object')
    for link in task.robot.right_entity.get_links():append(link,'robot')
    path=Path(directory)/'physical_after_control_only.npz'
    np.savez_compressed(path,**arrays)
    return dict(path=str(path),shapes=records,skipped=skipped,
                source='simulation_physical_geometry_after_control_stopped',feeds_back=False)


def capture_failure(session,args,kwargs,failed_result,pre_flags):
    """Copies EXACT loaded world and target, without changing them."""
    import yaml
    task,scene,planner=session.task,session.scene,session.planner
    motion=planner.motion_gen
    directory=scene.directory/'failure_snapshot'
    directory.mkdir()
    robot=task.robot
    joint_names=[j.get_name() for j in robot.right_entity.get_active_joints()]
    full_q=np.asarray(robot.right_entity.get_qpos()).copy()
    arm_q=full_q[[joint_names.index(n) for n in planner.active_joints_name]]
    end_pose=robot._trans_from_gripper_to_endlink(args[0],arm_tag='right').to_transformation_matrix()
    transform=world_to_planner(planner)
    world=motion.world_coll_checker.world_model
    if len(world.mesh)!=1 or len(world.cuboid)!=1:
        raise ValueError('Capture expects exactly the current combined mesh and table')
    mesh,table=world.mesh[0],world.cuboid[0]
    vertices,faces=mesh.get_mesh_data()
    cache=motion.world_coll_checker._wp_mesh_cache[mesh.name]
    cached_vertices=cache.vertices.numpy()
    cached_faces=cache.faces.numpy().reshape(-1,3)
    if not np.allclose(cached_vertices,vertices,atol=2e-7) or not np.array_equal(cached_faces,faces):
        raise RuntimeError('Loaded Warp geometry differs from configured world; do not claim exact replay')
    voxel_vertices,voxel_faces=voxel_surface(scene.keys,scene.pitch)
    voxel_vertices=transform_points(voxel_vertices,transform)
    nv,nf=len(voxel_vertices),len(voxel_faces)
    if not np.allclose(np.asarray(vertices)[:nv],voxel_vertices,atol=1e-9):
        raise RuntimeError('Cannot split observed geometry from static-left geometry')
    payload=dict(mesh_vertices=np.asarray(vertices),mesh_faces=np.asarray(faces),
        mesh_pose=np.asarray(mesh.pose),voxel_vertex_count=nv,voxel_face_count=nf,
        table_pose=np.asarray(table.pose),table_dims=np.asarray(table.dims),
        joints_actual=arm_q,joints_planner_rounded=np.round(arm_q,5),
        full_qpos=full_q,root_pose=robot.right_entity_origion_pose.to_transformation_matrix(),
        world_to_planner=transform,target_ee_world=np.asarray(args[0]),
        target_end_planner=transform@end_pose,map_keys=scene.keys.copy(),map_pitch=scene.pitch)
    visual=getattr(task,'_obstacle_audit_visual',None)
    if visual is not None:payload['visual_object_pose']=np.asarray(visual[0])
    np.savez_compressed(directory/'planning_inputs.npz',**payload)
    with open(planner.yml_path) as stream:robot_config=yaml.safe_load(stream)
    constraints=kwargs.get('constraint_pose')
    if constraints is not None:constraints=robot.get_constraint_pose(constraints,arm_tag='right')
    dependencies=[Path(planner.yml_path),Path(robot.right_urdf_path)]
    sphere_path=robot_config['robot_cfg']['kinematics']['collision_spheres']
    if isinstance(sphere_path,str):dependencies.append(Path(sphere_path))
    metadata=dict(failed_result=failed_result,constraint_pose=constraints,
        pre_plan_cost_flags=pre_flags,post_plan_cost_flags=cost_flags(motion),
        robot_config=robot_config,robot_config_path=str(planner.yml_path),
        robot_urdf=str(robot.right_urdf_path),joint_names=joint_names,
        arm_joint_names=list(planner.active_joints_name),locked_joints=session.last_locks,
        right_gripper_command=float(robot.get_right_gripper_val()),
        delta_matrix=np.asarray(robot.right_delta_matrix).tolist(),
        gripper_bias=float(robot.right_gripper_bias),
        initial_map_path=scene.audit[0]['map_path'],map_path=scene.audit[-1]['map_path'],
        latest_visual_quality=None if visual is None else visual[1],
        exact_loaded_warp_geometry_verified=True,candidate_actions_executed=False,
        fingerprints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies})
    # Separate file: explicitly NEVER an input to the controller/visual map.
    metadata['independent_physical_audit']=physical_snapshot(task,directory)
    dump(directory/'metadata.json',metadata)
    return str(directory)


def cost_flags(motion):
    return dict(self_collision=bool(motion.rollout_fn.robot_self_collision_constraint.enabled),
                world_collision=bool(motion.rollout_fn.primitive_collision_constraint.enabled))
