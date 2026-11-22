"""Opt-in RGB-D occupied surfaces, calibrated robot self filtering, and CuRobo world.

No object actors, simulator masks, contacts, or task scores are accepted here.
Unseen space is NOT certified free. The map is refreshed before each motion;
old surfaces are cleared only by a measured free-space ray, not by occlusion.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        result[:3, :3] = Rotation.from_euler('xyz',
            np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return result


class RobotGeometry:
    """URDF collision hulls at measured joint angles, not simulator link poses."""
    def __init__(self, urdf):
        import trimesh
        urdf = Path(urdf).resolve()
        tree = ET.parse(urdf).getroot()
        self.joints = list(tree.findall('joint'))
        self.root = next(link.get('name') for link in tree.findall('link')
                         if link.get('name') not in {j.find('child').get('link') for j in self.joints})
        self.hulls = []
        self.visual_hulls = []
        for link in tree.findall('link'):
            # Robot images contain the visual exterior, which is not always
            # identical to the simplified physical collision hull (D435 case).
            visual = (link.findall('visual') if link.get('name').startswith(('fr_','fl_'))
                      or link.get('name') in ('right_camera','left_camera') else [])
            collisions = link.findall('collision')
            for collision in collisions+visual:
                geometry = collision.find('geometry')
                mesh = geometry.find('mesh')
                if mesh is not None:
                    path = urdf.parent / mesh.get('filename')
                    model = trimesh.load(path, force='mesh', process=False)
                    vertices = np.asarray(model.vertices) * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                elif geometry.find('box') is not None:
                    vertices = trimesh.creation.box(np.fromstring(geometry.find('box').get('size'), sep=' ')).vertices
                elif geometry.find('cylinder') is not None:
                    value = geometry.find('cylinder')
                    vertices = trimesh.creation.cylinder(float(value.get('radius')), float(value.get('length'))).vertices
                elif geometry.find('sphere') is not None:
                    vertices = trimesh.creation.icosphere(radius=float(geometry.find('sphere').get('radius'))).vertices
                else:
                    raise ValueError('Unsupported robot collision geometry')
                vertices = transform_points(vertices, origin(collision.find('origin')))
                hull = ConvexHull(vertices)
                value = (link.get('name'), vertices[hull.vertices], hull.equations)
                (self.hulls if collision in collisions else self.visual_hulls).append(value)

    def frames(self, joint_values, root_pose):
        frames = {self.root: np.asarray(root_pose)}
        remaining = list(self.joints)
        while remaining:
            progressed = False
            for joint in remaining[:]:
                parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                if parent not in frames:
                    continue
                motion = np.eye(4)
                kind = joint.get('type')
                if kind != 'fixed':
                    if joint.get('name') not in joint_values:
                        raise ValueError('Missing measured robot joint: '+joint.get('name'))
                    value = float(joint_values[joint.get('name')])
                    axis = joint.find('axis')
                    axis = np.fromstring('1 0 0' if axis is None else axis.get('xyz'), sep=' ')
                    if kind in ('revolute', 'continuous'):
                        motion[:3, :3] = Rotation.from_rotvec(axis*value).as_matrix()
                    elif kind == 'prismatic':
                        motion[:3, 3] = axis*value
                    else:
                        raise ValueError('Unsupported robot joint type: '+str(kind))
                frames[child] = frames[parent] @ origin(joint.find('origin')) @ motion
                remaining.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError('Disconnected or cyclic robot URDF')
        return frames

    def self_mask(self, points, frames, margin=.006):
        points = np.asarray(points)
        inside = np.zeros(len(points), bool)
        for name, vertices, equations in self.hulls+self.visual_hulls:
            pose = frames[name]
            world_vertices = transform_points(vertices, pose)
            candidates = np.flatnonzero(~inside & np.all(points >= world_vertices.min(0)-margin, axis=1)
                                       & np.all(points <= world_vertices.max(0)+margin, axis=1))
            if not len(candidates):
                continue
            local = (points[candidates]-pose[:3, 3]) @ pose[:3, :3]
            inside[candidates] = np.all(local @ equations[:, :3].T + equations[:, 3] <= margin, axis=1)
        return inside


def inside_box(points, pose, half, margin=0.):
    local = (np.asarray(points)-pose[:3, 3]) @ pose[:3, :3]
    return np.all(np.abs(local) <= np.asarray(half)+margin, axis=1)


def voxel_surface(keys, pitch):
    """Closed boundary of occupied voxels, with internal faces removed."""
    keys = np.unique(np.asarray(keys, dtype=np.int64).reshape(-1, 3), axis=0)
    occupied = {tuple(key) for key in keys}
    vertices, faces = [], []
    corners = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                        [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    sides = [((-1,0,0),(0,4,7,3)),((1,0,0),(1,2,6,5)),
             ((0,-1,0),(0,1,5,4)),((0,1,0),(3,7,6,2)),
             ((0,0,-1),(0,3,2,1)),((0,0,1),(4,5,6,7))]
    for key in keys:
        for direction, indices in sides:
            if tuple(key+direction) in occupied:
                continue
            offset = len(vertices)
            vertices.extend((key+corners[list(indices)])*pitch)
            faces.extend(((offset,offset+1,offset+2),(offset,offset+2,offset+3)))
    vertices = np.asarray(vertices,float).reshape(-1,3)
    faces = np.asarray(faces,int).reshape(-1,3)
    # Warp receives raw arrays (CuRobo does not run trimesh processing here).
    # Weld neighbouring faces so the collision surface is topologically closed.
    vertices,inverse = np.unique(vertices,axis=0,return_inverse=True)
    return vertices,inverse[faces]


def observed_free(points, frame, clearance):
    """Clear historical occupancy only if a valid depth ray sees beyond it."""
    from traditional_place_empty_cup.geometry import depth_to_metres, homogeneous_transform
    camera = transform_points(points, homogeneous_transform(frame['extrinsic_cv']))
    projected = camera @ np.asarray(frame['intrinsic_cv']).T
    pixels = np.rint(projected[:, :2]/np.maximum(projected[:, 2:3], 1e-9)).astype(int)
    depth = depth_to_metres(frame['depth'])
    h, w = depth.shape
    valid = ((camera[:,2] > 0) & (pixels[:,0] >= 1) & (pixels[:,0] < w-1)
             & (pixels[:,1] >= 1) & (pixels[:,1] < h-1))
    selected = np.flatnonzero(valid)
    # All nine neighbouring rays must see beyond the voxel. A discontinuity
    # or an invalid ray must not erase an occluded obstacle.
    if len(selected):
        x,y = pixels[selected].T
        samples = np.stack([depth[y+dy,x+dx] for dx in (-1,0,1) for dy in (-1,0,1)])
        valid[selected] = np.all(np.isfinite(samples) & (samples > camera[selected,2]+clearance),axis=0)
    return valid


class RGBDObstacleMap:
    def __init__(self, robot, initial_target, directory, pitch=.01):
        self.robot = robot
        self.geometry = RobotGeometry(robot.right_urdf_path)
        self.root_pose = robot.right_entity_origion_pose.to_transformation_matrix()
        self.initial_target = np.asarray(initial_target).copy()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pitch = pitch
        self.keys = np.empty((0,3), np.int64)
        self.table_z = None
        self.table_bounds = None
        self.sequence = 0
        self.audit = []

    def robot_frames(self):
        entity = self.robot.right_entity
        values = dict(zip((j.get_name() for j in entity.get_active_joints()), entity.get_qpos()))
        return self.geometry.frames(values, self.root_pose)

    def update(self, task, held_pose=None):
        from adapters.cup_target_focus import world_points_image, visual_table_height
        names = ['head_camera','left_camera','right_camera']
        if 'front_camera' in getattr(task.cameras,'static_camera_name',()):
            names.append('front_camera')  # Already present/calibrated; no added sensor or robot motion.
        frames = {name:task.get_camera_observation(name) for name in names}
        for name,frame in frames.items():
            if self.sequence <= 1 or name == 'front_camera':
                np.savez_compressed(self.directory/f'frame_{self.sequence:03d}_{name}.npz',
                    **{key:frame[key] for key in ('rgb','depth','intrinsic_cv','extrinsic_cv')})
        if self.table_z is None:
            head = frames['head_camera']
            world, valid = world_points_image(head['depth'],head['intrinsic_cv'],head['extrinsic_cv'])
            self.table_z = visual_table_height(world,valid)
            plane = world[valid & (np.abs(world[...,2]-self.table_z)<.006)]
            if len(plane)<80:
                raise RuntimeError('No measured table extent for collision world')
            self.table_bounds = np.quantile(plane[:,:2],[.005,.995],axis=0)
            if np.min(np.diff(self.table_bounds,axis=0))<.1:
                raise RuntimeError('Insufficient observed table extent')
        robot_frames = self.robot_frames()
        measured, counts = [], {}
        for name, frame in frames.items():
            world, valid = world_points_image(frame['depth'],frame['intrinsic_cv'],frame['extrinsic_cv'])
            # Restrict the modeled tabletop workspace, not an object-identity ROI.
            valid &= ((np.abs(world[...,0]) < .8) & (world[...,1] > -.55)
                      & (world[...,1] < .7) & (world[...,2] > self.table_z+.006)
                      & (world[...,2] < self.table_z+.85))
            points = world[valid]
            _, indices = np.unique(np.floor(points/(self.pitch/2)).astype(np.int64),axis=0,return_index=True)
            points = points[indices]
            self_points = self.geometry.self_mask(points,robot_frames)
            target = (inside_box(points,self.initial_target,[.03,.03,.1],.004)
                      if self.sequence==0 else np.zeros(len(points),bool))
            if held_pose is not None:
                target |= inside_box(points,held_pose,[.03,.03,.1],.004)
            counts[name] = dict(depth_points=len(points),robot_removed=int(self_points.sum()),
                                target_removed=int(target.sum()))
            measured.append(points[~self_points & ~target])
        old = (self.keys+.5)*self.pitch
        free = np.zeros(len(old),bool)
        for frame in frames.values():
            free |= observed_free(old,frame,2*self.pitch)
        if held_pose is not None:
            free |= inside_box(old,held_pose,[.03,.03,.1],.004)
        self.keys = np.unique(np.concatenate((self.keys[~free],
            np.floor(np.concatenate(measured)/self.pitch).astype(np.int64))),axis=0)
        if not len(self.keys):
            raise RuntimeError('No measured RGB-D obstacle surfaces; refusing obstacle-free fallback')
        path = self.directory / f'map_{self.sequence:03d}.npz'
        np.savez_compressed(path,keys=self.keys,pitch=self.pitch,table_z=self.table_z,
                            table_bounds=self.table_bounds)
        audit = dict(sequence=self.sequence,occupied_voxels=len(self.keys),voxel_m=self.pitch,
            historical_voxels_cleared=int(free.sum()),views=counts,map_path=str(path),
            uses_object_truth=False,uses_robot_identity_mask=False,unknown_space_is_free_claimed=False)
        self.audit.append(audit)
        self.sequence += 1
        return audit


def world_to_planner(planner, arm='right'):
    """Exactly the SAME frame bias/ALOHA calibration as CuroboPlanner targets."""
    value = np.linalg.inv(planner.robot_origion_pose.to_transformation_matrix())
    bias = np.eye(4)
    bias[:3,3] = planner.frame_bias
    result = bias @ value
    if 'aloha-agilex' in planner.yml_path:
        correction = np.eye(4)
        correction[:3,:3] = Rotation.from_euler('z',-.01 if arm=='right' else -.02).as_matrix()
        result = correction @ result
    return result


def measured_locked_joints(robot, names):
    """The gripper COMMAND (0/1) is not its measured prismatic joint position."""
    entity = robot.right_entity
    positions = dict(zip((j.get_name() for j in entity.get_active_joints()),entity.get_qpos()))
    if any(name not in positions for name in names):
        raise ValueError('Missing measured locked-joint position')
    return {name:float(positions[name]) for name in names}


def dense_joint_path(current, positions, max_step_rad=.01):
    current, positions = np.asarray(current,dtype=float),np.asarray(positions,dtype=float)
    if (current.ndim!=1 or positions.ndim!=2 or positions.shape[1:]!=current.shape
        or not len(positions) or not np.isfinite(current).all() or not np.isfinite(positions).all()
        or not np.isfinite(max_step_rad) or max_step_rad<=0):
        raise ValueError('A nonempty finite joint path and positive step are required')
    path = [current]
    for value in positions:
        start = path[-1]
        count = max(1,int(np.ceil(np.max(np.abs(value-start))/max_step_rad)))
        path.extend(start+(value-start)*fraction/count for fraction in range(1,count+1))
    return np.asarray(path)


def replace_collision_world(motion, world, *, synchronize=None):
    """CuRobo's Warp cache is keyed by NAME, not by vertex content.

    update_world alone reuses the old geometry for an unchanged mesh name.
    Synchronize before retiring Warp meshes; retain preallocated collision
    tensors so captured CUDA graph references stay valid.
    """
    if synchronize is None:
        import torch
        synchronize = lambda:torch.cuda.synchronize(motion.tensor_args.device)
    synchronize()
    motion.clear_world_cache()
    motion.update_world(world)


def collision_world(scene, planner):
    from curobo.geom.types import Mesh, Cuboid, WorldConfig
    transform = world_to_planner(planner)
    vertices, faces = voxel_surface(scene.keys,scene.pitch)
    vertices = transform_points(vertices,transform)
    # The static left arm is a known robot obstacle during right-only placement.
    extra_vertices, extra_faces = [vertices], [faces]
    frames = scene.robot_frames()
    size = len(vertices)
    for name, local, _ in scene.geometry.hulls:
        if not name.startswith('fl_') and name != 'left_camera':
            continue
        hull = ConvexHull(local)
        # ConvexHull simplices have arbitrary winding; trimesh fixes each solid.
        import trimesh
        mesh = trimesh.Trimesh(vertices=local,faces=hull.simplices,process=True)
        mesh.fix_normals()
        extra_vertices.append(transform_points(transform_points(mesh.vertices,frames[name]),transform))
        extra_faces.append(mesh.faces+size)
        size += len(mesh.vertices)
    table = np.eye(4)
    table[:3,3] = [*scene.table_bounds.mean(0),scene.table_z-.025]
    table = transform @ table
    quaternion = Rotation.from_matrix(table[:3,:3]).as_quat()
    return WorldConfig(mesh=[Mesh(name='rgbd_observed_and_static_left_arm',pose=[0,0,0,1,0,0,0],
        vertices=np.concatenate(extra_vertices),faces=np.concatenate(extra_faces))],
        cuboid=[Cuboid(name='visual_table',dims=[*np.diff(scene.table_bounds,axis=0)[0],.05],
        pose=[*table[:3,3],quaternion[3],*quaternion[:3]])])


class ObstaclePlanningSession:
    """Temporary right-arm world. Original planner/world restored even on failure."""
    def __init__(self,task,scene,result):
        self.task,self.scene,self.result = task,scene,result

    def __enter__(self):
        import yaml
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        robot = self.task.robot
        if robot.communication_flag:
            raise RuntimeError('RGB-D obstacle adapter requires in-process CuRobo; no blind fallback')
        self.planner = robot.right_planner
        self.original_motion = self.planner.motion_gen
        self.original_plan = robot.right_plan_path
        with open(self.planner.yml_path) as stream:
            self.robot_config = yaml.safe_load(stream)
        self.lock_names = list(self.robot_config['robot_cfg']['kinematics'].get('lock_joints',{}))
        self.last_locks = None
        self.result.update(enabled=True,geometry_source='RGBD_and_known_robot_URDF_measured_joints',
            arrival_gates_unchanged=True,object_truth_used=False,maps=self.scene.audit,plans=[])
        world = collision_world(self.scene,self.planner)
        cached = getattr(self.planner,'_rgbd_motion',None)
        if cached is None:
            config = MotionGenConfig.load_from_robot_config(self.planner.yml_path,world,
                interpolation_dt=1/250,num_trajopt_seeds=4,num_graph_seeds=4,
                collision_cache={'mesh':2,'obb':2},evaluate_interpolated_trajectory=True)
            cached = MotionGen(config)
            cached.warmup()
            self.planner._rgbd_motion = cached
        else:
            replace_collision_world(cached,world)
            cached.reset()
        self.planner.motion_gen = cached
        robot.right_plan_path = self.plan
        return self

    def update(self,held_pose):
        entry = self.scene.update(self.task,held_pose)
        replace_collision_world(self.planner.motion_gen,collision_world(self.scene,self.planner))
        return entry

    def plan(self,*args,**kwargs):
        from curobo.types.robot import JointState
        motion = self.planner.motion_gen
        locks = measured_locked_joints(self.task.robot,self.lock_names)
        # The legacy YAML fixes both fingers at 4 cm (open), even while the
        # actual gripper holds the block. Synchronize geometry, not thresholds.
        if locks != self.last_locks:
            motion.update_locked_joints(locks,self.robot_config)
            self.last_locks = locks
        result = self.original_plan(*args,**kwargs)
        entry = dict(status=result['status'],planner_status=result.get('planner_status'),
                     measured_locked_joints=locks)
        self.result['plans'].append(entry)
        current = np.asarray(self.task.robot.get_right_arm_real_jointState())[:6]
        current_state = JointState.from_position(motion.tensor_args.to_device(current[None]),
                                                joint_names=self.planner.active_joints_name)
        spheres = motion.kinematics.get_state(current_state.position).get_link_spheres().detach().cpu().numpy()
        diagnostic_path = self.scene.directory/f'plan_start_{len(self.result["plans"])-1:03d}.npz'
        np.savez_compressed(diagnostic_path,joints=current,spheres_planner_frame=spheres,
                            world_to_planner=world_to_planner(self.planner))
        entry['robot_geometry_debug_path'] = str(diagnostic_path)
        if result['status']=='Success':
            positions = np.asarray(result['position'])
            # Include the measured start and densify any large joint jumps.
            path = dense_joint_path(current,positions)
            feasible = []
            for begin in range(0,len(path),128):
                state = JointState.from_position(motion.tensor_args.to_device(np.asarray(path[begin:begin+128])),
                                                joint_names=self.planner.active_joints_name)
                metrics = motion.check_constraints(state)
                feasible.extend(metrics.feasible.detach().cpu().numpy().reshape(-1).tolist())
            entry.update(full_arm_samples_checked=len(path),full_arm_path_feasible=bool(all(feasible)),
                         max_checked_joint_increment_rad=.01)
            if not all(feasible):
                return dict(status='Fail',planner_status='RGBD_FULL_ARM_PATH_RECHECK_FAILED')
        return result

    def __exit__(self,*exc):
        self.task.robot.right_plan_path = self.original_plan
        self.planner.motion_gen = self.original_motion


class MeasuredObstacleObserver:
    """Update current occupancy only AFTER a held-object observation succeeds.

    A rigid transfer prediction is a search prior, not a segmentation mask.
    Inserting a current frame using that prior can leave held-object ghost
    voxels that remain occluded after the real fit arrives.
    """
    def __init__(self,observer,planning):
        self.observer,self.planning = observer,planning

    def observe(self,prior):
        pose,quality = self.observer.observe(prior)
        if not quality.get('accepted'):
            raise RuntimeError('Unaccepted held observation cannot authorize a background map update')
        quality = dict(quality,obstacle_map=self.planning.update(pose))
        return pose,quality

