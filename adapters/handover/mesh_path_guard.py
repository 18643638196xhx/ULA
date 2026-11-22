"""Known-URDF hulls and visually bound primitive against measured occupied space.

No simulator actors, link poses, object IDs, contacts or task scores. CuRobo's
small spheres remain a SEARCH approximation; they never authorize execution.
The separate hull checker closes their geometry coverage gap for accepted paths.
It is not a claim that unseen space is free or that the grasp cannot slip.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull

from adapters.handover.rgbd_obstacles import transform_points
from adapters.handover.pose_search import rigid


@dataclass(frozen=True)
class GuardMargins:
    sweep_m: float=.002
    object_uncertainty_m: float=.005
    max_path_samples: int=20000

    def validate(self):
        if (not np.isfinite(self.sweep_m) or self.sweep_m<=0 or
            not np.isfinite(self.object_uncertainty_m) or self.object_uncertainty_m<0 or
            self.max_path_samples<2):raise ValueError('Invalid bounded geometry guard margins')


def bounded_joint_path(current,positions,lever_bounds,margin,max_samples):
    """A conservative displacement bound, rather than endpoint-only testing.

    For each revolute joint, lever_bounds bounds every point's distance to that
    axis throughout the joint interpolation. Sum(L*abs(dq)) bounds movement.
    Each interval is <= margin; checking static clearance margin and self
    clearance 2*margin at its beginning therefore covers that interpolation.
    """
    current,positions=np.asarray(current,float),np.asarray(positions,float)
    lever_bounds=np.asarray(lever_bounds,float)
    if (current.ndim!=1 or positions.ndim!=2 or positions.shape[1:]!=current.shape or
        not len(positions) or lever_bounds.shape!=current.shape or
        not np.isfinite(current).all() or not np.isfinite(positions).all() or
        not np.isfinite(lever_bounds).all() or np.any(lever_bounds<0) or
        not np.isfinite(margin) or margin<=0 or max_samples<2):
        raise ValueError('Finite joint path and displacement bounds are required')
    out=[current]
    for endpoint in positions:
        start=out[-1]
        count=max(1,int(np.ceil(np.abs(endpoint-start)@lever_bounds/margin)))
        if len(out)+count>max_samples:
            raise RuntimeError('Geometry guard path-sample budget exhausted')
        out.extend(start+(endpoint-start)*fraction/count for fraction in range(1,count+1))
    return np.asarray(out)


def lever_bounds(geometry,joint_names,values,extra_radius_by_link=None):
    """Triangle-inequality link lengths bound reach for any intermediate angle."""
    parent={j.find('child').get('link'):j for j in geometry.joints}
    active=set(joint_names)
    by_name={j.get('name'):j for j in geometry.joints}
    if any(name not in by_name or by_name[name].get('type') not in ('revolute','continuous')
           for name in joint_names):
        raise ValueError('This motion bound requires named revolute arm joints')
    radii={}
    for name,vertices,_ in geometry.hulls:
        if name.startswith('fr_') or name=='right_camera':
            radii[name]=max(radii.get(name,0.),float(np.max(np.linalg.norm(vertices,axis=1))))
    for name,radius in (extra_radius_by_link or {}).items():
        if not np.isfinite(radius) or radius<0:raise ValueError('Invalid attached-object reach')
        radii[name]=max(radii.get(name,0.),radius)
    bounds={name:0. for name in joint_names}
    for link,radius in radii.items():
        reach=radius
        while link in parent:
            joint=parent[link];name=joint.get('name')
            if name in active:bounds[name]=max(bounds[name],reach)
            element=joint.find('origin')
            xyz=np.fromstring('0 0 0' if element is None else element.get('xyz','0 0 0'),sep=' ')
            reach+=np.linalg.norm(xyz)
            if joint.get('type')=='prismatic':
                if name not in values:raise ValueError('Missing measured locked prismatic joint')
                axis=joint.find('axis')
                reach+=abs(values[name])*np.linalg.norm(np.fromstring(
                    '1 0 0' if axis is None else axis.get('xyz'),sep=' '))
            link=joint.find('parent').get('link')
    return np.asarray([bounds[name] for name in joint_names])


def active_ancestors(geometry,link,joint_names):
    """Equal active ancestry means rigid relative pose during this arm path.

    Other arm/finger joints are held at their measured values for this check.
    Their possible real motion remains an explicitly unmodelled tracking error.
    """
    parent={joint.find('child').get('link'):joint for joint in geometry.joints}
    active=set(joint_names);result=set()
    while link in parent:
        joint=parent[link]
        if joint.get('name') in active:result.add(joint.get('name'))
        link=joint.find('parent').get('link')
    return frozenset(result)


class VisualMeshPathGuard:
    def __init__(self,scene,joint_names,robot_config,*,half=(.03,.03,.1),ee_link='fr_link6',
                 gripping_links=('fr_link7','fr_link8'),margins=GuardMargins()):
        import mplib
        from mplib.collision_detection import fcl
        import trimesh
        margins.validate()
        self.fcl,self.Pose=fcl,mplib.Pose
        self.scene,self.joint_names,self.ee_link=scene,list(joint_names),ee_link
        self.margins=margins
        self.half=np.asarray(half,float)
        if self.half.shape!=(3,) or not np.isfinite(self.half).all() or np.any(self.half<=0):
            raise ValueError('A task-provided positive primitive half extent is required')
        self.gripping_links=set(gripping_links)
        self.ignore=set()
        for first,others in robot_config['robot_cfg']['kinematics'].get('self_collision_ignore',{}).items():
            self.ignore.update(frozenset((first,other)) for other in others)
        self.shapes=[]
        for name,vertices,_ in scene.geometry.hulls:
            if not (name.startswith(('fr_link','fl_link')) or name in
                    ('fr_base_link','fl_base_link','right_camera','left_camera')):continue
            hull=ConvexHull(vertices)
            mesh=trimesh.Trimesh(vertices=vertices,faces=hull.simplices,process=True)
            mesh.fix_normals()
            shape=fcl.Convex(np.asarray(mesh.vertices,np.float64),np.asarray(mesh.faces,np.int32),False)
            obj=fcl.CollisionObject(shape,self.Pose())
            self.shapes.append(dict(name=name,vertices=np.asarray(mesh.vertices),object=obj,
                                    right=name.startswith('fr_') or name=='right_camera'))
        if ee_link not in {item['name'] for item in self.shapes}:
            raise ValueError('TCP parent link missing from full geometry guard')
        self.ancestors={item['name']:active_ancestors(scene.geometry,item['name'],self.joint_names)
                        for item in self.shapes}
        self.link_to_object=None
        self.released_pose=None
        self.sequence=None

    def _object(self,geometry,pose):
        return self.fcl.CollisionObject(geometry,self.Pose(np.asarray(pose,dtype=float)))

    def refresh(self):
        if self.sequence==self.scene.sequence:return
        if (not len(self.scene.keys) or self.scene.table_z is None or
            self.scene.table_bounds is None):raise RuntimeError('Missing measured occupied world')
        pad=self.margins.sweep_m
        self.centres=(np.asarray(self.scene.keys)+.5)*self.scene.pitch
        self.voxel_half=self.scene.pitch/2+pad
        geometry=self.fcl.Box(*([2*self.voxel_half]*3))
        self.voxels=[]
        for centre in self.centres:
            pose=np.eye(4);pose[:3,3]=centre
            self.voxels.append(self._object(geometry,pose))
        table_pose=np.eye(4)
        table_pose[:3,3]=[*self.scene.table_bounds.mean(0),self.scene.table_z-.025]
        dims=np.r_[np.diff(self.scene.table_bounds,axis=0)[0],.05]
        self.table=self._object(self.fcl.Box(*(dims+2*pad)),table_pose)
        self.table_lower=table_pose[:3,3]-dims/2-pad
        self.table_upper=table_pose[:3,3]+dims/2+pad
        self.sequence=self.scene.sequence

    def bind(self,pose,values):
        frames=self.scene.geometry.frames(values,self.scene.root_pose)
        self.link_to_object=np.linalg.solve(frames[self.ee_link],rigid(pose))
        self.released_pose=None

    def release(self,values):
        if self.link_to_object is None:raise RuntimeError('No visual object binding at release')
        frames=self.scene.geometry.frames(values,self.scene.root_pose)
        self.released_pose=frames[self.ee_link]@self.link_to_object
        self.link_to_object=None

    def _world_hit(self,obj,lo,hi,static):
        mask=np.all(self.centres+self.voxel_half>=lo,axis=1)&np.all(self.centres-self.voxel_half<=hi,axis=1)
        for index in np.flatnonzero(mask):
            if self.fcl.collide(obj,self.voxels[index]).is_collision():
                return 'observed_voxel',int(index)
        if np.all(hi>=self.table_lower) and np.all(lo<=self.table_upper):
            if self.fcl.collide(obj,self.table).is_collision():return 'visual_table',None
        for other in static:
            if self._close(obj,lo,hi,other,self.margins.sweep_m):return other['name'],None
        return None

    def _close(self,obj,lo,hi,other,margin):
        if np.any(lo>other['upper']+margin) or np.any(hi<other['lower']-margin):return False
        return (self.fcl.collide(obj,other['object']).is_collision() or
                self.fcl.distance(obj,other['object']).min_distance<=margin)

    def relative_margin(self,first,second):
        # Do not add TWO absolute motion bounds to a pair which moves rigidly
        # together. Their relative sweep is exactly zero under this model.
        # Geometry and the object's 5 mm estimation envelope still apply.
        return (0. if self.ancestors[first]==self.ancestors[second]
                else 2*self.margins.sweep_m)

    def check_state(self,q,values):
        values=dict(values);values.update(zip(self.joint_names,q))
        frames=self.scene.geometry.frames(values,self.scene.root_pose)
        right,static=[],[]
        for shape in self.shapes:
            item=dict(shape)
            item['object'].set_pose(self.Pose(np.asarray(frames[item['name']],dtype=float)))
            points=transform_points(item['vertices'],frames[item['name']])
            item.update(lower=points.min(0),upper=points.max(0))
            (right if item['right'] else static).append(item)
        for index,item in enumerate(right):
            hit=self._world_hit(item['object'],item['lower'],item['upper'],static)
            if hit:return dict(kind='robot_environment',link=item['name'],obstacle=hit[0],voxel_index=hit[1])
            for other in right[:index]:
                if item['name']==other['name'] or frozenset((item['name'],other['name'])) in self.ignore:continue
                if self._close(item['object'],item['lower'],item['upper'],other,
                               self.relative_margin(item['name'],other['name'])):
                    return dict(kind='robot_self',link=item['name'],obstacle=other['name'])
        if self.link_to_object is not None:
            pose=frames[self.ee_link]@self.link_to_object
        elif self.released_pose is not None:
            pose=self.released_pose
        else:
            return dict(kind='missing_visual_object_binding')
        half=self.half+self.margins.object_uncertainty_m
        corners=np.array([[x,y,z] for x in (-half[0],half[0]) for y in (-half[1],half[1])
                          for z in (-half[2],half[2])])
        points=transform_points(corners,pose)
        obj=self._object(self.fcl.Box(*(2*half)),pose)
        lo,hi=points.min(0),points.max(0)
        if self.link_to_object is not None:
            hit=self._world_hit(obj,lo,hi,static)
            if hit:return dict(kind='carried_object_environment',obstacle=hit[0],voxel_index=hit[1])
        for item in right:
            # Contact with measured fingers is the intended manipulation. No
            # exemption for the wrist, camera, elbow or environmental clutter.
            if item['name'] in self.gripping_links:continue
            margin=(self.relative_margin(self.ee_link,item['name']) if self.link_to_object is not None
                    else self.margins.sweep_m)
            if self._close(obj,lo,hi,item,margin):
                return dict(kind='object_robot',obstacle=item['name'])
        return None

    def check_path(self,current,positions,values):
        self.refresh()
        extra={}
        if self.link_to_object is not None:
            half=self.half+self.margins.object_uncertainty_m
            extra[self.ee_link]=float(np.linalg.norm(self.link_to_object[:3,3])+np.linalg.norm(half))
        bounds=lever_bounds(self.scene.geometry,self.joint_names,values,extra)
        path=bounded_joint_path(current,positions,bounds,self.margins.sweep_m,self.margins.max_path_samples)
        result=dict(accepted=False,checked_samples=0,sweep_displacement_bound_m=self.margins.sweep_m,
            object_uncertainty_padding_m=self.margins.object_uncertainty_m,
            robot_geometry='known_URDF_full_convex_hulls_not_search_spheres',
            carried_object_checked=self.link_to_object is not None,
            released_object_checked=self.released_pose is not None,
            intentional_object_finger_contacts_only=sorted(self.gripping_links),
            environment='visual_occupied_voxels_visual_table_known_left_arm',
            truth_used=False,frozen_observed_world_only=True,unseen_space_certified_free=False,
            rigid_grasp_assumed=True,actual_tracking_error_certified=False)
        for index,q in enumerate(path):
            hit=self.check_state(q,values)
            result['checked_samples']+=1
            if hit:
                result.update(first_rejected_sample=index,collision=hit,start_rejected=index==0)
                return result
        result['accepted']=True
        return result
