"""Known-instance model preparation from creation parameters, NEVER world poses.

Only this evaluation harness sees the scene creator. The visual worker receives
RGB-D and explicitly supplied static CAD. Articulation CAD is built at a static
declared initial joint configuration, not from live articulation/renderer meshes.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    return value


@contextmanager
def record_creations():
    """Record whitelisted model identity/geometry parameters; discard pose/scene."""
    module=importlib.import_module('envs.utils.create_actor')
    names=('create_actor','create_box','create_visual_box','create_sapien_urdf_obj')
    originals={name:getattr(module,name) for name in names};records={};restored=[]
    def wrapper(kind,original):
        signature=inspect.signature(original)
        @wraps(original)
        def call(*args,**kwargs):
            arguments=signature.bind(*args,**kwargs);arguments.apply_defaults()
            result=original(*args,**kwargs)
            if result is not None:
                allowed=('modelname','model_id','modelid','scale','half_size','color')
                records[id(result)]=dict(kind=kind,**{key:json_value(arguments.arguments[key])
                    for key in allowed if key in arguments.arguments})
            return result
        return call
    replacements={name:wrapper(name,value) for name,value in originals.items()}
    # rand_create_actor imports creator aliases into its own globals. Patch
    # those aliases in this private process as well; no source file is edited.
    for owner in list(sys.modules.values()):
        if owner is None or not getattr(owner,'__name__','').startswith('envs'):continue
        for name,original in originals.items():
            if getattr(owner,name,None) is original:
                restored.append((owner,name,original));setattr(owner,name,replacements[name])
    try:yield records
    finally:
        for owner,name,original in reversed(restored):setattr(owner,name,original)


def _color(mesh):
    if hasattr(mesh.visual,'to_color'):
        try:mesh.visual=mesh.visual.to_color()
        except IndexError:
            mesh.visual.uv=np.clip(np.asarray(mesh.visual.uv),0.,1.)
            mesh.visual=mesh.visual.to_color()
    return mesh


def _load(path):
    import trimesh
    value=trimesh.load(str(path),force='scene',process=False)
    return trimesh.util.concatenate([_color(p) for p in value.dump()])


def _origin(node):
    value=np.eye(4)
    if node is not None:
        value[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
        value[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
    return value


def urdf_directory(root,modelname,modelid):
    parent=Path(root)/'assets/objects'/modelname
    if modelid is None:return parent
    options=sorted((p for p in parent.iterdir() if p.is_dir() and p.name!='visual'),
        key=lambda p:int(re.search(r'\d+',p.name).group()) if re.search(r'\d+',p.name) else 0)
    if modelid<len(options):return options[modelid]
    found=next((p for p in options if p.name==str(modelid)),None)
    if found is None:raise FileNotFoundError('Unknown articulation asset variant')
    return found


def assemble_urdf(directory,*,initial_joint_fraction=None):
    """Static FK from URDF origins + metadata init_qpos; no simulator object."""
    import trimesh
    directory=Path(directory);source=directory/'mobility.urdf'
    metadata=directory/'model_data.json'
    data=json.loads(metadata.read_text()) if metadata.exists() else {}
    scale=np.asarray(data.get('scale',1.),float)
    root=ET.parse(source).getroot();transforms={};joint_values={};sources=[source]
    if metadata.exists():sources.append(metadata)
    children={j.find('child').get('link') for j in root.findall('joint')}
    for link in root.findall('link'):
        if link.get('name') not in children:transforms[link.get('name')]=np.eye(4)
    active=[j for j in root.findall('joint') if j.get('type')!='fixed']
    init=data.get('init_qpos',[])
    for index,joint in enumerate(active):
        q=float(init[index]) if index<len(init) else 0.
        if index==0 and initial_joint_fraction is not None:
            limit=joint.find('limit')
            if limit is None:raise ValueError('Initial joint fraction requires static URDF limits')
            low=float(limit.get('lower','0'));high=float(limit.get('upper','0'))
            q=low+(high-low)*initial_joint_fraction
        joint_values[joint.get('name')]=q
    pending=list(root.findall('joint'))
    while pending:
        ready=[j for j in pending if j.find('parent').get('link') in transforms]
        if not ready:raise ValueError('Disconnected URDF kinematic tree')
        for joint in ready:
            value=_origin(joint.find('origin'));motion=np.eye(4)
            axis_node=joint.find('axis');axis=np.fromstring(axis_node.get('xyz','1 0 0') if axis_node is not None else '1 0 0',sep=' ')
            q=joint_values.get(joint.get('name'),0.)
            if joint.get('type') in ('revolute','continuous'):
                motion[:3,:3]=Rotation.from_rotvec(axis/np.linalg.norm(axis)*q).as_matrix()
            elif joint.get('type')=='prismatic':motion[:3,3]=axis*q
            elif joint.get('type')!='fixed':raise ValueError('Unsupported URDF joint type')
            transforms[joint.find('child').get('link')]=transforms[joint.find('parent').get('link')]@value@motion
        pending=[j for j in pending if j not in ready]
    parts=[]
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            mesh_node=visual.find('geometry/mesh')
            if mesh_node is None:continue
            file=directory/mesh_node.get('filename');sources.append(file)
            mesh=_load(file)
            mesh.apply_scale(np.fromstring(mesh_node.get('scale','1 1 1'),sep=' '))
            mesh.apply_transform(transforms[link.get('name')]@_origin(visual.find('origin')))
            mesh.apply_scale(scale);parts.append(mesh)
    if not parts:raise ValueError('No visual meshes in URDF')
    return trimesh.util.concatenate(parts),dict(scale=json_value(scale),static_joint_values=joint_values,
        initial_joint_fraction=initial_joint_fraction,sources=[str(p) for p in dict.fromkeys(sources)],
        live_joint_state_read=False,world_pose_read=False,scope='initial_configuration_whole_object_only')


class AssetPreparer:
    def __init__(self,robotwin,output):
        self.robotwin=Path(robotwin);self.output=Path(output);self.cache={}
        self.output.mkdir(parents=True,exist_ok=True)

    def prepare(self,description,*,initial_joint_fraction=None):
        import trimesh
        signature=json.dumps(dict(description=description,initial_joint_fraction=initial_joint_fraction),sort_keys=True)
        if signature in self.cache:return self.cache[signature]
        identity=hashlib.sha256(signature.encode()).hexdigest()[:24]
        kind=description['kind'];modelname=description.get('modelname','box')
        audit=dict(creation_parameters=description,known_instance_identity_supplied=True,
            pose_or_mask_supplied=False,world_pose_read=False)
        if kind=='create_actor':
            directory=self.robotwin/'assets/objects'/modelname
            model_id=description['model_id'];suffix='' if model_id is None else str(model_id)
            metadata=directory/f'model_data{suffix}.json'
            mesh=next((p for folder in (directory/'visual',directory) for ext in ('glb','obj')
                if (p:=folder/f'base{suffix}.{ext}').is_file()),None)
            if mesh is None:raise FileNotFoundError('Known visual mesh missing: '+str(directory))
            scale=json.loads(metadata.read_text())['scale'] if metadata.exists() else description.get('scale',[1,1,1])
            audit.update(source_mesh=str(mesh),source_metadata=str(metadata),scale=scale)
            specification=dict(mesh_path=str(mesh),scale=np.broadcast_to(scale,(3,)).tolist())
        elif kind in ('create_box','create_visual_box'):
            half=np.asarray(description['half_size'],float)
            if half.shape!=(3,) or np.any(half<=0):raise ValueError('Invalid supplied primitive dimensions')
            mesh=trimesh.creation.box(extents=2*half)
            rgb=np.asarray(description.get('color') or [0.5,0.5,0.5],float)[:3]
            mesh.visual.vertex_colors=np.r_[np.rint(np.clip(rgb,0,1)*255).astype(np.uint8),255]
            file=self.output/(identity+'.glb');mesh.export(file)
            specification=dict(mesh_path=str(file));audit.update(primitive='box',known_full_extents_m=(2*half).tolist())
        elif kind=='create_sapien_urdf_obj':
            directory=urdf_directory(self.robotwin,modelname,description.get('modelid'))
            mesh,assembly=assemble_urdf(directory,initial_joint_fraction=initial_joint_fraction)
            file=self.output/(identity+'.glb');mesh.export(file)
            specification=dict(mesh_path=str(file));audit.update(assembly=assembly)
        else:raise ValueError('Unknown creation kind: '+kind)
        audit['cad_sha256']=hashlib.sha256(Path(specification['mesh_path']).read_bytes()).hexdigest()
        value=dict(cad=specification,audit=audit);self.cache[signature]=value
        return value
