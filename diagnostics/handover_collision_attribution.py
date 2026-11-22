"""Independent attribution at collision witnesses; no planner or actions invoked."""
import json
from pathlib import Path
import numpy as np
import yaml

from adapters.handover.rgbd_obstacles import transform_points,dense_joint_path
from diagnostics.handover_obstacle_audit import PhysicalAudit
from diagnostics.handover_obstacle_snapshot import dump
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix


def analyze_witness(audit,metadata,data,q,link,partner):
    frames=audit.frames(q)
    config=metadata['robot_config']['robot_cfg']['kinematics']
    definitions=config['collision_spheres']
    if isinstance(definitions,str):definitions=yaml.safe_load(Path(definitions).read_text())['collision_spheres']
    buffer=config.get('collision_sphere_buffer',0.)
    if isinstance(buffer,dict):buffer=buffer.get(link,0.)
    spheres=definitions.get(link,[])
    centers=transform_points(np.array([s['center'] for s in spheres]).reshape(-1,3),frames[link])
    radii=np.array([s['radius']+buffer for s in spheres])
    partners=[other for other in audit.static if other['record']['name']==partner]
    model_hits=[];partner_hits=[]
    voxels=(data['map_keys']+.5)*float(data['map_pitch']);half=float(data['map_pitch'])/2
    table=np.linalg.inv(data['world_to_planner'])@pose_wxyz_to_matrix(data['table_pose'])
    table_half=data['table_dims']/2
    for index,(center,radius) in enumerate(zip(centers,radii)):
        sphere=audit.fcl.CollisionObject(audit.fcl.Sphere(float(radius)),audit.Pose(center))
        if any(audit.fcl.collide(sphere,other['object']).is_collision() for other in partners):partner_hits.append(index)
        distance=np.linalg.norm(np.maximum(np.abs(voxels-center)-half,0),axis=1)-radius
        local=(center-table[:3,3])@table[:3,:3]
        delta=np.abs(local)-table_half
        table_distance=np.linalg.norm(np.maximum(delta,0))+min(delta.max(),0)-radius
        if distance.min()<0 or table_distance<0:
            n=int(np.argmin(distance));model_hits.append(dict(sphere=index,voxel_distance_m=float(distance[n]),
                table_distance_m=float(table_distance),closest_voxel=voxels[n].tolist()))
    # Does the actual link hit occupied VISUAL voxels that its spheres miss?
    link_hits=[]
    cube_geometry=audit.fcl.Box(np.full(3,float(data['map_pitch'])))
    for value in audit.robot:
        if value['record']['name']!=link:continue
        pose=frames[link]
        value['object'].set_pose(audit.Pose(np.asarray(pose,dtype=float)))
        vertices=transform_points(value['vertices'],pose)
        selected=np.flatnonzero(np.all(voxels>=vertices.min(0)-half,axis=1)&
                                np.all(voxels<=vertices.max(0)+half,axis=1))
        for index in selected:
            cube=audit.fcl.CollisionObject(cube_geometry,audit.Pose(voxels[index]))
            if audit.fcl.collide(value['object'],cube).is_collision():
                link_hits.append(voxels[index].tolist())
                if len(link_hits)>=5:break
        if len(link_hits)>=5:break
    robot_undercoverage=not partner_hits
    visible_link_miss=bool(link_hits) and not model_hits
    return dict(link=link,physical_partner=partner,q=np.asarray(q).tolist(),
        physical_link_partner_intersection_verified=True,
        model_spheres_intersect_physical_partner=bool(partner_hits),
        model_sphere_partner_hit_indices=partner_hits,
        model_spheres_overlap_recorded_voxels_or_table=model_hits,
        physical_link_intersects_recorded_visual_voxels=bool(link_hits),example_voxels=link_hits,
        physical_link_exceeds_sphere_collision_envelope_at_partner=robot_undercoverage,
        visible_occupied_space_missed_by_link_sphere_model=visible_link_miss,
        scope='first collision witness, not an exhaustive error attribution')


def rejected_physical_free_goal(audit,metadata,data,q):
    frames=audit.frames(q)
    config=metadata['robot_config']['robot_cfg']['kinematics']
    definitions=config['collision_spheres']
    if isinstance(definitions,str):definitions=yaml.safe_load(Path(definitions).read_text())['collision_spheres']
    buffer=config.get('collision_sphere_buffer',0.)
    voxels=(data['map_keys']+.5)*float(data['map_pitch']);half=float(data['map_pitch'])/2
    overlaps=[]
    for link,values in definitions.items():
        margin=buffer.get(link,0.) if isinstance(buffer,dict) else buffer
        for value in values:
            center=transform_points(np.asarray(value['center'])[None],frames[link])[0]
            radius=value['radius']+margin
            distances=np.linalg.norm(np.maximum(np.abs(voxels-center)-half,0),axis=1)-radius
            n=int(np.argmin(distances))
            if distances[n]<0:
                overlaps.append(dict(link=link,sphere_center=center.tolist(),sphere_radius_m=float(radius),
                    voxel_center=voxels[n].tolist(),overlap_m=float(-distances[n])))
    overlaps.sort(key=lambda x:-x['overlap_m'])
    for value in overlaps[:4]:
        value['nearest_background_shapes']=audit.nearest(value['voxel_center'],background_only=True)
    return dict(q=np.asarray(q).tolist(),physical_intersections=audit.collisions(q),
                largest_voxel_overlaps=overlaps[:4],
                note='Physical nonintersection alone does not waive configured safety padding')


def attribute(audit_root,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(candidate_actions_executed=False,physical_truth_audit_only=True,samples=[])
    for audit_path in sorted(Path(audit_root).glob('scene_*/audit.json')):
        result=json.loads(audit_path.read_text())
        if result.get('status')!='finished':continue
        snapshot=Path(result['snapshot']);metadata=json.loads((snapshot/'metadata.json').read_text())
        data=dict(np.load(snapshot/'planning_inputs.npz'));physical=PhysicalAudit(metadata,data)
        entries=[]
        for path,summary in result.get('independent_path_geometry',{}).items():
            if summary['geometric_collision_free']:continue
            dense=dense_joint_path(data['joints_planner_rounded'],np.load(path)['positions'])
            # One witness per physical pair, rather than pretending endpoint
            # collisions explain all intermediate-path intersections.
            remaining=set(summary['collision_pairs'])
            for q in dense:
                if not remaining:break
                pairs=physical.collisions(q)
                for link,partner in pairs:
                    key=' / '.join((link,partner))
                    if key not in remaining:continue
                    value=analyze_witness(physical,metadata,data,q,link,partner)
                    value['path_file']=path;entries.append(value);remaining.remove(key)
        free=next((x for x in result.get('empty_world_solution_physical_collisions',[]) if not x['pairs']),None)
        rejection=None if free is None else rejected_physical_free_goal(physical,metadata,data,np.asarray(free['q']))
        report['samples'].append(dict(audit=str(audit_path),witnesses=entries,
                                     physically_free_but_model_rejected_goal=rejection))
        dump(output/'attribution.json',report)
    report['status']='finished'
    dump(output/'attribution.json',report)
    return report
