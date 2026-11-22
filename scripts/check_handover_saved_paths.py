#!/usr/bin/env python3
"""Guard old paths using ONLY captured RGB-D and robot proprioception.

Never opens physical_after_control_only.npz, constructs a simulator, or moves a
robot. An old robot/background-only witness may fail the stricter held-body or
self checks; report that instead of treating it as a recovered task.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.handover.rgbd_obstacles import RobotGeometry,transform_points
from adapters.handover.mesh_path_guard import VisualMeshPathGuard
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'adapters/handover/mesh_path_guard.py',
             ROOT/'adapters/handover/pose_search.py',ROOT/'adapters/handover/rgbd_obstacles.py']
    def hashes():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report=dict(status='running',sources_before=hashes(),paths=[],physical_truth_used=False,
                simulation_constructed=False,actions_executed=False)
    def save():
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    save()
    for path in sorted(args.audit_root.glob('scene_*/yaw_*.npz')):
        audit=json.loads((path.parent/'audit.json').read_text())
        snapshot=Path(audit['snapshot'])
        metadata=json.loads((snapshot/'metadata.json').read_text())
        data=dict(np.load(snapshot/'planning_inputs.npz'))
        geometry=RobotGeometry(metadata['robot_urdf'])
        table=np.linalg.inv(data['world_to_planner'])@pose_wxyz_to_matrix(data['table_pose'])
        dx,dy,dz=data['table_dims']
        corners=np.array([[x,y,z] for x in (-dx/2,dx/2) for y in (-dy/2,dy/2) for z in (-dz/2,dz/2)])
        world=transform_points(corners,table)
        scene=SimpleNamespace(geometry=geometry,root_pose=data['root_pose'],sequence=1,
            keys=data['map_keys'],pitch=float(data['map_pitch']),table_z=float(world[:,2].max()),
            table_bounds=np.array([world[:,:2].min(0),world[:,:2].max(0)]))
        guard=VisualMeshPathGuard(scene,metadata['arm_joint_names'],metadata['robot_config'])
        values=dict(zip(metadata['joint_names'],data['full_qpos']))
        guard.bind(data['visual_object_pose'],values)
        started=time.perf_counter()
        result=guard.check_path(data['joints_actual'],np.load(path)['positions'],values)
        row=dict(path=str(path),snapshot=str(snapshot),elapsed_s=time.perf_counter()-started,guard=result)
        report['paths'].append(row);save()
        print(json.dumps(row,ensure_ascii=False),flush=True)
    report.update(status='finished',sources_after=hashes(),sources_unchanged=report['sources_before']==hashes())
    save()


if __name__=='__main__':main()
