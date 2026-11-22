#!/usr/bin/env python3
"""Isolated free-yaw placement + URDF/held-object guard, not a new VLA model."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
SPEC=importlib.util.spec_from_file_location('safe_yaw_obstacle_entry',
    ROOT/'scripts/eval_pi05_handover_rgbd_obstacles.py')
entry=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)
from adapters.handover.adaptive_place import adaptive_place
from adapters.handover.pose_search import SupportYawExecutor
from adapters.handover.safe_pose_planning import SafePosePlanningSession


def main():
    entry.ObstaclePlanningSession=SafePosePlanningSession
    executor=SupportYawExecutor()
    def place(*args,observer,on_stage=None,**kwargs):
        def stage(name):
            if name=='release':observer.planning.release()
            if on_stage is not None:on_stage(name)
        return adaptive_place(*args,observer=observer,on_stage=stage,
                              segment_executor=executor,**kwargs)
    entry.adaptive_place=place
    original_sources=entry.full.shared.initial.sources
    def sources():
        files=[Path(__file__),ROOT/'scripts/eval_pi05_handover_safe_yaw.sh',
            ROOT/'adapters/handover/pose_search.py',ROOT/'adapters/handover/mesh_path_guard.py',
            ROOT/'adapters/handover/safe_pose_planning.py']
        return dict(original_sources(),**{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    entry.full.shared.initial.sources=sources
    original_dump=entry.full.shared.dump
    def dump(path,value):
        if Path(path).name=='report.json':
            value.update(placement_geometry='task_free_yaw_with_full_URDF_and_carried_primitive_guard',
                task_free_yaw_search=True,independent_mesh_path_guard=True,
                carried_object_path_guard=True,uses_object_truth_for_guard=False,
                new_candidate_execution_requires_mesh_guard=True,
                original_VLA_checkpoints_unchanged=True)
        return original_dump(path,value)
    entry.full.shared.dump=dump
    original_main=entry.full.main
    def full_main():
        # The reused entry sets its own file name; detached children must run
        # THIS entry, otherwise they silently lose both search and guard.
        entry.full.__file__=str(Path(__file__).resolve())
        return original_main()
    entry.full.main=full_main
    entry.main()


if __name__=='__main__':main()
