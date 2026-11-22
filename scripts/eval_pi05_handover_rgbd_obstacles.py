#!/usr/bin/env python3
"""Isolated RGB-D full-arm obstacle-planning regression; other entries unchanged."""
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
SPEC=importlib.util.spec_from_file_location('rgbd_obstacles_full_harness',ROOT/'scripts/eval_pi05_handover_full.py')
full=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)
from adapters.handover.adaptive_place import adaptive_place
from adapters.handover.temporal_vision import TemporalHeldRGBDObserver
from adapters.handover.rgbd_obstacles import RGBDObstacleMap, ObstaclePlanningSession, MeasuredObstacleObserver
from adapters.support_placement_goal import SupportPlacementGoal


def main():
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    counter=itertools.count()
    original_sources=full.shared.initial.sources
    def sources():
        files=[Path(__file__),ROOT/'scripts/eval_pi05_handover_rgbd_obstacles.sh',
               ROOT/'scripts/eval_pi05_handover_full.py',ROOT/'adapters/constraint_feedback.py',
               ROOT/'adapters/support_placement_goal.py']
        return dict(original_sources(),**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    full.shared.initial.sources=sources
    original_dump=full.shared.dump
    def dump(path,value):
        if Path(path).name=='report.json' and value.get('stage')=='through_right_take_and_traditional_platform_placement':
            value.update(placement_geometry='relaxed_visual_tracking_with_RGBD_full_arm_obstacle_planning',
                rgbd_obstacles_enabled=True,obstacle_stage='right_traditional_placement_only',
                arrival_and_release_gates_unchanged=True,object_truth_used_for_obstacles=False,
                robot_identity_masks_used=False,unknown_space_certified_free=False,
                manipulation_target_excluded_until_controlled_withdraw_completes=True,
                sample_scope=('selected_collision_regression_seeds_not_overall_success_rate'
                    if value.get('selected_scene_seeds') is not None else 'all_initialized_random_scenes'))
        original_dump(path,value)
    full.shared.dump=dump
    original_approach=full.shared.approach
    def approach(task,visual,result):
        directory=output/'rgbd_obstacles'/f'scene_{next(counter):03d}'
        target=np.asarray(visual.cup_world_pose,dtype=float)
        scene=RGBDObstacleMap(task.robot,target,directory)
        scene.update(task)
        task._ula_rgbd_scene=scene
        result['rgbd_obstacle_initial_map']=scene.audit[-1]
        return original_approach(task,visual,result)
    full.shared.approach=approach
    goal=SupportPlacementGoal((0.,0.,-.1),.008,(.003,.018),.01)
    def place(task,binding,platform,result,*,on_stage=None):
        scene=task._ula_rgbd_scene
        result.update(vision_debug_dir=str(scene.directory/'held_vision'),rgbd_obstacles={})
        base=TemporalHeldRGBDObserver(task,scene.directory/'held_vision',min_initial_visible_span_m=.03,
            allow_partial_initialization=True,resolve_initial_axis_ambiguity_with_prior=True)
        # Bootstrap the GPU world from the initial static map. Do NOT insert
        # current points with an unmeasured transfer-prediction target mask.
        # The first accepted visual fit refreshes the world before planning.
        with ObstaclePlanningSession(task,scene,result['rgbd_obstacles']) as planning:
            def stage(name):
                # Opening the gripper is an intentional contact transition.
                # Do not reclassify the manipulated target as static clutter
                # while the hand is still withdrawing alongside it. This
                # adapter does NOT claim target-contact collision certification.
                if on_stage is not None:
                    on_stage(name)
            return adaptive_place(task,binding,platform,result,feedback_goal=goal,
                observer=MeasuredObstacleObserver(base,planning),on_stage=stage)
    full.place_on_platform=place
    full.__file__=str(Path(__file__).resolve())
    full.main()


if __name__=='__main__':main()
