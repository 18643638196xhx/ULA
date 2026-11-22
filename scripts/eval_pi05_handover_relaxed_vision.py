#!/usr/bin/env python3
"""Opt-in ablation: shorter initial surfaces and partial visual bootstrap.

All motion/release checks remain unchanged. A provisional bootstrap is not a
full visual anchor and cannot by itself authorize release. Never used by the
default constraint-feedback evaluation entry.
"""
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
SPEC=importlib.util.spec_from_file_location('relaxed_vision_full_harness',ROOT/'scripts/eval_pi05_handover_full.py')
full=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)
from adapters.handover.adaptive_place import adaptive_place
from adapters.handover.temporal_vision import TemporalHeldRGBDObserver
from adapters.support_placement_goal import SupportPlacementGoal


def main():
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    counter=itertools.count()
    original_sources=full.shared.initial.sources
    def sources():
        files=[Path(__file__),ROOT/'scripts/eval_pi05_handover_relaxed_vision.sh',
            ROOT/'scripts/eval_pi05_handover_full.py',ROOT/'adapters/constraint_feedback.py',
            ROOT/'adapters/support_placement_goal.py']
        return dict(original_sources(),**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    full.shared.initial.sources=sources
    original_dump=full.shared.dump
    def dump(path,value):
        if Path(path).name=='report.json' and value.get('stage')=='through_right_take_and_traditional_platform_placement':
            value.update(placement_geometry='experimental_shorter_view_and_partial_prediction_bootstrap',
                initial_visible_span_m=.03,allow_partial_initialization=True,
                ambiguous_initial_axis_uses_transfer_prediction=True,
                full_visual_anchor_required_before_release=True,
                initial_rigid_prediction_can_authorize_release_alone=False,
                nominal_motion_and_release_tolerances_unchanged=True,
                sample_scope=('user_selected_previous_visual_failure_seeds_not_overall_success_rate'
                    if value.get('selected_scene_seeds') is not None else
                    'all_initialized_random_scenes_including_stage_failures'))
        original_dump(path,value)
    full.shared.dump=dump
    goal=SupportPlacementGoal((0.,0.,-.1),.008,(.003,.018),.01)
    def place(task,binding,platform,result,*,on_stage=None):
        directory=output/'adaptive_vision'/f'placement_{next(counter):03d}'
        result.update(vision_debug_dir=str(directory),initial_visible_span_m=.03,
            experimental_partial_initialization=True)
        return adaptive_place(task,binding,platform,result,feedback_goal=goal,
            observer=TemporalHeldRGBDObserver(task,directory,min_initial_visible_span_m=.03,
                allow_partial_initialization=True,resolve_initial_axis_ambiguity_with_prior=True),on_stage=on_stage)
    full.place_on_platform=place
    full.__file__=str(Path(__file__).resolve())
    full.main()


if __name__=='__main__':main()
