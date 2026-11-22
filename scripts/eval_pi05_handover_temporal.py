#!/usr/bin/env python3
"""Isolated full-task comparison using history-conditioned RGB-D placement."""
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
SPEC=importlib.util.spec_from_file_location('temporal_handover_full_harness',ROOT/'scripts/eval_pi05_handover_full.py')
full=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)
from adapters.handover.adaptive_place import adaptive_place
from adapters.handover.temporal_vision import TemporalHeldRGBDObserver


def main():
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    counter=itertools.count()
    original_sources=full.shared.initial.sources
    def sources():
        files=[Path(__file__),ROOT/'scripts/eval_pi05_handover_temporal.sh',ROOT/'scripts/eval_pi05_handover_full.py']
        return dict(original_sources(),**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    full.shared.initial.sources=sources
    original_dump=full.shared.dump
    def dump(path,value):
        if Path(path).name=='report.json' and value.get('stage')=='through_right_take_and_traditional_platform_placement':
            value.update(placement_geometry='post_handover_visual_anchor_robot_motion_and_current_surfaces',
                orientation_selection='minimum_required_tilt_yaw_free',
                initial_rigid_binding_used_as_search_prior_only=True,
                final_release_requires_position_observable=False,
                final_release_requires_bounded_temporal_confidence_and_current_surfaces=True,
                historical_confidence_is_engineering_assumption_not_physical_grasp_proof=True)
        original_dump(path,value)
    full.shared.dump=dump
    def place(task,binding,platform,result,*,on_stage=None):
        directory=output/'adaptive_vision'/f'placement_{next(counter):03d}'
        result['vision_debug_dir']=str(directory)
        return adaptive_place(task,binding,platform,result,
            observer=TemporalHeldRGBDObserver(task,directory),on_stage=on_stage)
    full.place_on_platform=place
    full.__file__=str(Path(__file__).resolve())
    full.main()


if __name__=='__main__':main()
