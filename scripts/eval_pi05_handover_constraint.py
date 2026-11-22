#!/usr/bin/env python3
"""Full-task regression with shared task-independent constraint feedback."""
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
SPEC=importlib.util.spec_from_file_location('constraint_handover_full_harness',ROOT/'scripts/eval_pi05_handover_full.py')
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
        files=[Path(__file__),ROOT/'scripts/eval_pi05_handover_constraint.sh',
            ROOT/'scripts/eval_pi05_handover_full.py',ROOT/'adapters/constraint_feedback.py',
            ROOT/'adapters/support_placement_goal.py']
        return dict(original_sources(),**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    full.shared.initial.sources=sources
    original_dump=full.shared.dump
    def dump(path,value):
        if Path(path).name=='report.json' and value.get('stage')=='through_right_take_and_traditional_platform_placement':
            value.update(placement_geometry='post_handover_visual_history_and_generic_constraint_feedback',
                orientation_selection='minimum_required_tilt_yaw_free',
                final_release_requires_bounded_temporal_confidence_and_current_surfaces=True,
                constraint_feedback='task_independent_ball_interval_robust_goal_and_progress_budget',
                nominal_release_tolerances_unchanged=True,
                universal_task_migration_validated=False)
        original_dump(path,value)
    full.shared.dump=dump
    # Task-owned specification. No task name/seed appears in the shared engine.
    goal=SupportPlacementGoal(support_anchor_local=(0.,0.,-.1),horizontal_tolerance_m=.008,
        gap_interval_m=(.003,.018),preferred_gap_m=.01)
    def place(task,binding,platform,result,*,on_stage=None):
        directory=output/'adaptive_vision'/f'placement_{next(counter):03d}'
        result['vision_debug_dir']=str(directory)
        return adaptive_place(task,binding,platform,result,feedback_goal=goal,
            observer=TemporalHeldRGBDObserver(task,directory),on_stage=on_stage)
    full.place_on_platform=place
    full.__file__=str(Path(__file__).resolve())
    full.main()


if __name__=='__main__':main()
