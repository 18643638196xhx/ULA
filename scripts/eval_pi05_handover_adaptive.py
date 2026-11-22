#!/usr/bin/env python3
"""Isolated full-task harness with task-constrained visual placement feedback."""
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tra_pipe/RoboTwin")]
SPEC=importlib.util.spec_from_file_location("adaptive_handover_full_harness",ROOT/"scripts/eval_pi05_handover_full.py")
full=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)
from adapters.handover.adaptive_place import adaptive_place
from adapters.handover.held_vision import HeldRGBDObserver


def main():
    # The base parser still validates all paths, checkpoints and disjoint seeds.
    output=Path(sys.argv[sys.argv.index("--output")+1]).resolve()
    counter=itertools.count()
    original_sources=full.shared.initial.sources
    def sources():
        files=[Path(__file__),ROOT/"scripts/eval_pi05_handover_adaptive.sh",ROOT/"scripts/eval_pi05_handover_full.py"]
        return dict(original_sources(),**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    full.shared.initial.sources=sources
    original_dump=full.shared.dump
    def dump(path,value):
        if Path(path).name=="report.json" and value.get("stage")=="through_right_take_and_traditional_platform_placement":
            value.update(placement_geometry="fresh_RGBD_task_constrained_visual_feedback_known_primitive",
                orientation_selection="minimum_required_tilt_yaw_free",rotation_or_preserve_decided_by_measurement=True,
                initial_rigid_binding_used_as_search_prior_only=True,final_release_requires_position_observable=True)
        original_dump(path,value)
    full.shared.dump=dump
    def place(task,binding,platform,result,*,on_stage=None):
        directory=output/"adaptive_vision"/f"placement_{next(counter):03d}"
        result["vision_debug_dir"]=str(directory)
        return adaptive_place(task,binding,platform,result,
            observer=HeldRGBDObserver(task,directory),on_stage=on_stage)
    full.place_on_platform=place
    full.__file__=str(Path(__file__).resolve())
    full.main()


if __name__=="__main__":
    main()
