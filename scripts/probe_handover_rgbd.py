#!/usr/bin/env python3
"""Diagnostic only: save real camera channels at handover and transport end.

Stops before descent/release. Does not use actor poses to compute any target.
This is not a success-rate experiment and may run while new modules are edited.
"""
import importlib.util
import itertools
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tra_pipe/RoboTwin")]
SPEC=importlib.util.spec_from_file_location("held_rgbd_probe_full",ROOT/"scripts/eval_pi05_handover_full.py")
full=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)
original=full.place_on_platform
output=Path(sys.argv[sys.argv.index("--output")+1]).resolve()
counter=itertools.count()


def place(task,binding,platform,result,*,on_stage=None):
    directory=output/"rgbd_probe"/f"placement_{next(counter):03d}"
    directory.mkdir(parents=True,exist_ok=True)
    def save(label):
        for name in ("head_camera","left_camera","right_camera"):
            camera=task.get_camera_observation(name)
            np.savez_compressed(directory/f"{label}_{name}.npz",
                rgb=camera["rgb"],depth=camera["depth"],intrinsic=camera["intrinsic_cv"],
                world_to_camera=camera["extrinsic_cv"])
        from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
        np.savez_compressed(directory/f"{label}_prior.npz",
            predicted=pose_wxyz_to_matrix(task.get_arm_pose("right"))@binding,
            hand=pose_wxyz_to_matrix(task.get_arm_pose("right")))
    save("handover")
    def callback(name):
        save(name)
        if on_stage is not None:
            on_stage(name)
        if name=="move_above_pad":
            raise RuntimeError("RGB-D diagnostic complete; stop before lowering or release")
    original(task,binding,platform,result,on_stage=callback)


full.place_on_platform=place
# The reused main spawns its own __file__ for --detach; keep the probe wrapper
# as that child entry, otherwise the wrapper would silently be bypassed.
full.__file__=str(Path(__file__).resolve())
full.main()
