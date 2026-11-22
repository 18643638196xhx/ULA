"""Sensor-only fixed workspace camera check; no inference or task actions."""
import argparse
from copy import deepcopy
import importlib
import os
from pathlib import Path
import random
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate_unified_vision50 as runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT):p.error('Output must stay under ULA')
    out.mkdir(parents=True,exist_ok=False)
    runner.rendering_runtime();runner.block_planner_import();os.chdir(runner.ROBOTWIN)
    cls=getattr(importlib.import_module('envs.put_bottles_dustbin'),'put_bottles_dustbin')
    task=cls();config=runner.scene_configuration('put_bottles_dustbin',True)
    config['left_embodiment_config']['static_camera_list'][-1].update(
        position=[0.,0.,2.2],forward=[0.,0.,-1.],left=[-1.,0.,0.])
    try:
        random.seed(0)
        with runner.no_planners():task.setup_demo(now_ep_num=0,seed=0,**deepcopy(config))
        for camera in ('head_camera','workspace_camera'):
            value=task.get_camera_observation(camera)
            Image.fromarray(value['rgb']).save(out/(camera+'.png'))
            np.savez_compressed(out/(camera+'.npz'),rgb=value['rgb'],depth=value['depth'],
                intrinsic=value['intrinsic_cv'],world_to_camera=value['extrinsic_cv'])
    finally:task.close_env()


if __name__=='__main__':main()
