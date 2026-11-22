#!/usr/bin/env python3
"""Posthoc one-frame-per-task sensor diagnosis, NEVER changes or re-scores a run.

No detector/CAD inference, task actions or object-pose access. Scene identity
labels are used solely to diagnose raw versus alpha-scaled geometric depth.
Only exact RGB AND saved-depth reproductions support paired causal claims.
"""
import argparse
from copy import deepcopy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.unified_vision import RGBDFrame
from adapters.unified_vision.geometry import world_grid,voxelize
from diagnostics.vision50_catalog import targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--shards',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT):parser.error('Output must stay under ULA')
    output.mkdir(parents=True,exist_ok=True)
    destination=output/f'shard_{args.shard:02d}.json'
    if destination.exists():parser.error('Refuse to overwrite a completed sensor diagnostic')
    report=json.loads(args.analysis.read_text())
    spec=importlib.util.spec_from_file_location('ula_sensor_scene',ROOT/'scripts/evaluate_unified_vision50.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.rendering_runtime();runner.block_planner_import();os.chdir(runner.ROBOTWIN)
    from adapters.cup_initial_audit import no_task_actions
    names=list(report['tasks'])[args.shard::args.shards];rows=[]
    for name in names:
        representative=report['tasks'][name]['representative'];directory=Path(representative['directory'])
        row=dict(task=name,seed=representative['seed'],source_directory=str(directory),
            stage='recreate',no_task_actions=True,no_detector_or_CAD_inference=True,targets={})
        task=None
        try:
            module=importlib.import_module('envs.'+name);task=getattr(module,name)()
            config=runner.scene_configuration(name);random.seed(row['seed'])
            index=int(directory.name.split('_')[1])
            with runner.no_planners():task.setup_demo(now_ep_num=index,seed=row['seed'],**deepcopy(config))
            with no_task_actions(task):
                observation=task.get_camera_observation('head_camera')
                camera=task.cameras.static_camera_list[task.cameras.static_camera_name.index('head_camera')]
                rgba=camera.get_picture('Color');alpha=rgba[...,3]
                raw_depth=-camera.get_picture('Position')[...,2]
                labels=camera.get_picture('Segmentation')[...,1]
                with np.load(directory/'rgbd.npz',allow_pickle=False) as archived:
                    row.update(rgb_identical=bool(np.array_equal(observation['rgb'],archived['rgb'])),
                        saved_depth_max_abs_difference_mm=float(np.max(np.abs(observation['depth']-archived['depth']))))
                row['exact_paired_reproduction']=row['rgb_identical'] and row['saved_depth_max_abs_difference_mm']==0.
                frame_raw=RGBDFrame('head','initial',observation['rgb'],raw_depth,observation['intrinsic_cv'],observation['extrinsic_cv'],'m')
                frame_saved=RGBDFrame('head','initial',observation['rgb'],observation['depth'],observation['intrinsic_cv'],observation['extrinsic_cv'],'mm')
                raw_grid,raw_valid=world_grid(frame_raw);saved_grid,saved_valid=world_grid(frame_saved)
                for key,actor in targets(task,name):
                    mask=np.isin(labels,runner.entity_ids(actor));a=alpha[mask]
                    value=dict(visible_pixels=int(mask.sum()),alpha_scaled_depth_input=True)
                    if len(a):
                        changed=mask&raw_valid&(np.abs(frame_raw.depth_m-frame_saved.depth_m)>.001)
                        value.update(alpha_min=float(a.min()),alpha_quantiles=np.quantile(a,[0,.1,.5,.9,1]).tolist(),
                            alpha_below_099_pixels=int(np.sum(a<.99)),depth_changed_more_than_1mm_pixels=int(changed.sum()),
                            median_abs_depth_change_m=float(np.median(np.abs(frame_raw.depth_m[mask]-frame_saved.depth_m[mask]))),
                            max_abs_depth_change_m=float(np.max(np.abs(frame_raw.depth_m[mask]-frame_saved.depth_m[mask]))))
                        for label,grid,valid in (('raw_geometry',raw_grid,raw_valid),('saved_geometry',saved_grid,saved_valid)):
                            points=voxelize(grid[mask&valid],.003)
                            if len(points)>=3:
                                bounds=np.quantile(points,[.01,.99],axis=0)
                                value[label]=dict(points=len(points),bounds_world=bounds.tolist(),
                                    visible_extents_m=(bounds[1]-bounds[0]).tolist())
                    row['targets'][key]=value
                row['stage']='done'
        except Exception as error:
            row.update(stage='error',error=f'{type(error).__name__}: {error}');traceback.print_exc()
        finally:
            if task is not None:
                try:task.close_env()
                except Exception as error:row['close_error']=str(error)
        rows.append(row);destination.write_text(json.dumps(dict(scope=__doc__,rows=rows),indent=2,allow_nan=False)+'\n')
        print(f'[sensor] {name} seed={row["seed"]} stage={row["stage"]} exact={row.get("exact_paired_reproduction")}',flush=True)


if __name__=='__main__':main()
