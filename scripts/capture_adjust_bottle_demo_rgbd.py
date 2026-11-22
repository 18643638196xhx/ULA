"""Recreate one official adjust_bottle demo seed and capture only RGB-D.

The saved sensor is for offline demo calibration. No live actor pose, contact,
segmentation label, or success flag is read by this script.
"""

import argparse
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "tra_pipe/RoboTwin"
sys.path[:0] = [str(ROOT), str(ROBOTWIN)]
spec = importlib.util.spec_from_file_location("ula_adjust_bottle_vision_harness",
                                              ROOT/"scripts/evaluate_unified_vision50.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
block_planner_import = harness.block_planner_import
no_planners = harness.no_planners
rendering_runtime = harness.rendering_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    collection = Path("/data2/xuhuixin/RoboTwin/data_official50/demo_clean/adjust_bottle/aloha_agilex")
    seeds = [int(value) for value in (collection / "seed.txt").read_text().split()]
    seed = seeds[args.episode]
    demo_path = collection / "data" / f"episode_{args.episode:07d}.hdf5"
    with h5py.File(demo_path) as demo:
        source = np.asarray(Image.open(io.BytesIO(bytes(demo["vision/cam_head/colors"][0]))).convert("RGB"))

    rendering_runtime()
    block_planner_import()
    os.chdir(ROBOTWIN)
    from scripts.run_place_empty_cup_vision import load_task_args
    from envs.adjust_bottle import adjust_bottle
    config = load_task_args(str(ROBOTWIN / "env_cfg/task_config/demo_clean.yml"), 0, arm="auto")
    config.update(task_name="adjust_bottle", eval_mode=True, eval_video_log=False,
                  save_data=False, need_plan=False, enable_mplib=False,
                  head_camera_only=True, fast_eval_mode=True)
    config["data_type"].update(mesh_segmentation=False, actor_segmentation=False, pointcloud=False)
    task = adjust_bottle()
    try:
        random.seed(seed)
        with no_planners():
            task.setup_demo(now_ep_num=args.episode, seed=seed, **deepcopy(config))
            camera = task.get_camera_observation("head_camera")
            np.savez_compressed(args.output/"rgbd.npz", rgb=camera["rgb"], depth=camera["depth"],
                                intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"])
            replay = np.asarray(camera["rgb"])[..., :3]
            Image.fromarray(replay).save(args.output/"replay_initial_head.png")
            Image.fromarray(source).save(args.output/"official_initial_head.png")
            if source.shape != replay.shape:
                raise ValueError("Head camera resolution differs from official demo")
            diff = np.abs(source.astype(float) - replay.astype(float))
            result = dict(episode=args.episode, seed=seed, model_identity_from_scene_info=
                json.loads((collection/"scene_info.json").read_text())[f"episode_{args.episode}"]["info"]["{A}"],
                mean_absolute_rgb_difference=float(diff.mean()),
                median_absolute_rgb_difference=float(np.median(diff)),
                max_absolute_rgb_difference=float(diff.max()),
                exact_rgb_replay=bool(np.array_equal(source, replay)),
                scope="sensor_capture_only_no_actor_pose_or_labels_or_robot_actions")
            (args.output/"report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
            print(json.dumps(result, ensure_ascii=False))
    finally:
        task.close_env()


if __name__ == "__main__":
    main()
