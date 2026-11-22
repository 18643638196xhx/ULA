#!/usr/bin/env python3
"""Replay saved initial RGB-D, without moving a robot or claiming task success."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("focused_replay_runner",
    ROOT/"scripts/run_move_pillbottle_pad_focused_randomized.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,8")
    parser.add_argument("--precise", action="store_true")
    args = parser.parse_args()
    args.baseline, args.output = args.baseline.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(mode="saved_rgbd_perception_replay_not_task_success",
                  precise_cad=bool(args.precise), observations=[])
    worker = runner.FocusedWorker(ROOT, args.output/"vision_worker", model_id=1)
    worker.placement_policy = "visual"
    try:
        for seed in map(int, args.seeds.split(",")):
            sources = list(args.baseline.glob(f"episode_*_seed{seed}/perception/rgbd.npz"))
            if len(sources) != 1:
                raise ValueError(f"Expected one saved RGB-D for seed {seed}")
            directory = args.output/f"seed{seed}"/"perception"
            directory.mkdir(parents=True, exist_ok=True)
            with np.load(sources[0], allow_pickle=False) as frame:
                payload = dict(frame)
            if not args.precise:
                payload["coarse_only"] = np.asarray(True)
            request, result = directory/"rgbd.npz", directory/"estimate.npz"
            np.savez_compressed(request, **payload)
            record = dict(seed=seed, accepted=False, source=str(sources[0]))
            try:
                value = worker.estimate_files(request, result)
                record.update(accepted=True, audit=value.observation_audit,
                    bounds=value.cup_bounds_world.tolist(), pad=value.coaster_world_center.tolist(),
                    timings=value.timings_s)
                from PIL import Image, ImageDraw
                image = Image.fromarray(payload["rgb"][..., :3])
                draw = ImageDraw.Draw(image)
                draw.rectangle(value.observation_audit["cup_mask_bbox_pixels"], outline="lime", width=2)
                pad_box = value.observation_audit["pad_focus"]["box"]
                draw.rectangle(pad_box, outline="cyan", width=2)
                image.save(directory/"selected_targets.png")
            except RuntimeError as error:
                record["error"] = str(error)
            report["observations"].append(record)
            report.update(worker=worker.summary(),
                accepted=sum(item["accepted"] for item in report["observations"]))
            (args.output/"report.json").write_text(json.dumps(report, indent=2))
            print(f"seed={seed} accepted={record['accepted']} error={record.get('error')}", flush=True)
    finally:
        worker.close()
        report["worker"] = worker.summary()
        (args.output/"report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
