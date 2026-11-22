#!/usr/bin/env python3
"""Offline RGB-D replay to diagnose targeting, not task-success evaluation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new replay output directory")
    args.output.mkdir(parents=True)
    import torch
    import numpy as np
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    from adapters.cup_target_focus import FocusedCupPerception
    vision = FocusedCupPerception(ROOT/"tra_pipe/RoboTwin")
    results = []
    for seed in args.seeds:
        source = args.source/f"episode_{seed:03d}_seed{seed}"/"perception/rgbd.npz"
        item = dict(seed=seed, accepted=False)
        vision.reset_episode()
        try:
            with np.load(source, allow_pickle=False) as frame:
                value = vision.estimate(**dict(frame))
            item.update(accepted=True, observation=value.observation_audit,
                cup_bounds_world=value.cup_bounds_world.tolist(), pad=value.coaster_world_center.tolist())
            np.savez_compressed(args.output/f"seed{seed}_estimate.npz", cup_points_world=value.cup_points_world,
                cup_bounds_world=value.cup_bounds_world, coaster_world_center=value.coaster_world_center)
        except (ValueError, RuntimeError) as error:
            item["error"] = str(error)
        results.append(item)
        print(json.dumps(item), flush=True)
    (args.output/"report.json").write_text(json.dumps(dict(offline_replay_only=True, results=results), indent=2)+"\n")


if __name__ == "__main__":
    main()
