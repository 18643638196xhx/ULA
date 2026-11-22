#!/usr/bin/env python3
"""Isolated persistent focused cup perception plus known-CAD registration."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--fp-iterations", type=int, default=5)
    parser.add_argument("--backend", default="foundationpose")
    args = parser.parse_args()
    import torch
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    from adapters.cup_known_model import KnownCupVision
    spec = importlib.util.spec_from_file_location("known_cup_protocol", ROOT/"scripts/perceive_rgbd.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    def estimate_file(vision, source, destination):
        path = Path(source)
        initial = path.parent.name == "perception" and path.name == "rgbd.npz"
        if initial:
            vision.reset_episode()
        require_cad = initial or path.parent.name == "placement_perception"
        with np.load(source, allow_pickle=False) as frame:
            value = vision.estimate(**dict(frame), require_cad=require_cad)
        payload = dict(cup_bounds_world=value.cup_bounds_world, cup_points_world=value.cup_points_world,
            coaster_world_center=value.coaster_world_center,
            timings_json=np.asarray(json.dumps(value.timings_s)),
            observation_audit_json=np.asarray(json.dumps(value.observation_audit)))
        if value.cup_world_pose is not None:
            payload["cup_world_pose"] = value.cup_world_pose
        np.savez(destination, **payload)
    protocol.estimate_file = estimate_file
    return protocol.serve(lambda: KnownCupVision(ROOT/"tra_pipe/RoboTwin", args.fp_iterations, args.debug_dir))


if __name__ == "__main__":
    raise SystemExit(main())
