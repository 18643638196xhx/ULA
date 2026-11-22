#!/usr/bin/env python3
"""Cup registration protocol/backend with explicit bottle asset and appearance."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--backend", default="foundationpose")
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--fp-iterations", type=int, default=5)
    args = parser.parse_args()
    from adapters.pillbottle_focus.cup_method import CupMethodBottleVision
    original = load_module("bottle_cup_method_loader", ROOT/"scripts/perceive_pillbottle_known_model.py")
    protocol = load_module("bottle_cup_method_protocol", ROOT/"scripts/perceive_rgbd.py")
    def factory():
        args.model_id, args.object_prompt, args.target_prompt = 1, "pill bottle", "blue square pad"
        args.geometry_guard = False
        vision = original.make_vision(args)
        # All classes are Python adapters over the same loaded model stack.
        # Change dispatch only in this worker, avoiding a second model load.
        vision.__class__ = CupMethodBottleVision
        vision.reset_episode()
        return vision
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
    return protocol.serve(factory)


if __name__ == "__main__":
    raise SystemExit(main())
