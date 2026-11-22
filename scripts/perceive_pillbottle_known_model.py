#!/usr/bin/env python3
"""Persistent, isolated FoundationPose worker for one explicitly known bottle."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT/"tra_pipe/RoboTwin"
sys.path[:0] = [str(ROOT), str(ROBOTWIN)]


def make_vision(args):
    import torch
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get(
        "PILLBOTTLE_VISION_TORCH_FRACTION", ".10")), 0)
    from adapters.pillbottle_known_vision import KnownBottleVision

    assets = ROBOTWIN/"assets/objects/080_pillbottle"
    vision = KnownBottleVision(
        dino_config=str(ROBOTWIN/"third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        dino_checkpoint=str(ROBOTWIN/"third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"),
        dino_text_encoder=str(ROBOTWIN/"third_party/GroundingDINO/weights/bert-base-uncased"),
        sam2_checkpoint=str(ROBOTWIN/"third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"),
        sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        cup_mesh=str(assets/f"visual/base{args.model_id}.glb"),
        cup_metadata=str(assets/f"model_data{args.model_id}.json"),
        foundationpose_root=str(ROBOTWIN/"third_party/FoundationPose"),
        groundingdino_root=str(ROBOTWIN/"third_party/GroundingDINO"),
        sam2_root=str(ROBOTWIN/"third_party/sam2"),
        fp_iterations=args.fp_iterations, debug_dir=str(args.debug_dir),
        cup_prompt=args.object_prompt, coaster_prompt=args.target_prompt)
    vision.last_accepted_world_pose = None
    vision.geometry_guard = getattr(args, "geometry_guard", False)
    vision.table_height_world = None
    vision.object_luminance_floor = 0.
    # Fixed random sampling is only CAD surface discretization, not a scene pose.
    state = np.random.get_state()
    try:
        np.random.seed(0)
        vision.model_samples = vision.mesh.sample(12000)
    finally:
        np.random.set_state(state)
    return vision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--fp-iterations", type=int, default=5)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--object-prompt", default="pill bottle")
    parser.add_argument("--target-prompt", default="blue square pad")
    parser.add_argument("--geometry-guard", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("ula_known_pill_protocol", ROOT/"scripts/perceive_rgbd.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    def estimate_file(vision, source, destination):
        # The original approach request starts a new random scene. Never carry
        # one episode's visual pose into another, even in a persistent worker.
        if Path(source).parent.name == "perception" and Path(source).name == "rgbd.npz":
            vision.last_accepted_world_pose = None
            vision.table_height_world = None
            vision.object_luminance_floor = 0.
        with np.load(source, allow_pickle=False) as frame:
            value = vision.estimate(**dict(frame))
        payload = dict(cup_bounds_world=value.cup_bounds_world, cup_points_world=value.cup_points_world,
            coaster_world_center=value.coaster_world_center,
            timings_json=np.asarray(json.dumps(value.timings_s)),
            observation_audit_json=np.asarray(json.dumps(value.observation_audit)))
        if value.cup_world_pose is not None:
            payload["cup_world_pose"] = value.cup_world_pose
        np.savez(destination, **payload)
    protocol.estimate_file = estimate_file
    return protocol.serve(lambda: make_vision(args))


if __name__ == "__main__":
    raise SystemExit(main())
