#!/usr/bin/env python3
"""Serial RGB-D-only handover vision worker. Simulator inputs are forbidden."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--backend", default="pointcloud")
    parser.add_argument("--fp-iterations", type=int, default=5)
    args = parser.parse_args()
    import torch
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    from adapters.handover.initial_vision import HandoverInitialVision
    spec = importlib.util.spec_from_file_location("handover_initial_protocol", ROOT/"scripts/perceive_rgbd.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    def estimate_file(vision, source, destination):
        with np.load(source, allow_pickle=False) as frame:
            if set(frame.files) != {"rgb", "depth", "intrinsic", "world_to_camera"}:
                raise ValueError("Handover initial vision accepts only RGB-D and calibration")
            value = vision.estimate(**dict(frame))
        payload = dict(cup_bounds_world=value.cup_bounds_world, cup_points_world=value.cup_points_world,
            coaster_world_center=value.coaster_world_center, block_mask=value.block_mask,
            platform_mask=value.platform_mask,
            observation_audit_json=np.asarray(json.dumps(value.observation_audit)),
            timings_json=np.asarray(json.dumps(value.timings_s)))
        if value.cup_world_pose is not None:
            payload["cup_world_pose"] = value.cup_world_pose
        np.savez_compressed(destination, **payload)
    protocol.estimate_file = estimate_file
    return protocol.serve(lambda: HandoverInitialVision(ROOT/"tra_pipe/RoboTwin"))


if __name__ == "__main__":
    raise SystemExit(main())
