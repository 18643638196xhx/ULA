#!/usr/bin/env python3
"""Independent initial-only vision protocol, no action model or known CAD."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
SPEC = importlib.util.spec_from_file_location("ula_cup_initial_protocol", ROOT/"scripts/perceive_rgbd.py")
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def load(*args, **kwargs):
    import torch
    from adapters.cup_initial_perception import InitialCupPerception
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    return InitialCupPerception(ROOT/"tra_pipe/RoboTwin")


def estimate_file(vision, input_path, output_path):
    with np.load(input_path, allow_pickle=False) as frame:
        # Deliberately no simulator IDs, masks or poses in the input schema.
        if set(frame.files) != {"rgb", "depth", "intrinsic", "world_to_camera"}:
            raise ValueError("Initial vision accepts RGB-D and calibration only")
        result = vision.estimate(**dict(frame))
    np.savez_compressed(output_path, cup_bounds_world=result.cup_bounds_world,
        cup_points_world=result.cup_points_world, coaster_world_center=result.coaster_world_center,
        cup_mask=result.cup_mask, pad_mask=result.pad_mask,
        observation_audit_json=np.asarray(json.dumps(result.observation_audit)),
        timings_json=np.asarray(json.dumps(result.timings_s)))


if __name__ == "__main__":
    protocol.load_perception, protocol.estimate_file = load, estimate_file
    raise SystemExit(protocol.main())
