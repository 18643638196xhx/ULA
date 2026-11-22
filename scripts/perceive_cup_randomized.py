#!/usr/bin/env python3
"""Allocator-limited entry to the unchanged RGB-D perception protocol."""
from pathlib import Path
import importlib.util
import os
import runpy


if __name__ == "__main__":
    import torch
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    source = Path(__file__).with_name("perceive_rgbd.py")
    if os.environ.get("ULA_CUP_TARGET_FOCUS") != "1":
        runpy.run_path(str(source), run_name="__main__")
    else:
        spec = importlib.util.spec_from_file_location("focused_cup_protocol", source)
        protocol = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol)
        def load(fp_iterations, debug_dir, backend="pointcloud"):
            if backend != "pointcloud":
                raise ValueError("Cup target focus requires pointcloud perception")
            from adapters.cup_target_focus import FocusedCupPerception
            value = FocusedCupPerception(source.parents[1]/"tra_pipe/RoboTwin")
            value.reset_episode()
            return value
        original_estimate = protocol.estimate_file
        def estimate(vision, input_path, output_path):
            path = Path(input_path)
            if path.parent.name == "perception" and path.name == "rgbd.npz":
                vision.reset_episode()
            return original_estimate(vision, input_path, output_path)
        protocol.load_perception, protocol.estimate_file = load, estimate
        raise SystemExit(protocol.main())
