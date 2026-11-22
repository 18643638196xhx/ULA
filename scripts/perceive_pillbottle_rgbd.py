#!/usr/bin/env python3
"""Independent persistent DINO/SAM worker for bottle/pad RGB-D."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-prompt", default="pill bottle")
    parser.add_argument("--target-prompt", default="blue square pad")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("ula_pillbottle_worker_protocol", ROOT/"scripts/perceive_rgbd.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    def factory():
        import torch
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get(
            "PILLBOTTLE_VISION_TORCH_FRACTION", ".08")), 0)
        from adapters.pillbottle import PillbottleVision
        return PillbottleVision(ROOT/"tra_pipe/RoboTwin", args.object_prompt, args.target_prompt)
    return protocol.serve(factory)


if __name__ == "__main__":
    raise SystemExit(main())
