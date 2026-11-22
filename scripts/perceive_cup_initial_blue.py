#!/usr/bin/env python3
"""Reuse strict RGB-D-only protocol with the caption-only blue-cup class."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
SPEC = importlib.util.spec_from_file_location("ula_blue_initial_protocol", ROOT/"scripts/perceive_cup_initial.py")
initial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(initial)


def load(*args, **kwargs):
    import torch
    from adapters.cup_initial_blue_prompt import BlueInitialCupPerception
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    return BlueInitialCupPerception(ROOT/"tra_pipe/RoboTwin")


if __name__ == "__main__":
    initial.protocol.load_perception = load
    initial.protocol.estimate_file = initial.estimate_file
    raise SystemExit(initial.protocol.main())
