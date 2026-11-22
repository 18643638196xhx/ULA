#!/usr/bin/env python3
"""Independent vision worker; reuse protocol, replace only initial perception."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.microwave import perception
from adapters.microwave.panel_tracking import ApplianceFirstVision

SPEC = importlib.util.spec_from_file_location(
    "ula_microwave_panel_protocol", ROOT/"scripts/perceive_microwave_rgbd.py")
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)

if __name__ == "__main__":
    perception.MicrowaveVision = ApplianceFirstVision
    protocol.main()
