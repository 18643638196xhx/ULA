#!/usr/bin/env python3
"""Persistent sensor-only worker for the independent CAD-ranked experiment."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tra_pipe/RoboTwin")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--fp-iterations", type=int, default=5)
    parser.add_argument("--backend", default="foundationpose")
    args = parser.parse_args()
    if not args.serve or args.fp_iterations < 1:
        parser.error("This worker requires --serve and positive --fp-iterations")

    import torch
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    from perception_cad_search.pillbottle_cad_search import PillbottleCadSearchVision
    from adapters.unified_vision.contracts import serializable

    protocol_spec = importlib.util.spec_from_file_location(
        "cad_search_protocol", ROOT / "scripts/perceive_rgbd.py")
    protocol = importlib.util.module_from_spec(protocol_spec)
    protocol_spec.loader.exec_module(protocol)

    def estimate_file(vision, source, destination):
        with np.load(source, allow_pickle=False) as raw:
            payload = dict(raw)
        if "model_id" not in payload:
            raise ValueError("Every CAD-search request must carry the current model_id")
        model_id = int(np.asarray(payload.pop("model_id")).item())
        object_asset = (str(np.asarray(payload.pop("object_asset")).item())
                        if "object_asset" in payload else vision.object_asset)
        cup_only = bool(np.asarray(payload.pop("cup_only", False)).item())
        support_only = bool(np.asarray(payload.pop("support_only", False)).item())
        refine_cup = bool(np.asarray(payload.pop("refine_cup", False)).item())
        value = vision.estimate(
            **payload, model_id=model_id, object_asset=object_asset,
            cup_only=cup_only, support_only=support_only, refine_cup=refine_cup,
        )
        output = dict(
            cup_world_pose=value.cup_world_pose,
            cup_bounds_world=value.cup_bounds_world,
            cup_points_world=value.cup_points_world,
            coaster_world_center=value.coaster_world_center,
            coaster_points_world=value.coaster_points_world,
            observation_audit_json=np.asarray(json.dumps(serializable(value.observation_audit))),
            timings_json=np.asarray(json.dumps(serializable(value.timings_s))),
        )
        np.savez(destination, **output)

    protocol.estimate_file = estimate_file
    return protocol.serve(lambda: PillbottleCadSearchVision(
        ROOT,
        args.debug_dir.parent,
        args.fp_iterations,
        object_asset=os.environ.get("ULA_CAD_SEARCH_OBJECT_ASSET", "080_pillbottle"),
        object_prompt=os.environ.get("ULA_CAD_SEARCH_OBJECT_PROMPT", "orange pill bottle"),
        support_prompt=os.environ.get("ULA_CAD_SEARCH_SUPPORT_PROMPT", "blue square pad"),
    ))


if __name__ == "__main__":
    raise SystemExit(main())
