#!/usr/bin/env python3
"""Isolated persistent known-CAD perception worker (asset/prompt from env)."""
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
    import torch
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    from adapters.known_cad_model import KnownCadVision
    spec = importlib.util.spec_from_file_location("known_protocol", ROOT / "scripts/perceive_rgbd.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)

    cache = dict(vision=None, key=None)

    def get_vision(model_id, object_asset):
        key = (object_asset, model_id)
        if cache["vision"] is None or cache["key"] != key:
            cache["vision"] = KnownCadVision(
                ROOT / "tra_pipe/RoboTwin", args.fp_iterations, args.debug_dir,
                object_asset=object_asset,
                object_prompt=os.environ["ULA_KNOWN_OBJECT_PROMPT"],
                support_prompt=os.environ["ULA_KNOWN_SUPPORT_PROMPT"],
                model_id=model_id)
            cache["key"] = key
        return cache["vision"]

    def estimate_file(_factory_vision, source, destination):
        path = Path(source)
        with np.load(source, allow_pickle=False) as frame:
            payload = dict(frame)
        model_id = int(payload.pop("model_id", 0)) if "model_id" in payload else 0
        # Task-selected support colour, forwarded per request (see run_ula_task).
        if "support_color" in payload:
            try:
                from adapters.unified_perception import PERCEPTION_SPEC
                PERCEPTION_SPEC["support_color"] = str(payload.pop("support_color"))
            except Exception:
                payload.pop("support_color", None)
        object_asset = (str(payload.pop("object_asset")) if "object_asset" in payload
                        else os.environ["ULA_KNOWN_OBJECT_ASSET"])
        vision = get_vision(model_id, object_asset)
        initial = path.parent.name == "perception" and path.name == "rgbd.npz"
        if initial:
            vision.reset_episode()
        require_cad = initial or path.parent.name == "placement_perception"
        value = vision.estimate(**payload, require_cad=require_cad)
        payload = dict(cup_bounds_world=value.cup_bounds_world,
                       cup_points_world=value.cup_points_world,
                       coaster_world_center=value.coaster_world_center,
                       timings_json=np.asarray(json.dumps(value.timings_s)),
                       observation_audit_json=np.asarray(json.dumps(value.observation_audit)))
        if value.cup_world_pose is not None:
            payload["cup_world_pose"] = value.cup_world_pose
        np.savez(destination, **payload)

    protocol.estimate_file = estimate_file

    class _LazyVision:
        """Create the real vision on first use.

        The served instance used to be built eagerly with model_id=0, which fails
        at startup for assets whose models start at 1 (e.g. 080_pillbottle has no
        base0.glb).  estimate_file() already selects the per-request instance, so
        the served one only needs to exist for attribute access.
        """

        __slots__ = ()

        def __getattr__(self, name):
            # Delegate to the instance that estimate_file() selected for the
            # current request. Reading it from the cache (instead of an env
            # variable) keeps the per-episode model id intact: the id lives in
            # each request payload, not in the worker's environment.
            if name == "load_timings_s":
                # Read once at server startup, before any request has created the
                # real instance; report zeros until the first model load.
                return {"model_stack_load_total": 0.0}
            vision = cache["vision"]
            if vision is None:
                raise AttributeError(f"vision not initialised yet while accessing {name!r}")
            return getattr(vision, name)

    return protocol.serve(lambda: _LazyVision())


if __name__ == "__main__":
    raise SystemExit(main())
