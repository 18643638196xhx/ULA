#!/usr/bin/env python3
"""Observe the unchanged cup-method bottle pipeline, retaining partial failures.

Instrumentation is confined to this serial worker. No simulator labels/poses
are accepted; target matching and pose errors are evaluated in the parent only.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]


def load(name, path):
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
    import adapters.pillbottle_focus.frontend as frontend
    from adapters.pillbottle_focus.cup_method import CupMethodBottleVision, BottleGeometryFrontend
    from adapters.pillbottle_known_vision import surface_mask
    original = load("initial_bottle_model_loader", ROOT/"scripts/perceive_pillbottle_known_model.py")
    protocol = load("initial_bottle_protocol", ROOT/"scripts/perceive_rgbd.py")
    trace = {}
    choose, measure = frontend.choose_unambiguous, BottleGeometryFrontend.estimate
    def observed_choose(*a, **kw):
        value = choose(*a, **kw)
        if "mask" in value:
            trace["pad"] = value
        return value
    def observed_measure(self, *a, **kw):
        value = measure(self, *a, **kw)
        trace["bottle"] = value
        return value
    frontend.choose_unambiguous = observed_choose
    BottleGeometryFrontend.estimate = observed_measure
    def factory():
        args.model_id, args.object_prompt, args.target_prompt = 1, "pill bottle", "blue square pad"
        args.geometry_guard = False
        vision = original.make_vision(args)
        vision.__class__ = CupMethodBottleVision
        vision.reset_episode()
        return vision
    def estimate_file(vision, source, destination):
        trace.clear()
        vision.reset_episode()
        with np.load(source, allow_pickle=False) as frame:
            if set(frame.files) != {"rgb", "depth", "intrinsic", "world_to_camera"}:
                raise ValueError("Initial audit accepts only RGB-D and calibration")
            inputs = dict(frame)
        started, error, value = time.perf_counter(), None, None
        try:
            value = vision.estimate(**inputs, require_cad=True)
        except (RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        coarse, pad = trace.get("bottle"), trace.get("pad")
        shape = inputs["rgb"].shape[:2]
        bottle_mask, pad_mask = np.zeros(shape, bool), np.zeros(shape, bool)
        points, bounds = np.empty((0, 3)), np.full((2, 3), np.nan)
        center = np.full(3, np.nan)
        audit = dict(bottle_accepted=coarse is not None, pad_accepted=pad is not None,
            bottle_attempted=pad is not None, cad_attempted=coarse is not None,
            cad_accepted=bool(value is not None and value.cup_world_pose is not None),
            perception_error=error, diagnostic_only_instrumentation=True,
            inference_uses_simulator_truth=False, source="unchanged_cup_method_initial_pipeline")
        if pad is not None:
            center, pad_mask = pad["center"].copy(), pad["mask"].copy()
        if coarse is not None:
            points, bounds = coarse.cup_points_world, coarse.cup_bounds_world
            bottle_mask = surface_mask(points, inputs["intrinsic"], inputs["world_to_camera"], shape)
            audit["frontend_audit"] = coarse.observation_audit
        audit["failure_stage"] = (None if audit["cad_accepted"] else "pad_recognition" if pad is None
            else "bottle_recognition" if coarse is None else "cad_registration")
        timings = dict(perception_total=time.perf_counter()-started)
        payload = dict(cup_bounds_world=bounds, cup_points_world=points, coaster_world_center=center,
            cup_mask=bottle_mask, pad_mask=pad_mask,
            observation_audit_json=np.asarray(json.dumps(audit)), timings_json=np.asarray(json.dumps(timings)))
        if audit["cad_accepted"]:
            payload["cup_world_pose"] = value.cup_world_pose
        np.savez_compressed(destination, **payload)
    protocol.estimate_file = estimate_file
    try:
        return protocol.serve(factory)
    finally:
        frontend.choose_unambiguous, BottleGeometryFrontend.estimate = choose, measure


if __name__ == "__main__":
    raise SystemExit(main())
