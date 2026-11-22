#!/usr/bin/env python3
"""One-shot or persistent RGB-D estimation in the Python 3.10 vision environment."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def load_perception(fp_iterations, debug_dir, backend="foundationpose"):
    root = Path(__file__).resolve().parents[1] / "tra_pipe/RoboTwin"
    if backend == "pointcloud":
        sys.path.insert(0, str(root.parents[1]))
        from adapters.rgbd_perception import GroundedPointCloudPerception

        return GroundedPointCloudPerception(root)
    from traditional_place_empty_cup.perception import GroundedFoundationPerception

    return GroundedFoundationPerception(
        dino_config=str(root / "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        dino_checkpoint=str(root / "third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"),
        dino_text_encoder=str(root / "third_party/GroundingDINO/weights/bert-base-uncased"),
        sam2_checkpoint=str(root / "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"),
        sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        cup_mesh=str(root / "assets/objects/021_cup/visual/base0.glb"),
        cup_metadata=str(root / "assets/objects/021_cup/model_data0.json"),
        foundationpose_root=str(root / "third_party/FoundationPose"),
        groundingdino_root=str(root / "third_party/GroundingDINO"),
        sam2_root=str(root / "third_party/sam2"),
        fp_iterations=fp_iterations, debug_dir=str(debug_dir),
    )


def estimate_file(perception, input_path, output_path):
    # Always process this frame afresh; the selected backend determines outputs.
    with np.load(input_path, allow_pickle=False) as frame:
        estimate = perception.estimate(**dict(frame))
    timings = dict(estimate.timings_s)
    payload = dict(coaster_world_center=estimate.coaster_world_center,
                   timings_json=np.asarray(json.dumps(timings)))
    if getattr(estimate, "coaster_points_world", None) is not None:
        payload["coaster_points_world"] = estimate.coaster_points_world
    if hasattr(estimate, "cup_world_pose"):
        payload["cup_world_pose"] = estimate.cup_world_pose
    else:
        payload.update(cup_bounds_world=estimate.cup_bounds_world,
                       observation_audit_json=np.asarray(json.dumps(estimate.observation_audit)))
        if getattr(estimate, "cup_points_world", None) is not None:
            payload["cup_points_world"] = estimate.cup_points_world
    np.savez(output_path, **payload)


def serve(factory):
    # Reserve a protocol fd BEFORE loading native libraries. Redirect Python
    # and native stdout to stderr so CUDA/FP logging cannot corrupt JSON replies.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        def send(message):
            protocol.write(json.dumps(message) + "\n")
            protocol.flush()

        try:
            perception = factory()
            send(dict(status="ready", pid=os.getpid(),
                      model_load_s=perception.load_timings_s["model_stack_load_total"]))
        except Exception as error:
            traceback.print_exc()
            send(dict(status="error", error=f"{type(error).__name__}: {error}"))
            return 1
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                if request["op"] == "shutdown":
                    return 0
                if request["op"] != "estimate":
                    raise ValueError("Unknown worker operation")
                estimate_file(perception, request["input"], request["output"])
                send(dict(status="ok", id=request["id"]))
            except Exception as error:
                traceback.print_exc()
                send(dict(status="error", id=request.get("id"),
                          error=f"{type(error).__name__}: {error}"))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--fp-iterations", type=int, default=5)
    parser.add_argument("--backend", choices=("foundationpose", "pointcloud"), default="foundationpose")
    args = parser.parse_args()
    if args.serve:
        if args.debug_dir is None:
            parser.error("--serve requires --debug-dir")
        return serve(lambda: load_perception(args.fp_iterations, args.debug_dir, args.backend))
    if args.input is None or args.output is None:
        parser.error("one-shot mode requires --input and --output")
    perception = load_perception(args.fp_iterations, args.output.parent / "debug", args.backend)
    estimate_file(perception, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
