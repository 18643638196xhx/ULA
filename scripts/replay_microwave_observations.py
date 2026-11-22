#!/usr/bin/env python3
"""Replay saved RGB-D with one bounded vision worker; no scene or robot motion."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.microwave.geometry import estimate_hinge, fit_panel
from adapters.microwave.perception import MicrowaveWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-gpu", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new replay output directory")
    args.output.mkdir(parents=True)
    with np.load(args.input, allow_pickle=False) as frame:
        camera = dict(rgb=frame["rgb"], depth=frame["depth"],
                      intrinsic_cv=frame["intrinsic"], extrinsic_cv=frame["world_to_camera"])
    report = dict(input=str(args.input), attempts=[], simulator_started=False, robot_motion=False)
    worker = MicrowaveWorker(ROOT, args.output/"worker", vision_gpu=args.vision_gpu)
    try:
        for index, (door, handle) in enumerate([
                ("microwave door", "microwave handle"), ("microwave door", "handle"),
                ("door", "door handle"), ("microwave oven door", "vertical handle")]):
            for refined in (False, True):
                audit = dict(door_prompt=door, handle_prompt=handle, refined=refined)
                report["attempts"].append(audit)
                try:
                    estimate = worker.estimate_frame(camera, args.output, f"pair{index}_{int(refined)}",
                        refined=refined, door_prompt_override=door, handle_prompt_override=handle)
                    hint = estimate["camera_world"]-np.median(estimate["door_points"], axis=0)
                    panel = fit_panel(estimate["door_points"], hint)
                    hinge = estimate_hinge(panel, estimate["handle_world"])
                    audit.update(accepted=True, vision=estimate["vision_audit"],
                                 handle_world=estimate["handle_world"].tolist(),
                                 hinge_world=hinge["hinge_world"].tolist())
                except (RuntimeError, ValueError) as error:
                    audit.update(accepted=False, error=str(error))
                print(json.dumps(audit), flush=True)
    finally:
        worker.close()
        report["worker"] = worker.summary()
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
