#!/usr/bin/env python3
"""Read-only replay of two recorded failure frames with the new SAM prompts.

Writes independent diagnostics; does not run/reset a simulation or read CAD.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.perception import ExternalPerception, PerceptionWorker
from adapters.reobservation import observation_quality
from adapters.pointcloud_registration import RegistrationConfig, estimate_pointcloud_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"results/early_handoff_no_trial_slip_50eps_20261024_212703")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(source=str(args.source.resolve()), cases=[])
    with PerceptionWorker(ROOT, args.output/"vision_worker", backend="pointcloud") as worker:
        for seed, stage in [(3, "placement_perception"), (42, "perception")]:
            episode = args.source/f"episode_{seed:03d}_seed{seed}"
            with np.load(episode/stage/"rgbd.npz", allow_pickle=False) as values:
                frame = {k: values[k].copy() for k in values.files}
            with np.load(episode/"perception/estimate.npz", allow_pickle=False) as values:
                reference_points = values["cup_points_world"].copy()
                reference_bounds = values["cup_bounds_world"].copy()
            facade = ExternalPerception(ROOT, args.output/f"seed{seed}", worker=worker)
            camera = dict(rgb=frame["rgb"], depth=frame["depth"], intrinsic_cv=frame["intrinsic"],
                          extrinsic_cv=frame["world_to_camera"])
            case = dict(seed=seed, stage=stage, variants=[])
            report["cases"].append(case)
            for refined in (False, True):
                item = dict(refined=refined)
                case["variants"].append(item)
                try:
                    estimate = facade.supplementary(camera, "refined" if refined else "raw",
                                                     cup_only=False, refine_cup=refined)
                    item["quality"] = observation_quality(estimate, reference_bounds if seed == 3 else None)
                    if seed == 3:
                        _, item["registration"] = estimate_pointcloud_motion(reference_points,
                            estimate.cup_points_world, RegistrationConfig(max_rmse_m=.012, max_tilt_deg=35.))
                except (RuntimeError, ValueError) as error:
                    item["error"] = f"{type(error).__name__}: {error}"
                print(json.dumps(dict(seed=seed, **item), ensure_ascii=False), flush=True)
        report["worker"] = worker.summary()
    report["worker"] = worker.summary()
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
