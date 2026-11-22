#!/usr/bin/env python3
"""CPU-only replay of saved closed-gripper frames, not task success scoring."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.microwave.panel_tracking import attach_reference, track_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Never overwrite a prior replay")
    args.output.mkdir(parents=True)
    batch = json.loads((args.input/"report.json").read_text())
    results = []
    for index, episode in enumerate(batch["episodes"]):
        if episode.get("failure_stage") != "visual_preopening_baseline":
            continue
        directory = args.input/f"episode_{index:03d}_seed{episode['seed']}"
        accepted = next(a for a in episode["observations"] if a.get("accepted"))
        path = directory/"perception"/f"{accepted['label']}_{accepted['camera']}"/"observation.npz"
        with np.load(path, allow_pickle=False) as data:
            reference = dict(door_points=data["door_points"], handle_world=data["handle_world"],
                             panel=accepted["panel"], vision_audit={})
        reference = attach_reference(reference)
        reference["tracking_corners"] = reference["tracking_root"]["corners"]
        attempts = []
        for camera_name in ("head_camera", "left_camera"):
            path = directory/"perception"/f"closed_0_{camera_name}"/"rgbd.npz"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as data:
                camera = dict(rgb=data["rgb"], depth=data["depth"], intrinsic_cv=data["intrinsic"],
                              extrinsic_cv=data["world_to_camera"])
            attempt = dict(camera=camera_name)
            attempts.append(attempt)
            try:
                value = track_panel(camera, reference)
                attempt.update(accepted=True, audit=value["vision_audit"])
                break
            except (RuntimeError, ValueError) as error:
                attempt.update(accepted=False, error=str(error))
        results.append(dict(seed=episode["seed"], accepted=any(a.get("accepted") for a in attempts),
                            attempts=attempts))
    report = dict(simulator_started=False, robot_motion=False,
                  not_a_task_success_rate=True, episodes=results,
                  accepted=sum(e["accepted"] for e in results), total=len(results))
    (args.output/"report.json").write_text(json.dumps(report, indent=2, default=lambda a:
        a.item() if isinstance(a, np.generic) else a.tolist())+"\n")
    print(json.dumps(report, default=lambda a: a.item() if isinstance(a, np.generic) else a.tolist()))


if __name__ == "__main__":
    main()
