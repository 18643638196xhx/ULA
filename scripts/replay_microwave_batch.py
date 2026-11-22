#!/usr/bin/env python3
"""CPU regression on saved RGB-D, not a new physical task evaluation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.microwave.geometry import (fit_panel, isolate_door_surface, panel_corners,
    update_panel_corners, hinge_motion, verify_step)
from adapters.microwave.perception import initial_depth_frame, reference_depth_frame


def camera(path):
    with np.load(path, allow_pickle=False) as f:
        return dict(rgb=f["rgb"], depth=f["depth"], intrinsic_cv=f["intrinsic"],
                    extrinsic_cv=f["world_to_camera"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.mkdir(parents=True)
    source = json.loads((args.source/"report.json").read_text())
    report = dict(source=str(args.source), simulator_started=False, robot_motion=False,
                  uses_cached_detection_boxes=True, initial=[], opening=[])
    for index, episode in enumerate(source["episodes"]):
        seed = episode["seed"]
        base = args.source/f"episode_{index:03d}_seed{seed}"/"perception"
        if episode["failure_stage"] == "traditional_approach":
            row = dict(seed=seed)
            report["initial"].append(row)
            try:
                directory = base/"initial_0_head_camera"
                with np.load(directory/"observation.npz", allow_pickle=False) as f:
                    box = json.loads(str(f["audit_json"]))["boxes"]["door"]
                result = initial_depth_frame(camera(directory/"rgbd.npz"), box)
                panel = fit_panel(result["door_points"], result["camera_world"]-np.median(result["door_points"], 0))
                cloud, _ = isolate_door_surface(result["door_points"], panel, result["handle_world"])
                fit_panel(cloud, panel["normal"])
                row.update(accepted=True, audit=result["handle_audit"])
            except FileNotFoundError as error:
                row.update(accepted=None, validated=False, error=str(error))
            except (RuntimeError, ValueError) as error:
                row.update(accepted=False, error=str(error))
            print(json.dumps(dict(kind="initial", **row)), flush=True)
        if episode["failure_stage"] != "traditional_opening":
            continue
        row = dict(seed=seed, frames=[])
        report["opening"].append(row)
        try:
            first = next(a for a in episode["observations"] if a.get("accepted"))
            with np.load(base/f"{first['label']}_{first['camera']}"/"observation.npz", allow_pickle=False) as f:
                points, handle, centre = f["door_points"], f["handle_world"], f["camera_world"]
            hint = centre-np.median(points, 0)
            panel = fit_panel(points, hint)
            points, _ = isolate_door_surface(points, panel, handle)
            panel = fit_panel(points, hint)
            reference = dict(door_points=points, panel=panel, handle_world=handle,
                             tracking_corners=panel_corners(panel))
            hinge = episode["hinge_estimate"]
            labels = ["closed_0"]+[f"open_{i:02d}_check0" for i in range(len(episode["traditional_opening"]["steps"]))]
            for number, label in enumerate(labels):
                delta = (0. if number == 0 else
                    episode["traditional_opening"]["steps"][number-1]["commanded_step_deg"])
                motion = np.eye(4) if number == 0 else hinge_motion(hinge["hinge_world"],
                    np.radians(delta)*hinge["opening_sign"])
                errors = []
                for view in ("head_camera", "left_camera"):
                    try:
                        result = reference_depth_frame(camera(base/f"{label}_{view}"/"rgbd.npz"), reference, motion)
                        result["panel"] = fit_panel(result["door_points"], hinge["initial_normal"])
                        result["tracking_corners"] = update_panel_corners(reference["tracking_corners"],
                            reference["panel"], result["panel"], reference["handle_world"], result["handle_world"])
                        break
                    except (RuntimeError, ValueError, FileNotFoundError) as error:
                        errors.append(f"{view}: {error}")
                else:
                    raise RuntimeError(f"{label}: {errors}")
                frame = dict(label=label, camera=view, width_m=result["panel"]["width_m"])
                if number:
                    frame["decision"] = verify_step(hinge, reference, result, motion, delta)
                row["frames"].append(frame)
                reference = result
            row["geometry_replayed_to_stop_frame"] = True
        except (RuntimeError, ValueError, FileNotFoundError) as error:
            row.update(geometry_replayed_to_stop_frame=False, error=str(error))
        print(json.dumps(dict(kind="opening", **row)), flush=True)
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
