#!/usr/bin/env python3
"""Convert one verified ULA local-task collection to local LeRobot v2.1."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.support_placement_data import CAMERAS, MOTORS


def build(collection, output, repo_id, published_output=None):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    report = json.loads((collection / "report.json").read_text())
    if report["run_status"] != "finished" or not report.get("sources_unchanged"):
        raise ValueError("Collection must be complete with unchanged sources")
    rows = report["successful_trajectories"]
    if len(rows) != report["requested_successful_episodes"]:
        raise ValueError("Incomplete collection")
    if len({row["seed"] for row in rows}) != len(rows) or {row["split"] for row in rows} != {"train", "val"}:
        raise ValueError("Need unique seeds and disjoint episode-level splits")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(rows[0]["path"]) as probe:
        _, height, width, channels = probe["images/cam_high"].shape
        if channels != 3:
            raise ValueError("Expected RGB camera data")
    features = {
        "observation.state": dict(dtype="float32", shape=(14,), names=[MOTORS]),
        "action": dict(dtype="float32", shape=(14,), names=[MOTORS]),
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = dict(
            dtype="image", shape=(3, height, width), names=["channels", "height", "width"])
    dataset_root = output / "lerobot" / repo_id
    dataset = LeRobotDataset.create(
        repo_id=repo_id, root=dataset_root, fps=15, robot_type="aloha",
        features=features, use_videos=False)
    manifest, episodes, global_index = [], [], 0
    try:
        for episode, row in enumerate(rows):
            with h5py.File(row["path"]) as source:
                state = source["observation.state"][()]
                action = source["action"][()]
                if state.shape != action.shape or state.shape[1:] != (14,):
                    raise ValueError("Invalid local demonstration shape")
                for frame in range(len(state)):
                    sample = {
                        "observation.state": state[frame].astype(np.float32),
                        "action": action[frame].astype(np.float32),
                        "task": report["attempts"][0]["instruction"],
                    }
                    for camera in CAMERAS:
                        sample[f"observation.images.{camera}"] = source[f"images/{camera}"][frame]
                    dataset.add_frame(sample)
                    manifest.append(dict(
                        global_index=global_index, episode_index=episode, frame_index=frame,
                        split=row["split"], task=report["task"], seed=row["seed"],
                        sample_id=f"{report['task']}_{episode}_{frame}"))
                    global_index += 1
            dataset.save_episode()
            episodes.append(dict(row, episode_index=episode, length=len(state)))
            print(f"[clean-dataset] task={report['task']} {episode+1}/{len(rows)} "
                  f"seed={row['seed']} frames={len(state)} split={row['split']}", flush=True)
    finally:
        if dataset.image_writer is not None:
            dataset.stop_image_writer()
    manifest_path = output / "split_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest))
    instruction = report["attempts"][0]["instruction"]
    published_output = Path(published_output).resolve() if published_output else output
    published_dataset_root = published_output / "lerobot" / repo_id
    metadata = dict(
        schema="ula_support_placement_lerobot_v1", collection=str(collection),
        task=report["task"], task_config=report["task_config"], repo_id=repo_id,
        dataset_root=str(published_dataset_root),
        lerobot_root=str(published_dataset_root.parent),
        manifest=str(published_output / "split_manifest.jsonl"),
        episodes=episodes, frames=global_index,
        splits=dict(Counter(row["split"] for row in rows)), instruction=instruction,
        nominal_fps=15,
        recording_boundary="after_visual_localization_and_traditional_pregrasp",
        state_semantics="commanded_drive_targets_matching_runtime",
        action_semantics="end_of_interval_command_paired_with_start_image",
        model_inputs="three_RGB_views_commanded_robot_state_language_only",
        teacher="official_task_planner_from_ULA_pregrasp",
        teacher_uses_simulator_truth=True, ground_truth_in_model_inputs=False,
        traditional_approach_in_action_labels=False,
    )
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--published-output", type=Path,
                        help="Final directory recorded in metadata when building atomically in staging")
    args = parser.parse_args()
    build(args.collection.resolve(), args.output.resolve(), args.repo_id,
          args.published_output)
