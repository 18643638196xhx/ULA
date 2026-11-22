#!/usr/bin/env python3
"""Convert verified short demonstrations to standalone local LeRobot v2.1."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.handover.pickup_data import CAMERAS, INSTRUCTION, MOTORS


def build(collection, output, repo_id):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    report = json.loads((collection/"report.json").read_text())
    if report["run_status"] != "finished" or not report.get("sources_unchanged"):
        raise ValueError("Collection must finish with unchanged source files")
    rows = report["successful_trajectories"]
    if len(rows) != report["requested_successful_episodes"]:
        raise ValueError("Incomplete collection")
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds) or not {r["split"] for r in rows} == {"train", "val"}:
        raise ValueError("Need unique seeds and separate episode-level train/val splits")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(rows[0]["path"]) as probe:
        _, height, width, channels = probe["images/cam_high"].shape
        if channels != 3:
            raise ValueError("Expected RGB images")
    features = {"observation.state": dict(dtype="float32", shape=(14,), names=[MOTORS]),
                "action": dict(dtype="float32", shape=(14,), names=[MOTORS])}
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = dict(dtype="image", shape=(3,height,width),
                                                       names=["channels","height","width"])
    dataset_root = output/"lerobot"/repo_id
    dataset = LeRobotDataset.create(repo_id=repo_id, root=dataset_root, fps=15,
                                   robot_type="aloha", features=features, use_videos=False)
    manifest, episodes = [], []
    global_index = 0
    try:
        for episode, row in enumerate(rows):
            with h5py.File(row["path"]) as source:
                state, actions = source["observation.state"][()], source["action"][()]
                if state.shape != actions.shape or state.shape[1:] != (14,) or not np.isfinite(actions).all():
                    raise ValueError("Invalid state/action demonstration")
                for frame in range(len(state)):
                    sample = {"observation.state": state[frame].astype(np.float32),
                              "action": actions[frame].astype(np.float32), "task": INSTRUCTION}
                    for camera in CAMERAS:
                        sample[f"observation.images.{camera}"] = source[f"images/{camera}"][frame]
                    dataset.add_frame(sample)
                    # Split metadata only: no synthesized WAM routes or teacher features.
                    manifest.append(dict(global_index=global_index, episode_index=episode,
                        frame_index=frame, split=row["split"], family="handover_left_pickup",
                        seed=row["seed"], sample_id=f"pickup_{episode}_{frame}"))
                    global_index += 1
            dataset.save_episode()
            episodes.append(dict(row, episode_index=episode, length=len(state)))
            print(f"[pickup-dataset] {episode+1}/{len(rows)} seed={row['seed']} frames={len(state)} split={row['split']}", flush=True)
    finally:
        if dataset.image_writer is not None:
            dataset.stop_image_writer()
    manifest_path = output/"split_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(r)+"\n" for r in manifest))
    (output/"metadata.json").write_text(json.dumps(dict(num_stages=4, num_route_tokens=768,
        purpose="episode_split_only_no_WAM_target_supervision"), indent=2))
    metadata = dict(schema="ula_pickup_lerobot_v1", collection=str(collection), repo_id=repo_id,
        dataset_root=str(dataset_root), lerobot_root=str(dataset_root.parent),
        manifest=str(manifest_path), episodes=episodes, frames=global_index,
        splits=dict(Counter(r["split"] for r in rows)), instruction=INSTRUCTION,
        nominal_fps=15, physical_timestamps="raw_HDF5_audit.physics_step_before_and_after_at_250Hz",
        state_semantics="commanded_drive_targets_matching_runtime_not_measured_joint_positions",
        action_semantics="end_of_interval_command_target_paired_with_start_of_interval_image",
        model_inputs="three_RGB_views_commanded_robot_state_language_only",
        ground_truth_in_model_inputs=False, teacher="traditional_RGBD_geometry_expert_success_filtered_by_simulation",
        WAM_teacher_targets_generated=False)
    (output/"dataset.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="ula_handover_pickup50_v1")
    args = parser.parse_args()
    build(args.collection.resolve(), args.output.resolve(), args.repo_id)
