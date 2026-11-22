#!/usr/bin/env python3
"""Replay one official RoboTwin demonstration and capture keyframe RGB-D.

The script reads only the recorded robot actions, scene seed and calibrated
camera channels.  It never reads actor poses, renderer labels, contacts or the
task success flag.  Captures are written to a fresh directory together with
RGB replay and robot-state errors so downstream geometry extraction can reject
an inaccurate replay instead of silently treating it as the demonstration.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import random
import sys

import h5py
import numpy as np
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "tra_pipe/RoboTwin"
DEFAULT_DEMOS = Path("/data2/xuhuixin/RoboTwin/data_official50/demo_clean")
CAMERA_TO_HDF5 = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
    "front_camera": "cam_third_view",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def visit_stages(stages):
    for stage in stages:
        yield stage
        yield from visit_stages(stage.get("children", []))


def required_frames(rule_path: Path, explicit=()) -> list[int]:
    """Collect the minimum visual frames needed by primitive geometry."""
    rule = yaml.safe_load(rule_path.read_text())
    frame_count = int(rule["frame_count"])
    frames = {0, frame_count - 1, *(int(value) for value in explicit)}
    keyframes_path = rule_path.parent / rule.get("keyframes_file", "keyframes.json")
    keyframes = json.loads(keyframes_path.read_text()) if keyframes_path.is_file() else {}
    stages = list(visit_stages(rule.get("stages", [])))
    first_grasp = {}
    for stage in stages:
        if stage.get("primitive") == "grasp" and stage.get("arm"):
            first_grasp[stage["arm"]] = min(first_grasp.get(stage["arm"], frame_count),
                                             int(stage["demo_frames"][0]))
    for stage in stages:
        primitive = stage.get("primitive")
        first, last = (int(value) for value in stage.get("demo_frames", (0, 0)))
        if primitive == "move":
            if (stage.get("gripper_state") == "closed_candidate"
                    or last <= first_grasp.get(stage.get("arm"), frame_count)):
                frames.add(last)
        elif primitive in {"grasp", "release", "place", "handover", "open"}:
            frames.update((first, last))
        elif primitive in {"push", "pull", "rotate", "press", "strike"}:
            frames.update(int(keyframes[key]["frame"]) for key in stage.get("keyframe_ids", [])
                          if key in keyframes)
            frames.update((first, last))
        event = stage.get("recognition", {}).get("event", {})
        frames.update(int(event[key]) for key in ("start_frame", "end_frame") if key in event)
    if not frames or min(frames) < 0 or max(frames) >= frame_count:
        raise ValueError("Requested frame lies outside the demonstration")
    return sorted(frames)


def official_rgb(dataset, frame: int) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(bytes(dataset[frame]))).convert("RGB"))


def task_configuration(task_name: str) -> dict:
    specification = importlib.util.spec_from_file_location(
        "ula_demo_keyframe_config", ROBOTWIN / "scripts/run_place_empty_cup_vision.py")
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    config = module.load_task_args(
        str(ROBOTWIN / "env_cfg/task_config/demo_clean.yml"), 0, arm="auto")
    config.update(task_name=task_name, eval_mode=True, eval_video_log=False,
                  save_data=False, need_plan=False, enable_mplib=False,
                  head_camera_only=False, fast_eval_mode=True, render_freq=0)
    # ``load_task_args`` belongs to the old head-camera-only cup runner and
    # explicitly disables wrist cameras.  Keyframe geometry needs the same
    # head/left-wrist/right-wrist views recorded by the official demo, so undo
    # that runner-specific override before the environment creates cameras.
    config["camera"]["collect_head_camera"] = True
    config["camera"]["collect_wrist_camera"] = True
    config["data_type"].update(mesh_segmentation=False, actor_segmentation=False,
                               pointcloud=False)
    return config


def replay_target(task, demo, frame: int, frequency: float, physics_steps: int) -> None:
    """Interpolate recorded drive targets over one sampled interval.

    Official HDF5 actions are sampled every ``physics_steps`` simulation
    steps.  Jumping immediately to the next sampled target preserves logged
    drive targets but changes the contact trajectory enough to lose a grasp.
    Linear interpolation reconstructs the missing dense controller targets
    without reading any simulator object state.
    """
    arm_values = {}
    for arm in ("left", "right"):
        state = np.asarray(demo[f"state/{arm}_arm_joint_states"][frame], float)
        target = np.asarray(demo[f"action/{arm}_arm_joint_states"][frame], float)
        gripper_state = float(
            np.asarray(demo[f"state/{arm}_ee_joint_states"][frame]).reshape(-1)[0])
        gripper_target = float(
            np.asarray(demo[f"action/{arm}_ee_joint_states"][frame]).reshape(-1)[0])
        arm_values[arm] = state, target, (target - state) * frequency, gripper_state, gripper_target
    for step in range(physics_steps):
        alpha = float(step + 1) / float(physics_steps)
        for arm, (state, target, velocity, gripper_state, gripper_target) in arm_values.items():
            task.robot.set_arm_joints(state + alpha * (target - state), velocity, arm)
            task.robot.set_gripper(
                gripper_state + alpha * (gripper_target - gripper_state),
                arm,
                gripper_eps=0.,
            )
        task.scene.step()


def save_capture(task, demo, frame: int, cameras: list[str], directory: Path) -> dict:
    destination = directory / f"frame_{frame:04d}"
    destination.mkdir()
    audit = {"frame": frame, "cameras": {}}
    for camera_name in cameras:
        value = task.get_camera_observation(camera_name)
        path = destination / f"{camera_name}.npz"
        np.savez_compressed(path, rgb=value["rgb"][..., :3], depth=value["depth"],
                            intrinsic=value["intrinsic_cv"],
                            world_to_camera=value["extrinsic_cv"],
                            cam2world_gl=value["cam2world_gl"])
        camera_audit = {"path": str(path.resolve()), "sha256": digest(path)}
        hdf5_name = CAMERA_TO_HDF5.get(camera_name)
        if hdf5_name and f"vision/{hdf5_name}/colors" in demo:
            expected = official_rgb(demo[f"vision/{hdf5_name}/colors"], frame)
            actual = np.asarray(value["rgb"])[..., :3]
            if expected.shape != actual.shape:
                camera_audit["rgb_shape_mismatch"] = [list(expected.shape), list(actual.shape)]
            else:
                difference = np.abs(expected.astype(float) - actual.astype(float))
                camera_audit.update(rgb_mae=float(difference.mean()),
                                    rgb_median_absolute_error=float(np.median(difference)),
                                    exact_rgb=bool(np.array_equal(expected, actual)))
        calibration_path = f"vision/{hdf5_name}/extrinsics_matrix" if hdf5_name else ""
        if calibration_path in demo:
            # RoboTwin's official HDF5 calls this ``extrinsics_matrix`` but
            # stores SAPIEN's OpenGL camera-to-world model matrix.
            expected_pose = np.asarray(demo[calibration_path][frame], float)
            actual_pose = np.asarray(value["cam2world_gl"], float)
            delta = np.linalg.inv(expected_pose) @ actual_pose
            cosine = np.clip((np.trace(delta[:3, :3]) - 1.) / 2., -1., 1.)
            camera_audit.update(
                camera_pose_translation_error_m=float(np.linalg.norm(delta[:3, 3])),
                camera_pose_rotation_error_deg=float(np.degrees(np.arccos(cosine))),
            )
        audit["cameras"][camera_name] = camera_audit
    robot_error = {}
    for arm in ("left", "right"):
        actual = np.asarray(getattr(task.robot, f"get_{arm}_arm_jointState")(), float)[:-1]
        expected = np.asarray(demo[f"state/{arm}_arm_joint_states"][frame], float)
        robot_error[arm] = {
            "joint_l2_rad": float(np.linalg.norm(actual - expected)),
            "joint_max_abs_rad": float(np.max(np.abs(actual - expected))),
        }
    audit["robot_state_error"] = robot_error
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", action="append", choices=tuple(CAMERA_TO_HDF5),
                        help="Repeat for synchronized views; defaults to head+two wrists")
    parser.add_argument("--frame", type=int, action="append", default=[])
    parser.add_argument("--physics-steps-per-frame", type=int, default=15)
    args = parser.parse_args()
    if args.episode < 0 or args.physics_steps_per_frame < 1:
        parser.error("Invalid episode or physics step count")
    args.rule = args.rule.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    if not args.rule.is_file():
        parser.error("Rule file does not exist")
    cameras = args.camera or ["head_camera", "left_camera", "right_camera"]
    frames = required_frames(args.rule, args.frame)
    collection = args.demos / args.task / "aloha_agilex"
    demo_path = collection / "data" / f"episode_{args.episode:07d}.hdf5"
    seeds = [int(value) for value in (collection / "seed.txt").read_text().split()]
    if args.episode >= len(seeds) or not demo_path.is_file():
        parser.error("Official demonstration episode is unavailable")

    args.output.mkdir(parents=True)
    report = {
        "status": "running", "task": args.task, "episode": args.episode,
        "seed": seeds[args.episode], "source_demo": str(demo_path.resolve()),
        "source_demo_sha256": digest(demo_path), "requested_frames": frames,
        "cameras": cameras, "physics_steps_per_frame": args.physics_steps_per_frame,
        "uses_actor_pose": False, "uses_renderer_labels": False,
        "uses_contact_truth": False, "task_success_evaluated": False,
        "captures": [],
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    sys.path[:0] = [str(ROOT), str(ROBOTWIN)]
    harness_spec = importlib.util.spec_from_file_location(
        "ula_demo_keyframe_harness", ROOT / "scripts/evaluate_unified_vision50.py")
    harness = importlib.util.module_from_spec(harness_spec)
    assert harness_spec.loader is not None
    harness_spec.loader.exec_module(harness)
    harness.rendering_runtime()
    harness.block_planner_import()
    previous = Path.cwd()
    task = None
    try:
        os.chdir(ROBOTWIN)
        module = importlib.import_module("envs." + args.task)
        task = getattr(module, args.task)()
        random.seed(seeds[args.episode])
        with h5py.File(demo_path, "r") as demo, harness.no_planners():
            task.setup_demo(now_ep_num=args.episode, seed=seeds[args.episode],
                            **deepcopy(task_configuration(args.task)))
            frequency = float(demo["additional_info/frequency"][()])
            requested = set(frames)
            last = max(frames)
            for frame in range(last + 1):
                if frame in requested:
                    capture = save_capture(task, demo, frame, cameras, args.output)
                    report["captures"].append(capture)
                    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                    head = capture["cameras"].get("head_camera", {})
                    print(f"[demo-rgbd] frame={frame}/{last} head_rgb_mae={head.get('rgb_mae')}", flush=True)
                if frame < last:
                    replay_target(task, demo, frame, frequency, args.physics_steps_per_frame)
        report["status"] = "finished"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if task is not None:
            try:
                task.close_env()
            except Exception as error:
                report["close_error"] = f"{type(error).__name__}: {error}"
        os.chdir(previous)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
