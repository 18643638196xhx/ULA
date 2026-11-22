#!/usr/bin/env python3
"""Capture one labelled demo_clean RGB-D template for each RoboTwin task.

The official policy HDF5 stores RGB but no depth, so it cannot directly yield
a metric point cloud.  This isolated data-preparation script regenerates one
initial scene with the same ``demo_clean`` task configuration, saves measured
head-camera RGB-D, and uses renderer identity masks *offline* to isolate each
declared task target.  Those masks never enter randomized-scene inference.

In a real-robot version, the renderer mask corresponds to a one-time manual
annotation of the clean demonstration frame.  This script executes no task
action, loads no CAD, and writes only to a fresh output directory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "tra_pipe/RoboTwin"
sys.path[:0] = [str(ROOT), str(ROBOTWIN), str(ROBOTWIN / "envs/curobo/src")]

from adapters.unified_vision import RGBDFrame  # noqa: E402
from adapters.unified_vision.geometry import voxelize, world_grid  # noqa: E402
from diagnostics.vision50_catalog import TASK_TARGETS, targets  # noqa: E402


_HARNESS_SPEC = importlib.util.spec_from_file_location(
    "demo_template_scene_harness", ROOT / "scripts/evaluate_unified_vision50.py"
)
HARNESS = importlib.util.module_from_spec(_HARNESS_SPEC)
assert _HARNESS_SPEC.loader is not None
_HARNESS_SPEC.loader.exec_module(HARNESS)


def clean_scene_configuration(task_name: str) -> dict:
    specification = importlib.util.spec_from_file_location(
        "demo_template_clean_config", ROBOTWIN / "scripts/run_place_empty_cup_vision.py"
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    config = module.load_task_args(
        str(ROBOTWIN / "env_cfg/task_config/demo_clean.yml"), 0, arm="auto"
    )
    config.update(
        task_name=task_name,
        eval_mode=True,
        eval_video_log=False,
        save_data=False,
        need_plan=False,
        enable_mplib=False,
        head_camera_only=True,
        fast_eval_mode=True,
    )
    config["data_type"].update(
        mesh_segmentation=False,
        actor_segmentation=False,
        pointcloud=False,
    )
    return config


def _template_points(frame: RGBDFrame, mask: np.ndarray, voxel_m: float) -> np.ndarray:
    grid, valid = world_grid(frame)
    selected = valid & np.asarray(mask, bool)
    points = voxelize(grid[selected], voxel_m)
    if len(points) < 40:
        raise RuntimeError(f"target_has_only_{len(points)}_visible_depth_points")
    return points


def _audit_image(rgb: np.ndarray, masks: dict[str, np.ndarray], path: Path) -> None:
    from PIL import Image

    colors = np.asarray(
        ([0, 210, 255], [255, 70, 170], [80, 255, 100], [255, 180, 0],
         [170, 100, 255], [30, 230, 160]), dtype=float
    )
    canvas = np.asarray(rgb)[..., :3].copy()
    for index, mask in enumerate(masks.values()):
        selected = np.asarray(mask, bool)
        canvas[selected] = (
            0.55 * canvas[selected] + 0.45 * colors[index % len(colors)]
        ).astype(np.uint8)
    Image.fromarray(canvas).save(path)


def capture_task(args) -> None:
    name = args.worker_task
    directory = args.output / "tasks" / name
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "task": name,
        "status": "running",
        "gpu": args.worker_gpu,
        "configuration": "demo_clean",
        "seed_start": args.seed,
        "max_attempts": args.max_attempts,
        "uses_CAD": False,
        "robot_task_actions_executed": 0,
        "template_segmentation_source": "offline_renderer_identity_mask",
        "runtime_randomized_inference_receives_template_points_only": True,
        "official_demo_hdf5_used": False,
        "reason_official_hdf5_not_used": "official_demo_clean_HDF5_has_RGB_but_no_depth",
        "attempts": [],
    }
    HARNESS.rendering_runtime()
    HARNESS.block_planner_import()
    os.chdir(ROBOTWIN)
    module = importlib.import_module("envs." + name)
    task = getattr(module, name)()
    config = clean_scene_configuration(name)
    from envs.utils.create_actor import UnStableError
    from adapters.cup_initial_audit import no_task_actions

    def save() -> None:
        HARNESS.dump(directory / "report.json", report)

    save()
    try:
        with HARNESS.no_planners():
            for index in range(args.max_attempts):
                seed = args.seed + index
                attempt = {"seed": seed, "status": "running"}
                report["attempts"].append(attempt)
                try:
                    random.seed(seed)
                    task.setup_demo(now_ep_num=index, seed=seed, **deepcopy(config))
                    with no_task_actions(task):
                        observation = task.get_camera_observation("head_camera")
                        sensor = directory / "rgbd.npz"
                        np.savez_compressed(
                            sensor,
                            rgb=observation["rgb"],
                            depth=observation["depth"],
                            intrinsic=observation["intrinsic_cv"],
                            world_to_camera=observation["extrinsic_cv"],
                        )
                        frame = RGBDFrame.from_npz(
                            sensor, camera="head", capture_id="clean_template", depth_unit="mm"
                        )
                        camera = task.cameras.static_camera_list[
                            task.cameras.static_camera_name.index("head_camera")
                        ]
                        labels = camera.get_picture("Segmentation")[..., 1].copy()
                        masks = {}
                        arrays = {}
                        entries = {}
                        for key, actor in targets(task, name):
                            mask = np.isin(labels, HARNESS.entity_ids(actor))
                            points = _template_points(frame, mask, args.voxel_m)
                            masks[key] = mask
                            array = key + "_points_world"
                            arrays[array] = points
                            bounds = np.quantile(points, (0.01, 0.99), axis=0)
                            entries[key] = {
                                "path": str((directory / "template.npz").resolve()),
                                "array": array,
                                "point_count": len(points),
                                "visible_bounds_world": bounds,
                                "visible_diameter_m": float(np.linalg.norm(bounds[1] - bounds[0])),
                            }
                        np.savez_compressed(directory / "template.npz", **arrays)
                        np.savez_compressed(directory / "offline_template_masks.npz", **masks)
                        _audit_image(observation["rgb"], masks, directory / "template_audit.png")
                        attempt.update(status="captured", target_count=len(entries))
                        report.update(
                            status="finished",
                            selected_seed=seed,
                            sensor=str(sensor.resolve()),
                            sensor_sha256=HARNESS.digest(sensor),
                            template=str((directory / "template.npz").resolve()),
                            template_sha256=HARNESS.digest(directory / "template.npz"),
                            targets=entries,
                        )
                    save()
                    return
                except UnStableError as error:
                    attempt.update(status="initialization_rejected", error=str(error))
                except RuntimeError as error:
                    # A valid clean scene can still hide one declared target.
                    # Try another clean seed rather than storing an empty model.
                    attempt.update(status="template_rejected", error=str(error))
                finally:
                    try:
                        task.close_env()
                    except Exception as error:
                        attempt["close_error"] = str(error)
                    save()
        raise RuntimeError("clean_template_capture_attempts_exhausted")
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise


def _manifest(args, selected: list[str]) -> dict:
    tasks_value = {}
    errors = {}
    for name in selected:
        report_path = args.output / "tasks" / name / "report.json"
        if not report_path.is_file():
            errors[name] = "missing_report"
            continue
        value = json.loads(report_path.read_text())
        if value.get("status") != "finished":
            errors[name] = value.get("error", value.get("status"))
            continue
        tasks_value[name] = {
            key: {"path": target["path"], "array": target["array"]}
            for key, target in value["targets"].items()
        }
    return {
        "schema_version": 1,
        "source": "regenerated_demo_clean_initial_RGBD",
        "not_official_HDF5_trajectory": True,
        "reason": "official demo HDF5 omits depth",
        "template_supervision": "offline_renderer_identity_mask_equivalent_to_one_time_annotation",
        "CAD_used": False,
        "runtime_truth_used": False,
        "tasks": tasks_value,
        "errors": errors,
        "complete": len(tasks_value) == len(selected),
        "requested_tasks": selected,
    }


def coordinate(args) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    selected = args.tasks or list(TASK_TARGETS)
    gpus = [int(value) for value in args.gpus.split(",")]
    report = {
        "status": "running",
        "requested_tasks": selected,
        "gpus": gpus,
        "active": {},
        "tasks": {},
    }
    pending = list(selected)
    active = {}
    completed = {}
    started = time.perf_counter()

    def save() -> None:
        summaries = {}
        for name in selected:
            path = args.output / "tasks" / name / "report.json"
            if path.is_file():
                value = json.loads(path.read_text())
                summaries[name] = {key: value.get(key) for key in
                                   ("status", "gpu", "selected_seed", "targets", "error")}
            else:
                summaries[name] = {"status": "pending"}
        report.update(
            tasks=summaries,
            active={str(gpu): {"pid": item[0].pid, "task": item[1]}
                    for gpu, item in active.items()},
            completed=sum(value["status"] == "finished" for value in summaries.values()),
            elapsed_s=time.perf_counter() - started,
        )
        HARNESS.dump(args.output / "report.json", report)
        HARNESS.dump(args.output / "manifest.json", _manifest(args, selected))

    save()
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active:
                    process, name, log = active[gpu]
                    if process.poll() is None:
                        continue
                    log.close()
                    completed[name] = process.returncode
                    del active[gpu]
                if pending:
                    name = pending.pop(0)
                    log = (args.output / "logs" / f"{name}.log").open("x")
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-task", name,
                        "--worker-gpu", str(gpu),
                        "--seed", str(args.seed),
                        "--max-attempts", str(args.max_attempts),
                        "--voxel-m", str(args.voxel_m),
                        "--output", str(args.output),
                    ]
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=HARNESS.environment(gpu),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    active[gpu] = (process, name, log)
            save()
            if active:
                time.sleep(2)
        manifest = _manifest(args, selected)
        report.update(
            status="finished" if manifest["complete"] else "finished_with_errors",
            process_exit_codes=completed,
        )
        save()
    except BaseException as error:
        for process, _name, log in active.values():
            if process.poll() is None:
                process.terminate()
            log.close()
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_TARGETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--voxel-m", type=float, default=0.002)
    parser.add_argument("--worker-task", choices=tuple(TASK_TARGETS))
    parser.add_argument("--worker-gpu", type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output == ROOT or not args.output.is_relative_to(ROOT):
        parser.error("--output must be a new directory under ULA")
    if args.output.exists() and not args.worker_task:
        parser.error("Refusing to overwrite an existing output directory")
    if args.seed < 0 or args.max_attempts < 1 or not 0.0005 <= args.voxel_m <= 0.01:
        parser.error("Invalid seed, attempt budget, or voxel size")
    try:
        gpus = [int(value) for value in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must be comma-separated integers")
    if not gpus or len(gpus) != len(set(gpus)):
        parser.error("--gpus must contain distinct indices")
    if args.tasks and len(args.tasks) != len(set(args.tasks)):
        parser.error("Duplicate task")
    if args.worker_task:
        try:
            capture_task(args)
        except BaseException:
            traceback.print_exc()
            raise
    else:
        coordinate(args)


if __name__ == "__main__":
    main()
