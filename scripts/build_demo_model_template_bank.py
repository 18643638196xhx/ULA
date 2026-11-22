#!/usr/bin/env python3
"""Build a one-clean-RGB-D-example-per-model template bank.

This is an isolated extension of ``build_demo_clean_pointcloud_templates50.py``.
It does not change the old single-template manifest.  For dynamic bottle
assets it deliberately instantiates every model variant in a clean scene,
captures measured head-camera RGB-D, and uses an offline renderer identity
mask to cut out the demonstrated object.  Randomized-scene inference receives
only the resulting point clouds and never receives ``model_id`` or a renderer
mask.

The three currently supported tasks cover the two important cases:

* ``move_pillbottle_pad``: five variants of ``080_pillbottle``;
* ``pick_dual_bottles``: two fixed ``001_bottle`` variants;
* ``pick_diverse_bottles``: twenty ``001_bottle`` variants, each paired with a
  known-stable companion during clean capture.

One scene may contain two target instances, so one clean episode can provide
one example for each of two models.  No robot action is executed by this
localization-only data preparation script.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "tra_pipe/RoboTwin"
sys.path[:0] = [str(ROOT), str(ROBOTWIN), str(ROBOTWIN / "envs/curobo/src")]

# Install the scene-only planner guard BEFORE loading the base builder: that
# module imports envs.* at module scope, so installing the guard afterwards is
# rejected ("Install scene-only planner guard before importing tasks").
sys.path.insert(0, str(ROOT))
from diagnostics.planner_guard import block_planner_import as _guard  # noqa: E402
_guard()

_BASE_SPEC = importlib.util.spec_from_file_location(
    "single_demo_template_builder",
    ROOT / "scripts/build_demo_clean_pointcloud_templates50.py",
)
BASE = importlib.util.module_from_spec(_BASE_SPEC)
assert _BASE_SPEC.loader is not None
_BASE_SPEC.loader.exec_module(BASE)

from adapters.unified_vision import RGBDFrame  # noqa: E402
from diagnostics.vision50_catalog import targets  # noqa: E402


TASK_SPECS = {
    "move_pillbottle_pad": {
        "choice_pool": tuple(range(1, 6)),
        "captures": [
            {"forced_choices": [model], "role_models": {"pillbottle": model}}
            for model in range(1, 6)
        ],
        "roles": {
            "pillbottle": {
                "family": "080_pillbottle",
                "model_attribute": "pillbottle_id",
                "models": tuple(range(1, 6)),
            },
            "pad": {"family": "pad", "static": True},
        },
        "distinct_groups": [],
        "role_orders": [],
    },
    "pick_dual_bottles": {
        "choice_pool": None,
        "captures": [{
            "forced_choices": [],
            "role_models": {"bottle1": 13, "bottle2": 16},
        }],
        "roles": {
            "bottle1": {
                "family": "001_bottle",
                "fixed_model": 13,
                "models": (13,),
            },
            "bottle2": {
                "family": "001_bottle",
                "fixed_model": 16,
                "models": (16,),
            },
        },
        "distinct_groups": [["bottle1", "bottle2"]],
        "role_orders": [{
            "roles": ["bottle1", "bottle2"],
            "axis": "x",
            "ascending": True,
            "source": "stable_relation_observed_in_clean_task_scenes",
        }],
    },
    "pick_diverse_bottles": {
        "choice_pool": tuple(range(20)),
        "captures": [
            {
                "model": model,
                "alternatives": [
                    {
                        "forced_choices": [model, 1 if model == 0 else 0],
                        "role_models": {
                            "bottle1": model,
                            "bottle2": 1 if model == 0 else 0,
                        },
                    },
                    {
                        "forced_choices": [1 if model == 0 else 0, model],
                        "role_models": {
                            "bottle1": 1 if model == 0 else 0,
                            "bottle2": model,
                        },
                    },
                ],
            }
            for model in range(20)
        ],
        "roles": {
            # Both roles receive the same 20-model family bank at runtime.
            # Their identities are assigned jointly by x order, not by model.
            "bottle1": {
                "family": "001_bottle",
                "model_attribute": "bottle1_id",
                "models": tuple(range(20)),
                "share_family_bank": True,
            },
            "bottle2": {
                "family": "001_bottle",
                "model_attribute": "bottle2_id",
                "models": tuple(range(20)),
                "share_family_bank": True,
            },
        },
        "distinct_groups": [["bottle1", "bottle2"]],
        "role_orders": [{
            "roles": ["bottle1", "bottle2"],
            "axis": "x",
            "ascending": True,
            "source": "stable_relation_observed_in_clean_task_scenes",
        }],
    },
}


def _dump(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(BASE.HARNESS.serializable(value), ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _choice_result(value: int, size):
    if size is None:
        return value
    return np.full(size, value, dtype=int)


@contextmanager
def _forced_model_choices(pool, values: list[int]):
    """Force only choice calls whose population exactly matches an asset pool."""
    if pool is None:
        yield []
        return
    original = np.random.choice
    remaining = list(values)
    consumed = []
    expected = np.asarray(pool)

    def choice(a, size=None, replace=True, p=None):
        array = np.asarray(a)
        if remaining and array.shape == expected.shape and np.array_equal(array, expected):
            value = remaining.pop(0)
            consumed.append(value)
            return _choice_result(value, size)
        return original(a, size=size, replace=replace, p=p)

    np.random.choice = choice
    try:
        yield consumed
    finally:
        np.random.choice = original
    if remaining:
        raise RuntimeError(f"task_did_not_request_forced_models:{remaining}")


def _actual_model(task, role: str, role_spec: dict) -> int | str:
    if "fixed_model" in role_spec:
        return role_spec["fixed_model"]
    attribute = role_spec.get("model_attribute")
    if not attribute:
        return "static"
    return int(getattr(task, attribute))


def _model_key(family: str, model) -> str:
    return family if model == "static" else f"{family}/base{model}"


def _capture(args, task_name: str) -> dict:
    specification = TASK_SPECS[task_name]
    task_directory = args.output / "tasks" / task_name
    task_directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "task": task_name,
        "configuration": "demo_clean",
        "uses_CAD": False,
        "robot_task_actions_executed": 0,
        "template_segmentation_source": "offline_renderer_identity_mask",
        "runtime_randomized_inference_receives_model_id": False,
        "runtime_randomized_inference_receives_renderer_masks": False,
        "one_clean_RGBD_example_per_asset_model": True,
        "captures": [],
    }
    _dump(task_directory / "report.json", report)

    BASE.HARNESS.rendering_runtime()
    # The guard is already installed at import time above; the base module's own
    # block_planner_import() would now raise because a stub is present.
    if 'envs.robot.planner' not in sys.modules:
        BASE.HARNESS.block_planner_import()
    os.chdir(ROBOTWIN)
    task_class = getattr(importlib.import_module("envs." + task_name), task_name)
    config = BASE.clean_scene_configuration(task_name)
    from envs.utils.create_actor import UnStableError
    from adapters.cup_initial_audit import no_task_actions

    # Collect one record per family/model.  A static support is intentionally
    # stored once even when it appears in every forced-model scene.
    family_templates: dict[str, dict[str, dict]] = {}
    role_observations: dict[str, list[np.ndarray]] = {
        role: [] for role in specification["roles"]
    }
    try:
        with BASE.HARNESS.no_planners():
            for capture_index, plan in enumerate(specification["captures"]):
                capture_directory = task_directory / f"capture_{capture_index:03d}"
                capture_directory.mkdir()
                alternatives = plan.get("alternatives", [plan])
                capture_report = {
                    "capture": capture_index,
                    "requested_model": plan.get("model"),
                    "role_model_alternatives": [
                        value["role_models"] for value in alternatives
                    ],
                    "attempts": [],
                }
                report["captures"].append(capture_report)
                captured = False
                for alternative_index, active_plan in enumerate(alternatives):
                    for retry in range(args.max_attempts):
                        seed = (args.seed + capture_index * args.seed_stride
                                + alternative_index * args.max_attempts + retry)
                        attempt = {
                            "seed": seed,
                            "alternative": alternative_index,
                            "requested_role_models": active_plan["role_models"],
                            "status": "running",
                        }
                        capture_report["attempts"].append(attempt)
                        task = task_class()
                        try:
                            random.seed(seed)
                            np.random.seed(seed)
                            with _forced_model_choices(
                                specification["choice_pool"], active_plan["forced_choices"]
                            ) as consumed:
                                task.setup_demo(
                                    now_ep_num=capture_index, seed=seed, **deepcopy(config)
                                )
                            actual = {
                                role: _actual_model(task, role, role_spec)
                                for role, role_spec in specification["roles"].items()
                            }
                            for role, expected in active_plan["role_models"].items():
                                if actual[role] != expected:
                                    raise RuntimeError(
                                        f"forced_model_mismatch:{role}:expected={expected}:actual={actual[role]}"
                                    )
                            with no_task_actions(task):
                                observation = task.get_camera_observation("head_camera")
                                sensor = capture_directory / "rgbd.npz"
                                np.savez_compressed(
                                    sensor,
                                    rgb=observation["rgb"],
                                    depth=observation["depth"],
                                    intrinsic=observation["intrinsic_cv"],
                                    world_to_camera=observation["extrinsic_cv"],
                                )
                                frame = RGBDFrame.from_npz(
                                    sensor,
                                    camera="head",
                                    capture_id=f"clean_model_template_{capture_index}",
                                    depth_unit="mm",
                                )
                                camera = task.cameras.static_camera_list[
                                    task.cameras.static_camera_name.index("head_camera")
                                ]
                                labels = camera.get_picture("Segmentation")[..., 1].copy()
                                actors = dict(targets(task, task_name))
                                masks = {}
                                arrays = {}
                                entries = []
                                for role, role_spec in specification["roles"].items():
                                    actor = actors[role]
                                    mask = np.isin(labels, BASE.HARNESS.entity_ids(actor))
                                    points = BASE._template_points(frame, mask, args.voxel_m)
                                    masks[role] = mask
                                    model = actual[role]
                                    family = role_spec["family"]
                                    key = _model_key(family, model)
                                    role_observations[role].append(np.median(points, axis=0))
                                    already = key in family_templates.setdefault(family, {})
                                    if already:
                                        continue
                                    array = f"{role}_points_world"
                                    arrays[array] = points
                                    bounds = np.quantile(points, (0.01, 0.99), axis=0)
                                    entry = {
                                        "model_key": key,
                                        "model_id": model,
                                        "family": family,
                                        "demonstrated_role": role,
                                        "path": str((capture_directory / "templates.npz").resolve()),
                                        "array": array,
                                        "point_count": len(points),
                                        "visible_bounds_world": bounds,
                                        "visible_diameter_m": float(np.linalg.norm(bounds[1] - bounds[0])),
                                        "capture_seed": seed,
                                    }
                                    family_templates[family][key] = entry
                                    entries.append(entry)
                                if arrays:
                                    np.savez_compressed(capture_directory / "templates.npz", **arrays)
                                np.savez_compressed(
                                    capture_directory / "offline_template_masks.npz", **masks
                                )
                                BASE._audit_image(
                                    observation["rgb"], masks,
                                    capture_directory / "template_audit.png",
                                )
                            attempt.update(
                                status="captured",
                                forced_choices_consumed=consumed,
                                actual_role_models=actual,
                            )
                            capture_report.update(
                                status="captured",
                                selected_seed=seed,
                                selected_alternative=alternative_index,
                                selected_role_models=active_plan["role_models"],
                                sensor=str(sensor.resolve()),
                                templates=entries,
                            )
                            captured = True
                            break
                        except UnStableError as error:
                            attempt.update(status="initialization_rejected", error=str(error))
                        except RuntimeError as error:
                            attempt.update(status="capture_rejected", error=str(error))
                        finally:
                            try:
                                task.close_env()
                            except Exception as error:
                                attempt["close_error"] = str(error)
                            _dump(task_directory / "report.json", report)
                    if captured:
                        break
                if not captured:
                    raise RuntimeError(
                        f"model_template_capture_exhausted:{task_name}:capture={capture_index}"
                    )

        role_templates = {}
        for role, role_spec in specification["roles"].items():
            family = role_spec["family"]
            wanted = {
                _model_key(family, model) for model in role_spec.get("models", ("static",))
            }
            if role_spec.get("share_family_bank"):
                wanted = set(family_templates[family])
            selected = [
                value for key, value in sorted(family_templates[family].items())
                if key in wanted
            ]
            if len(selected) != len(wanted):
                missing = sorted(wanted - {item["model_key"] for item in selected})
                raise RuntimeError(f"missing_model_templates:{role}:{missing}")
            role_templates[role] = {
                "family": family,
                "templates": selected,
            }

        role_partitions = []
        axes = {"x": 0, "y": 1, "z": 2}
        for relation in specification["role_orders"]:
            roles = relation["roles"]
            if len(roles) != 2:
                continue
            left, right = roles
            left_values = role_observations[left]
            right_values = role_observations[right]
            if not left_values or len(left_values) != len(right_values):
                continue
            axis = axes[relation["axis"]]
            split = float(np.median([
                (a[axis] + b[axis]) / 2.0
                for a, b in zip(left_values, right_values)
            ]))
            if all(a[axis] < split < b[axis]
                   for a, b in zip(left_values, right_values)):
                role_partitions.append({
                    "roles": roles,
                    "axis": relation["axis"],
                    "split_world": split,
                    "sides": ["lower", "upper"],
                    "source": "relation_measured_from_clean_RGBD_demonstration",
                })

        report.update(
            status="finished",
            role_templates=role_templates,
            distinct_groups=specification["distinct_groups"],
            role_orders=specification["role_orders"],
            role_partitions=role_partitions,
            role_position_samples={
                role: values for role, values in role_observations.items()
            },
        )
        _dump(task_directory / "report.json", report)
        return report
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        _dump(task_directory / "report.json", report)
        raise


def _manifest(output: Path, selected: list[str], reports: dict[str, dict]) -> dict:
    tasks = {}
    errors = {}
    for task in selected:
        report = reports.get(task)
        if not report or report.get("status") != "finished":
            errors[task] = None if report is None else report.get("error", report.get("status"))
            continue
        tasks[task] = {
            "targets": report["role_templates"],
            "distinct_groups": report["distinct_groups"],
            "role_orders": report["role_orders"],
            "role_partitions": report["role_partitions"],
        }
    return {
        "schema_version": 2,
        "kind": "clean_RGBD_asset_model_template_bank",
        "source": "regenerated_demo_clean_initial_RGBD",
        "one_clean_RGBD_example_per_asset_model": True,
        "official_demo_HDF5_used": False,
        "reason": "official demo_clean HDF5 omits depth; synchronized clean RGB-D sidecars are captured",
        "offline_template_supervision": "renderer_identity_mask_equivalent_to_one_time_annotation",
        "runtime_model_id_used": False,
        "runtime_renderer_truth_used": False,
        "CAD_used": False,
        "tasks": tasks,
        "errors": errors,
        "complete": len(tasks) == len(selected),
        "requested_tasks": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_SPECS),
                        default=list(TASK_SPECS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--voxel-m", type=float, default=0.002)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output == ROOT or not args.output.is_relative_to(ROOT):
        parser.error("--output must be a new directory under ULA")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output directory")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("Duplicate task")
    if args.max_attempts < 1 or args.seed_stride < args.max_attempts:
        parser.error("seed stride must be at least max attempts")
    if not 0.0005 <= args.voxel_m <= 0.01:
        parser.error("voxel size must be in [0.0005, 0.01] m")

    # Match the existing ULA convention: --gpu is a physical GPU index and
    # the task process sees it as local cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output.mkdir(parents=True)
    reports = {}
    try:
        for task in args.tasks:
            reports[task] = _capture(args, task)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        manifest = _manifest(args.output, args.tasks, reports)
        _dump(args.output / "manifest.json", manifest)
        _dump(args.output / "report.json", {
            "status": "finished" if manifest["complete"] else "finished_with_errors",
            "gpu": args.gpu,
            "tasks": {
                task: {key: report.get(key) for key in ("status", "error")}
                for task, report in reports.items()
            },
            "manifest": str((args.output / "manifest.json").resolve()),
        })


if __name__ == "__main__":
    main()
