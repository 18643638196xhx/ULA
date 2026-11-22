#!/usr/bin/env python3
"""Isolated localization probe: scene setup + localize object/support, nothing else.

Does not move any arm and does not run VLA.  Reports per seed:
  - whether the joint head-camera query localized object+support;
  - the object/support status strings (available/ambiguous/not_observed/...);
  - whether the multi-camera fallback (placement_observation) recovered a failed joint query.
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "tra_pipe/RoboTwin"
sys.path[:0] = [str(ROOT), str(ROBOTWIN)]

from adapters.task_rules import load_support_placement_rule
from adapters.ula_task_registry import TASKS
from adapters.unified_perception import (
    PERCEPTION_SPEC,
    UnifiedPerceptionWorker,
    UnifiedVisionPerception,
)


def _load_module(path):
    spec = importlib.util.spec_from_file_location("ula_unified_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_task_adapter(spec):
    os.chdir(ROBOTWIN)
    env_module = importlib.import_module(spec["env_module"])
    env_class = getattr(env_module, spec["env_attr"])

    class TaskAdapter(env_class):
        def load_actors(self, *args, **kwargs):
            super().load_actors(*args, **kwargs)
            self.cup = getattr(self, spec["object_actor"])
            self.coaster = getattr(self, spec["support_actor"])
            for key, asset in (("object_cad", spec.get("object_asset")),
                               ("support_cad", spec.get("support_asset"))):
                if not asset:
                    continue
                attr = asset.get("model_id_attr")
                model_id = getattr(self, attr, 0) if attr else 0
                model_dir = asset.get("model_dir")
                if model_dir is None and asset.get("model_dir_attr"):
                    model_dir = str(getattr(self, asset["model_dir_attr"], "") or "")
                if not model_dir:
                    continue
                scale = asset.get("scale")
                if scale is None:
                    metadata = (ROBOTWIN / "assets/objects" / model_dir
                                / f"model_data{model_id}.json")
                    scale = float(json.loads(metadata.read_text())["scale"][0])
                if isinstance(scale, (int, float)):
                    scale = (scale, scale, scale)
                PERCEPTION_SPEC[key] = dict(
                    mesh_path=(f"tra_pipe/RoboTwin/assets/objects/{model_dir}"
                               f"/visual/base{model_id}.glb"),
                    scale=tuple(scale),
                )

    env_module.__dict__[spec["env_attr"]] = TaskAdapter
    cup_module = importlib.import_module("envs.place_empty_cup")
    cup_module.place_empty_cup = TaskAdapter
    return TaskAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-shader", choices=("default", "rt"), default="default")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output = args.output.resolve()   # absolute before chdir into RoboTwin
    spec = TASKS[args.task]
    os.chdir(ROBOTWIN)
    load_support_placement_rule(ROOT / spec["rule"])
    _build_task_adapter(spec)
    PERCEPTION_SPEC.update(
        object_prompt=spec["object_prompt"],
        support_prompt=spec["support_prompt"],
        object_hints=spec.get("object_hints") or {},
        support_hints=spec.get("support_hints") or {},
        object_cad=None,
        support_cad=spec.get("support_cad"),
    )
    base = _load_module(ROOT / spec["base"])
    config = base.task_config(argparse.Namespace(arm="auto", camera_shader=args.camera_shader))
    config["task_name"] = spec["env_attr"]

    env_module = importlib.import_module(spec["env_module"])
    env_class = getattr(env_module, spec["env_attr"])
    task = env_class()
    worker = UnifiedPerceptionWorker(ROOT, args.output / "vision_worker")

    report = dict(task=args.task, config=os.environ.get("ULA_TASK_CONFIG", "demo_clean"),
                  episodes=[])
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            record = dict(seed=seed, localized=False, joint_head=None,
                          object_status=None, support_status=None,
                          recovered_by_multiview=False, error=None,
                          object_bounds=None, support_center=None)
            try:
                task.setup_demo(now_ep_num=index, seed=seed, **config)
                perception = UnifiedVisionPerception(
                    ROOT, args.output / "perception", worker=worker)
                head = task.get_camera_observation("head_camera")
                estimate = None
                try:
                    estimate = perception.estimate(
                        head["rgb"], head["depth"],
                        head["intrinsic_cv"], head["extrinsic_cv"])
                    record["joint_head"] = "success"
                except RuntimeError as error:
                    record["joint_head"] = f"failed: {error}"
                    arm = str(getattr(task, "arm_tag", "left") or "left")
                    estimate = perception.placement_observation(task, arm)
                    record["recovered_by_multiview"] = True
                audit = estimate.observation_audit
                record["localized"] = True
                record["object_status"] = audit.get("object_status")
                record["support_status"] = audit.get("support_status")
                record["object_bounds"] = np.asarray(estimate.cup_bounds_world).tolist()
                record["support_center"] = np.asarray(estimate.coaster_world_center).tolist()
            except Exception as error:  # noqa: BLE001
                record["error"] = f"{type(error).__name__}: {error}"
            report["episodes"].append(record)
            print(f"[probe] seed={seed} localized={record['localized']} "
                  f"joint={record['joint_head']} err={(record['error'] or '')[:60]}",
                  flush=True)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, default=str) + "\n")
    finally:
        worker.close()
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n")

    episodes = report["episodes"]
    localized = [e for e in episodes if e["localized"]]
    unstable = [e for e in episodes if "UnStableError" in str(e.get("error") or "")]
    print(f"SUMMARY localized={len(localized)}/{len(episodes)} "
          f"scene_init_failures={len(unstable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
