#!/usr/bin/env python3
"""Dump why a pillbottle object query is 'ambiguous': scene image + candidates."""
import argparse, importlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "tra_pipe/RoboTwin"
sys.path[:0] = [str(ROOT), str(ROBOTWIN)]

from adapters.ula_task_registry import TASKS
from adapters.unified_perception import PERCEPTION_SPEC, UnifiedPerceptionWorker, UnifiedVisionPerception
from probe_localization import _build_task_adapter, _load_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="+", default=[2])
    ap.add_argument("--output", type=Path, default=Path("/tmp/pb_ambig"))
    args = ap.parse_args()

    spec = TASKS["move_pillbottle_pad"]
    os.chdir(ROBOTWIN)
    _build_task_adapter(spec)
    PERCEPTION_SPEC.update(
        object_prompt=spec["object_prompt"], support_prompt=spec["support_prompt"],
        object_hints=spec.get("object_hints") or {}, support_hints=spec.get("support_hints") or {},
        object_cad=None, support_cad=spec.get("support_cad"))
    base = _load_module(ROOT / spec["base"])
    config = base.task_config(argparse.Namespace(arm="auto", camera_shader="default"))
    config["task_name"] = spec["env_attr"]

    env_module = importlib.import_module(spec["env_module"])
    task = getattr(env_module, spec["env_attr"])()
    worker = UnifiedPerceptionWorker(ROOT, args.output / "vision_worker")
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        for seed in args.seed:
            task.setup_demo(now_ep_num=0, seed=seed, **config)
            perception = UnifiedVisionPerception(ROOT, args.output / "perception", worker=worker)
            head = task.get_camera_observation("head_camera")
            import imageio.v2 as imageio
            imageio.imwrite(args.output / f"pb_seed{seed}_head.png", np.asarray(head["rgb"]))
            scene = perception._observe(head["rgb"], head["depth"], head["intrinsic_cv"],
                                        head["extrinsic_cv"], perception.queries, "dbg")
            obj = scene.targets.get("object")
            sup = scene.targets.get("support")
            print(f"=== seed {seed} ===")
            print("object.status:", getattr(obj, "status", None))
            print("object.reasons:", getattr(obj, "reasons", None))
            cands = getattr(obj, "candidate_records", None) or getattr(obj, "candidates", None) or []
            print("object.candidate_records:", json.dumps(cands, default=str)[:2500])
            print("support.status:", getattr(sup, "status", None))
    finally:
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
