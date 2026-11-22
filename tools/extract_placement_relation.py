"""Extract the object-to-support placement relation from the official demo.

`functional_matrix` answers a static question: in which pose, and with which
point, should the object meet its support?  For cup and pillbottle the asset
metadata happens to answer it.  For fan, mouse and stapler it does not — and that
missing field is what makes `placement.py` fail with `IndexError`.

Rather than hand-annotating every CAD file, run the task's own demonstration once
and record where the object ends up relative to its support:

    T_support_object = inverse(T_world_support) @ T_world_object

This is a *static task prior*, extracted offline.  It encodes upright vs. lying,
which face is down, and which point meets the support centre.  At runtime only the
support is localised visually and the stored relation is applied — no simulator
pose is read during execution.

Usage:
    python tools/extract_placement_relation.py place_shoe [--seeds 0,1,2]
    python tools/extract_placement_relation.py move_stapler_pad --write
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# A valid demo leaves the object on the support: a few mm of planar offset and
# at most a couple of centimetres of height difference.
MAX_PLANAR_OFFSET_M = 0.05
MAX_HEIGHT_OFFSET_M = 0.03

ROOT = Path("/data1/xuhuixin/ula")
SIM = ROOT / "tra_pipe/RoboTwin"
sys.path[:0] = [str(ROOT), str(SIM)]
# RoboTwin's env modules load assets by relative path.
os.chdir(SIM)


def _spec(task_name):
    from adapters.ula_task_registry import TASKS
    if task_name not in TASKS:
        raise SystemExit(f"unknown task: {task_name}")
    return TASKS[task_name]


def _matrix(pose) -> np.ndarray:
    """sapien.Pose -> 4x4 homogeneous matrix."""
    if pose is None:
        raise ValueError("pose is None")
    if hasattr(pose, "to_transformation_matrix"):
        return np.asarray(pose.to_transformation_matrix(), dtype=float)
    array = np.asarray(pose, dtype=float)
    if array.shape == (4, 4):
        return array
    raise ValueError(f"unsupported pose {type(pose)}")


def _rotation_angle(a, b):
    cosine = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _rotation_medoid(rotations):
    distances = np.asarray([
        [_rotation_angle(left, right) for right in rotations]
        for left in rotations
    ])
    index = int(np.argmin(np.sum(distances, axis=1)))
    return np.asarray(rotations[index]), float(np.mean(distances[index]))


def _task_config(arm="auto", name=None):
    """Same config the runner uses, so the demo matches real episodes.

    RoboTwin's ``scripts`` package shadows the ULA one, so load the helper by
    file path instead of by module name.
    """
    import importlib.util
    cfg_name = name or os.environ.get("ULA_TASK_CONFIG", "demo_clean")
    path = SIM / "scripts/run_place_empty_cup_vision.py"
    spec = importlib.util.spec_from_file_location("_rpev", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_task_args(str(SIM / f"env_cfg/task_config/{cfg_name}.yml"), 0, arm=arm)


def extract(task_name: str, seeds, write=False):
    spec = _spec(task_name)
    module = importlib.import_module(spec["env_module"])
    env_class = getattr(module, spec["env_attr"])
    object_actor = spec["object_actor"]
    support_actor = spec["support_actor"]

    relations, records = [], []
    for seed in seeds:
        env = env_class()
        try:
            env.setup_demo(now_ep_num=0, seed=seed, **_task_config())
            env.play_once()
            object_pose = _matrix(getattr(env, object_actor).get_pose())
            support_pose = _matrix(getattr(env, support_actor).get_pose())
            relation = np.linalg.inv(support_pose) @ object_pose
            # Reject demos whose own placement failed: the object would sit far
            # from the support and poison the averaged relation.
            offset = relation[:3, 3]
            on_support = (np.linalg.norm(offset[:2]) < MAX_PLANAR_OFFSET_M
                          and abs(offset[2]) < MAX_HEIGHT_OFFSET_M)
            if not on_support:
                print(f"  seed {seed}: REJECTED, object not on support "
                      f"(offset={np.round(offset, 4).tolist()})")
                continue
            relations.append(relation)
            records.append(dict(seed=seed,
                                object_pose=object_pose.tolist(),
                                support_pose=support_pose.tolist(),
                                relation=relation.tolist()))
            print(f"  seed {seed}: object t={np.round(object_pose[:3, 3], 4).tolist()} "
                  f"support t={np.round(support_pose[:3, 3], 4).tolist()}")
        except Exception as error:
            import traceback
            print(f"  seed {seed}: FAILED {type(error).__name__}: {error}")
            traceback.print_exc()
        finally:
            try:
                env.close_env()
            except Exception:
                pass

    if not relations:
        raise SystemExit("no successful demo extraction")

    stack = np.stack(relations)
    mean = stack.mean(axis=0)
    # Spread across seeds tells us whether the relation is genuinely constant.
    translation_spread = float(np.max(np.ptp(stack[:, :3, 3], axis=0)))
    rotation_spread = float(np.max(np.ptp(stack[:, :3, :3], axis=(0, 1))))
    print(f"\n  跨 seed 平移极差: {translation_spread*1000:.1f} mm")
    print(f"  跨 seed 旋转极差: {rotation_spread:.4f}")
    print(f"  relation t = {np.round(mean[:3, 3], 4).tolist()}")

    world_rotation, world_rotation_spread = _rotation_medoid(
        [np.asarray(record["object_pose"])[:3, :3] for record in records])
    relative_rotation, relative_rotation_spread = _rotation_medoid(
        [np.asarray(record["relation"])[:3, :3] for record in records])
    if len(records) < 2:
        orientation_frame = "support_relative"
    else:
        orientation_frame = (
            "world" if world_rotation_spread <= relative_rotation_spread + np.deg2rad(1.0)
            else "support_relative")
    # Never element-wise average a rotation matrix: the result need not belong
    # to SO(3).  Translation can be averaged, while orientation uses an observed
    # medoid from successful demonstrations.
    mean[:3, :3] = relative_rotation

    payload = dict(
        task=task_name, object_actor=object_actor, support_actor=support_actor,
        source="demo_extraction", uses_runtime_truth=False,
        seeds=list(seeds), translation_spread_m=round(translation_spread, 5),
        rotation_spread=round(rotation_spread, 5),
        orientation_frame=orientation_frame,
        world_orientation=world_rotation.tolist(),
        world_orientation_spread_rad=round(world_rotation_spread, 6),
        support_relative_orientation=relative_rotation.tolist(),
        support_relative_orientation_spread_rad=round(relative_rotation_spread, 6),
        object_to_support=mean.tolist(), samples=records)
    out = ROOT / "rules" / "placement_relations" / f"{task_name}.json"
    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"  written -> {out}")
    else:
        print("  (dry run; pass --write to save)")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"=== extracting placement relation for {args.task} ===")
    extract(args.task, seeds, write=args.write)


if __name__ == "__main__":
    main()
