#!/usr/bin/env python3
"""Isolated randomized-scene evaluation of the unchanged visual cup ULA."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
SPEC = importlib.util.spec_from_file_location("ula_randomized_cup_base", ROOT/"scripts/run_place_empty_cup.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
SCENE = ROOT/"tra_pipe/RoboTwin/env_cfg/task_config/demo_randomized.yml"


def randomized_config(clean):
    with SCENE.open() as stream:
        source = yaml.safe_load(stream)
    config = deepcopy(clean)
    config["domain_randomization"] = deepcopy(source["domain_randomization"])
    config["eval_instruction"] = source["eval_instruction"]
    # Runtime sensor and control settings are preserved, including depth=True.
    return config


def fingerprints():
    files = [ROOT/"scripts/run_place_empty_cup.py", ROOT/"scripts/run_place_empty_cup.sh",
        ROOT/"scripts/perceive_rgbd.py", ROOT/"tra_pipe/RoboTwin/envs/place_empty_cup.py",
        ROOT/"tra_pipe/RoboTwin/envs/_base_task.py",
        ROOT/"tra_pipe/RoboTwin/envs/robot/robot.py", ROOT/"tra_pipe/RoboTwin/envs/robot/planner.py"]
    files.extend(sorted((ROOT/"adapters").glob("*.py")))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def configure_runtime():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise ValueError("Expose exactly one shared GPU")
    import torch
    torch.cuda.set_per_process_memory_fraction(.08, 0)
    import sapien
    from sapien.pysapien.render import SapienRenderer as NativeRenderer
    device = sapien.Device("cuda:0")
    if not device.can_render():
        raise RuntimeError("Selected GPU cannot render")
    class SharedRenderer(sapien.SapienRenderer):
        _worldla_gpu = gpu
        def __init__(self, **kwargs):
            NativeRenderer.__init__(self, device)
    sapien.SapienRenderer = SharedRenderer
    return dict(gpu=gpu, jax_allocator_fraction=.40, torch_allocator_fraction=.08,
        vision_allocator_fraction=.10, hard_total_memory_limit=False, compute_isolation=False)


class SharedVisionWorker(base.PerceptionWorker):
    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(ROOT/"scripts/perceive_cup_randomized.py")
        return command, cwd, environment

    def estimate_files(self, request, result):
        if Path(request).parent.name == "perception" and Path(request).name == "rgbd.npz":
            self.focus_cup_bounds, self.focus_pad_center = None, None
        value = super().estimate_files(request, result)
        if os.environ.get("ULA_CUP_TARGET_FOCUS") == "1":
            import numpy as np
            self.focus_cup_bounds = value.cup_bounds_world.copy()
            if np.isfinite(value.coaster_world_center).all():
                self.focus_pad_center = value.coaster_world_center.copy()
        return value


def main():
    args = base.parse_args()
    targeting = os.environ.get("ULA_CUP_TARGET_FOCUS") == "1"
    vla_focus = os.environ.get("ULA_CUP_VLA_FOCUS") == "1"
    if vla_focus and not targeting:
        raise ValueError("VLA focus requires accepted focused target tracking")
    if args.mode != "hybrid" or args.approach_mode != "pointcloud" or args.grasp_handoff != "visual-trial-lift":
        raise ValueError("This comparison requires hybrid + pointcloud + visual-trial-lift")
    if args.seed_cache:
        raise ValueError("Randomized comparison uses raw consecutive seeds, not clean accepted-seed caches")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory; prior results are not overwritten")
    textures = ROOT/"tra_pipe/RoboTwin/assets/background_texture/unseen"
    if not any(textures.glob("*.png")):
        raise RuntimeError("Required unseen background textures are missing")
    args.output.mkdir(parents=True, exist_ok=True)
    resource = configure_runtime()
    before = fingerprints()
    focus_audit = dict(calls=0, applied_calls=0)
    original_policy = base.WorldLA
    if vla_focus:
        class FocusedPolicy(original_policy):
            def step(self, task, observation):
                from adapters.cup_vla_focus import focused_observation
                focused, audit = focused_observation(task, observation, worker)
                focus_audit["calls"] += 1
                focus_audit["applied_calls"] += int(audit["applied"])
                focus_audit["last"] = audit
                return super().step(task, focused)
        base.WorldLA = FocusedPolicy
    original_config, original_dump = base.task_config, base.dump_json
    config = randomized_config(original_config(args))
    run_status = "running"
    def metadata():
        return dict(task_config="demo_randomized", effective_domain_randomization=deepcopy(config["domain_randomization"]),
            evaluation_protocol="randomized_scenes_fixed_instruction_fixed_left_arm_not_full_official_benchmark",
            instruction_sampling="fixed_same_as_clean_baseline_not_unseen_instruction_sampling",
            seed_filtering=False, planned_episodes=args.episodes, run_status=run_status,
            target_focus_enabled=targeting, vla_soft_background_focus=vla_focus,
            vla_focus_audit=deepcopy(focus_audit),
            retained_priors=["robot_model_and_state", "camera_calibration", "configured_table_collision_box"],
            resource_policy=resource, protected_files_before=before,
            scene_config_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(),
            randomized_implementation_hashes={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                "scripts/run_place_empty_cup_randomized.py", "scripts/run_place_empty_cup_randomized.sh",
                "scripts/perceive_cup_randomized.py", "adapters/cup_target_focus.py",
                "adapters/cup_vla_focus.py")})
    def dump(path, value):
        if Path(path).name == "report.json":
            value.update(metadata(), completed_episodes=len(value["episodes"]))
        return original_dump(path, value)
    base.task_config = lambda unused: deepcopy(config)
    base.dump_json = dump
    worker = SharedVisionWorker(ROOT, args.output/"vision_worker", args.fp_iterations, backend="pointcloud")
    previous = Path.cwd()
    try:
        code = base.run_batch(args, worker)
        run_status = "finished"
        return code
    finally:
        worker.close()
        path = args.output/"report.json"
        base.task_config, base.dump_json = original_config, original_dump
        base.WorldLA = original_policy
        os.chdir(previous)
        if path.exists():
            report = json.loads(path.read_text())
            after = fingerprints()
            report.update(metadata(), completed_episodes=len(report["episodes"]),
                perception_worker=worker.summary(), protected_files_after=after,
                protected_files_unchanged=before == after)
            if run_status != "finished":
                report["run_status"] = "interrupted_or_error"
            original_dump(path, report)


if __name__ == "__main__":
    raise SystemExit(main())
