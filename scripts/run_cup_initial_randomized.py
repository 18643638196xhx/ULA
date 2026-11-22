#!/usr/bin/env python3
"""Only initial perception, never approach/grasp/place or task success scoring."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
SPEC = importlib.util.spec_from_file_location("ula_initial_cup_scene", ROOT/"scripts/run_place_empty_cup_randomized.py")
scene = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scene)
from adapters.cup_initial_perception import InitialCupWorker
from adapters.cup_initial_audit import no_task_actions, mask_metrics, evaluation_labels, overlay, contact_sheets


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda a: a.item() if isinstance(a, np.generic) else a.tolist())+"\n")


def summarize(records):
    initialized = [r for r in records if r["initialized"]]
    return dict(attempted_scenes=len(records), initialized_scenes=len(initialized),
        initialization_failures=[r["seed"] for r in records if not r["initialized"]],
        cup_outputs=sum(r.get("cup_output", False) for r in initialized),
        pad_outputs=sum(r.get("pad_output", False) for r in initialized),
        both_outputs=sum(r.get("cup_output", False) and r.get("pad_output", False) for r in initialized),
        evaluation_available=sum("evaluation_only" in r for r in initialized),
        cup_identity_matches=sum(r.get("evaluation_only", {}).get("cup", {}).get("identity_match", False)
                                 for r in initialized),
        pad_identity_matches=sum(r.get("evaluation_only", {}).get("pad", {}).get("identity_match", False)
                                 for r in initialized),
        both_identity_matches=sum(all(r.get("evaluation_only", {}).get(k, {}).get("identity_match", False)
                                      for k in ("cup", "pad")) for r in initialized),
        task_success_not_evaluated=True)


def fingerprints():
    values = scene.fingerprints()
    for name in ("scripts/run_cup_initial_randomized.py", "scripts/perceive_cup_initial.py",
                 "scripts/run_place_empty_cup_randomized.sh", "adapters/cup_initial_audit.py",
                 "tra_pipe/RoboTwin/env_cfg/task_config/demo_randomized.yml"):
        values[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return values


def run_episode(task, worker, config, output, index, seed):
    import imageio.v2 as imageio
    directory = output/f"episode_{index:03d}_seed{seed}"
    directory.mkdir()
    record = dict(seed=seed, initialized=False, failure_stage=None, error=None,
                  scope="initial_head_rgbd_perception_only", task_actions_executed=0,
                  vla_loaded=False, task_success_evaluated=False)
    started = time.perf_counter()
    try:
        task.setup_demo(now_ep_num=index, seed=seed, **deepcopy(config))
        record["initialized"] = True
        with no_task_actions(task):
            camera = task.get_camera_observation("head_camera")
            imageio.imwrite(directory/"initial_head_camera.png", camera["rgb"])
            request, result = directory/"rgbd.npz", directory/"prediction.npz"
            np.savez_compressed(request, rgb=camera["rgb"], depth=camera["depth"],
                                intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"])
            inference_start = time.perf_counter()
            estimate = worker.estimate_files(request, result)
            record["perception_elapsed_s"] = time.perf_counter()-inference_start
            audit = estimate.observation_audit
            record.update(cup_output=audit["cup_accepted"], pad_output=audit["pad_accepted"],
                          perception_audit=audit, timings_s=estimate.timings_s,
                          cup_bounds_world=estimate.cup_bounds_world,
                          pad_center_world=estimate.coaster_world_center)
            with np.load(result, allow_pickle=False) as data:
                cup_mask, pad_mask = data["cup_mask"].copy(), data["pad_mask"].copy()
            overlay(camera["rgb"], cup_mask, pad_mask,
                    f"seed {seed} | cup:{'output' if record['cup_output'] else 'missing'} "
                    f"pad:{'output' if record['pad_output'] else 'missing'}", directory/"overlay_head_camera.png")
            # Freeze the prediction first. Evaluation labels cannot cause retries.
            dump(directory/"prediction.json", record)
            try:
                truth_cup, truth_pad, ids = evaluation_labels(task, "head_camera")
                evaluation = dict(source="renderer_identity_after_frozen_prediction_only",
                                  feeds_back_into_perception=False, entity_ids=ids,
                                  cup=mask_metrics(cup_mask, truth_cup),
                                  pad=mask_metrics(pad_mask, truth_pad),
                                  identity_threshold="visible_mask_IoU_at_least_0.5_not_task_success")
                record["evaluation_only"] = evaluation
                np.savez_compressed(directory/"evaluation_only_masks.npz", cup=truth_cup, pad=truth_pad)
                def label(name):
                    if not record[f"{name}_output"]:
                        return "missing"
                    return "match" if evaluation[name]["identity_match"] else "wrong"
                overlay(camera["rgb"], cup_mask, pad_mask,
                        f"seed {seed} | cup:{label('cup')} pad:{label('pad')}",
                        directory/"audit_head_camera.png", truth_masks=(truth_cup, truth_pad))
            except Exception as error:
                record["evaluation_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        record.update(failure_stage="perception" if record["initialized"] else "setup",
                      error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        record["elapsed_s"] = time.perf_counter()-started
        dump(directory/"result.json", record)
        # Also release partially initialized scenes after setup rejection.
        try:
            task.close_env()
        except Exception as error:
            record["close_env_error"] = str(error)
            dump(directory/"result.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seed < 0:
        parser.error("Positive episode count and nonnegative initial seed required")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    config = scene.randomized_config(scene.base.task_config(
        argparse.Namespace(arm="left", camera_shader="default")))
    # Keep the same construction/RNG path as the full cup comparison. Planner
    # initialization is retained, but all task actions are forbidden afterward.
    before = fingerprints()
    resources = scene.configure_runtime()
    worker = InitialCupWorker(ROOT, args.output/"vision_worker")
    report = dict(task="place_empty_cup", scope="initial_perception_only",
                  task_config="demo_randomized", planned_scenes=args.episodes, run_status="running",
                  task_success_not_evaluated=True, uses_vla=False, task_actions_enabled=False,
                  color_is_hard_gate=False, uses_prepared_object_model=False,
                  inference_inputs=["rgb", "depth", "intrinsic", "world_to_camera"],
                  evaluation_truth_is_post_prediction_only=True, seed_filtering=False,
                  camera_scope="head_camera_only_no_active_view_motion",
                  effective_domain_randomization=config["domain_randomization"],
                  resources=resources, sources_before=before, episodes=[])
    previous = Path.cwd()
    try:
        os.chdir(scene.base.ROBOTWIN)
        from envs.place_empty_cup import place_empty_cup
        task = place_empty_cup()
        for index in range(args.episodes):
            record = run_episode(task, worker, config, args.output, index, args.seed+index)
            report["episodes"].append(record)
            report.update(completed_scenes=len(report["episodes"]), counts=summarize(report["episodes"]),
                          perception_worker=worker.summary())
            dump(args.output/"report.json", report)
            print(f"[cup-initial] seed={record['seed']} cup={record.get('cup_output')} "
                  f"pad={record.get('pad_output')} error={record['error']}", flush=True)
        report["run_status"] = "finished"
    finally:
        worker.close()
        os.chdir(previous)
        after = fingerprints()
        report.update(sources_after=after, sources_unchanged=before == after,
                      completed_scenes=len(report["episodes"]), counts=summarize(report["episodes"]),
                      perception_worker=worker.summary())
        if report["run_status"] != "finished":
            report["run_status"] = "interrupted_or_error"
        dump(args.output/"report.json", report)
        contact_sheets(args.output, report["episodes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
