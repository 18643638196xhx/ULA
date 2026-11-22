#!/usr/bin/env python3
"""Initial handover-block visual localization only; N initialized random scenes."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
SPEC = importlib.util.spec_from_file_location("handover_initial_shared",
    ROOT/"scripts/run_pillbottle_initial_randomized.py")
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
from adapters.cup_initial_audit import no_task_actions, evaluation_labels, overlay, contact_sheets
from adapters.handover.initial_geometry import yaw_error_modulo_square_degrees
from adapters.perception import PerceptionWorker


class HandoverInitialWorker(PerceptionWorker):
    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(ROOT/"scripts/perceive_handover_initial.py")
        selected = os.environ.get("HANDOVER_VISION_GPU", os.environ.get("CUDA_VISIBLE_DEVICES"))
        if selected is not None:
            if not selected.isdigit():
                raise ValueError("Expose one physical vision GPU")
            environment["CUDA_VISIBLE_DEVICES"] = selected
        return command, cwd, environment


def task_config():
    method = shared.method
    config = method.cup.randomized.randomized_config(method.base.task_config(
        argparse.Namespace(arm="auto", camera_shader="default")))
    config.update(task_name="handover_block", force_arm=None, active_arm=None)
    config["data_type"].update(mesh_segmentation=False, actor_segmentation=False)
    return config


def sources():
    paths = [Path(__file__), ROOT/"scripts/run_handover_initial_randomized.sh",
        ROOT/"scripts/perceive_handover_initial.py", ROOT/"scripts/perceive_rgbd.py",
        ROOT/"scripts/run_pillbottle_initial_randomized.py",
        ROOT/"scripts/run_move_pillbottle_pad_cup_method.py",
        ROOT/"scripts/run_place_empty_cup.py", ROOT/"scripts/run_place_empty_cup_randomized.py",
        ROOT/"scripts/run_place_empty_cup_randomized.sh",
        ROOT/"tra_pipe/RoboTwin/envs/handover_block.py",
        ROOT/"tra_pipe/RoboTwin/envs/_base_task.py", ROOT/"tra_pipe/RoboTwin/envs/utils/create_actor.py",
        ROOT/"tra_pipe/RoboTwin/env_cfg/task_config/demo_randomized.yml",
        *sorted((ROOT/"adapters").glob("*.py")),
        *sorted((ROOT/"adapters/pillbottle_focus").glob("*.py")),
        *sorted((ROOT/"adapters/handover").glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def evaluate_prediction(task, value, block_mask, platform_mask):
    # Called only after RGB-D prediction files are persisted. Nothing returned
    # here is passed back to perception, candidate choice, or any robot action.
    proxy = SimpleNamespace(cameras=task.cameras, cup=task.box, coaster=task.target_box)
    actual_block, actual_platform, ids = evaluation_labels(proxy, "head_camera")
    block = shared.identity_metrics(block_mask, actual_block)
    platform = shared.identity_metrics(platform_mask, actual_platform)
    audit = dict(source="simulation_after_frozen_prediction_only", feeds_back=False,
        block=block, platform=platform, entity_ids=ids)
    both = block["identity_match"] and platform["identity_match"]
    if value.cup_world_pose is not None:
        actual_pose = np.asarray(task.box.get_pose().to_transformation_matrix())
        audit.update(block_origin_translation_error_m=float(np.linalg.norm(
            value.cup_world_pose[:3, 3]-actual_pose[:3, 3])),
            block_yaw_error_mod90_deg=yaw_error_modulo_square_degrees(value.cup_world_pose, actual_pose),
            actual_up_tilt_deg=float(np.degrees(np.arccos(np.clip(actual_pose[2, 2], -1., 1.)))))
    if np.isfinite(value.coaster_world_center).all():
        actual_top = np.asarray(task.target_box.get_functional_point(1, "pose").p)
        audit.update(platform_xy_error_m=float(np.linalg.norm(value.coaster_world_center[:2]-actual_top[:2])),
            platform_top_z_error_m=float(abs(value.coaster_world_center[2]-actual_top[2])))
    localized = bool(both and audit.get("block_origin_translation_error_m", np.inf) <= .02
                     and audit.get("platform_xy_error_m", np.inf) <= .02)
    return audit, dict(both_identity_match=bool(both), localization_verified=localized,
        localization_and_yaw_verified=bool(localized and audit.get("block_yaw_error_mod90_deg", np.inf) <= 10.)), (
            actual_block, actual_platform)


def summarize(records):
    valid = [r for r in records if r["initialized"]]
    keys = ("block_output", "platform_output", "both_output", "both_identity_match",
            "localization_verified", "localization_and_yaw_verified")
    counts = {key: sum(bool(r.get(key)) for r in valid) for key in keys}
    return dict(valid_scenes=len(valid), raw_attempts=len(records),
        initialization_rejections=[r["seed"] for r in records if not r["initialized"]],
        counts=counts, rates={key: value/len(valid) if valid else None for key, value in counts.items()},
        evaluation_available=sum("evaluation_only" in r for r in valid),
        block_no_output_seeds=[r["seed"] for r in valid if not r["block_output"]],
        platform_no_output_seeds=[r["seed"] for r in valid if not r["platform_output"]],
        output_but_not_verified_seeds=[r["seed"] for r in valid
                                     if r.get("both_output") and not r.get("localization_verified")],
        task_actions_executed=sum(r["task_actions_executed"] for r in valid))


def run_episode(task, worker, config, output, index, seed):
    from PIL import Image
    directory = output/f"episode_{index:03d}_seed{seed}"
    directory.mkdir()
    record = dict(seed=seed, initialized=False, task_actions_executed=0, vla_loaded=False,
        task_success_evaluated=False, failure_stage=None, error=None,
        block_output=False, platform_output=False, both_output=False)
    started = time.perf_counter()
    try:
        task.setup_demo(now_ep_num=index, seed=seed, **deepcopy(config))
        record["initialized"] = True
        with no_task_actions(task):
            frame = task.get_camera_observation("head_camera")
            Image.fromarray(frame["rgb"][..., :3]).save(directory/"initial_head_camera.png")
            np.savez_compressed(directory/"rgbd.npz", rgb=frame["rgb"], depth=frame["depth"],
                intrinsic=frame["intrinsic_cv"], world_to_camera=frame["extrinsic_cv"])
            value = worker.estimate_files(directory/"rgbd.npz", directory/"prediction.npz")
            audit = value.observation_audit
            record.update(block_output=audit["block_output"], platform_output=audit["platform_output"],
                both_output=audit["block_output"] and audit["platform_output"],
                perception_audit=audit, timings_s=value.timings_s)
            record["failure_stage"] = None if record["both_output"] else "initial_perception"
            with np.load(directory/"prediction.npz", allow_pickle=False) as prediction:
                block_mask, platform_mask = prediction["block_mask"], prediction["platform_mask"]
                overlay(frame["rgb"], block_mask, platform_mask,
                    f"seed {seed} | block:{record['block_output']} platform:{record['platform_output']}",
                    directory/"overlay_head_camera.png")
                shared.dump(directory/"prediction.json", record)
                # Frozen prediction is on disk before any simulator labels or
                # actor poses below are read. Evaluation never changes output.
                try:
                    evaluation, checks, actual = evaluate_prediction(task, value, block_mask, platform_mask)
                    record.update(checks, evaluation_only=evaluation)
                    np.savez_compressed(directory/"evaluation_only_masks.npz", block=actual[0], platform=actual[1])
                    overlay(frame["rgb"], block_mask, platform_mask,
                        f"seed {seed} | IDs:{checks['both_identity_match']} position:{checks['localization_verified']}",
                        directory/"audit_head_camera.png", truth_masks=actual)
                except Exception as error:
                    record["evaluation_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        record.update(failure_stage="infrastructure" if record["initialized"] else "setup",
                      error=f"{type(error).__name__}: {error}")
    finally:
        record["elapsed_s"] = time.perf_counter()-started
        shared.dump(directory/"result.json", record)
        try:
            task.close_env()
        except Exception as error:
            record["close_env_error"] = str(error)
            shared.dump(directory/"result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("Invalid scene count or seed")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        parser.error('Evaluation outputs must stay under ULA')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh output directory")
    if args.detach:
        with args.output.with_suffix('.log').open('x') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                *[value for value in sys.argv[1:] if value!='--detach']],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                close_fds=True,start_new_session=True)
        import json
        print(json.dumps(dict(pid=process.pid,output=str(args.output),
            log=str(args.output.with_suffix('.log')),detached=True)),flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    before = sources()
    resource = shared.method.cup.randomized.configure_runtime()
    config = task_config()
    report = dict(task="handover_block", scope="initial_visual_localization_only", run_status="running",pid=os.getpid(),
        requested_valid_scenes=args.episodes, raw_seed_start=args.seed, task_config="demo_randomized",
        task_actions_enabled=False, uses_vla=False, task_success_evaluated=False,
        resources=resource, effective_domain_randomization=config["domain_randomization"],
        perception="RGBD_color_depth_semantic_SAM_and_initial_upright_cuboid_geometry",
        uses_prepared_cad_file=False, uses_foundationpose=False, uses_declared_shape_dimensions=True,
        geometry_priors=dict(block_size_m=[.06, .06, .20], block_initially_upright=True,
            horizontal_table=True, platform_approximately_square=True),
        evaluation_truth_used_after_frozen_prediction_only=True,
        identity_rule="at_least_20_target_pixels_and_prediction_precision_at_least_0.8",
        position_audit_rule="block_origin_3D_error_and_platform_XY_error_each_at_most_2cm",
        yaw_audit_rule="square_symmetry_modulo_90deg_error_at_most_10deg_separate_from_position",
        scoring_thresholds_are_not_inference_gates=True,
        denominator="all_successfully_initialized_scenes_including_perception_failures",
        sources_before=before, episodes=[])
    worker = HandoverInitialWorker(ROOT, args.output/"vision_worker", backend="pointcloud")
    previous = Path.cwd()
    try:
        os.chdir(ROOT/"tra_pipe/RoboTwin")
        from envs.handover_block import handover_block
        task = handover_block()
        while sum(r["initialized"] for r in report["episodes"]) < args.episodes:
            index = len(report["episodes"])
            if index >= max(200, 5*args.episodes):
                raise RuntimeError("Too many scene initialization rejections")
            record = run_episode(task, worker, config, args.output, index, args.seed+index)
            report["episodes"].append(record)
            report.update(summary=summarize(report["episodes"]), worker=worker.summary())
            shared.dump(args.output/"report.json", report)
            print(f"[handover-initial] seed={record['seed']} valid={report['summary']['valid_scenes']}/{args.episodes} "
                f"block={record['block_output']} platform={record['platform_output']} "
                f"position_verified={record.get('localization_verified')} stage={record['failure_stage']}", flush=True)
            if record["failure_stage"] == "infrastructure" or record.get("evaluation_error"):
                raise RuntimeError(record.get("error") or record["evaluation_error"])
        report["run_status"] = "finished"
    finally:
        worker.close()
        os.chdir(previous)
        report.update(summary=summarize(report["episodes"]), worker=worker.summary(), sources_after=sources())
        report["sources_unchanged"] = before == report["sources_after"]
        if report["run_status"] != "finished":
            report["run_status"] = "interrupted_or_error"
        shared.dump(args.output/"report.json", report)
        contact_sheets(args.output, report["episodes"])


if __name__ == "__main__":
    main()
