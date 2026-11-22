#!/usr/bin/env python3
"""N initialized random scenes: frozen initial perception/pose, no robot actions."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
SPEC = importlib.util.spec_from_file_location("initial_pill_cup_method",
    ROOT/"scripts/run_move_pillbottle_pad_cup_method.py")
method = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(method)
from adapters.cup_initial_audit import no_task_actions, evaluation_labels, mask_metrics, overlay, contact_sheets


def dump(path, value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        return item.tolist()
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode)+"\n")


class InitialWorker(method.BottleWorker):
    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(ROOT/"scripts/perceive_pillbottle_initial_audit.py")
        return command, cwd, environment


def identity_metrics(predicted, actual):
    result = mask_metrics(predicted, actual)
    # Partial visible surfaces should not be mislabeled as the wrong object
    # merely because erosion/depth filtering reduces whole-object mask IoU.
    result["mask_iou_at_least_half"] = result.pop("identity_match")
    result["identity_match"] = bool(result["target_visible"] and
        result["intersection_pixels"] >= 20 and result["target_precision"] >= .8)
    return result


def summarize(records):
    valid = [r for r in records if r["initialized"]]
    def count(key):
        return sum(bool(r.get(key)) for r in valid)
    numbers = {key: count(key) for key in ("bottle_output", "pad_output", "cad_output",
        "both_identity_match", "localization_verified")}
    return dict(valid_scenes=len(valid), raw_attempts=len(records),
        initialization_rejections=[r["seed"] for r in records if not r["initialized"]],
        counts=numbers, rates={k: v/len(valid) if valid else None for k, v in numbers.items()},
        evaluation_available=sum("evaluation_only" in r for r in valid),
        failures={stage: [r["seed"] for r in valid if r.get("failure_stage") == stage]
            for stage in ("pad_recognition", "bottle_recognition", "cad_registration", "infrastructure")})


def sources():
    paths = [Path(__file__), ROOT/"scripts/run_pillbottle_initial_randomized.sh",
        ROOT/"scripts/perceive_pillbottle_initial_audit.py",
        ROOT/"scripts/run_move_pillbottle_pad_cup_method.py",
        ROOT/"scripts/perceive_pillbottle_known_model.py",
        ROOT/"tra_pipe/RoboTwin/envs/move_pillbottle_pad.py",
        ROOT/"tra_pipe/RoboTwin/env_cfg/task_config/demo_randomized.yml",
        *sorted((ROOT/"adapters").glob("*.py")),
        *sorted((ROOT/"adapters/pillbottle_focus").glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_episode(task, worker, config, output, index, seed):
    from PIL import Image
    directory = output/f"episode_{index:03d}_seed{seed}"
    directory.mkdir()
    record = dict(seed=seed, initialized=False, task_actions_executed=0, vla_loaded=False,
        task_success_evaluated=False, failure_stage=None, error=None,
        bottle_output=False, pad_output=False, cad_output=False)
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
            record.update(bottle_output=audit["bottle_accepted"], pad_output=audit["pad_accepted"],
                cad_output=audit["cad_accepted"], failure_stage=audit["failure_stage"],
                bottle_attempted=audit["bottle_attempted"], cad_attempted=audit["cad_attempted"],
                error=audit["perception_error"], perception_audit=audit, timings_s=value.timings_s)
            with np.load(directory/"prediction.npz", allow_pickle=False) as prediction:
                bottle_mask, pad_mask = prediction["cup_mask"], prediction["pad_mask"]
                overlay(frame["rgb"], bottle_mask, pad_mask,
                    f"seed {seed} | bottle:{record['bottle_output']} pad:{record['pad_output']} CAD:{record['cad_output']}",
                    directory/"overlay_head_camera.png")
                dump(directory/"prediction.json", record)
                # Prediction has been persisted. All simulator reads BELOW are
                # independent evaluation only; there is no subsequent inference.
                try:
                    actual_bottle, actual_pad, ids = evaluation_labels(task, "head_camera")
                    bottle_score, pad_score = identity_metrics(bottle_mask, actual_bottle), identity_metrics(pad_mask, actual_pad)
                    evaluation = dict(source="simulation_after_frozen_prediction_only", feeds_back=False,
                        bottle=bottle_score, pad=pad_score, entity_ids=ids)
                    record["both_identity_match"] = bottle_score["identity_match"] and pad_score["identity_match"]
                    if value.cup_world_pose is not None:
                        true_pose = np.asarray(task.pillbottle.get_pose().to_transformation_matrix())
                        true_pad = np.asarray(task.pad.get_pose().p)
                        evaluation.update(bottle_origin_translation_error_m=float(np.linalg.norm(
                            value.cup_world_pose[:3, 3]-true_pose[:3, 3])),
                            pad_xy_error_m=float(np.linalg.norm(value.coaster_world_center[:2]-true_pad[:2])))
                        record["localization_verified"] = bool(record["both_identity_match"] and
                            evaluation["bottle_origin_translation_error_m"] <= .02 and evaluation["pad_xy_error_m"] <= .02)
                    else:
                        record["localization_verified"] = False
                    record["evaluation_only"] = evaluation
                    np.savez_compressed(directory/"evaluation_only_masks.npz", bottle=actual_bottle, pad=actual_pad)
                    overlay(frame["rgb"], bottle_mask, pad_mask,
                        f"seed {seed} | bottleID:{bottle_score['identity_match']} padID:{pad_score['identity_match']} CAD:{record['cad_output']}",
                        directory/"audit_head_camera.png", truth_masks=(actual_bottle, actual_pad))
                except Exception as error:
                    record["evaluation_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        record.update(failure_stage="infrastructure" if record["initialized"] else "setup",
                      error=f"{type(error).__name__}: {error}")
    finally:
        record["elapsed_s"] = time.perf_counter()-started
        dump(directory/"result.json", record)
        try:
            task.close_env()
        except Exception as error:
            record["close_env_error"] = str(error)
            dump(directory/"result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("Invalid scene count/seed")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    before = sources()
    resources = method.cup.randomized.configure_runtime()
    config = method.bottle_config(method.cup.randomized.randomized_config(
        method.base.task_config(argparse.Namespace(arm="auto", camera_shader="default"))))
    report = dict(task="move_pillbottle_pad", scope="initial_perception_and_cad_only",
        requested_valid_scenes=args.episodes, run_status="running", resources=resources,
        task_actions_enabled=False, uses_vla=False, task_success_evaluated=False,
        known_model="080_pillbottle/base1", task_config="demo_randomized",
        evaluation_truth_used_after_frozen_prediction_only=True,
        identity_rule="at_least_20_target_pixels_and_prediction_precision_at_least_0.8",
        position_audit_rule="bottle_CAD_origin_3D_and_pad_XY_error_each_at_most_2cm",
        position_audit_threshold_is_not_a_control_gate=True,
        effective_domain_randomization=config["domain_randomization"], sources_before=before, episodes=[])
    worker = InitialWorker(ROOT, args.output/"vision_worker", backend="foundationpose")
    previous = Path.cwd()
    try:
        os.chdir(ROOT/"tra_pipe/RoboTwin")
        import envs.move_pillbottle_pad as task_module
        from adapters.pillbottle_focus.cup_method import task_class
        task = task_class(task_module.move_pillbottle_pad, task_module)()
        while sum(r["initialized"] for r in report["episodes"]) < args.episodes:
            index = len(report["episodes"])
            if index >= max(200, 5*args.episodes):
                raise RuntimeError("Too many setup rejections")
            record = run_episode(task, worker, config, args.output, index, args.seed+index)
            report["episodes"].append(record)
            report.update(summary=summarize(report["episodes"]), worker=worker.summary())
            dump(args.output/"report.json", report)
            print(f"[pill-initial] seed={record['seed']} valid={report['summary']['valid_scenes']}/{args.episodes} "
                f"bottle={record['bottle_output']} pad={record['pad_output']} CAD={record['cad_output']} "
                f"stage={record['failure_stage']}", flush=True)
            if record["failure_stage"] == "infrastructure":
                raise RuntimeError(record["error"])
        report["run_status"] = "finished"
    finally:
        worker.close()
        os.chdir(previous)
        after = sources()
        report.update(summary=summarize(report["episodes"]), worker=worker.summary(),
            sources_after=after, sources_unchanged=before == after)
        if report["run_status"] != "finished":
            report["run_status"] = "interrupted_or_error"
        dump(args.output/"report.json", report)
        contact_sheets(args.output, report["episodes"])


if __name__ == "__main__":
    main()
