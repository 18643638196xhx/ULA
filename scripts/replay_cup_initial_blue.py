#!/usr/bin/env python3
"""Same frozen initial frames, blue-cup caption, no simulator or task actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.cup_initial_blue_prompt import BlueInitialCupWorker
from adapters.cup_initial_audit import mask_metrics, overlay, contact_sheets


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist())+"\n")


def counts(records):
    available = [e for e in records if e["source_frame_available"]]
    summary = dict(source_cases=len(records), available_frames=len(available),
        source_unavailable_seeds=[e["seed"] for e in records if not e["source_frame_available"]],
        evaluated_frames=sum("evaluation_only" in e for e in available),
        both_identity_matches=sum(all(e.get("evaluation_only", {}).get(k, {}).get("identity_match", False)
                                      for k in ("cup", "pad")) for e in available))
    for name in ("cup", "pad"):
        summary[name+"_outputs"] = sum(e.get(name+"_output", False) for e in available)
        summary[name+"_identity_matches"] = sum(e.get("evaluation_only", {}).get(name, {}).get("identity_match", False)
                                                for e in available)
        summary[name+"_wrong_seeds"] = [e["seed"] for e in available if e.get(name+"_output", False)
            and name in e.get("evaluation_only", {}) and not e["evaluation_only"][name]["identity_match"]]
        summary[name+"_missing_seeds"] = [e["seed"] for e in available
                                          if not e.get(name+"_output", False)]
    return summary


def fingerprints(source_report):
    paths = set(source_report["sources_before"])
    paths.update(("adapters/cup_initial_blue_prompt.py", "scripts/perceive_cup_initial_blue.py",
                  "scripts/replay_cup_initial_blue.py"))
    return {name: digest(ROOT/name) for name in sorted(paths)}


def run_frame(worker, source, output, original, index):
    seed = original["seed"]
    directory = output/f"episode_{index:03d}_seed{seed}"
    directory.mkdir()
    record = dict(seed=seed, source_frame_available=bool(original["initialized"]),
                  task_actions_executed=0, vla_loaded=False, simulator_started=False,
                  task_success_evaluated=False, error=None)
    if not record["source_frame_available"]:
        record.update(failure_stage="source_unavailable", source_setup_error=original["error"])
        dump(directory/"result.json", record)
        return record
    src = source/f"episode_{index:03d}_seed{seed}"
    request, result = directory/"rgbd.npz", directory/"prediction.npz"
    shutil.copy2(src/"rgbd.npz", request)
    shutil.copy2(src/"initial_head_camera.png", directory/"initial_head_camera.png")
    record["source_input_sha256"] = digest(src/"rgbd.npz")
    record["copied_input_sha256"] = digest(request)
    if record["source_input_sha256"] != record["copied_input_sha256"]:
        raise RuntimeError("Paired comparison requires byte-identical RGB-D input")
    try:
        with np.load(request, allow_pickle=False) as frame:
            if set(frame.files) != {"rgb", "depth", "intrinsic", "world_to_camera"}:
                raise ValueError("Frozen input must contain RGB-D and calibration only")
            rgb = frame["rgb"].copy()
        start = time.perf_counter()
        estimate = worker.estimate_files(request, result)
        record.update(perception_elapsed_s=time.perf_counter()-start,
                      cup_output=estimate.observation_audit["cup_accepted"],
                      pad_output=estimate.observation_audit["pad_accepted"],
                      perception_audit=estimate.observation_audit, timings_s=estimate.timings_s,
                      cup_bounds_world=estimate.cup_bounds_world,
                      pad_center_world=estimate.coaster_world_center)
        with np.load(result, allow_pickle=False) as data:
            cup, pad = data["cup_mask"].copy(), data["pad_mask"].copy()
        overlay(rgb, cup, pad, f"seed {seed} | blue cup caption", directory/"overlay_head_camera.png")
        dump(directory/"prediction.json", record)
        # Independent OLD labels are opened only after this NEW prediction is frozen.
        with np.load(src/"evaluation_only_masks.npz", allow_pickle=False) as data:
            truth_cup, truth_pad = data["cup"].copy(), data["pad"].copy()
        record["evaluation_only"] = dict(source="same_frozen_frame_labels_after_new_prediction",
            feeds_back_into_perception=False, cup=mask_metrics(cup, truth_cup),
            pad=mask_metrics(pad, truth_pad), identity_threshold="visible_mask_IoU_at_least_0.5_not_task_success")
        np.savez_compressed(directory/"evaluation_only_masks.npz", cup=truth_cup, pad=truth_pad)
        def label(name):
            if not record[name+"_output"]:
                return "missing"
            return "match" if record["evaluation_only"][name]["identity_match"] else "wrong"
        overlay(rgb, cup, pad, f"seed {seed} | cup:{label('cup')} pad:{label('pad')}",
                directory/"audit_head_camera.png", truth_masks=(truth_cup, truth_pad))
        with np.load(src/"prediction.npz", allow_pickle=False) as data:
            record["paired_pad_mask_identical"] = bool(np.array_equal(pad, data["pad_mask"]))
    except Exception as error:
        record.update(error=f"{type(error).__name__}: {error}", failure_stage="replay")
    dump(directory/"result.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.input, args.output = args.input.resolve(), args.output.resolve()
    source = json.loads((args.input/"report.json").read_text())
    if source["run_status"] != "finished" or not source["task_success_not_evaluated"]:
        parser.error("Require a completed initial-only source batch")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        parser.error("Expose one explicitly selected shared GPU")
    args.output.mkdir(parents=True, exist_ok=True)
    before = fingerprints(source)
    worker = BlueInitialCupWorker(ROOT, args.output/"vision_worker")
    report = dict(task="place_empty_cup", scope="paired_initial_rgbd_replay_only",
        run_status="running", baseline_directory=str(args.input),
        planned_source_cases=len(source["episodes"]), cup_semantic_caption="blue cup .",
        baseline_cup_semantic_caption="cup . mug .", changed_component="cup_semantic_caption_only",
        color_is_hard_gate=False, no_generic_caption_fallback=True,
        same_rgbd_inputs=True, simulator_started=False, task_actions_enabled=False,
        task_success_not_evaluated=True, uses_vla=False, evaluation_truth_is_post_prediction_only=True,
        baseline_counts=source["counts"], shared_gpu=gpu, sources_before=before, episodes=[])
    try:
        for index, original in enumerate(source["episodes"]):
            record = run_frame(worker, args.input, args.output, original, index)
            report["episodes"].append(record)
            report.update(counts=counts(report["episodes"]), completed_source_cases=len(report["episodes"]),
                          perception_worker=worker.summary())
            dump(args.output/"report.json", report)
            print(f"[cup-blue-initial] seed={record['seed']} cup={record.get('cup_output')} "
                  f"pad={record.get('pad_output')} error={record['error']}", flush=True)
        report["run_status"] = "finished"
    finally:
        worker.close()
        after = fingerprints(source)
        report.update(counts=counts(report["episodes"]), completed_source_cases=len(report["episodes"]),
                      sources_after=after, sources_unchanged=before == after,
                      perception_worker=worker.summary())
        if report["run_status"] != "finished":
            report["run_status"] = "interrupted_or_error"
        dump(args.output/"report.json", report)
        contact_sheets(args.output, report["episodes"])
    return int(any(e.get("error") for e in report["episodes"]))


if __name__ == "__main__":
    raise SystemExit(main())
