#!/usr/bin/env python3
"""Existing panel-tracking ULA, direct normal opening, N initialized scenes.

There is no extra lift/probe in this microwave controller. Its first ordinary
3-degree opening increment historically has the audit label ``trial_pull``.
We keep that controller, feedback, and thresholds unchanged; this is a fresh
evaluation, NOT a no-feedback or no-trial ablation of a different algorithm.
"""
from collections import Counter
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "ula_microwave_direct_panel", ROOT/"scripts/run_open_microwave_panel_randomized.py")
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)
base = panel.randomized.base


def progress(records):
    valid = [e for e in records if e.get("failure_stage") != "setup"]
    rejected = [e for e in records if e.get("failure_stage") == "setup"]
    return dict(
        raw_seed_attempts=len(records), completed_valid_episodes=len(valid),
        initialization_rejections=len(rejected),
        initialization_rejection_seeds=[e["seed"] for e in rejected],
        successes=sum(bool(e["success"]) for e in valid),
        success_rate=sum(bool(e["success"]) for e in valid)/len(valid) if valid else None,
        official_successes=sum(bool(e.get("task_success")) for e in valid),
        official_success_rate=sum(bool(e.get("task_success")) for e in valid)/len(valid) if valid else None,
        failure_stages=dict(Counter(e.get("failure_stage") or "final_official_score"
                                   for e in valid if not e["success"])),
        statistics=base.summarize_episodes(valid),
        timing_statistics=base.summarize_timings(valid))


def run_valid(args):
    """Called inside the original runtime/config/truth guards and task cwd."""
    from envs.open_microwave import open_microwave

    before = base.protected_hashes()
    policy = base.WorldLA(ROOT, args.model_dir, args.asset_checkpoint,
                          blend=args.blend, inference_seed=args.inference_seed)
    worker = base.MicrowaveWorker(ROOT, args.output/"vision_worker",
        door_prompt=args.door_prompt, handle_prompt=args.handle_prompt,
        vision_gpu=args.vision_gpu,
        torch_fraction=float(os.environ.get("MICROWAVE_VISION_MEMORY_FRACTION", ".10")))
    config, task = base.task_config(), open_microwave()
    report = dict(task="open_microwave", mode=args.mode, configuration=vars(args), episodes=[],
        sequence="traditional_approach -> vla_close_candidate -> normal_incremental_traditional_opening",
        requested_valid_episodes=args.episodes, denominator="successfully_initialized_scenes",
        setup_rejections_replaced_by_next_consecutive_seed=True,
        seed_filtering=False, evaluation_valid_sample_size_stop=True,
        vla_used=True, closure_controller="vla", extra_trial_lift_executed=False,
        extra_trial_pull_executed=False, first_opening_increment_is_normal_task_motion=True,
        historical_first_increment_audit_label="trial_pull",
        opening_feedback_retained=True, controller_and_thresholds_changed=False,
        comparison_interpretation="fresh_repeat_of_existing_panel_controller_not_no_trial_ablation",
        control_uses_object_truth=False, runtime_truth_success_disabled=True,
        policy_oracle_diagnostic_truth_disabled=True, evaluation_uses_object_truth=True,
        success_definition="controller_completed_and_official_task_score_passed",
        protected_files_before=before, resource_policy=base.resource_policy())
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        base.dump(args.output/"report.json", dict(report, **progress([])))
        max_attempts = max(200, args.episodes*5)
        while progress(report["episodes"])["completed_valid_episodes"] < args.episodes:
            index = len(report["episodes"])
            if index >= max_attempts:
                raise RuntimeError("Too many rejected initializations to fill valid-scene target")
            record = base.run_episode(task, policy, worker, args, config, index)
            report["episodes"].append(record)
            report.update(progress(report["episodes"]), perception_worker=worker.summary())
            base.dump(args.output/"report.json", report)
            print(f"[ula-microwave-direct] valid={report['completed_valid_episodes']}/{args.episodes} "
                  f"seed={record['seed']} success={record['success']} "
                  f"stage={record['failure_stage']} error={record['error']}", flush=True)
            error = (record.get("error") or "").lower()
            if any(word in error for word in ("out of memory", "resource_exhausted", "cuda error",
                                               "vulkan", "incompatible driver")):
                raise RuntimeError("Infrastructure failure; stop rather than publish as method failure")
    finally:
        worker.close()
        report.update(progress(report["episodes"]), perception_worker=worker.summary(),
                      protected_files_after=base.protected_hashes())
        report["placement_files_unchanged_during_run"] = before == report["protected_files_after"]
        base.dump(args.output/"report.json", report)
    return int(any(not e["success"] for e in report["episodes"] if e.get("failure_stage") != "setup"))


def main(argv=None):
    original_run = base._run_in_robotwin
    original_fingerprints = panel.randomized.fingerprints

    def fingerprints():
        hashes = original_fingerprints()
        for name in ("scripts/run_open_microwave_direct_randomized.py",
                     "scripts/run_open_microwave_direct_randomized.sh",
                     "scripts/microwave_direct_python.sh"):
            hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        return hashes

    base._run_in_robotwin = run_valid
    panel.randomized.fingerprints = fingerprints
    try:
        return panel.main(argv)
    finally:
        base._run_in_robotwin = original_run
        panel.randomized.fingerprints = original_fingerprints


if __name__ == "__main__":
    raise SystemExit(main())
