#!/usr/bin/env python3
"""Read-only aggregation of the finished direct-opening experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


def summarize(report, old=None):
    valid = [e for e in report["episodes"] if e.get("failure_stage") != "setup"]
    rejected = [e["seed"] for e in report["episodes"] if e.get("failure_stage") == "setup"]
    successful = [e for e in valid[1:] if e["success"]]
    result = dict(
        run_status=report.get("run_status"), valid_episodes=len(valid),
        raw_attempts=len(report["episodes"]), initialization_rejection_seeds=rejected,
        complete_successes=sum(bool(e["success"]) for e in valid),
        complete_success_rate=sum(bool(e["success"]) for e in valid)/len(valid) if valid else None,
        official_successes=sum(bool(e.get("task_success")) for e in valid),
        official_success_rate=sum(bool(e.get("task_success")) for e in valid)/len(valid) if valid else None,
        first_valid_seed=valid[0]["seed"] if valid else None,
        warm_successful_samples=len(successful),
        warm_mean_completion_s=mean(e["elapsed_s"] for e in successful) if successful else None,
        failures=[dict(seed=e["seed"], stage=e.get("failure_stage"), error=e.get("error"),
                       official_success=e.get("task_success")) for e in valid if not e["success"]],
        failure_stages=dict(Counter(e.get("failure_stage") or "final_official_score"
                                   for e in valid if not e["success"])),
        worker=report.get("perception_worker"),
        sources_unchanged=report.get("randomized_sources_unchanged"),
        placement_unchanged=report.get("placement_files_unchanged_during_run"),
        algorithm_changed=False, interpretation="repeat_not_no_trial_ablation")
    if old is not None:
        previous = {e["seed"]: e for e in old["episodes"] if e.get("failure_stage") != "setup"}
        common = [e for e in valid if e["seed"] in previous]
        result["common_scene_comparison"] = dict(
            valid_scenes=len(common), old_successes=sum(bool(previous[e["seed"]]["success"]) for e in common),
            new_successes=sum(bool(e["success"]) for e in common),
            newly_successful=[e["seed"] for e in common if e["success"] and not previous[e["seed"]]["success"]],
            newly_failed=[e["seed"] for e in common if not e["success"] and previous[e["seed"]]["success"]],
            caution="same_scenes_not_exact_paired_actions_or_isolated_hardware")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    old = json.loads(args.previous.read_text()) if args.previous else None
    print(json.dumps(summarize(report, old), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
