#!/usr/bin/env python3
"""Read-only statistics and paired approach-time comparison for ranked ULA."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


def average(values):
    values = list(values)
    return mean(values) if values else None


def valid_episodes(report):
    return [e for e in report["episodes"] if e.get("failure_stage") != "setup"]


def phase(e, name):
    return e.get("timing_profile", {}).get("stages", {}).get(name, {})


def summarize(report, previous=None):
    valid = valid_episodes(report)
    warm = [e for e in valid[1:] if e["success"]]
    approached = [e for e in valid[1:] if e["approach"].get("complete")]
    result = dict(run_status=report.get("run_status"), completed_valid_episodes=len(valid),
        raw_attempts=len(report["episodes"]),
        initialization_rejection_seeds=[e["seed"] for e in report["episodes"] if e.get("failure_stage") == "setup"],
        successes=sum(bool(e["success"]) for e in valid),
        success_rate=sum(bool(e["success"]) for e in valid)/len(valid) if valid else None,
        official_successes=sum(bool(e.get("task_success")) for e in valid),
        official_success_rate=sum(bool(e.get("task_success")) for e in valid)/len(valid) if valid else None,
        warm_successful_samples=len(warm), warm_mean_completion_s=average(e["elapsed_s"] for e in warm),
        warm_success_stage_means={key: average(phase(e, key).get("wall_s", 0) for e in warm)
            for key in sorted({k for e in warm for k in e["timing_profile"]["stages"]})},
        warm_success_approach_components={key: average(phase(e, "traditional_approach").get("components_s", {}).get(key, 0) for e in warm)
            for key in sorted({k for e in warm for k in phase(e, "traditional_approach").get("components_s", {})})},
        failures=[dict(seed=e["seed"], stage=e.get("failure_stage"), error=e.get("error"),
                       official_success=e.get("task_success"),
                       last_angle=e["traditional_opening"].get("last_visual_angle_deg"))
                  for e in valid if not e["success"]],
        selected_pitches=dict(Counter(str(e["approach"].get("approach_pitch_deg")) for e in approached)),
        selected_budgets=dict(Counter(e["approach"].get("selected_budget") for e in approached)),
        mean_standoff_search_calls=average(len(e["approach"].get("search_attempts", [])) for e in approached),
        sources_unchanged=report.get("randomized_sources_unchanged"),
        placement_unchanged=report.get("placement_files_unchanged_during_run"),
        perception_worker=report.get("perception_worker"))
    if previous is not None:
        old = {e["seed"]: e for e in valid_episodes(previous)}
        common = [e for e in valid if e["seed"] in old]
        first = valid[0]["seed"] if valid else None
        paired = [e for e in common if e["seed"] != first and
                  e["approach"].get("complete") and old[e["seed"]]["approach"].get("complete")]
        result["comparison"] = dict(common_valid_scenes=len(common),
            previous_successes=sum(bool(old[e["seed"]]["success"]) for e in common),
            current_successes=sum(bool(e["success"]) for e in common),
            newly_successful=[e["seed"] for e in common if e["success"] and not old[e["seed"]]["success"]],
            newly_failed=[e["seed"] for e in common if not e["success"] and old[e["seed"]]["success"]],
            paired_approach_samples=len(paired),
            paired_old_approach_s=average(phase(old[e["seed"]], "traditional_approach")["wall_s"] for e in paired),
            paired_new_approach_s=average(phase(e, "traditional_approach")["wall_s"] for e in paired),
            paired_old_motion_planning_s=average(phase(old[e["seed"]], "traditional_approach")["components_s"].get("motion_planning", 0) for e in paired),
            paired_new_motion_planning_s=average(phase(e, "traditional_approach")["components_s"].get("motion_planning", 0) for e in paired),
            paired_new_ik_s=average(phase(e, "traditional_approach")["components_s"].get("ik_candidate_screening", 0) for e in paired),
            caveat="same_scenes_not_identical_actions_or_isolated_hardware")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    previous = json.loads(args.previous.read_text()) if args.previous else None
    print(json.dumps(summarize(report, previous), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
