"""Unambiguous episode-level success/time statistics for pilot experiments."""

from statistics import mean


def summarize_episodes(episodes):
    def summary(records):
        successes = [record for record in records if record["success"]]
        return dict(
            episodes=len(records), successes=len(successes),
            success_rate=len(successes) / len(records) if records else None,
            mean_attempt_elapsed_s=mean(record["elapsed_s"] for record in records) if records else None,
            mean_successful_completion_s=mean(record["elapsed_s"] for record in successes) if successes else None,
            successful_seeds=[record["seed"] for record in successes],
            failed_seeds=[record["seed"] for record in records if not record["success"]],
        )

    return dict(
        all_episodes=summary(episodes),
        excluding_first_episode=summary(episodes[1:]),
        timing_definition="elapsed_s: episode setup through final evaluation/snapshots; "
                          "includes lazy vision worker startup once per batch (first vision episode). "
                          "Completion mean includes successful episodes only; attempt mean includes failures.",
    )
