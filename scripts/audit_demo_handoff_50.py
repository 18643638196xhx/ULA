"""Audit whether one demo per RoboTwin task supports a shared handoff envelope.

This is an offline trajectory audit only. It never reads simulator actor poses,
plans motion, or claims object-relative geometry from RGB-only demonstrations.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DEMOS = Path("/data2/xuhuixin/RoboTwin/data_official50/demo_clean")


def closure_segments(gripper):
    """Return final-closing ramps, excluding preparatory partial closure."""
    g = np.asarray(gripper, dtype=float).reshape(-1)
    if not np.isfinite(g).all() or len(g) < 4:
        raise ValueError("Invalid gripper series")
    falling = np.flatnonzero(np.diff(g) < -0.005) + 1
    if not len(falling):
        return []
    blocks = np.split(falling, np.flatnonzero(np.diff(falling) > 2) + 1)
    result = []
    for block in blocks:
        start, end = int(block[0] - 1), int(block[-1])
        if g[start] - g[end] >= 0.25 and g[end] <= 0.1:
            result.append((start, end))
    return result


def angle_deg(q1, q2):
    q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
    norms = np.linalg.norm(q1) * np.linalg.norm(q2)
    if norms < 1e-12:
        raise ValueError("Invalid pose quaternion")
    dot = abs(float(np.dot(q1, q2) / norms))
    return float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))))


def candidate_before_event(poses, event, *, preferred=.08, lower=.05, upper=.12):
    """Find a measured pose in the last approach that is 5–12 cm from the event TCP."""
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not 0 < event < len(poses):
        raise ValueError("Expected valid [N,7] world-frame TCP poses and event index")
    reference = poses[event]
    options = []
    for frame in range(event):
        distance = float(np.linalg.norm(poses[frame, :3] - reference[:3]))
        if lower <= distance <= upper:
            angle = angle_deg(poses[frame, 3:], reference[3:])
            options.append((abs(distance - preferred), -frame, frame, distance, angle))
    if not options:
        return None
    _, _, frame, distance, angle = min(options)
    return dict(frame=frame, distance_to_event_tcp_m=distance,
                orientation_difference_deg=angle,
                world_tcp_pose_xyz_wxyz=poses[frame].astype(float).tolist())


def audit_task(task_dir):
    demo = next(task_dir.glob("*/data/episode_0000000.hdf5"), None)
    if demo is None:
        return dict(task=task_dir.name, status="missing_demo")
    with h5py.File(demo) as file:
        frequency = int(file["additional_info/frequency"][()])
        arms = {}
        for arm in ("left", "right"):
            g = np.asarray(file[f"state/{arm}_ee_joint_states"]).reshape(-1)
            poses = np.asarray(file[f"state/{arm}_ee_poses"])
            segments = closure_segments(g)
            events = []
            for start, end in segments:
                candidate = candidate_before_event(poses, start) if start else None
                events.append(dict(closure_start_frame=start, closure_end_frame=end,
                    closure_start_s=start/frequency, closure_end_s=end/frequency,
                    gripper_start=float(g[start]), gripper_end=float(g[end]),
                    handoff_candidate=candidate))
            arms[arm] = dict(closure_events=events, tcp_path_length_m=float(
                np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()))
        events = [(v["closure_start_frame"], arm, v) for arm, data in arms.items()
                  for v in data["closure_events"]]
        events.sort()
        first_frame = events[0][0] if events else None
        first_events = [dict(arm=arm, **event) for frame, arm, event in events
                        if frame == first_frame]
        usable = bool(first_events and all(e["handoff_candidate"] is not None for e in first_events))
        return dict(task=task_dir.name, demo=str(demo), frames=len(file["state/left_ee_poses"]),
                    frequency_hz=frequency, arms=arms, first_closure_events=first_events,
                    first_event_handoff_extractable=usable,
                    status="candidate_extracted" if usable else
                    "no_final_closure" if not first_events else "no_5_to_12cm_candidate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [audit_task(task) for task in sorted(args.demos.iterdir()) if task.is_dir()]
    by_status = {name: sum(row["status"] == name for row in rows)
                 for name in sorted({row["status"] for row in rows})}
    report = dict(scope="one_official_clean_demo_per_task_offline_robot_state_only",
        envelope=dict(tcp_distance_m=[.05, .12], preferred_m=.08,
                      actual_robot_to_demo_handoff_orientation_tolerance_deg=15,
                      demo_handoff_to_interaction_orientation_is_not_limited=True),
        tasks=len(rows), by_status=by_status,
        candidate_tasks=[r["task"] for r in rows if r["first_event_handoff_extractable"]],
        limitations=["HDF5 has no depth or object pose: no object-relative handoff target",
                     "Recorded TCP proximity does not prove collision clearance or motion-planning reachability",
                     "First final gripper closure may be a control preparation rather than object interaction"],
        rows=rows)
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    lines = ["task,status,arm,event_frame,handoff_frame,distance_m,angle_deg"]
    for row in rows:
        for event in row.get("first_closure_events", [None]):
            event = event or {}
            candidate = event.get("handoff_candidate") or {}
            lines.append(",".join(str(x) for x in (row["task"], row["status"],
                event.get("arm", ""), event.get("closure_start_frame", ""),
                candidate.get("frame", ""), candidate.get("distance_to_event_tcp_m", ""),
                candidate.get("orientation_difference_deg", ""))))
    (args.output/"summary.csv").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(tasks=len(rows), by_status=by_status), ensure_ascii=False))


if __name__ == "__main__":
    main()
