#!/usr/bin/env python3
"""Compile one RoboTwin demonstration per task into executable rule skeletons.

Only recorded robot state and task-language metadata are used.  Object-relative
targets are deliberately left as runtime visual anchors because the official
HDF5 demonstrations do not contain object poses.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DEMOS = Path("/data2/xuhuixin/RoboTwin/data_official50/demo_clean")
DEFAULT_DESCRIPTIONS = Path(__file__).resolve().parents[1]/"tra_pipe/RoboTwin/description/task_instruction"


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=serializable)+"\n")


def angle_deg(q1, q2):
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    dot = abs(float(q1 @ q2)/(np.linalg.norm(q1)*np.linalg.norm(q2)))
    return float(np.degrees(2*np.arccos(np.clip(dot, 0., 1.))))


def transition_segments(gripper, direction):
    """Find complete open/close ramps; ignore small preparatory changes."""
    values = np.asarray(gripper, float).reshape(-1)
    delta = np.diff(values)
    indices = np.flatnonzero(delta < -.005 if direction == "close" else delta > .005)+1
    if not len(indices):
        return []
    blocks = np.split(indices, np.flatnonzero(np.diff(indices) > 2)+1)
    events = []
    for block in blocks:
        start, end = int(block[0]-1), int(block[-1])
        amount = float(values[start]-values[end])
        complete = (amount >= .25 and values[end] <= .1) if direction == "close" else (
            amount <= -.25 and values[end] >= .9)
        if complete:
            events.append(dict(kind=direction, start_frame=start, end_frame=end,
                start_value=float(values[start]), end_value=float(values[end])))
    return events


def motion_summary(poses, start, end, frequency):
    poses = np.asarray(poses, float)
    start, end = int(max(0, start)), int(min(len(poses)-1, end))
    if end < start:
        start, end = end, start
    path = poses[start:end+1, :3]
    delta = poses[end, :3]-poses[start, :3]
    return dict(start_frame=start, end_frame=end,
        duration_s=float((end-start)/frequency),
        delta_world_xyz_m=delta.tolist(), displacement_m=float(np.linalg.norm(delta)),
        vertical_delta_m=float(delta[2]),
        path_length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path)>1 else 0.,
        orientation_change_deg=angle_deg(poses[start, 3:], poses[end, 3:]))


def approach_reference(poses, event_frame, preferred=.08, lower=.05, upper=.12):
    poses = np.asarray(poses, float)
    options = []
    for frame in range(int(event_frame)):
        distance = float(np.linalg.norm(poses[frame, :3]-poses[event_frame, :3]))
        if lower <= distance <= upper:
            options.append((abs(distance-preferred), -frame, frame, distance))
    if not options:
        return None
    _, _, frame, distance = min(options)
    return dict(frame=int(frame), world_tcp_pose_xyz_wxyz=poses[frame].tolist(),
        interaction_world_tcp_pose_xyz_wxyz=poses[event_frame].tolist(),
        distance_to_interaction_m=distance,
        orientation_difference_deg=angle_deg(poses[frame, 3:], poses[event_frame, 3:]),
        requires_runtime_visual_anchor=True)


def semantic_family(description):
    text = description.lower()
    if "handover" in text or "hand over" in text or "pass" in text:
        return "handover"
    if "open" in text:
        return "articulated_open"
    if "stack" in text:
        return "stack"
    if any(word in text for word in ("place", "put", "hang", "onto", "on the")):
        return "pick_transport_place"
    if "shake" in text:
        return "grasp_and_oscillate"
    if any(word in text for word in ("press", "click", "switch", "stamp", "scan", "beat")):
        return "contact_manipulation"
    if any(word in text for word in ("rotate", "adjust", "head-up", "upright")):
        return "grasp_and_reorient"
    if any(word in text for word in ("pick", "grab", "lift")):
        return "grasp_and_move"
    if any(word in text for word in ("dump", "pour")):
        return "grasp_and_pour"
    return "trajectory_unspecified"


def first_later(events, kind, frame):
    return next((e for e in events if e["kind"] == kind and e["start_frame"] > frame), None)


def detect_handovers(arms):
    transfers = []
    for receiver in ("left", "right"):
        donor = "right" if receiver == "left" else "left"
        for receive in [e for e in arms[receiver]["events"] if e["kind"] == "close"]:
            prior_donor_closes = [e for e in arms[donor]["events"]
                                  if e["kind"] == "close" and e["end_frame"] < receive["start_frame"]]
            if not prior_donor_closes:
                continue
            donor_close = prior_donor_closes[-1]
            donor_release = first_later(arms[donor]["events"], "open", receive["end_frame"])
            if donor_release is None:
                continue
            receiver_release = first_later(arms[receiver]["events"], "open", receive["end_frame"])
            if receiver_release and receiver_release["start_frame"] <= donor_release["end_frame"]:
                continue
            transfers.append(dict(donor_arm=donor, receiver_arm=receiver,
                donor_grasp=donor_close, receiver_grasp=receive,
                overlap_start_frame=receive["end_frame"],
                donor_release=donor_release,
                overlap_frames=int(donor_release["start_frame"]-receive["end_frame"]),
                rule="receiver_close_then_donor_release"))
    return transfers


def build_phases(arms, handovers, frames, frequency):
    phases = []
    consumed = set()
    for transfer in handovers:
        donor, receiver = transfer["donor_arm"], transfer["receiver_arm"]
        dc, rc, dr = transfer["donor_grasp"], transfer["receiver_grasp"], transfer["donor_release"]
        for arm, event in ((donor, dc), (receiver, rc)):
            key = (arm, event["start_frame"], "close"); consumed.add(key)
            phases.extend([
                dict(controller="traditional", operation="approach_reference", arm=arm,
                     reference=event["approach_reference"]),
                dict(controller="VLA", operation="local_grasp", arm=arm,
                     start_frame=event["start_frame"], end_frame=event["end_frame"]),
            ])
        consumed.add((donor, dr["start_frame"], "open"))
        phases.extend([
            dict(controller="traditional", operation="carry_to_handover_region", arm=donor,
                 motion=motion_summary(arms[donor]["poses"], dc["end_frame"], rc["start_frame"], frequency)),
            dict(controller="VLA", operation="handover_overlap", arms=[donor, receiver],
                 condition="receiver grasp is established before donor opens",
                 start_frame=transfer["overlap_start_frame"], end_frame=dr["start_frame"]),
            dict(controller="VLA", operation="donor_release", arm=donor,
                 start_frame=dr["start_frame"], end_frame=dr["end_frame"]),
        ])
    for arm in ("left", "right"):
        events = arms[arm]["events"]
        for event in events:
            key = (arm, event["start_frame"], event["kind"])
            if key in consumed:
                continue
            if event["kind"] == "close":
                phases.append(dict(controller="traditional", operation="approach_reference", arm=arm,
                                   reference=event["approach_reference"]))
                phases.append(dict(controller="VLA", operation="local_interaction_close", arm=arm,
                    start_frame=event["start_frame"], end_frame=event["end_frame"]))
                following = first_later(events, "open", event["end_frame"])
                end = following["start_frame"] if following else frames-1
                phases.append(dict(controller="traditional", operation="closed_gripper_motion", arm=arm,
                    motion=motion_summary(arms[arm]["poses"], event["end_frame"], end, frequency)))
            else:
                phases.append(dict(controller="VLA", operation="release_or_open", arm=arm,
                    start_frame=event["start_frame"], end_frame=event["end_frame"]))
    def phase_start(phase):
        if "start_frame" in phase:
            return phase["start_frame"]
        if phase.get("motion"):
            return phase["motion"].get("start_frame", -1)
        if phase.get("reference"):
            return phase["reference"].get("frame", -1)
        return -1
    phases.sort(key=phase_start)
    return phases


def compile_task(task_dir, descriptions):
    demo = next(task_dir.glob("*/data/episode_0000000.hdf5"), None)
    if demo is None:
        return dict(task=task_dir.name, status="missing_demo")
    description_path = descriptions/f"{task_dir.name}.json"
    language = json.loads(description_path.read_text()) if description_path.exists() else {}
    description = language.get("full_description", task_dir.name.replace("_", " "))
    with h5py.File(demo) as file:
        frequency = int(file["additional_info/frequency"][()])
        frames = len(file["state/left_ee_poses"])
        arms = {}
        for arm in ("left", "right"):
            poses = np.asarray(file[f"state/{arm}_ee_poses"], float)
            gripper = np.asarray(file[f"state/{arm}_ee_joint_states"], float).reshape(-1)
            events = transition_segments(gripper, "close")+transition_segments(gripper, "open")
            events.sort(key=lambda e: e["start_frame"])
            for event in events:
                event["approach_reference"] = (approach_reference(poses, event["start_frame"])
                                                if event["kind"] == "close" else None)
            arms[arm] = dict(poses=poses, gripper=gripper, events=events,
                initially_closed=bool(gripper[0] <= .1),
                final_closed=bool(gripper[-1] <= .1),
                tcp_path_length_m=float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()))
        family = semantic_family(description)
        overlap_candidates = detect_handovers(arms)
        handovers = overlap_candidates if family == "handover" else []
        phases = build_phases(arms, handovers, frames, frequency)
        active = [a for a in ("left", "right") if arms[a]["tcp_path_length_m"] > .05]
        result = dict(task=task_dir.name, status="compiled", demo=str(demo),
            description=description, schema=language.get("schema", ""),
            rule_family="handover" if handovers else family,
            frames=frames, frequency_hz=frequency, active_arms=active,
            handover_detected=bool(handovers), handovers=handovers, phases=phases,
            non_handover_overlap_candidates=(overlap_candidates if not handovers else []),
            arms={arm: {k: v for k, v in data.items() if k not in ("poses", "gripper")}
                  for arm, data in arms.items()},
            runtime_contract=dict(
                visual_inputs=["task target identity", "current RGB-D target pose or geometric anchor"],
                traditional_control=["move to transformed reference", "execute measured coarse motion rule"],
                VLA_control=["local contact/grasp/release segments"],
                handover_guard=("receiver must establish grasp before donor release"
                                if handovers else None)),
            limitations=[
                "World TCP poses are demonstration evidence, not reusable world targets.",
                "HDF5 contains no object pose; every approach reference requires a runtime visual anchor.",
                "Gripper closure is an event boundary, not proof that an object was securely grasped.",
            ])
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    task_output = args.output/"tasks"
    task_output.mkdir()
    rules = []
    for task_dir in sorted(p for p in args.demos.iterdir() if p.is_dir()):
        rule = compile_task(task_dir, args.descriptions)
        rules.append(rule)
        write_json(task_output/f"{task_dir.name}.json", rule)
    handovers = [r["task"] for r in rules if r.get("handover_detected")]
    families = {name: sum(r.get("rule_family") == name for r in rules)
                for name in sorted({r.get("rule_family") for r in rules})}
    report = dict(scope="one clean demonstration per RoboTwin official task",
        tasks=len(rules), compiled=sum(r["status"] == "compiled" for r in rules),
        handover_tasks=handovers, rule_families=families, rules=rules)
    write_json(args.output/"rules.json", report)
    lines = ["task,status,rule_family,active_arms,handover,phase_count"]
    for rule in rules:
        lines.append(",".join((rule["task"], rule["status"], rule.get("rule_family", ""),
            "+".join(rule.get("active_arms", [])), str(rule.get("handover_detected", False)),
            str(len(rule.get("phases", []))))))
    (args.output/"summary.csv").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(tasks=len(rules), compiled=report["compiled"],
                          handover_tasks=handovers, rule_families=families), ensure_ascii=False))


if __name__ == "__main__":
    main()
