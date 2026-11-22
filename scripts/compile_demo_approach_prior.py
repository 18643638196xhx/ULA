#!/usr/bin/env python3
"""Compile a robot-only RoboTwin demo into a visual-anchor approach prior.

Official demonstrations contain EE trajectories but no object pose/depth.  The
result therefore preserves the demonstrated interaction-to-approach direction
and EE orientation relative to a runtime visual bounds-top anchor.  It does not
claim to be a full object-frame transform.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def pose_matrix(value):
    value = np.asarray(value, float)
    quaternion = value[3:]
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    result = np.eye(4)
    result[:3, :3] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    result[:3, 3] = value[:3]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rule = json.loads(args.compiled_rule.read_text(encoding="utf-8"))
    phase = next((item for item in rule.get("phases", [])
                  if item.get("operation") == "approach_reference" and item.get("reference")), None)
    if phase is None:
        raise RuntimeError("Compiled rule has no approach reference")
    reference = phase["reference"]
    approach = pose_matrix(reference["world_tcp_pose_xyz_wxyz"])
    interaction = pose_matrix(reference["interaction_world_tcp_pose_xyz_wxyz"])
    value = dict(
        schema="ula_demo_relative_approach_v1", task=rule["task"],
        mode="visual_bounds_top_anchor", arm=phase["arm"],
        interaction_to_approach_world_m=(approach[:3, 3]-interaction[:3, 3]).tolist(),
        approach_ee_rotation_world=approach[:3, :3].tolist(),
        provenance=dict(
            demonstration=rule["demo"], approach_frame=reference["frame"],
            interaction_distance_m=reference["distance_to_interaction_m"],
            source_fields="recorded_robot_EE_states_only",
            simulator_object_pose_used=False, demonstration_depth_used=False,
        ),
        runtime_contract=dict(
            anchor="current_RGBD_visible_bounds_top_center",
            exact_candidate_first=True,
            neighbours="current_RGBD_point_spacing",
            geometry_bounds_candidates="fallback_only",
        ),
        limitation=("Official HDF5 has no object pose; this is a demonstrated approach "
                    "direction relative to a visual anchor, not a full object-frame pose."),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
