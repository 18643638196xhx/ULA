"""Reuse the just-verified measured surface; independently refresh only the pad.

No scene step/action may occur between verification and using this cache.
An initial-to-current lift consistency test rejects contradictory local ICP.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adapters.handoff import handoff_ready
from adapters.pointcloud_placement import finish_pointcloud_placement
from adapters.pointcloud_registration import estimate_pointcloud_motion


def load_surface(path):
    with np.load(path, allow_pickle=False) as data:
        return SimpleNamespace(cup_points_world=data["cup_points_world"].copy(),
            cup_bounds_world=data["cup_bounds_world"].copy(),
            observation_audit=json.loads(str(data["observation_audit_json"])),
            timings_s=json.loads(str(data["timings_json"])))


def lift_consistency(initial, current, config, *, minimum_lift=.03):
    motion, registration = estimate_pointcloud_motion(initial.cup_points_world, current.cup_points_world, config)
    result = dict(ready=False, registration=registration, uses_object_truth=False,
        minimum_lift_m=minimum_lift, support_anchor_is_approximate=True)
    if motion is None:
        result["reason"] = "initial_to_current_registration_rejected"
        return result
    anchor = np.asarray(initial.cup_bounds_world, dtype=float).mean(0)
    anchor[2] = float(initial.observation_audit["table_height_m"])
    held = motion[:3, :3]@anchor+motion[:3, 3]
    lift = float(held[2]-anchor[2])
    result.update(ready=lift >= minimum_lift, estimated_anchor_lift_m=lift,
        reason="consistent_lift" if lift >= minimum_lift else "current_surface_contradicts_lifted_grasp",
        initial_support_anchor_world=anchor.tolist(), current_support_anchor_world=held.tolist())
    return result


class VerifiedSurfacePerception:
    def __init__(self, task, arm, pad_perception, verification):
        if not verification.get("complete") or not verification.get("verified_surface_path"):
            raise RuntimeError("No verified surface is available for placement")
        self.task, self.arm, self.pad = task, arm, pad_perception
        self.output = pad_perception.output
        self.pose = np.asarray(verification["verified_surface_ee_pose"], dtype=float)
        self.surface = load_surface(verification["verified_surface_path"])
        self.source = verification["verified_surface_path"]

    def estimate(self, rgb, depth, intrinsic, world_to_camera):
        arrival = handoff_ready(self.task.get_arm_pose(self.arm), self.pose, True, .005, 2.)
        if not arrival.ready:
            raise RuntimeError("Verified surface cache invalidated by robot motion")
        target = self.pad.pad_only(dict(rgb=rgb, depth=depth, intrinsic_cv=intrinsic,
                                       extrinsic_cv=world_to_camera))
        if not np.isfinite(target.coaster_world_center).all():
            raise RuntimeError("No current visual pad target")
        return SimpleNamespace(cup_bounds_world=self.surface.cup_bounds_world,
            cup_points_world=self.surface.cup_points_world, coaster_world_center=target.coaster_world_center,
            observation_audit=dict(self.surface.observation_audit,
                measured_surface_source="just_verified_trial_lift_same_robot_pose",
                reused_surface_path=str(self.source), fresh_pad_observation=True),
            timings_s=target.timings_s)


def finish_verified_placement(task, perception, arm, initial_path, verification, result, *,
                              registration_config, release_gap_m=.003):
    cached = VerifiedSurfacePerception(task, arm, perception, verification)
    result["global_lift_consistency"] = lift_consistency(load_surface(initial_path),
        cached.surface, registration_config)
    if not result["global_lift_consistency"]["ready"]:
        raise RuntimeError("Placement rejected: current measured surface does not confirm a lifted bottle")
    result["object_surface_source"] = "verified_trial_lift_cache_no_intervening_motion"
    finish_pointcloud_placement(task, cached, arm, initial_path, result,
        registration_config=registration_config, release_gap_m=release_gap_m, reobserve=False)
