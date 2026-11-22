"""Provisional command handoff followed by measured RGB-D trial-lift evidence.

No object actor pose, functional point, physics contact, CAD or identity mask is
read here. Camera calibration and robot encoder/kinematic state are still used.
This is a visual heuristic, not a force/tactile grasp certificate.
"""

import os
from copy import deepcopy

import numpy as np

from adapters import pointcloud_registration
from adapters.early_grasp import slower_trajectory
from adapters.handoff import handoff_ready
from adapters.reobservation import observation_quality


class ObjectTruthTerminationGuard:
    """Suppress simulator task-success checks throughout control, restore exactly."""

    def __init__(self, task):
        self.task = task
        self.active = False

    def __enter__(self):
        self.had_instance_method = "check_success" in vars(self.task)
        self.original = self.task.check_success
        self.previous_success = self.task.eval_success
        self.task.check_success = lambda: False
        self.task.eval_success = False
        self.active = True
        return self

    def close(self):
        if not self.active:
            return
        if self.had_instance_method:
            self.task.check_success = self.original
        else:
            del self.task.check_success
        self.task.eval_success = self.previous_success
        self.active = False

    def __exit__(self, *exc):
        self.close()


class ClosedCommandMonitor:
    """Closing command is only permission to TRY lifting, never grasp evidence."""

    def __init__(self, *, stable_steps=3, gripper_threshold=.05):
        if stable_steps < 1 or not np.isfinite(gripper_threshold) or not 0 < gripper_threshold <= .6:
            raise ValueError("Invalid provisional command thresholds")
        self.stable_steps, self.threshold = stable_steps, gripper_threshold
        self.count = 0

    def update(self, gripper_command, step):
        eligible = np.isfinite(gripper_command) and 0 <= gripper_command < self.threshold
        self.count = self.count+1 if eligible else 0
        return dict(ready=self.count >= self.stable_steps,
                    detector="closed_command_provisional_handoff", uses_simulator_truth=False,
                    gripper_command=float(gripper_command), gripper_threshold=self.threshold,
                    gripper_signal_source="robot_control_state_not_measured_aperture_or_force",
                    stable_steps=self.count, required_stable_steps=self.stable_steps,
                    after_vla_step=int(step), lifted_grasp_verified=False)


class ClosedCommandRobotLiftMonitor(ClosedCommandMonitor):
    """Keep VLA control after closure until the closed hand moves upward.

    This is still only a *candidate* handoff: encoder/kinematic motion cannot
    prove that an object is attached.  It prevents the scheduler from
    interrupting on the first close command; RGB-D co-motion must verify the
    grasp afterwards.
    """

    def __init__(self, pose_supplier, *, minimum_upward_m=.03,
                 max_actions_after_closure=80, stable_steps=3,
                 gripper_threshold=.05):
        super().__init__(stable_steps=stable_steps,
                         gripper_threshold=gripper_threshold)
        if (not callable(pose_supplier) or not np.isfinite(minimum_upward_m)
                or minimum_upward_m <= 0 or not isinstance(max_actions_after_closure, int)
                or max_actions_after_closure < 1):
            raise ValueError("Invalid robot-lift handoff configuration")
        self.pose_supplier = pose_supplier
        self.minimum_upward_m = float(minimum_upward_m)
        self.max_actions_after_closure = max_actions_after_closure
        self.closed_pose = None
        self.first_closure_step = None

    def update(self, gripper_command, step):
        decision = super().update(gripper_command, step)
        pose = np.asarray(self.pose_supplier(), dtype=float)
        if pose.shape[0] < 3 or not np.isfinite(pose[:3]).all():
            raise ValueError("Robot-lift handoff requires a finite EE pose")
        if decision["stable_steps"] == 0:
            self.closed_pose = None
        elif decision["ready"] and self.closed_pose is None:
            self.closed_pose = pose[:3].copy()
            if self.first_closure_step is None:
                self.first_closure_step = int(step)
        upward = (0.0 if self.closed_pose is None
                  else float(pose[2] - self.closed_pose[2]))
        decision.update(
            ready=bool(decision["ready"] and upward >= self.minimum_upward_m),
            detector="closed_command_plus_robot_upward_motion_candidate",
            closure_gate_reached=bool(self.closed_pose is not None),
            robot_upward_motion_m=upward,
            required_robot_upward_motion_m=self.minimum_upward_m,
            lifted_grasp_verified=False,
            verification_required="fresh_RGBD_object_hand_comotion",
            first_closure_step=self.first_closure_step,
            max_actions_after_closure=self.max_actions_after_closure,
        )
        if (self.first_closure_step is not None and not decision["ready"]
                and int(step) - self.first_closure_step >= self.max_actions_after_closure):
            raise RuntimeError(
                "VLA closed the gripper but produced no upward hand motion "
                f"within {self.max_actions_after_closure} actions")
        return decision


def visual_comotion(reference_points, after_points, ee_displacement, *, minimum_lift=.03):
    """Rigid objects translate: compare surface centroids directly.

    ICP registration on a handful of points (a 3 cm bottle mostly hidden by the
    gripper) routinely latched onto background and reported 27 cm lateral motion.
    A grasped rigid body only translates, so the centroid difference is both
    simpler and far more stable than a fitted transform.
    """
    points = np.asarray(reference_points, dtype=float)
    after = np.asarray(after_points, dtype=float)
    ee = np.asarray(ee_displacement, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3
            or after.ndim != 2 or after.shape[1:] != (3,) or len(after) < 3
            or ee.shape != (3,)
            or not np.isfinite(points).all() or not np.isfinite(after).all()
            or not np.isfinite(ee).all() or minimum_lift <= 0):
        raise ValueError("Invalid visual co-motion inputs")
    displacement = np.median(after, axis=0) - np.median(points, axis=0)
    distance = float(np.linalg.norm(displacement))
    ee_distance = float(np.linalg.norm(ee))
    ratio = distance/max(ee_distance, 1e-9)
    cosine = float(np.dot(displacement, ee)/max(distance*ee_distance, 1e-9))
    ready = (ee[2] >= minimum_lift and displacement[2] >= minimum_lift
             and .5 <= ratio <= 1.6 and cosine >= .8)
    return dict(ready=bool(ready), observed_surface_displacement_m=displacement.tolist(),
                ee_displacement_m=ee.tolist(), motion_magnitude_ratio=ratio,
                motion_direction_cosine=cosine, minimum_visual_lift_m=minimum_lift,
                ratio_range=[.5, 1.6], minimum_direction_cosine=.8,
                relative_slip_gate_enabled=False, uses_simulator_truth=False)


def verify_visual_trial_lift(task, perception, arm, result, *, lift_m=.05,
                             required_lift_m=.03, registration_config=None):
    if (not np.isfinite(lift_m) or not .03 <= lift_m <= .10
            or not 0 < required_lift_m <= lift_m):
        raise ValueError("Invalid visual trial-lift distances")
    config = registration_config or pointcloud_registration.RegistrationConfig()
    result.update(complete=False, uses_simulator_truth_for_verification=False,
                  uses_object_truth_for_target=False, slip_check_enabled=False,
                  detector="rgbd_registered_surface_comotion", segments=[], observations=[],
                  failure_policy="stop_without_release_or_placement", lift_distance_m=lift_m,
                  robot_region_filtering=False, force_or_tactile_sensing=False)
    names = ["head_camera", f"{arm}_camera"]
    # Save both BEFORE moving; wrist fallback must have its own matching frame.
    frames = {name: deepcopy(task.get_camera_observation(name)) for name in names}
    references = {}

    def observe(frame, label, refined, reference_bounds=None):
        estimate = perception.supplementary(frame, label, cup_only=True, refine_cup=refined)
        quality = observation_quality(estimate, reference_bounds)
        if not quality["geometry_ok"]:
            raise RuntimeError(f"Untrusted trial cup geometry: {quality}")
        return estimate, quality

    # Require credible visual baseline BEFORE applying any trial lift.
    for name in names:
        audit = dict(stage="baseline", camera=name, refined=True)
        result["observations"].append(audit)
        try:
            ref, audit["quality"] = observe(frames[name], f"trial_before_{name}", True)
            references[name] = ref
            break
        except (ValueError, RuntimeError) as error:
            audit["error"] = str(error)
    if not references:
        raise RuntimeError("Visual trial rejected before motion: no credible cup baseline")
    start = np.asarray(task.get_arm_pose(arm), dtype=float)
    for index in range(3):
        target = start.copy()
        target[2] += lift_m*(index+1)/3
        plan = getattr(task.robot, f"{arm}_plan_path")(target)
        audit = dict(target_ee_pose=target.tolist(), status=plan["status"])
        result["segments"].append(audit)
        if plan["status"] != "Success":
            raise RuntimeError("Visual trial lift planning failed")
        controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
        controls[f"{arm}_arm"] = slower_trajectory(plan, 2)
        moved = bool(task.take_dense_action(controls))
        arrival = handoff_ready(task.get_arm_pose(arm), target, moved and task.plan_success)
        audit["arrival"] = arrival.to_dict()
        if not arrival.ready:
            raise RuntimeError("Visual trial lift failed robot arrival check")
    ee_displacement = np.asarray(task.get_arm_pose(arm))[:3]-start[:3]
    # Bounded changed-prompt retry, then matched wrist baseline/current pair.
    for name in names:
        if name not in references:
            audit = dict(stage="fallback_baseline", camera=name, refined=True)
            result["observations"].append(audit)
            try:
                ref, audit["quality"] = observe(frames[name], f"trial_before_{name}", True)
                references[name] = ref
            except (ValueError, RuntimeError) as error:
                audit["error"] = str(error)
                continue
        ref = references[name]
        for refined in (False, True):
            audit = dict(stage="after_lift", camera=name, refined=refined)
            result["observations"].append(audit)
            try:
                after, audit["quality"] = observe(task.get_camera_observation(name),
                    f"trial_after_{name}_{int(refined)}", refined, ref.cup_bounds_world)
                motion, registration = pointcloud_registration.estimate_pointcloud_motion(
                    ref.cup_points_world, after.cup_points_world, config)
                audit["registration"] = registration
                # Registration is still recorded for diagnosis, but the verdict no
                # longer depends on it (it failed on small occluded point clouds).
                decision = visual_comotion(ref.cup_points_world, after.cup_points_world,
                                           ee_displacement, minimum_lift=required_lift_m)
                audit["comotion"] = dict(decision)
                if not decision["ready"]:
                    if os.environ.get("ULA_LAYERED_TRIAL_LIFT") != "1":
                        continue   # stock behaviour: any failed co-motion aborts
                    # Layered policy.  Only a CLEAR failure aborts: the grasped
                    # surface hardly rose at all, or its motion points somewhere
                    # else entirely.  A grasp that co-moves plausibly but misses
                    # the strict cup thresholds (small objects, wide flat objects)
                    # is granted provisionally so the placement can still try;
                    # the placement result then decides the episode.
                    vertical = float(decision["observed_surface_displacement_m"][2])
                    cosine = float(decision["motion_direction_cosine"])
                    if vertical < 0.01 or cosine < 0.5:
                        audit["aborted"] = dict(clear_failure=True, vertical_rise_m=vertical,
                                                direction_cosine=cosine)
                        continue
                    decision.update(ready=True, provisional=True, lifted_grasp_verified=False,
                                    detector="rgbd_registered_surface_comotion",
                                    verification_controller="traditional_visual_trial_lift",
                                    provisional_reason="ambiguous_object_hand_co_motion",
                                    matching_camera=name, ee_pose=list(task.get_arm_pose(arm)))
                    result.update(complete=True, lifted_grasp_verified=False,
                                  provisional_handoff=True, selected_camera=name)
                    return decision
                decision.update(detector="rgbd_registered_surface_comotion",
                    verification_controller="traditional_visual_trial_lift",
                    lifted_grasp_verified=True, matching_camera=name,
                    ee_pose=list(task.get_arm_pose(arm)))
                result.update(complete=True, lifted_grasp_verified=True, selected_camera=name)
                return decision
            except (ValueError, RuntimeError) as error:
                audit["error"] = str(error)
    raise RuntimeError("Visual trial lift did not establish credible cup/hand co-motion")
