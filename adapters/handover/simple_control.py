"""Minimal five-stage ULA: command-only handoffs, assumed rigid attachments.

No visual grasp confirmation, contact query, trial lift, slip check, simulator
object pose or postgrasp object localization is used by this controller.
"""
from types import SimpleNamespace

import numpy as np

from adapters.grasp import GraspComplete
from adapters.handoff import handoff_ready
from adapters.visual_grasp import ClosedCommandMonitor
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix, matrix_to_pose_wxyz


def block_metadata():
    # Explicit primitive geometry/contact conventions from the public task
    # definition, not from the live simulator actor or its current pose.
    rotations = [
        [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
        [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    ]
    contacts = []
    for height in (.7, -.7):
        for rotation in rotations:
            matrix = np.eye(4)
            matrix[:3, :3], matrix[2, 3] = rotation, height
            contacts.append(matrix.tolist())
    functions = []
    for height in (-1., 1.):
        matrix = np.diag([1., -1., -1., 1.])
        matrix[2, 3] = height
        functions.append(matrix.tolist())
    return dict(center=[0, 0, 0], extents=[.03, .03, .1], scale=[.03, .03, .1],
        contact_points_pose=contacts, contact_points_group=[list(range(8))],
        contact_points_mask=[True], functional_matrix=functions,
        transform_matrix=np.eye(4).tolist())


def assumed_attachment(ee_world, object_world):
    return np.linalg.solve(np.asarray(ee_world), np.asarray(object_world))


def propagated_object(task, arm, attachment):
    return pose_wxyz_to_matrix(task.get_arm_pose(arm))@attachment


def approach(task, visual, result, *, pregrasp_distance=.07):
    from envs.utils import ArmTag
    from traditional_place_empty_cup.controller import EstimatedActor
    arm = ArmTag("left")
    if not task.move(task.close_gripper(arm, pos=.6)):
        raise RuntimeError("Traditional left gripper preparation failed")
    actor = EstimatedActor(visual.cup_world_pose, block_metadata())
    action_arm, actions = task.grasp_actor(actor, arm, pre_grasp_dis=pregrasp_distance,
        grasp_dis=0., contact_point_id=[0, 1, 2, 3])
    if not actions or actions[0].action != "move":
        raise RuntimeError("No visual pre-grasp action generated")
    target = np.asarray(actions[0].target_pose)
    moved = bool(task.move((action_arm, [actions[0]])))
    decision = handoff_ready(task.get_arm_pose(arm), target, moved and task.plan_success)
    result.update(decision.to_dict(), arm="left", target_ee_pose=target.tolist(),
        actual_ee_pose=list(task.get_arm_pose(arm)), object_pose_source="initial_RGBD",
        executes_final_grasp=False, uses_object_truth=False)
    if not decision.ready:
        raise RuntimeError(f"Traditional pre-grasp not reached: {decision.reason}")


def rollout_phase(task, policy, phase, result, *, left_attachment=None, stable_steps=3):
    """Only robot gripper control states authorize stage changes, not grasp proof."""
    if phase not in ("left_grasp", "handover"):
        raise ValueError("Unknown VLA phase")
    if phase == "handover" and left_attachment is None:
        raise ValueError("Handover needs the assumed left-hand attachment")
    monitor = ClosedCommandMonitor(stable_steps=stable_steps, gripper_threshold=.05)
    original = task.take_action
    right_attachment = None
    release_count = 0
    result.update(phase=phase, ready=False, start_vla_step=int(task.take_action_cnt),
        total_budget=int(task.step_lim), handoff_verified=False, uses_object_truth=False,
        uses_contacts=False, uses_postgrasp_vision=False, trial_lift=False,
        signal_source="robot_control_state_not_measured_force_or_object_motion", commands=[])

    def action(values, *args, **kwargs):
        nonlocal right_attachment, release_count
        values = np.asarray(values)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("Expected finite dual-arm 14-D action")
        original(values, *args, **kwargs)
        left = float(task.robot.get_left_gripper_val())
        right = float(task.robot.get_right_gripper_val())
        decision = monitor.update(left if phase == "left_grasp" else right, task.take_action_cnt)
        result["commands"].append(dict(step=int(task.take_action_cnt), left=left, right=right))
        if phase == "left_grasp":
            ready = decision["ready"]
        else:
            if not 0 <= right < .05:
                right_attachment, release_count = None, 0
            if decision["ready"] and right_attachment is None:
                predicted = propagated_object(task, "left", left_attachment)
                right_attachment = assumed_attachment(pose_wxyz_to_matrix(task.get_arm_pose("right")), predicted)
                result["right_closed_candidate_step"] = int(task.take_action_cnt)
                result["right_attachment_source"] = "left_rigid_motion_hypothesis_at_right_close_not_observed_grasp"
            release_count = release_count+1 if decision["ready"] and left > .5 else 0
            ready = release_count >= stable_steps
        if ready:
            result.update(ready=True, after_vla_step=int(task.take_action_cnt),
                gripper_decision=decision, left_release_command_steps=release_count,
                left_ee_pose=list(task.get_arm_pose("left")), right_ee_pose=list(task.get_arm_pose("right")))
            if right_attachment is not None:
                result["assumed_right_ee_to_block"] = right_attachment.tolist()
            raise GraspComplete()
    task.take_action = action
    try:
        policy.reset()  # traditional intervention invalidates the cached action tail
        result["policy_cache_reset_at_phase_start"] = True
        while task.take_action_cnt < task.step_lim:
            before = task.take_action_cnt
            policy.step(task, task.get_obs())
            print(f"[handover-ULA] {phase} step={task.take_action_cnt}/{task.step_lim}", flush=True)
            if task.take_action_cnt == before:
                raise RuntimeError("VLA returned without executing an action")
    except GraspComplete:
        print(f"[handover-ULA] {phase} command handoff step={task.take_action_cnt}", flush=True)
    finally:
        task.take_action = original
        result.update(end_vla_step=int(task.take_action_cnt),
                      vla_steps=int(task.take_action_cnt)-result["start_vla_step"])
    if not result["ready"]:
        raise RuntimeError(f"No {phase} command transition within shared VLA budget")
    return right_attachment


def carry_to_middle(task, left_attachment, result):
    from envs.utils import ArmTag, Action
    arm = ArmTag("left")
    result.update(complete=False, actions=[], uses_object_truth=False,
        object_motion_source="assumed_rigid_attachment_to_left_hand", trial_lift=False,
        lift_purpose="normal_transport_not_grasp_verification",
        middle_functional_target_world=[0., 0., .9, 0., 1., 0., 0.])
    moved = bool(task.move(task.move_by_displacement(arm, z=.1)))
    result["actions"].append(dict(action="normal_transport_lift_10cm", success=moved))
    if not moved or not task.plan_success:
        raise RuntimeError("Traditional transport lift failed")
    # Official target is the BOTTOM functional pose, not the block center.
    # Its rotation diag(1,-1,-1) cancels the local functional frame rotation.
    desired_block = np.eye(4)
    desired_block[:3, 3] = [0., 0., 1.]
    target = matrix_to_pose_wxyz(desired_block@np.linalg.inv(left_attachment))
    result["target_ee_pose"] = target.tolist()
    moved = bool(task.move((arm, [Action(arm, "move", target_pose=target)])))
    result["actions"].append(dict(action="move_to_middle", success=moved))
    if not moved or not task.plan_success:
        raise RuntimeError("Traditional move to handover pose failed")
    result.update(complete=True, actual_ee_pose=list(task.get_arm_pose("left")),
        predicted_block_pose=propagated_object(task, "left", left_attachment).tolist())


class RigidAttachmentPerception:
    """State propagation for the placement API, explicitly NOT a fresh observation."""
    def __init__(self, task, attachment, platform):
        self.task, self.attachment = task, attachment
        self.platform = np.asarray(platform).copy()

    def estimate(self, *unused, **kwargs):
        return SimpleNamespace(cup_world_pose=propagated_object(self.task, "right", self.attachment),
            coaster_world_center=self.platform.copy(), timings_s={},
            observation_audit=dict(source="initial_RGBD_and_assumed_rigid_right_hand_attachment",
                fresh_object_measurement=False, uses_object_truth=False))


def place_right(task, attachment, platform, metadata_path, result):
    from adapters.placement import finish_placement
    predicted = RigidAttachmentPerception(task, attachment, platform)
    try:
        finish_placement(task, predicted, "right", metadata_path, result, upright_reference=np.eye(4))
    finally:
        # Override the legacy function's generic label: this experiment does
        # NOT measure the held block again or verify a right-hand grasp.
        result.update(geometry_source="initial_RGBD_and_assumed_rigid_right_hand_attachment",
            orientation_source="declared_initial_upright_block", fresh_object_measurement=False,
            uses_object_truth_for_planning=False, grasp_verified=False)
