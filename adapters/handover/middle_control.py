"""Carry an already lifted block using a declared rigid-grasp hypothesis.

Planning uses initial RGB-D and robot-only closure history, never actor poses.
The initial object is assumed upright/stationary until the recorded closure;
sliding, early object motion or tipping invalidate that approximation.
"""
import numpy as np

from adapters.handoff import handoff_ready
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix,matrix_to_pose_wxyz

MIDDLE_CENTER=np.array([0.,0.,1.])


def closure_attachment(
    initial_visual_pose,
    commands,
    stable_steps=3,
    *,
    min_ee_to_block_m=0.04,
    max_ee_to_block_m=0.25,
):
    if not commands:
        raise ValueError("No robot closure history")
    # Pick the start of the earliest contiguous closed run of length >= stable_steps:
    # avoids both early 1-step glitches and late re-closures after the block has lifted.
    earliest_run = []
    cur_run = []
    for command in commands:
        if 0 <= float(command["left"]) < 0.05:
            cur_run.append(command)
        else:
            if len(cur_run) >= stable_steps:
                earliest_run = cur_run
                break
            cur_run = []
    if not earliest_run and len(cur_run) >= stable_steps:
        earliest_run = cur_run
    if len(earliest_run) < stable_steps:
        raise RuntimeError("No sustained left closed-command run")
    anchor = earliest_run[0]
    hand = pose_wxyz_to_matrix(anchor["left_ee_pose"])
    attachment = np.linalg.solve(hand, np.asarray(initial_visual_pose, dtype=float))
    ee_to_block_dist = float(np.linalg.norm(attachment[:3, 3]))
    if not (min_ee_to_block_m <= ee_to_block_dist <= max_ee_to_block_m):
        raise RuntimeError(
            f"Implausible closure_attachment distance {ee_to_block_dist:.3f} m "
            f"(expected [{min_ee_to_block_m:.2f}, {max_ee_to_block_m:.2f}] m) "
            f"at closure_step={anchor['step']}"
        )
    return attachment, dict(
        source="initial_RGBD_at_start_of_earliest_sustained_closed_command_run",
        closure_step=anchor["step"],
        closed_command_samples=len(earliest_run),
        closure_ee_pose=anchor["left_ee_pose"],
        initial_visual_block_pose=np.asarray(initial_visual_pose, dtype=float).tolist(),
        assumed_ee_to_block=attachment.tolist(),
        ee_to_block_distance_m=ee_to_block_dist,
        uses_object_truth=False,
        assumptions=["block_pose_unchanged_before_closure", "rigid_grasp_after_closure"],
        fresh_postgrasp_visual_measurement=False,
        physically_verified_attachment=False,
    )


def carry_to_middle(task,attachment,result):
    from envs.utils import ArmTag,Action
    result.update(complete=False,uses_object_truth=False,trial_lift=False,
        additional_lift_before_transport=False,goal_block_center=MIDDLE_CENTER.tolist(),
        goal_source="declared_official_handover_bottom_z_0.9_plus_half_height_0.1")
    if not 0<=float(task.robot.get_left_gripper_val())<.05:
        raise RuntimeError("Left gripper command is open before carry")
    goal=np.eye(4)
    goal[:3,3]=MIDDLE_CENTER
    target=matrix_to_pose_wxyz(goal@np.linalg.inv(attachment))
    result["target_ee_pose"]=target.tolist()
    result["predicted_block_pose_before"]= (
        pose_wxyz_to_matrix(task.get_arm_pose("left"))@attachment).tolist()
    arm=ArmTag("left")
    moved=bool(task.move((arm,[Action(arm,"move",target_pose=target)])))
    decision=handoff_ready(task.get_arm_pose("left"),target,moved and task.plan_success)
    result.update(complete=bool(moved and task.plan_success),
        actual_ee_pose=list(task.get_arm_pose("left")),robot_arrival=decision.to_dict(),
        predicted_block_pose_after=(pose_wxyz_to_matrix(task.get_arm_pose("left"))@attachment).tolist())
    # Robot arrival error is diagnostic, not proof of block arrival and not a
    # new threshold that rejects an actually successful object-zone outcome.
    if not result["complete"]:
        raise RuntimeError("Traditional middle transport planning/execution failed")


def hold_interval(task,steps=15):
    """Fixed robot-only hold; no contact, object observation or extra lift."""
    left=np.asarray(task.robot.get_left_arm_jointState()[:6])
    right=np.asarray(task.robot.get_right_arm_jointState()[:6])
    task.take_dense_action(dict(
        left_arm=dict(position=np.tile(left,(steps,1)),velocity=np.zeros((steps,6))),
        right_arm=dict(position=np.tile(right,(steps,1)),velocity=np.zeros((steps,6))),
        left_gripper=None,right_gripper=None),save_freq=None)
