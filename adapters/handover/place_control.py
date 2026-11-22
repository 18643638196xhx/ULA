"""Right-hand block placement: RGB-D platform and robot-only grasp hypothesis."""
import numpy as np

from adapters.handoff import handoff_ready
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix, matrix_to_pose_wxyz


def right_attachment(left_attachment, commands, stable_steps=3):
    """Latch at the last dual-closed sample in the FINAL right-closed run.

    Never bind the old table pose to the already moved hand. Assumes the block
    follows the left hand before release, then the right hand without slipping.
    Closed drive commands are timing signals, NOT evidence of actual contact.
    """
    final_run = []
    for command in reversed(commands):
        if not 0 <= command["right"] < .05:
            break
        final_run.append(command)
    if len(final_run) < stable_steps:
        raise RuntimeError("No sustained final right-closed command run")
    overlap = [command for command in final_run if 0 <= command["left"] < .05]
    if not overlap:
        raise RuntimeError("No dual-closed robot sample to transfer the rigid-grasp hypothesis")
    anchor = overlap[0]
    predicted = pose_wxyz_to_matrix(anchor["left_ee_pose"])@np.asarray(left_attachment)
    binding = np.linalg.solve(pose_wxyz_to_matrix(anchor["right_ee_pose"]), predicted)
    return binding,dict(source="last_dual_closed_robot_sample_in_final_right_closed_run",
        anchor_step=anchor["step"],anchor_phase_step=anchor["phase_step"],
        left_ee_pose=anchor["left_ee_pose"],right_ee_pose=anchor["right_ee_pose"],
        predicted_block_pose_at_transfer=predicted.tolist(),right_ee_to_block=binding.tolist(),
        uses_object_truth=False,physically_verified=False,fresh_object_vision=False,
        assumptions=["initial_RGBD_and_left_rigid_motion_valid_until_release",
                     "block_rigidly_attached_to_right_hand_after_transfer"])


def settle_endpoint(task,trajectory,target,result,max_steps=200):
    """Zero end velocity and let measured robot feedback converge, at most .8 s.

    Keeps the same planned endpoint. No object, contact, task-score query,
    teleport, extra lift, looser tolerance or replanning is involved.
    """
    joints=np.asarray(trajectory["position"][-1],dtype=float)
    if joints.shape!=(6,) or not np.isfinite(joints).all():
        raise ValueError("Expected six finite right-arm endpoint joint targets")
    result.update(ready=False,max_physics_steps=max_steps,zero_velocity_target=True,
        uses_object_truth=False,samples=[])
    consecutive=0
    for elapsed in range(0,max_steps+1,20):
        arrival=handoff_ready(task.get_arm_pose("right"),matrix_to_pose_wxyz(target),task.plan_success)
        measured=np.asarray(task.robot.get_right_arm_real_jointState(),dtype=float)[:6]
        error=float(np.max(np.abs(measured-joints)))
        good=arrival.ready and error<=.025
        consecutive=consecutive+1 if good else 0
        result["samples"].append(dict(physics_steps=elapsed,arrival=arrival.to_dict(),
            max_joint_error_rad=error,consecutive=consecutive,
            actual_ee_pose=list(task.get_arm_pose("right"))))
        if consecutive>=3:
            result.update(ready=True,physics_steps=elapsed)
            return arrival
        if elapsed<max_steps:
            hold=dict(status="Success",position=np.tile(joints,(20,1)),velocity=np.zeros((20,6)))
            task.take_dense_action(dict(left_arm=None,right_arm=hold,left_gripper=None,right_gripper=None),save_freq=None)
    result["physics_steps"]=max_steps
    return arrival


def slower_trajectory(trajectory,factor=2.):
    """Time scale only: preserve planned path/endpoints, reduce joint speeds."""
    if factor<1 or not np.isfinite(factor):
        raise ValueError("Trajectory time factor must be finite and >= 1")
    position=np.asarray(trajectory["position"])
    velocity=np.asarray(trajectory["velocity"])
    if len(position)<=1 or factor==1:
        return dict(trajectory)
    grid=np.linspace(0,len(position)-1,int(np.ceil((len(position)-1)*factor))+1)
    original=np.arange(len(position))
    return dict(trajectory,
        position=np.stack([np.interp(grid,original,position[:,i]) for i in range(position.shape[1])],axis=1),
        velocity=np.stack([np.interp(grid,original,velocity[:,i])/factor for i in range(velocity.shape[1])],axis=1))


def place_on_platform(task, binding, platform_top, result, *, on_stage=None):
    from envs.utils import ArmTag
    arm = ArmTag("right")
    binding, platform_top = np.asarray(binding), np.asarray(platform_top,dtype=float)
    if binding.shape != (4,4) or platform_top.shape != (3,) or not np.isfinite(platform_top).all() or not np.isfinite(binding).all():
        raise ValueError("Invalid estimated binding or visual platform center")
    result.update(complete=False,uses_object_truth_for_planning=False,
        geometry_source="initial_RGBD_robot_closure_history_and_rigid_attachment",
        fresh_object_measurement=False,known_block_size_m=[.06,.06,.20],
        platform_source="initial_RGBD_blue_platform_top",platform_top=platform_top.tolist(),
        release_gap_m=.003,trial_lift=False,trajectory_time_factor=4.,
        orientation_source="preserve_handover_robot_frame_not_unobserved_object_alignment",
        orientation_constraint=[1,1,1,0,0,0],actions=[],planning_attempts=[])
    if not 0 <= task.robot.get_right_gripper_val() < .05 or task.robot.get_left_gripper_val() <= .8:
        raise RuntimeError("Robot commands have not completed right-close/left-release transition")
    current = pose_wxyz_to_matrix(task.get_arm_pose("right"))@binding
    result["predicted_block_pose_before"] = current.tolist()
    plan_path = task.robot.right_plan_path
    entity = task.robot.right_entity
    names = [joint.get_name() for joint in entity.get_active_joints()]
    indices = [names.index(name) for name in task.robot.right_arm_joints_name]
    chosen = None
    for yaw in (0.,-.15,.15,-.30,.30,-.60,.60):
        cosine,sine = np.cos(yaw),np.sin(yaw)
        desired_block = np.eye(4)
        desired_block[:3,:3] = np.array([[cosine,-sine,0],[sine,cosine,0],[0,0,1]])@current[:3,:3]
        # Preserve the current held orientation; align its approximate bottom
        # anchor, rather than rotating an unobserved block into a CAD frame.
        desired_block[:3,3] = platform_top+[0,0,.003]+.1*desired_block[:3,2]
        desired = desired_block@np.linalg.inv(binding)
        for clearance in (.05,.08,.10):
            above = desired.copy()
            above[2,3] += clearance
            pre = plan_path(matrix_to_pose_wxyz(above),constraint_pose=[1,1,1,0,0,0])
            attempt = dict(yaw_rad=yaw,clearance_m=clearance,pre_status=pre["status"],
                           pre_planner_status=pre.get("planner_status"))
            result["planning_attempts"].append(attempt)
            if pre["status"] != "Success":
                if "INVALID_START" in str(pre.get("planner_status","")):
                    raise RuntimeError("Placement planner rejected VLA end state: "+str(pre.get("planner_status")))
                continue
            future = np.asarray(entity.get_qpos()).copy()
            future[indices] = pre["position"][-1]
            down = plan_path(matrix_to_pose_wxyz(desired),last_qpos=future,constraint_pose=[1,1,1,0,0,0])
            attempt["place_status"] = down["status"]
            attempt["place_planner_status"] = down.get("planner_status")
            if down["status"] == "Success":
                chosen = (("move_above_pad",pre,above),("lower_to_pad",down,desired))
                result.update(chosen_yaw_rad=yaw,chosen_clearance_m=clearance,
                    target_ee_pose=matrix_to_pose_wxyz(desired).tolist(),
                    desired_block_pose=desired_block.tolist())
                break
        if chosen:
            break
    if chosen is None:
        raise RuntimeError("No feasible transport/descent pair for the visual platform")
    for name,trajectory,target in chosen:
        controls=dict(left_arm=None,right_arm=slower_trajectory(trajectory,4.),left_gripper=None,right_gripper=None)
        moved=bool(task.take_dense_action(controls,save_freq=None))
        item=dict(action=name,moved=moved,settling={})
        result["actions"].append(item)
        decision=settle_endpoint(task,trajectory,target,item["settling"])
        item["arrival"]=decision.to_dict()
        if on_stage is not None:
            on_stage(name)  # Passive callback return is deliberately ignored.
        if not moved or not item["settling"]["ready"]:
            raise RuntimeError(f"Right arm did not reach {name}: {decision.reason}; not releasing")
    # No release until both trajectories have been planned AND actual robot FK
    # reaches their targets. This is an arrival check, not a grasp-truth check.
    for name,command in (("release",lambda:task.open_gripper(arm,pos=1.)),
                         ("withdraw_from_block",lambda:task.move_by_displacement(arm,z=.07,move_axis="arm"))):
        moved=bool(task.move(command()))
        result["actions"].append(dict(action=name,success=moved))
        if on_stage is not None:
            on_stage(name)
        if not moved or not task.plan_success:
            raise RuntimeError(f"Traditional placement failed during {name}")
    result["complete"] = True
