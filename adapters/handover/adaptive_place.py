"""Task-constrained, RGB-D-feedback right-hand placement, not an oracle controller.

Only the allowed support axis is constrained; yaw is free. Each small motion is
followed by a NEW measured view and a new grasp transform. Partial observations
alone cannot authorize release. An explicitly bounded temporal estimator can
retain a post-handover visual anchor during occlusion, without claiming that
the hidden bottom was observed in the current frame.
"""
import numpy as np

from adapters.handover.orientation_goal import SupportConstraint,project_orientation
from adapters.handover.place_control import settle_endpoint,slower_trajectory
from adapters.constraint_feedback import FeedbackBudget
from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix,matrix_to_pose_wxyz


def move_segment(task,target,audit,*,rotation=False):
    kwargs={} if rotation else dict(constraint_pose=[1,1,1,0,0,0])
    plan=task.robot.right_plan_path(matrix_to_pose_wxyz(target),**kwargs)
    audit.update(target_ee_pose=matrix_to_pose_wxyz(target).tolist(),
        motion="required_tilt_correction" if rotation else "orientation_preserving_translation",
        planner_status=plan.get("planner_status"),status=plan["status"],settling={})
    if plan["status"]!="Success":
        raise RuntimeError("Adaptive placement segment planning failed: "+str(plan.get("planner_status")))
    controls=dict(left_arm=None,right_arm=slower_trajectory(plan,4.),left_gripper=None,right_gripper=None)
    moved=bool(task.take_dense_action(controls,save_freq=None))
    settle_endpoint(task,plan,target,audit["settling"])
    if not moved or not audit["settling"]["ready"]:
        raise RuntimeError("Adaptive placement segment did not reach the measured robot target; no release")


def adaptive_place(task,binding,platform_top,result,*,observer,on_stage=None,max_iterations=40,
                   constraint=SupportConstraint(max_tilt_deg=5.),feedback_goal=None,feedback_budget=None,
                   segment_executor=None):
    from scipy.spatial.transform import Rotation
    from envs.utils import ArmTag
    platform=np.asarray(platform_top,dtype=float)
    binding=np.asarray(binding,dtype=float).copy()
    if platform.shape!=(3,) or not np.isfinite(platform).all() or binding.shape!=(4,4) or not np.isfinite(binding).all():
        raise ValueError("Finite visual platform and tracking prior required")
    if max_iterations<1:
        raise ValueError("Positive bounded feedback budget required")
    budget=feedback_budget if feedback_budget is not None else FeedbackBudget()
    normal=np.asarray(constraint.normal,dtype=float)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-9:
        raise ValueError('A finite support normal is required')
    normal=normal/np.linalg.norm(normal)
    def execute(target,action,*,rotation=False):
        # The default controller is unchanged. Opt-in planners receive the
        # measured grasp transform and the task's actual free support axis.
        if segment_executor is None:
            return move_segment(task,target,action,rotation=rotation)
        return segment_executor(task,target,action,rotation=rotation,
            binding=binding.copy(),visual=visual.copy(),normal=normal.copy(),
            anchor=(0.,0.,-.1) if feedback_goal is None else feedback_goal.support_anchor_local,
            position_usable=position_usable)
    result.update(complete=False,uses_object_truth_for_planning=False,
        geometry_source="fresh_three_view_RGBD_known_primitive_feedback",
        initial_rigid_attachment_is_search_prior_only=True,fresh_object_measurement=True,
        orientation_source="task_support_cone_minimum_rotation_yaw_free",
        max_iterations=max_iterations,max_translation_step_m=.025,max_rotation_step_deg=5.,
        task_max_tilt_deg=constraint.max_tilt_deg,platform_source="initial_RGBD_blue_platform",
        release_mode="visual_alignment_then_short_10mm_release_not_force_control",
        release_gap_m=.01,trial_lift=False,observations=[],actions=[],final_release_authorized=False,
        constraint_feedback_enabled=feedback_goal is not None)
    if feedback_goal is not None:
        result['constraint_feedback_goal']=dict(support_anchor_local=list(feedback_goal.support_anchor_local),
            horizontal_tolerance_m=feedback_goal.horizontal_tolerance_m,
            gap_interval_m=list(feedback_goal.gap_interval_m),preferred_gap_m=feedback_goal.preferred_gap_m)
    if not 0<=task.robot.get_right_gripper_val()<.05 or task.robot.get_left_gripper_val()<=.8:
        raise RuntimeError("Right-close/left-release robot transition incomplete")
    phase="transport"
    for index in range(max_iterations):
        hand=pose_wxyz_to_matrix(task.get_arm_pose("right"))
        prior=hand@binding
        attempts=[]
        visual=None
        for retry in range(2):
            try:
                visual,quality=observer.observe(prior)
                attempts.append(dict(retry=retry,accepted=True,quality=quality))
                break
            except RuntimeError as error:
                attempts.append(dict(retry=retry,accepted=False,error=str(error)))
        result["observations"].append(dict(iteration=index,phase=phase,attempts=attempts))
        if visual is None:
            raise RuntimeError("No trustworthy measured held-block axis after bounded re-observation; no release")
        if not quality.get("accepted"):
            raise RuntimeError("Observer returned an unaccepted held-object measurement")
        shift=float(np.linalg.norm(visual[:3,3]-prior[:3,3]))
        angle=float(np.degrees(np.arccos(np.clip(visual[:3,2]@prior[:3,2],-1,1))))
        if shift>.12 or angle>65.:
            raise RuntimeError("Measured red object is inconsistent with the tracked object; no release")
        observable=bool(quality.get("position_observable"))
        history=quality.get("temporal_tracking",{})
        historical_position=bool(history.get("release_usable"))
        position_usable=observable or historical_position
        binding=np.linalg.solve(hand,visual)
        goal_rotation,goal_audit=project_orientation(visual[:3,:3],constraint,
            uncertainty_deg=quality["axis_uncertainty_deg"])
        entry=result["observations"][-1]
        entry.update(visual_block_pose=visual.tolist(),right_ee_to_block=binding.tolist(),
            position_observable=observable,position_estimate_usable=position_usable,
            temporal_tracking=history,goal=goal_audit,
            change_from_kinematic_prior_m=shift,axis_change_from_prior_deg=angle)
        required=goal_audit["rotation_needed_deg"]
        if required>.5:
            fraction=min(1.,5./required)
            delta=Rotation.from_matrix(goal_rotation@visual[:3,:3].T)
            correction=Rotation.from_rotvec(delta.as_rotvec()*fraction).as_matrix()
            target=hand.copy()
            if position_usable:
                desired=visual.copy()
                desired[:3,:3]=correction@visual[:3,:3]
                target=desired@np.linalg.inv(binding)
            else:
                # Observable axis, hidden end: rotate around measured robot
                # hand, never pretend to know the hidden object centre.
                target[:3,:3]=correction@hand[:3,:3]
            action=dict(iteration=index,phase=phase,goal=goal_audit)
            result["actions"].append(action)
            execute(target,action,rotation=True)
            if on_stage is not None:
                on_stage(f"feedback_{index:02d}_tilt")
            continue
        bottom=visual[:3,3]-.1*visual[:3,2] if feedback_goal is None else feedback_goal.anchor_world(visual)
        relative=bottom-platform
        gap=float(relative@normal)
        xy_error=float(np.linalg.norm(relative-gap*normal))
        entry.update(estimated_bottom=bottom.tolist(),gap_m=gap,xy_error_m=xy_error)
        if phase=="transport" and xy_error<=.012 and gap<=.10:
            phase="descent"
        clearance=.08 if phase=="transport" else .01
        desired_bottom=platform+normal*clearance
        displacement=desired_bottom-bottom
        distance=float(np.linalg.norm(displacement))
        safe_axis=goal_audit["measured_tilt_deg"]+quality["axis_uncertainty_deg"]<=constraint.max_tilt_deg+.5
        feedback=None
        if feedback_goal is not None and phase=='descent':
            # The handover estimator currently reports world XY/Z margins.
            # A different support normal needs its covariance projected into
            # that frame by its estimator, not silently reused as XY/Z.
            if history and not np.allclose(normal,[0.,0.,1.]):
                raise ValueError('Task adapter must supply support-frame uncertainty for a tilted plane')
            feedback,target_anchor=feedback_goal.decide(visual,platform,normal,
                horizontal_uncertainty_m=history.get('bottom_xy_95_m',0.),
                normal_uncertainty_m=history.get('bottom_z_95_m',0.),
                observation_usable=position_usable and (historical_position if history else observable))
            feedback=budget.check(feedback)
            entry['constraint_feedback']=feedback.audit()
            if feedback.action=='stop':
                raise RuntimeError('Constraint feedback stopped: '+feedback.reason+'; no release')
            if feedback.action=='reobserve':
                result['actions'].append(dict(iteration=index,phase=phase,
                    action='constraint_reobserve',reason=feedback.reason,robot_moved=False))
                continue
            if feedback.action=='correct':
                desired_bottom=target_anchor
                displacement=desired_bottom-bottom
                distance=float(np.linalg.norm(displacement))
        release_ready=(feedback.action=='complete' if feedback is not None else
            xy_error<=.008 and .003<=gap<=.018)
        if phase=="descent" and release_ready and safe_axis:
            if not position_usable:
                raise RuntimeError("Axis visible but hidden bottom position remains unobservable at release; no release")
            if history and feedback is None:
                if not historical_position:
                    raise RuntimeError("Temporal bottom uncertainty or visual anchor age exceeds release budget; no release")
                if (xy_error+history['bottom_xy_95_m']>.008 or
                    gap-history['bottom_z_95_m']<.003 or gap+history['bottom_z_95_m']>.018):
                    raise RuntimeError("Estimated bottom confidence region is outside release bounds; no release")
            result.update(final_release_authorized=True,
                final_visual_block_pose=visual.tolist(),final_right_ee_to_block=binding.tolist(),
                final_visual_gap_m=gap,final_visual_xy_error_m=xy_error,
                final_orientation_decision=goal_audit,
                final_position_source="bounded_visual_history_and_current_surfaces" if history else "current_surfaces",
                final_temporal_tracking=history,
                final_constraint_feedback=None if feedback is None else feedback.audit())
            arm=ArmTag("right")
            for name,command in (("release",lambda:task.open_gripper(arm,pos=1.)),
                                 ("withdraw_from_block",lambda:task.move_by_displacement(arm,z=.07,move_axis="arm"))):
                moved=bool(task.move(command()))
                result["actions"].append(dict(action=name,success=moved))
                if on_stage is not None:
                    on_stage(name)
                if not moved or not task.plan_success:
                    raise RuntimeError(f"Adaptive placement failed during {name}")
            result["complete"]=True
            return
        if gap<-.01:
            raise RuntimeError("Measured object bottom is below visual support plane; no further descent or release")
        if distance<.001:
            raise RuntimeError("Release not observable/allowed at the current target; no blind release")
        target=hand.copy()
        target[:3,3]+=displacement*min(1.,.025/distance)
        action=dict(iteration=index,phase=phase,goal=goal_audit,
            position_source="fused_visual_history_and_current_surfaces" if history else (
                "measured_surfaces" if observable else "partial_axis_with_declared_longitudinal_prior"))
        if feedback is not None:action['constraint_feedback']=feedback.audit()
        result["actions"].append(action)
        execute(target,action)
        if on_stage is not None:
            on_stage(f"feedback_{index:02d}_{phase}")
    raise RuntimeError("Adaptive visual placement exceeded bounded feedback budget; no release")
