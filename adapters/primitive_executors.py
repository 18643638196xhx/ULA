"""Task-independent runtime executors for compiled ULA primitive stages."""
from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from adapters.primitive_geometry import (
    apply_relative_pose,
    hand_goal_for_object_goal,
    relative_pose,
)
from adapters.rule_compiler import CompiledStage
from adapters.runtime_frame_calibration import (
    FrameCalibration,
    matrix_to_pose_wxyz,
    pose_wxyz_to_matrix,
    reconcile_recorded_ee_to_plan_target,
    reconcile_recorded_ee_to_tcp,
    rotation_error_deg,
)


@dataclass
class StageExecutionOutcome:
    stage_id: str
    primitive: str
    controller: str
    executor: str
    success: bool
    reason: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeContext:
    """Shared mutable state passed across compiled stage executors."""

    robot_backend: Any
    calibration: FrameCalibration
    perception_fn: Callable[[str, str], np.ndarray | None]
    vla_fn: Callable[[CompiledStage, "RuntimeContext"], Mapping[str, Any]]
    handover_confirmation_fn: Callable[
        [CompiledStage, "RuntimeContext", Mapping[str, Any]], Mapping[str, Any]
    ] | None = None
    object_poses: dict[str, np.ndarray] = field(default_factory=dict)
    attachments: dict[str, np.ndarray | None] = field(default_factory=lambda: {"left": None, "right": None})
    held_objects: dict[str, str | None] = field(default_factory=lambda: {"left": None, "right": None})
    pos_tol_m: float = 0.025
    rot_tol_deg: float = 18.0
    preplace_lift_m: float = 0.06
    retreat_lift_m: float = 0.08
    table_z: float | None = None
    min_tcp_clearance_m: float = 0.035
    receiver_preapproach_standoff_m: float | None = None
    enable_approach_yaw_symmetry: bool = True
    handover_vla_handles_receiver_approach: bool = False
    object_radii_m: dict[str, float] = field(default_factory=dict)
    uses_object_truth: bool = False

    def observe_entity(self, entity_name: str, reason: str, *, allow_cached: bool = False) -> np.ndarray:
        pose = self.perception_fn(entity_name, reason)
        if pose is not None:
            mat = np.asarray(pose, dtype=float)
            self.object_poses[entity_name] = mat
            return mat
        if allow_cached and entity_name in self.object_poses:
            return self.object_poses[entity_name]
        raise RuntimeError(f"Perception failed to localize {entity_name!r} for {reason}")

    def current_ee_matrix(self, arm: str) -> np.ndarray:
        return pose_wxyz_to_matrix(self.robot_backend.get_arm_pose(arm))

    def estimate_postgrasp_attachment(self, arm: str, obj_name: str, stage_id: str) -> tuple[np.ndarray, str]:
        """Re-estimate T_ee_object after VLA grasp using fresh observation."""
        T_world_ee = self.current_ee_matrix(arm)
        fresh = self.perception_fn(obj_name, f"postgrasp:{stage_id}")
        if fresh is not None:
            T_world_obj = np.asarray(fresh, dtype=float)
            self.object_poses[obj_name] = T_world_obj
            return relative_pose(T_world_ee, T_world_obj), "fresh_visual_postgrasp"
        if obj_name in self.object_poses:
            return relative_pose(T_world_ee, self.object_poses[obj_name]), "fallback_last_visual_pose"
        raise RuntimeError(f"Cannot estimate post-grasp attachment for {obj_name!r} on arm {arm}")


def _check_pose_reached(actual_T: np.ndarray, target_T: np.ndarray, pos_tol_m: float, rot_tol_deg: float) -> tuple[bool, float, float]:
    pos_err = float(np.linalg.norm(actual_T[:3, 3] - target_T[:3, 3]))
    rot_err = rotation_error_deg(actual_T[:3, :3], target_T[:3, :3])
    return (pos_err <= pos_tol_m and rot_err <= rot_tol_deg), pos_err, rot_err


def _move_arm_to_recorded_ee_goal(
    ctx: RuntimeContext,
    arm: str,
    T_world_recorded_ee_goal: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    T_plan = reconcile_recorded_ee_to_plan_target(T_world_recorded_ee_goal, ctx.calibration, arm)
    target_wxyz = matrix_to_pose_wxyz(T_plan)
    moved = bool(ctx.robot_backend.move_to_pose(arm, target_wxyz))
    actual_T = ctx.current_ee_matrix(arm)
    reached, pos_err, rot_err = _check_pose_reached(actual_T, T_world_recorded_ee_goal, ctx.pos_tol_m, ctx.rot_tol_deg)
    settle_info = None
    settle = getattr(ctx.robot_backend, "settle_to_pose", None)
    if moved and not reached and callable(settle):
        settle_info = dict(
            settle(
                arm,
                matrix_to_pose_wxyz(T_world_recorded_ee_goal),
                ctx.pos_tol_m,
                ctx.rot_tol_deg,
            )
        )
        actual_T = ctx.current_ee_matrix(arm)
        reached, pos_err, rot_err = _check_pose_reached(
            actual_T,
            T_world_recorded_ee_goal,
            ctx.pos_tol_m,
            ctx.rot_tol_deg,
        )
    return {
        "label": label,
        "arm": arm,
        "moved": moved,
        "reached": bool(moved and reached),
        "pos_err_m": pos_err,
        "rot_err_deg": rot_err,
        "target_pose_wxyz": target_wxyz,
        "actual_pose_wxyz": matrix_to_pose_wxyz(actual_T),
        "settle": settle_info,
    }


def _rot_z(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    T = np.eye(4, dtype=float)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return T


def _rot_x(deg: float) -> np.ndarray:
    rad = np.radians(float(deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    T = np.eye(4, dtype=float)
    T[:3, :3] = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    return T


def _pivot_world_x_tilt(T_pose: np.ndarray, pivot_xyz: np.ndarray, tilt_deg: float) -> np.ndarray:
    if abs(float(tilt_deg)) < 1e-6:
        return np.asarray(T_pose, dtype=float).copy()
    T = np.asarray(T_pose, dtype=float).copy()
    p = np.asarray(pivot_xyz, dtype=float).reshape(3)
    R = _rot_x(float(tilt_deg))[:3, :3]
    T[:3, :3] = R @ T[:3, :3]
    T[:3, 3] = p + R @ (T[:3, 3] - p)
    return T


def _enforce_table_clearance(T_goal: np.ndarray, ctx: RuntimeContext, arm: str) -> tuple[np.ndarray, float]:
    if ctx.table_z is None:
        return T_goal, 0.0
    T_tcp = reconcile_recorded_ee_to_tcp(T_goal, ctx.calibration, arm)
    min_z = float(ctx.table_z) + float(ctx.min_tcp_clearance_m)
    z_low = min(float(T_goal[2, 3]), float(T_tcp[2, 3]))
    if z_low < min_z:
        lift = min_z - z_low
        lifted = T_goal.copy()
        lifted[2, 3] += lift
        return lifted, float(lift)
    return T_goal, 0.0


def _best_yaw_hand_goal_for_object_goal(
    T_world_obj_goal: np.ndarray,
    attachment: np.ndarray,
    ctx: RuntimeContext,
    arm: str,
) -> tuple[np.ndarray, float]:
    """Keep object XYZ + upright Z fixed while relaxing yaw around object Z.

    Selects the yaw that minimizes wrist rotation from the current arm pose
    while keeping the arm on its own workspace side and out of the head camera
    front (-Y). This is task-independent robot geometry, not task hardcoding.
    """
    cur = ctx.current_ee_matrix(arm)
    # Build upright object frame at goal XYZ (world Z up) so yaw rotates cleanly around Z.
    T_obj_base = np.eye(4, dtype=float)
    T_obj_base[:3, 3] = np.asarray(T_world_obj_goal, dtype=float)[:3, 3]
    obj_x = float(T_obj_base[0, 3])
    best_score = float("inf")
    best_goal = None
    best_yaw = 0.0
    inv_att = np.linalg.inv(np.asarray(attachment, dtype=float))
    for yaw_deg in range(0, 360, 15):
        T_obj_cand = T_obj_base @ _rot_z(float(yaw_deg))
        T_ee_cand = T_obj_cand @ inv_att
        rot_err = rotation_error_deg(T_ee_cand[:3, :3], cur[:3, :3])
        dist = float(np.linalg.norm(T_ee_cand[:3, 3] - cur[:3, 3]))
        ee_x, ee_y = float(T_ee_cand[0, 3]), float(T_ee_cand[1, 3])
        # Keep arm on its own side of the object and avoid parking in front of head camera (-Y).
        if arm == "left":
            side_pen = max(0.0, ee_x - (obj_x - 0.04)) * 8.0
        elif arm == "right":
            side_pen = max(0.0, (obj_x + 0.04) - ee_x) * 8.0
        else:
            side_pen = 0.0
        cam_pen = max(0.0, -0.04 - ee_y) * 6.0
        score = dist + 0.004 * rot_err + side_pen + cam_pen
        if score < best_score:
            best_score = score
            best_goal = T_ee_cand
            best_yaw = float(yaw_deg)
    assert best_goal is not None
    return best_goal, best_yaw


def _best_yaw_receiver_preapproach(
    T_world_obj: np.ndarray,
    T_object_receiver_precontact: np.ndarray,
    T_receiver_object_grasp: np.ndarray | None,
    ctx: RuntimeContext,
    receiver_arm: str,
    obj_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose a collision-safe receiver pre-approach yaw on the receiver's own side."""
    cur = ctx.current_ee_matrix(receiver_arm)
    T_obj_upright = np.eye(4, dtype=float)
    T_obj_upright[:3, 3] = np.asarray(T_world_obj, dtype=float)[:3, 3]
    obj_x = float(T_obj_upright[0, 3])
    best_score = float("inf")
    best_goal = None
    best_geom = None
    for yaw_deg in range(0, 360, 15):
        T_obj_cand = T_obj_upright @ _rot_z(float(yaw_deg))
        T_cand, geom = _receiver_safe_preapproach_goal(
            T_obj_cand,
            T_object_receiver_precontact,
            T_receiver_object_grasp,
            object_radius_m=ctx.object_radii_m.get(obj_name),
            fallback_clearance_m=(
                float(ctx.receiver_preapproach_standoff_m)
                if ctx.receiver_preapproach_standoff_m is not None
                else float(ctx.min_tcp_clearance_m)
            ),
        )
        rot_err = rotation_error_deg(T_cand[:3, :3], cur[:3, :3])
        dist = float(np.linalg.norm(T_cand[:3, 3] - cur[:3, 3]))
        ee_x, ee_y = float(T_cand[0, 3]), float(T_cand[1, 3])
        if receiver_arm == "right":
            side_pen = max(0.0, (obj_x + 0.04) - ee_x) * 8.0
        elif receiver_arm == "left":
            side_pen = max(0.0, ee_x - (obj_x - 0.04)) * 8.0
        else:
            side_pen = 0.0
        cam_pen = max(0.0, -0.04 - ee_y) * 6.0
        # Preserve the exact demonstrated yaw=0 orientation whenever it already sits
        # cleanly on the receiver's own side without blocking the head camera, rather
        # than rotating toward the receiver's resting home pose.
        if yaw_deg == 0 and side_pen == 0.0 and cam_pen == 0.0:
            geom_zero = dict(geom)
            geom_zero["selected_yaw_deg"] = 0.0
            return T_cand, geom_zero
        score = dist + 0.004 * rot_err + side_pen + cam_pen
        if score < best_score:
            best_score = score
            best_goal = T_cand
            best_geom = dict(geom)
            best_geom["selected_yaw_deg"] = float(yaw_deg)
    assert best_goal is not None and best_geom is not None
    return best_goal, best_geom


def _approach_candidates(T_ref: np.ndarray, T_rel: np.ndarray, ctx: RuntimeContext, arm: str) -> list[tuple[float, np.ndarray, float]]:
    yaws_other = (90.0, -90.0, 180.0, 45.0, -45.0, 135.0, -135.0)
    yaws_cardinal_other = (90.0, -90.0, 180.0)
    # Fallback schedule:
    # Preserve yaw=0 (wrist camera orientation + wrist_roll distribution for VLA) across forward pitch tilts
    # (0 -> 12 -> 18 -> 24 deg) and slight Z relaxation BEFORE twisting the wrist to +/-90/180 deg.
    tiers: list[tuple[int, float, float, tuple[float, ...]]] = [(0, 0.0, 0.0, (0.0,))]
    if ctx.enable_approach_yaw_symmetry:
        tiers.extend([
            (1, 12.0, 0.0, (0.0,)),
            (2, 18.0, 0.0, (0.0,)),
            (3, 24.0, 0.0, (0.0,)),
            (4, 0.0, -0.02, (0.0,)),
            (5, 12.0, -0.02, (0.0,)),
            (6, 0.0, 0.0, yaws_other),
            (7, 12.0, 0.0, yaws_cardinal_other),
            (8, 18.0, 0.0, yaws_cardinal_other),
        ])
    cur = ctx.current_ee_matrix(arm)
    pivot_xyz = np.asarray(T_ref, dtype=float)[:3, 3]
    cands: list[tuple[tuple[int, int, float], float, np.ndarray, float]] = []
    for tier_idx, tilt_deg, dz, yaws in tiers:
        for yaw in yaws:
            T_ref_yaw = T_ref @ _rot_z(yaw)
            T_goal = apply_relative_pose(T_ref_yaw, T_rel)
            if tilt_deg != 0.0:
                T_goal = _pivot_world_x_tilt(T_goal, pivot_xyz, tilt_deg)
            if dz != 0.0:
                T_goal = T_goal.copy()
                T_goal[2, 3] += dz
            T_safe, lift = _enforce_table_clearance(T_goal, ctx, arm)
            dist = float(np.linalg.norm(T_safe[:3, 3] - cur[:3, 3]))
            rot = rotation_error_deg(T_safe[:3, :3], cur[:3, :3])
            yaw_tier = 0 if yaw == 0.0 else 1
            score = dist + 0.002 * rot
            cands.append(((tier_idx, yaw_tier, score), yaw, T_safe, lift))
    cands.sort(key=lambda item: item[0])
    return [(yaw, T_safe, lift) for _, yaw, T_safe, lift in cands]


def _receiver_safe_preapproach_goal(
    T_world_object: np.ndarray,
    T_object_receiver_precontact: np.ndarray,
    T_receiver_object_grasp: np.ndarray | None,
    *,
    object_radius_m: float | None,
    fallback_clearance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move the receiver onto the demonstrated approach ray, not into contact.

    The demonstration supplies both the receiver pose just before contact and,
    when available, the receiver-to-object attachment after capture. Their
    translation difference defines the demonstrated final approach direction.
    The demonstrated relation already contains the object's size, so the
    waypoint adds only the robot-level TCP clearance. This avoids using the
    bounding-sphere radius of a tall object as an unrelated lateral retreat.
    """
    T_pre = np.asarray(T_object_receiver_precontact, dtype=float)
    if T_pre.shape != (4, 4):
        raise ValueError("Expected 4x4 receiver precontact relation")

    pre_xyz = T_pre[:3, 3]
    source = "demo_precontact_minus_receiver_attachment"
    demo_gap_m = None
    if T_receiver_object_grasp is not None:
        T_grasp = np.linalg.inv(np.asarray(T_receiver_object_grasp, dtype=float))
        approach_outward = pre_xyz - T_grasp[:3, 3]
        demo_gap_m = float(np.linalg.norm(approach_outward))
    else:
        approach_outward = pre_xyz.copy()
        source = "object_center_to_demo_precontact"

    norm = float(np.linalg.norm(approach_outward))
    if norm < 1e-5:
        approach_outward = pre_xyz.copy()
        norm = float(np.linalg.norm(approach_outward))
        source = "object_center_to_demo_precontact"
    if norm < 1e-5:
        raise ValueError("Cannot infer a non-degenerate receiver approach direction")

    radius = float(object_radius_m) if object_radius_m is not None else 0.0
    additional_standoff = float(fallback_clearance_m)
    T_safe_object = T_pre.copy()
    if os.environ.get("HANDOVER_ALIGN_RECEIVER_Y0", "0") == "1" and T_receiver_object_grasp is not None:
        T_grasp_align = np.linalg.inv(np.asarray(T_receiver_object_grasp, dtype=float))
        backoff = max(0.06, float(additional_standoff) + 0.07)
        outward_dir = -T_safe_object[:3, 0]
        outward_dir = outward_dir / max(float(np.linalg.norm(outward_dir)), 1e-6)
        aligned_xyz = T_grasp_align[:3, 3] + outward_dir * backoff
        aligned_xyz[2] = float(pre_xyz[2])
        T_safe_object[:3, 3] = aligned_xyz
    else:
        T_safe_object[:3, 3] = pre_xyz + approach_outward / norm * additional_standoff
    T_world_safe = np.asarray(T_world_object, dtype=float) @ T_safe_object
    return T_world_safe, {
        "source": source,
        "demo_precontact_to_grasp_m": demo_gap_m,
        "object_radius_m": radius if object_radius_m is not None else None,
        "fallback_clearance_m": float(fallback_clearance_m),
        "standoff_source": "robot_tcp_clearance_after_demo_precontact",
        "additional_standoff_m": float(additional_standoff),
        "T_object_receiver_safe": T_safe_object.tolist(),
        "T_world_receiver_safe": T_world_safe.tolist(),
    }


def execute_move(stage: CompiledStage, ctx: RuntimeContext) -> StageExecutionOutcome:
    arm = stage.arm
    if arm is None:
        return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, False, "missing_arm")

    if stage.role == "approach" and "approach" in stage.relations:
        rel = stage.relations["approach"]
        if ctx.handover_vla_handles_receiver_approach:
            held_by_other = any(
                h_arm != arm and h_obj == rel.reference and ctx.attachments.get(h_arm) is not None
                for h_arm, h_obj in ctx.held_objects.items()
            )
            if held_by_other:
                return StageExecutionOutcome(
                    stage.stage_id, stage.primitive, stage.controller, stage.executor,
                    True, "deferred_to_handover_vla",
                    {"arm": arm, "reference": rel.reference, "skipped_prehandover_curobo": True},
                )
        T_ref = ctx.observe_entity(rel.reference, f"move_approach:{stage.stage_id}", allow_cached=True)
        T_rel = np.asarray(rel.T_reference_controlled, dtype=float)
        attempts = []
        for yaw_deg, T_goal, lift_m in _approach_candidates(T_ref, T_rel, ctx, arm):
            info = _move_arm_to_recorded_ee_goal(ctx, arm, T_goal, label=f"approach_yaw{int(yaw_deg)}")
            info["yaw_deg"] = yaw_deg
            info["table_clearance_lift_m"] = lift_m
            attempts.append(info)
            if info["reached"]:
                return StageExecutionOutcome(
                    stage.stage_id, stage.primitive, stage.controller, stage.executor,
                    True, "ok", {**info, "attempts": len(attempts)},
                )
        last = attempts[-1] if attempts else {}
        return StageExecutionOutcome(
            stage.stage_id, stage.primitive, stage.controller, stage.executor,
            False, "approach_not_reached", {**last, "attempts": attempts},
        )

    if stage.role == "held_transport" and ("held_transport" in stage.relations or stage.reusable_object_goal is not None):
        attachment = ctx.attachments.get(arm)
        if attachment is None:
            return StageExecutionOutcome(
                stage.stage_id, stage.primitive, stage.controller, stage.executor,
                False, f"missing_runtime_attachment_for_{arm}",
            )
        if "held_transport" in stage.relations:
            rel = stage.relations["held_transport"]
            ref_name = rel.reference
            T_rel = np.asarray(rel.T_reference_controlled, dtype=float)
        else:
            ref_name = stage.reference_name or "target"
            T_rel = np.asarray(stage.reusable_object_goal, dtype=float)
        T_target = ctx.observe_entity(ref_name, f"held_transport:{stage.stage_id}", allow_cached=True)
        T_world_obj_goal = apply_relative_pose(T_target, T_rel)
        if ctx.enable_approach_yaw_symmetry:
            T_goal, sel_yaw = _best_yaw_hand_goal_for_object_goal(T_world_obj_goal, attachment, ctx, arm)
        else:
            T_goal = hand_goal_for_object_goal(T_target, T_rel, attachment)
            sel_yaw = 0.0
        info = _move_arm_to_recorded_ee_goal(ctx, arm, T_goal, label="held_transport")
        info["selected_yaw_deg"] = sel_yaw
        return StageExecutionOutcome(
            stage.stage_id, stage.primitive, stage.controller, stage.executor,
            info["reached"], "ok" if info["reached"] else "transport_not_reached", info,
        )

    cur = ctx.current_ee_matrix(arm)
    T_goal = cur.copy()
    if stage.motion_delta_xyz_m is not None:
        delta = np.clip(np.asarray(stage.motion_delta_xyz_m, dtype=float), -0.25, 0.25)
        T_goal[:3, 3] = cur[:3, 3] + delta
    else:
        T_goal[2, 3] = cur[2, 3] + ctx.retreat_lift_m
    info = _move_arm_to_recorded_ee_goal(ctx, arm, T_goal, label="retreat")
    return StageExecutionOutcome(
        stage.stage_id, stage.primitive, stage.controller, stage.executor,
        info["reached"], "ok" if info["reached"] else "retreat_not_reached", info,
    )


def execute_grasp(stage: CompiledStage, ctx: RuntimeContext) -> StageExecutionOutcome:
    arm = stage.arm
    obj_name = stage.object_name or "object"
    if arm is None:
        return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, False, "missing_arm")

    details: dict[str, Any] = {}
    if "precontact" in stage.relations:
        rel = stage.relations["precontact"]
        T_obj = ctx.observe_entity(rel.reference, f"grasp_precontact:{stage.stage_id}", allow_cached=True)
        T_rel = np.asarray(rel.T_reference_controlled, dtype=float)
        cur_ee = ctx.current_ee_matrix(arm)
        cands = _approach_candidates(T_obj, T_rel, ctx, arm)
        best_check = None
        for yaw_deg, T_pre, lift_m in cands:
            reached, pos_err, rot_err = _check_pose_reached(cur_ee, T_pre, ctx.pos_tol_m, ctx.rot_tol_deg)
            if best_check is None or pos_err < best_check["pos_err_m"]:
                best_check = {"reached": reached, "pos_err_m": pos_err, "rot_err_deg": rot_err, "yaw_deg": yaw_deg}
            if reached:
                break
        details["precontact_check"] = best_check
        if not best_check["reached"]:
            recovered = False
            for yaw_deg, T_pre, lift_m in cands:
                reapproach = _move_arm_to_recorded_ee_goal(ctx, arm, T_pre, label=f"grasp_precontact_yaw{int(yaw_deg)}")
                details["precontact_recovery"] = reapproach
                if reapproach["reached"]:
                    recovered = True
                    break
            if not recovered:
                return StageExecutionOutcome(
                    stage.stage_id, stage.primitive, stage.controller, stage.executor,
                    False, "precontact_not_reached", details,
                )

    vla_res = dict(ctx.vla_fn(stage, ctx))
    details["vla"] = vla_res
    if not bool(vla_res.get("success", False)):
        return StageExecutionOutcome(
            stage.stage_id, stage.primitive, stage.controller, stage.executor,
            False, str(vla_res.get("reason", "vla_grasp_failed")), details,
        )

    if vla_res.get("attachment") is not None:
        attachment = np.asarray(vla_res["attachment"], dtype=float)
        source = str(vla_res.get("attachment_source", "vla_provided_attachment"))
    else:
        attachment, source = ctx.estimate_postgrasp_attachment(arm, obj_name, stage.stage_id)
    ctx.attachments[arm] = attachment
    ctx.held_objects[arm] = obj_name
    details["attachment_source"] = source
    details["T_ee_object"] = attachment.tolist()
    return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, True, "ok", details)


def execute_release(stage: CompiledStage, ctx: RuntimeContext) -> StageExecutionOutcome:
    arm = stage.arm
    if arm is None:
        return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, False, "missing_arm")
    ok = bool(ctx.robot_backend.set_gripper(arm, stage.gripper_open_value))
    if ok:
        ctx.attachments[arm] = None
        ctx.held_objects[arm] = None
    return StageExecutionOutcome(
        stage.stage_id, stage.primitive, stage.controller, stage.executor,
        ok, "ok" if ok else "gripper_open_failed",
        {"arm": arm, "gripper_open_value": stage.gripper_open_value},
    )


def execute_handover(stage: CompiledStage, ctx: RuntimeContext) -> StageExecutionOutcome:
    donor = stage.donor_arm
    receiver = stage.receiver_arm
    obj_name = stage.object_name or "object"
    if not donor or not receiver:
        return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, False, "missing_handover_arms")

    details: dict[str, Any] = {"donor_arm": donor, "receiver_arm": receiver}

    # A handover constrains the donor wrist as well as the object.  Merely
    # moving the object to its demonstrated transfer pose preserves whatever
    # grasp relation the local policy happened to produce and can leave the
    # donor forearm blocking the receiver.  Reconstruct the demonstrated
    # target->donor relation from:
    #   target->object @ inverse(donor->object)
    # and explicitly align the donor before the receiver starts.
    donor_relation = stage.relations.get("donor_attachment")
    donor_attachment = ctx.attachments.get(donor)
    if (
        donor_relation is not None
        and stage.reusable_object_goal is not None
        and stage.reference_name is not None
        and donor_attachment is not None
    ):
        T_world_target = ctx.observe_entity(
            stage.reference_name,
            f"handover_donor_alignment:{stage.stage_id}",
            allow_cached=True,
        )
        T_target_object_demo = np.asarray(stage.reusable_object_goal, dtype=float)
        T_donor_object_demo = np.asarray(donor_relation.T_reference_controlled, dtype=float)
        T_target_donor_demo = T_target_object_demo @ np.linalg.inv(T_donor_object_demo)
        T_world_object_goal = T_world_target @ T_target_object_demo
        if ctx.enable_approach_yaw_symmetry:
            T_world_donor_goal, donor_yaw_deg = _best_yaw_hand_goal_for_object_goal(
                T_world_object_goal,
                np.asarray(donor_attachment, dtype=float),
                ctx,
                donor,
            )
        else:
            T_world_donor_goal = T_world_target @ T_target_donor_demo
            donor_yaw_deg = 0.0
        # Dual-arm coordination needs a tighter donor pose than ordinary
        # single-arm transport. Derive its translation tolerance from the
        # robot-level TCP clearance rather than a task-specific constant.
        coordination_pos_tol = min(
            ctx.pos_tol_m,
            max(0.005, 0.25 * ctx.min_tcp_clearance_m),
        )
        coordination_rot_tol = min(ctx.rot_tol_deg, 5.0)
        donor_reached, donor_pos_err, donor_rot_err = _check_pose_reached(
            ctx.current_ee_matrix(donor),
            T_world_donor_goal,
            coordination_pos_tol,
            coordination_rot_tol,
        )
        alignment: dict[str, Any] = {
            "source": "demo_target_object_and_donor_attachment",
            "reference": stage.reference_name,
            "T_target_donor_demo": T_target_donor_demo.tolist(),
            "T_world_donor_goal": T_world_donor_goal.tolist(),
            "initially_reached": donor_reached,
            "initial_pos_err_m": donor_pos_err,
            "initial_rot_err_deg": donor_rot_err,
            "coordination_pos_tol_m": coordination_pos_tol,
            "coordination_rot_tol_deg": coordination_rot_tol,
        }
        if not donor_reached:
            alignment["motion"] = _move_arm_to_recorded_ee_goal(
                ctx,
                donor,
                T_world_donor_goal,
                label="handover_donor_demo_pose",
            )
            donor_reached, donor_pos_err, donor_rot_err = _check_pose_reached(
                ctx.current_ee_matrix(donor),
                T_world_donor_goal,
                coordination_pos_tol,
                coordination_rot_tol,
            )
            alignment["motion"]["coordination_reached"] = donor_reached
            alignment["motion"]["coordination_pos_err_m"] = donor_pos_err
            alignment["motion"]["coordination_rot_err_deg"] = donor_rot_err
        alignment["reached"] = donor_reached
        details["donor_handover_alignment"] = alignment
        if not donor_reached:
            details["donor_released"] = False
            return StageExecutionOutcome(
                stage.stage_id,
                stage.primitive,
                stage.controller,
                stage.executor,
                False,
                "donor_handover_pose_not_reached",
                details,
            )
        # The object remains rigidly attached through the measured runtime
        # attachment; update its propagated pose after donor alignment.
        ctx.object_poses[obj_name] = ctx.current_ee_matrix(donor) @ np.asarray(
            donor_attachment, dtype=float
        )
    else:
        details["donor_handover_alignment"] = {
            "reached": False,
            "skipped": True,
            "reason": "missing_demo_donor_geometry_or_runtime_attachment",
        }

    if "receiver_precontact" in stage.relations:
        rel = stage.relations["receiver_precontact"]
        # Once the donor is holding the object, its current wrist and measured
        # runtime attachment are more reliable than a stale table observation.
        T_obj = None
        if ctx.attachments.get(donor) is not None:
            T_obj = ctx.current_ee_matrix(donor) @ ctx.attachments[donor]
        if T_obj is None:
            T_obj = ctx.perception_fn(rel.reference, f"handover_precontact:{stage.stage_id}")
        if T_obj is None:
            return StageExecutionOutcome(
                stage.stage_id, stage.primitive, stage.controller, stage.executor,
                False, "handover_object_pose_unavailable", details,
            )
        T_rel = np.asarray(rel.T_reference_controlled, dtype=float)

        # The learned policy owns only the final local contact. CuRobo first
        # establishes a collision-safe waypoint on the demonstrated approach
        # ray, after donor alignment has fixed the live object pose.
        if ctx.handover_vla_handles_receiver_approach:
            receiver_attachment_rel = stage.relations.get("receiver_attachment")
            T_receiver_object_demo = (
                np.asarray(receiver_attachment_rel.T_reference_controlled, dtype=float)
                if receiver_attachment_rel is not None
                else None
            )
            T_receiver_pre_for_safe = T_rel
            T_receiver_object_for_safe = T_receiver_object_demo
            # A local grasp policy may capture the donor object at a different
            # offset from the demonstration. Replaying both object-relative
            # hand poses would then change the inter-arm separation and can
            # make the demonstrated receiver pose physically collide with the
            # donor. Compensate the receiver's starting relation so the two
            # wrists reproduce their demonstrated relative geometry. The VLA
            # remains responsible for the final local object alignment.
            if donor_relation is not None and donor_attachment is not None:
                T_donor_object_demo = np.asarray(
                    donor_relation.T_reference_controlled, dtype=float
                )
                T_donor_object_runtime = np.asarray(donor_attachment, dtype=float)
                T_donor_receiver_pre_demo = T_donor_object_demo @ T_rel
                T_receiver_pre_for_safe = (
                    np.linalg.inv(T_donor_object_runtime)
                    @ T_donor_receiver_pre_demo
                )
                adaptation: dict[str, Any] = {
                    "source": "preserve_demo_donor_to_receiver_geometry",
                    "T_object_receiver_precontact_adapted": T_receiver_pre_for_safe.tolist(),
                    "runtime_vs_demo_donor_attachment_translation_m": float(
                        np.linalg.norm(
                            T_donor_object_runtime[:3, 3]
                            - T_donor_object_demo[:3, 3]
                        )
                    ),
                    "runtime_vs_demo_donor_attachment_rotation_deg": float(
                        rotation_error_deg(
                            T_donor_object_runtime[:3, :3],
                            T_donor_object_demo[:3, :3],
                        )
                    ),
                }
                if T_receiver_object_demo is not None:
                    T_object_receiver_grasp_demo = np.linalg.inv(
                        T_receiver_object_demo
                    )
                    T_donor_receiver_grasp_demo = (
                        T_donor_object_demo @ T_object_receiver_grasp_demo
                    )
                    T_object_receiver_grasp_adapted = (
                        np.linalg.inv(T_donor_object_runtime)
                        @ T_donor_receiver_grasp_demo
                    )
                    T_receiver_object_for_safe = np.linalg.inv(
                        T_object_receiver_grasp_adapted
                    )
                    adaptation["T_receiver_object_grasp_adapted"] = (
                        T_receiver_object_for_safe.tolist()
                    )
                details["receiver_preapproach_adaptation"] = adaptation
            try:
                if ctx.enable_approach_yaw_symmetry:
                    T_recv_safe, safe_geometry = _best_yaw_receiver_preapproach(
                        np.asarray(T_obj, dtype=float),
                        T_rel,
                        T_receiver_object_demo,
                        ctx,
                        receiver,
                        obj_name,
                    )
                else:
                    T_recv_safe, safe_geometry = _receiver_safe_preapproach_goal(
                        np.asarray(T_obj, dtype=float),
                        T_receiver_pre_for_safe,
                        T_receiver_object_for_safe,
                        object_radius_m=ctx.object_radii_m.get(obj_name),
                        fallback_clearance_m=(
                            float(ctx.receiver_preapproach_standoff_m)
                            if ctx.receiver_preapproach_standoff_m is not None
                            else float(ctx.min_tcp_clearance_m)
                        ),
                    )
            except (ValueError, np.linalg.LinAlgError) as exc:
                details["receiver_safe_preapproach"] = {
                    "reached": False,
                    "reason": f"geometry_error:{type(exc).__name__}:{exc}",
                }
                return StageExecutionOutcome(
                    stage.stage_id, stage.primitive, stage.controller, stage.executor,
                    False, "receiver_safe_preapproach_geometry_failed", details,
                )
            # RoboTwin's move() reports that a planned trajectory was played,
            # not that the simulated joints have physically converged at its
            # endpoint. Retry once from the measured state only when the first
            # playback remains outside tolerance. Both attempts target the
            # same collision-safe waypoint, never the contact pose.
            safe_geometry["motion_attempts"] = []
            for attempt_index in range(2):
                motion = _move_arm_to_recorded_ee_goal(
                    ctx,
                    receiver,
                    T_recv_safe,
                    label=f"receiver_safe_preapproach_{attempt_index + 1}",
                )
                safe_geometry["motion_attempts"].append(motion)
                if motion["reached"]:
                    break
            safe_geometry["motion"] = safe_geometry["motion_attempts"][-1]
            exact_reached = bool(safe_geometry["motion"]["reached"])
            local_handoff: dict[str, Any] = {
                "accepted": False,
                "reason": "exact_safe_waypoint_reached" if exact_reached else "outside_local_region",
            }
            if not exact_reached and T_receiver_object_for_safe is not None:
                T_object_receiver_actual = (
                    np.linalg.inv(np.asarray(T_obj, dtype=float))
                    @ ctx.current_ee_matrix(receiver)
                )
                T_object_receiver_grasp = np.linalg.inv(
                    T_receiver_object_for_safe
                )
                outward = (
                    T_receiver_pre_for_safe[:3, 3]
                    - T_object_receiver_grasp[:3, 3]
                )
                outward_norm = float(np.linalg.norm(outward))
                if outward_norm >= 1e-5:
                    outward_unit = outward / outward_norm
                    grasp_delta = (
                        T_object_receiver_actual[:3, 3]
                        - T_object_receiver_grasp[:3, 3]
                    )
                    axial_outward = float(np.dot(grasp_delta, outward_unit))
                    lateral_error = float(
                        np.linalg.norm(
                            grasp_delta - axial_outward * outward_unit
                        )
                    )
                    T_object_receiver_safe = np.asarray(
                        safe_geometry["T_object_receiver_safe"], dtype=float
                    )
                    safe_position_error = float(
                        np.linalg.norm(
                            T_object_receiver_actual[:3, 3]
                            - T_object_receiver_safe[:3, 3]
                        )
                    )
                    safe_rotation_error = float(
                        rotation_error_deg(
                            T_object_receiver_actual[:3, :3],
                            T_object_receiver_safe[:3, :3],
                        )
                    )
                    local_radius = float(
                        ctx.object_radii_m.get(obj_name, ctx.min_tcp_clearance_m)
                    )
                    accepted = bool(
                        axial_outward > 0.0
                        and safe_position_error <= local_radius
                        and safe_rotation_error <= ctx.rot_tol_deg
                    )
                    local_handoff = {
                        "accepted": accepted,
                        "reason": (
                            "outside_object_and_within_visual_local_region"
                            if accepted else "outside_local_region"
                        ),
                        "axial_outward_from_grasp_m": axial_outward,
                        "lateral_error_from_demo_approach_m": lateral_error,
                        "safe_position_error_m": safe_position_error,
                        "safe_rotation_error_deg": safe_rotation_error,
                        "local_region_radius_m": local_radius,
                        "uses_object_truth": False,
                        "uses_contact_truth": False,
                    }
            safe_geometry["exact_reached"] = exact_reached
            safe_geometry["local_handoff"] = local_handoff
            safe_geometry["reached"] = bool(
                exact_reached or local_handoff["accepted"]
            )
            details["receiver_safe_preapproach"] = safe_geometry
            if not safe_geometry["reached"]:
                return StageExecutionOutcome(
                    stage.stage_id, stage.primitive, stage.controller, stage.executor,
                    False, "receiver_safe_preapproach_not_reached", details,
                )
        else:
            cur_recv = ctx.current_ee_matrix(receiver)
            cands = _approach_candidates(np.asarray(T_obj, dtype=float), T_rel, ctx, receiver)
            already = None
            for yaw_deg, T_recv_pre, lift_m in cands:
                reached, pos_err, rot_err = _check_pose_reached(cur_recv, T_recv_pre, ctx.pos_tol_m, ctx.rot_tol_deg)
                if already is None or pos_err < already["pos_err_m"]:
                    already = {"reached": reached, "pos_err_m": pos_err, "rot_err_deg": rot_err, "yaw_deg": yaw_deg}
                if reached:
                    break
            details["receiver_precontact_check"] = already
            if not already["reached"]:
                pre_info = None
                for yaw_deg, T_recv_pre, lift_m in cands:
                    pre_info = _move_arm_to_recorded_ee_goal(ctx, receiver, T_recv_pre, label=f"receiver_precontact_yaw{int(yaw_deg)}")
                    pre_info["yaw_deg"] = yaw_deg
                    if pre_info["reached"]:
                        break
                details["receiver_precontact"] = pre_info
                if not pre_info or not pre_info["reached"]:
                    return StageExecutionOutcome(
                        stage.stage_id, stage.primitive, stage.controller, stage.executor,
                        False, "receiver_precontact_not_reached", details,
                    )

    vla_res = dict(ctx.vla_fn(stage, ctx))
    details["receiver_vla"] = vla_res
    if not bool(vla_res.get("success", False)):
        details["donor_released"] = False
        return StageExecutionOutcome(
            stage.stage_id, stage.primitive, stage.controller, stage.executor,
            False, str(vla_res.get("reason", "receiver_vla_take_failed")), details,
        )

    if vla_res.get("attachment") is not None:
        recv_attach = np.asarray(vla_res["attachment"], dtype=float)
        attach_src = str(vla_res.get("attachment_source", "vla_provided_attachment"))
    else:
        recv_attach, attach_src = ctx.estimate_postgrasp_attachment(receiver, obj_name, stage.stage_id)
    details["receiver_attachment_source"] = attach_src

    if ctx.handover_confirmation_fn is not None:
        try:
            confirmation = dict(ctx.handover_confirmation_fn(stage, ctx, vla_res))
        except Exception as exc:
            confirmation = {
                "accepted": False,
                "reason": f"confirmation_exception:{type(exc).__name__}:{exc}",
            }
    else:
        # Fail closed: a policy-reported close or an inferred attachment is not
        # physical evidence that the receiver actually captured the object.
        confirmation = {
            "accepted": bool(vla_res.get("receiver_attachment_confirmed", False)),
            "reason": str(vla_res.get("receiver_attachment_confirmation_reason", "no_confirmation_callback")),
        }
    confirmation.setdefault("uses_object_truth", False)
    details["receiver_attachment_confirmation"] = confirmation
    if not bool(confirmation.get("accepted", False)):
        details["donor_released"] = False
        return StageExecutionOutcome(
            stage.stage_id, stage.primitive, stage.controller, stage.executor,
            False, "receiver_attachment_unconfirmed", details,
        )

    ctx.attachments[receiver] = recv_attach
    ctx.held_objects[receiver] = obj_name

    donor_released = bool(ctx.robot_backend.set_gripper(donor, stage.gripper_open_value))
    details["donor_released"] = donor_released
    if not donor_released:
        return StageExecutionOutcome(
            stage.stage_id, stage.primitive, stage.controller, stage.executor,
            False, "donor_release_failed", details,
        )
    ctx.attachments[donor] = None
    ctx.held_objects[donor] = None
    return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, True, "ok", details)


def execute_place(stage: CompiledStage, ctx: RuntimeContext) -> StageExecutionOutcome:
    arm = stage.arm
    if arm is None:
        return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, False, "missing_arm")
    attachment = ctx.attachments.get(arm)
    if attachment is None:
        return StageExecutionOutcome(
            stage.stage_id, stage.primitive, stage.controller, stage.executor,
            False, f"missing_runtime_attachment_for_{arm}",
        )

    if "goal" in stage.relations:
        rel = stage.relations["goal"]
        target_name = rel.reference
        T_target_obj = np.asarray(rel.T_reference_controlled, dtype=float)
    elif stage.reusable_object_goal is not None:
        target_name = stage.reference_name or "target"
        T_target_obj = np.asarray(stage.reusable_object_goal, dtype=float)
    else:
        return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, False, "missing_place_goal")

    T_world_target = ctx.observe_entity(target_name, f"place_goal:{stage.stage_id}", allow_cached=True)
    yaws_coarse = (0.0, 90.0, -90.0, 180.0, 45.0, -45.0, 135.0, -135.0) if ctx.enable_approach_yaw_symmetry else (0.0,)
    yaws_fine = (
        (0.0, 90.0, -90.0, 180.0, 45.0, -45.0, 135.0, -135.0, 22.5, -22.5, 67.5, -67.5, 112.5, -112.5, 157.5, -157.5)
        if ctx.enable_approach_yaw_symmetry else (0.0,)
    )
    cur_ee = ctx.current_ee_matrix(arm)
    pivot_xyz = np.asarray(T_world_target, dtype=float)[:3, 3]

    def _build_place_cands(yaws_tuple: tuple[float, ...], tilt_deg: float) -> list[tuple[float, float, np.ndarray, float]]:
        out = []
        for yaw_deg in yaws_tuple:
            T_tgt_yaw = T_world_target @ _rot_z(yaw_deg)
            T_ee_cand = hand_goal_for_object_goal(T_tgt_yaw, T_target_obj, attachment)
            if tilt_deg != 0.0:
                T_ee_cand = _pivot_world_x_tilt(T_ee_cand, pivot_xyz, tilt_deg)
            T_ee_cand, lift_m = _enforce_table_clearance(T_ee_cand, ctx, arm)
            dist = float(np.linalg.norm(T_ee_cand[:3, 3] - cur_ee[:3, 3]))
            rot = rotation_error_deg(T_ee_cand[:3, :3], cur_ee[:3, :3])
            score = dist + 0.003 * rot + (0.0 if yaw_deg == 0.0 else 0.01)
            out.append((score, yaw_deg, T_ee_cand, lift_m))
        out.sort(key=lambda item: item[0])
        return out

    yaws_other_coarse = (90.0, -90.0, 180.0, 45.0, -45.0, 135.0, -135.0)
    yaws_other_fine = (
        90.0, -90.0, 180.0, 45.0, -45.0, 135.0, -135.0,
        22.5, -22.5, 67.5, -67.5, 112.5, -112.5, 157.5, -157.5,
    )
    base_lift = float(ctx.preplace_lift_m)
    # Always preserve the exact demonstrated yaw=0 orientation first (upright, then slight lift/pitch
    # relaxation) so non-axisymmetric objects (e.g. mouse, shoe) keep their demonstrated heading
    # whenever yaw=0 is reachable; only fall back to symmetric yaw rotations if yaw=0 fails IK.
    schedule: list[tuple[float, float, tuple[float, ...]]] = [(0.0, base_lift, (0.0,))]
    if base_lift > 0.036:
        schedule.append((0.0, 0.035, (0.0,)))
    for tilt_deg in (12.0, 18.0, 24.0):
        schedule.append((tilt_deg, base_lift, (0.0,)))
        if base_lift > 0.031:
            schedule.append((tilt_deg, 0.03, (0.0,)))
    if base_lift > 0.016:
        schedule.append((0.0, 0.015, (0.0,)))
    schedule.append((0.0, 0.0, (0.0,)))
    if ctx.enable_approach_yaw_symmetry:
        schedule.append((0.0, base_lift, yaws_other_coarse))
        if base_lift > 0.036:
            schedule.append((0.0, 0.035, yaws_other_coarse))
        for tilt_deg in (12.0, 18.0, 24.0):
            schedule.append((tilt_deg, base_lift, yaws_other_coarse))
            if base_lift > 0.031:
                schedule.append((tilt_deg, 0.03, yaws_other_coarse))
        for fallback_lift in (0.015, 0.0):
            if fallback_lift < base_lift - 1e-4:
                schedule.append((0.0, fallback_lift, yaws_other_fine))
        for tilt_deg in (12.0, 18.0, 24.0):
            schedule.append((tilt_deg, 0.0, yaws_other_coarse))

    details: dict[str, Any] = {"attempts": []}
    T_ee_goal = None
    pre_info = None
    cand_cache: dict[tuple[float, int], list[tuple[float, float, np.ndarray, float]]] = {}
    for tilt_deg, pre_lift, yaws_tuple in schedule:
        cache_key = (tilt_deg, len(yaws_tuple))
        if cache_key not in cand_cache:
            cand_cache[cache_key] = _build_place_cands(yaws_tuple, tilt_deg)
        for _, yaw_deg, T_ee_cand, lift_m in cand_cache[cache_key]:
            T_preplace = T_ee_cand.copy()
            T_preplace[2, 3] += pre_lift
            attempt_pre = _move_arm_to_recorded_ee_goal(
                ctx, arm, T_preplace,
                label=f"preplace_yaw{yaw_deg:g}_tilt{int(round(tilt_deg))}_lift{int(round(pre_lift * 1000))}mm",
            )
            attempt_pre["yaw_deg"] = yaw_deg
            attempt_pre["tilt_deg"] = tilt_deg
            attempt_pre["preplace_lift_m"] = pre_lift
            attempt_pre["table_clearance_lift_m"] = lift_m
            details["attempts"].append(attempt_pre)
            if attempt_pre["reached"]:
                pre_info = attempt_pre
                T_ee_goal = T_ee_cand
                details["selected_yaw_deg"] = yaw_deg
                details["selected_tilt_deg"] = tilt_deg
                details["selected_preplace_lift_m"] = pre_lift
                break
        if T_ee_goal is not None:
            break
    details["preplace"] = pre_info
    if T_ee_goal is None or pre_info is None:
        return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, False, "preplace_not_reached", details)

    place_info = _move_arm_to_recorded_ee_goal(ctx, arm, T_ee_goal, label="place_descend")
    details["place_descend"] = place_info
    if not place_info["reached"]:
        return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, False, "place_goal_not_reached", details)

    released = bool(ctx.robot_backend.set_gripper(arm, stage.gripper_open_value))
    details["released"] = released
    if not released:
        return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, False, "place_release_failed", details)
    ctx.attachments[arm] = None
    ctx.held_objects[arm] = None

    T_retreat = T_ee_goal.copy()
    T_retreat[2, 3] += ctx.retreat_lift_m
    details["postplace_retreat"] = _move_arm_to_recorded_ee_goal(ctx, arm, T_retreat, label="postplace_retreat")
    return StageExecutionOutcome(stage.stage_id, stage.primitive, stage.controller, stage.executor, True, "ok", details)


EXECUTORS = {
    "move": execute_move,
    "grasp": execute_grasp,
    "release": execute_release,
    "handover": execute_handover,
    "place": execute_place,
}


def execute_stage(stage: CompiledStage, ctx: RuntimeContext) -> StageExecutionOutcome:
    fn = EXECUTORS.get(stage.primitive)
    if fn is None:
        return StageExecutionOutcome(
            stage.stage_id, stage.primitive, stage.controller, stage.executor,
            False, f"no_executor_for_{stage.primitive}",
        )
    return fn(stage, ctx)
