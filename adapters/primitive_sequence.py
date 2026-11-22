"""Composable primitive sequence for multi-stage manipulation tasks.

Design rule: the VLA is used ONLY for grasping.  Every other action
(approach, translate, place, release, retreat) is traditional control.

A task is described as an ordered list of stages in its rule file, e.g.

    stages:
      - {arm: main,     op: reach,    target: object}
      - {arm: main,     op: grasp}                  # the only VLA step
      - {arm: opposite, op: reach,    target: container.contact_point}
      - {arm: opposite, op: grasp}
      - {arm: opposite, op: translate, displacement: [0, -0.16, 0], frame: world}
      - {arm: main,     op: translate, displacement: [0, 0, 0.15], frame: world}
      - {arm: main,     op: reach,    target: container.functional_point}
      - {arm: main,     op: release}

Nothing here embeds object identities, distances or task names: targets are CAD
annotations transformed by visually estimated poses, displacements come from the
rule, and the arm pair comes from the stage.
"""
from __future__ import annotations

import numpy as np

from adapters.handoff import handoff_ready


class StageError(RuntimeError):
    """Raised when a stage cannot be completed; carries the stage context."""

    def __init__(self, stage, reason):
        super().__init__(f"stage {stage.get('op')} ({stage.get('arm')}) failed: {reason}")
        self.stage = dict(stage)
        self.reason = reason


def _arm(task, ctx, role):
    """Resolve main/opposite arm roles against the context's chosen arm."""
    main = ctx["main_arm"]
    if role in (None, "main"):
        return main
    if role == "opposite":
        return "right" if str(main) == "left" else "left"
    return role


def _pose_matrix(pose):
    from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
    array = np.asarray(pose, dtype=float)
    if array.shape == (4, 4):
        return array
    return pose_wxyz_to_matrix(array)


# --------------------------------------------------------------------------- #
# primitives: every one takes (task, ctx, arm, ...) and returns an audit dict
# --------------------------------------------------------------------------- #

def reach(task, ctx, arm, pose, *, tolerance=None, label="reach"):
    """Traditional Cartesian move to an absolute end-effector pose."""
    tolerance = float(tolerance if tolerance is not None else ctx["tolerance"])
    target = np.asarray(pose, dtype=float).reshape(-1)
    moved = bool(task.move_by_pose(arm, target)) if hasattr(task, "move_by_pose") else False
    if not moved:
        # Fall back to the planner used everywhere else in this codebase.
        plan = getattr(task.robot, f"{arm}_plan_path")(target)
        if plan["status"] != "Success":
            raise StageError(dict(op="reach", arm=arm, label=label),
                             "traditional_planning_failed")
        controls = dict(left_arm=None, right_arm=None,
                        left_gripper=None, right_gripper=None)
        controls[f"{arm}_arm"] = plan
        if not task.take_dense_action(controls):
            raise StageError(dict(op="reach", arm=arm, label=label), "motion_failed")
    decision = handoff_ready(task.get_arm_pose(arm), target, bool(task.plan_success),
                            tolerance, ctx["orientation_tolerance"])
    audit = dict(op="reach", arm=str(arm), label=label,
                 target_pose=target.tolist(), ready=bool(decision.ready),
                 position_error_m=float(decision.position_error_m))
    if not decision.ready:
        raise StageError(dict(op="reach", arm=arm, label=label),
                         f"outside_position_tolerance:{decision.position_error_m:.4f}")
    return audit


def translate(task, ctx, arm, displacement, *, label="translate"):
    """Traditional relative move (drawer pull, lift, retreat)."""
    vector = np.asarray(displacement, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise StageError(dict(op="translate", arm=arm), "invalid_displacement")
    moved = bool(task.move_by_displacement(arm_tag=arm, x=float(vector[0]),
                                           y=float(vector[1]), z=float(vector[2])))
    audit = dict(op="translate", arm=str(arm), label=label,
                 displacement_m=vector.tolist(), moved=moved)
    if not moved:
        raise StageError(dict(op="translate", arm=arm), "displacement_failed")
    return audit


def place(task, ctx, arm, pose, *, label="place"):
    """Traditional move to the release pose (same motion primitive as reach)."""
    return reach(task, ctx, arm, pose, label=label)


def release(task, ctx, arm, *, label="release"):
    """Open the gripper and let the object settle."""
    opened = bool(task.move(task.open_gripper(arm)))
    audit = dict(op="release", arm=str(arm), label=label, opened=opened)
    if not opened:
        raise StageError(dict(op="release", arm=arm), "gripper_open_failed")
    return audit


def grasp(task, ctx, arm, *, label="grasp"):
    """The ONLY VLA step: local grasp of whatever the arm is aimed at.

    Returns the hand-off evidence as well, because "the gripper closed" is not
    the same claim as "the object is in the hand" (see primitive_result).
    """
    import os as _os
    from adapters.primitive_result import build_result, collect_grasp_evidence

    grasp_fn = ctx.get("grasp_fn")
    if grasp_fn is None:
        raise StageError(dict(op="grasp", arm=arm), "no_vla_grasp_available")
    audit = dict(grasp_fn(task, ctx, arm) or {})
    audit.update(op="grasp", arm=str(arm), label=label)
    evidence = collect_grasp_evidence(task, arm, audit)
    result = build_result(
        "grasp", arm, audit=audit, evidence=evidence,
        object_pose=ctx.get("object_pose"),
        ee_to_object=ctx.get("ee_to_object"),
        # Observation only by default; opt in once a real discriminator exists.
        requires_grasp_evidence=_os.environ.get("ULA_PRIMITIVE_STRICT_GRASP") == "1")
    audit["primitive_result"] = result.to_dict()
    audit["evidence"] = evidence
    if not result.permits_next_stage:
        raise StageError(dict(op="grasp", arm=arm),
                         f"grasp_not_permitted:{audit.get('reason') or 'no_evidence'}")
    return audit


PRIMITIVES = dict(reach=reach, grasp=grasp, translate=translate,
                  place=place, release=release)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def run_stages(task, ctx, stages, *, on_stage=None):
    """Execute an ordered stage list; stops at the first failing stage."""
    ctx = dict(ctx)
    ctx.setdefault("tolerance", 0.01)
    ctx.setdefault("orientation_tolerance", 0.1)
    executed = []
    for index, stage in enumerate(stages):
        op = str(stage.get("op", "")).strip()
        if op not in PRIMITIVES:
            raise StageError(dict(stage), f"unknown_primitive:{op}")
        arm = _arm(task, ctx, stage.get("arm"))
        stage_ctx = dict(op=op, arm=arm, index=index)
        try:
            if op == "reach":
                target = stage.get("pose")
                if target is None:
                    raise StageError(dict(stage), "reach_requires_resolved_pose")
                audit = reach(task, ctx, arm, target, tolerance=stage.get("tolerance"),
                              label=str(stage.get("label", op)))
            elif op == "grasp":
                audit = grasp(task, ctx, arm, label=str(stage.get("label", op)))
            elif op == "translate":
                audit = translate(task, ctx, arm, stage.get("displacement"),
                                  label=str(stage.get("label", op)))
            elif op == "place":
                target = stage.get("pose")
                if target is None:
                    raise StageError(dict(stage), "place_requires_resolved_pose")
                audit = place(task, ctx, arm, target, label=str(stage.get("label", op)))
            else:  # release
                audit = release(task, ctx, arm, label=str(stage.get("label", op)))
        except StageError as error:
            error.stage.update(stage_ctx)
            executed.append(dict(stage_ctx, ok=False, error=str(error)))
            if on_stage is not None:
                on_stage(executed[-1])
            raise
        executed.append(dict(stage_ctx, ok=True, audit=audit))
        if on_stage is not None:
            on_stage(executed[-1])
    return executed
