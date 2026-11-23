"""Compiled rule runtime scheduler, RoboTwin backend adapter, and stage audit."""
from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

import numpy as np

from adapters.primitive_executors import RuntimeContext, StageExecutionOutcome, execute_stage
from adapters.rule_compiler import CompiledRulePlan
from adapters.runtime_frame_calibration import pose_wxyz_to_matrix, rotation_error_deg


class RoboTwinRobotBackend:
    """Thin adapter exposing ``get_arm_pose``, ``move_to_pose``, and ``set_gripper``."""

    def __init__(self, task: Any):
        self.task = task

    def get_arm_pose(self, arm: str) -> list[float]:
        return [float(v) for v in self.task.get_arm_pose(arm)]

    def move_to_pose(self, arm: str, target_pose_wxyz: list[float]) -> bool:
        from envs.utils import Action, ArmTag
        if hasattr(self.task, "plan_success"):
            self.task.plan_success = True
        arm_tag = ArmTag(arm)
        action = Action(arm_tag, "move", target_pose=list(target_pose_wxyz))
        moved = bool(self.task.move((arm_tag, [action])))
        return bool(moved and getattr(self.task, "plan_success", True))

    def move_both_to_poses(
        self,
        left_pose_wxyz: list[float],
        right_pose_wxyz: list[float],
    ) -> bool:
        from envs.utils import Action, ArmTag
        if hasattr(self.task, "plan_success"):
            self.task.plan_success = True
        l_tag, r_tag = ArmTag("left"), ArmTag("right")
        l_act = Action(l_tag, "move", target_pose=list(left_pose_wxyz))
        r_act = Action(r_tag, "move", target_pose=list(right_pose_wxyz))
        moved = bool(self.task.move((l_tag, [l_act]), (r_tag, [r_act])))
        return bool(moved and getattr(self.task, "plan_success", True))

    def set_gripper(self, arm: str, open_value: float) -> bool:
        from envs.utils import ArmTag
        if hasattr(self.task, "plan_success"):
            self.task.plan_success = True
        arm_tag = ArmTag(arm)
        if float(open_value) >= 0.5:
            return bool(self.task.move(self.task.open_gripper(arm_tag, pos=float(open_value))))
        return bool(self.task.move(self.task.close_gripper(arm_tag, pos=float(open_value))))

    def settle_to_pose(
        self,
        arm: str,
        target_pose_wxyz: list[float],
        pos_tol_m: float,
        rot_tol_deg: float,
        *,
        max_steps: int = 60,
    ) -> dict[str, Any]:
        """Wait for simulated joint drives to converge after trajectory playback.

        RoboTwin's dense executor advances physics once per planned waypoint
        and returns immediately after the final drive target is written. The
        articulation can therefore still lag behind a valid planned endpoint.
        Real robots require the same measured-state arrival check.
        """
        target = pose_wxyz_to_matrix(target_pose_wxyz)
        last_pos_err = float("inf")
        last_rot_err = float("inf")
        for step in range(max_steps + 1):
            actual = pose_wxyz_to_matrix(self.get_arm_pose(arm))
            last_pos_err = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            last_rot_err = float(rotation_error_deg(actual[:3, :3], target[:3, :3]))
            if last_pos_err <= pos_tol_m and last_rot_err <= rot_tol_deg:
                return {
                    "reached": True,
                    "steps": step,
                    "pos_err_m": last_pos_err,
                    "rot_err_deg": last_rot_err,
                }
            if step < max_steps:
                self.task.scene.step()
        return {
            "reached": False,
            "steps": max_steps,
            "pos_err_m": last_pos_err,
            "rot_err_deg": last_rot_err,
        }


class CompiledRuleRunner:
    """Executes a ``CompiledRulePlan`` stage-by-stage with structured auditing."""

    def __init__(self, plan: CompiledRulePlan, ctx: RuntimeContext, *, on_stage_end=None):
        self.plan = plan
        self.ctx = ctx
        self.on_stage_end = on_stage_end

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        completed: set[str] = set()
        stage_records: list[dict[str, Any]] = []
        failure_stage: str | None = None
        failure_reason: str | None = None

        for stage in self.plan.stages:
            unmet = [d for d in stage.depends_on if d not in completed]
            if unmet:
                failure_stage = stage.stage_id
                failure_reason = f"unmet_dependencies:{unmet}"
                stage_records.append({
                    "stage_id": stage.stage_id,
                    "primitive": stage.primitive,
                    "controller": stage.controller,
                    "executor": stage.executor,
                    "success": False,
                    "reason": failure_reason,
                    "duration_s": 0.0,
                    "details": {"unmet_dependencies": unmet},
                })
                break

            t0 = time.perf_counter()
            try:
                outcome: StageExecutionOutcome = execute_stage(stage, self.ctx)
            except Exception as exc:
                outcome = StageExecutionOutcome(
                    stage.stage_id, stage.primitive, stage.controller, stage.executor,
                    False, f"exception:{type(exc).__name__}:{exc}",
                )
            elapsed = time.perf_counter() - t0
            rec = asdict(outcome)
            rec["duration_s"] = round(elapsed, 4)
            stage_records.append(rec)
            if self.on_stage_end is not None:
                try:
                    self.on_stage_end(stage, outcome)
                except Exception:
                    pass

            if not outcome.success:
                failure_stage = stage.stage_id
                failure_reason = outcome.reason
                break
            completed.add(stage.stage_id)

        return {
            "task": self.plan.task,
            "schema": self.plan.schema,
            "rule_source": self.plan.source_path,
            "control_uses_object_truth": self.ctx.uses_object_truth,
            "controller_complete": failure_stage is None,
            "failure_stage": failure_stage,
            "failure_reason": failure_reason,
            "completed_stages": len(completed),
            "total_stages": len(self.plan.stages),
            "duration_s": round(time.perf_counter() - started, 4),
            "stages": stage_records,
        }
