"""One bounded grasp retry, authorized by RGB-D table-support evidence only."""

import numpy as np

from adapters.multiview import clipped, move_ee_checked


class GraspStalled(RuntimeError):
    pass


def table_support_evidence(estimate,reference_bounds,reference_table_z):
    bounds = np.asarray(estimate.cup_bounds_world,dtype=float)
    reference = np.asarray(reference_bounds,dtype=float)
    if bounds.shape != (2,3) or not np.isfinite(bounds).all():
        return dict(supported=False,reason="invalid_bounds")
    table_z = float(estimate.observation_audit["table_height_m"])
    gap = float(bounds[0,2]-table_z)
    top_change = float(bounds[1,2]-reference[1,2])
    displacement = float(np.linalg.norm(bounds.mean(axis=0)[:2]-reference.mean(axis=0)[:2]))
    supported = (not clipped(estimate) and abs(table_z-reference_table_z) <= 0.015
                 and -0.003 <= gap <= 0.015 and abs(top_change) <= 0.015 and displacement <= 0.06)
    return dict(supported=bool(supported),clipped=clipped(estimate),bottom_table_gap_m=gap,
                visible_top_change_m=top_change,xy_displacement_m=displacement,
                uses_simulator_object_truth=False)


class VisualStallCheck:
    def __init__(self,task,perception,arm,reference_bounds,reference_table_z,record,
                 start_step,interval=120):
        self.task,self.perception,self.arm = task,perception,arm
        self.reference_bounds,self.table_z = reference_bounds,reference_table_z
        self.record = record
        self.next_step = start_step+interval
        self.interval = interval
        self.checks = 0

    def __call__(self):
        if self.task.take_action_cnt < self.next_step or self.checks >= 2:
            return
        self.next_step = self.task.take_action_cnt+self.interval
        self.checks += 1
        audit = dict(after_vla_step=self.task.take_action_cnt,supported=False,frames=[])
        self.record.setdefault("visual_stall_checks",[]).append(audit)
        centres = []
        for frame in range(2):
            if frame and hasattr(self.task,"delay"):
                self.task.delay(0.1)
            evidence = []
            try:
                camera = self.task.get_camera_observation("head_camera")
                estimate = self.perception.supplementary(camera,
                    f"check{self.checks}_frame{frame}_head",cup_only=False)
                evidence.append(("head_camera",estimate))
                if clipped(estimate):
                    camera = self.task.get_camera_observation(f"{self.arm}_camera")
                    wrist = self.perception.supplementary(camera,f"check{self.checks}_frame{frame}_wrist")
                    evidence.append((f"{self.arm}_camera",wrist))
            except (RuntimeError,ValueError,KeyError) as error:
                audit["error"] = f"{type(error).__name__}: {error}"
            candidates = []
            for name,estimate in evidence:
                decision = table_support_evidence(estimate,self.reference_bounds,self.table_z)
                audit["frames"].append(dict(frame=frame,camera=name,**decision))
                if decision["supported"]:
                    candidates.append(estimate)
            if not candidates:
                return
            centres.append(np.asarray(candidates[-1].cup_bounds_world).mean(axis=0))
        drift = float(np.linalg.norm(centres[1]-centres[0]))
        audit.update(supported=drift <= 0.005,between_frame_drift_m=drift)
        if audit["supported"]:
            raise GraspStalled("No stable grasp; two visual observations indicate cup remains on table")


def retreat_for_retry(task,arm,audit):
    # Only called after the visual support gate; no opening if support is uncertain.
    if not audit.get("visual_support_evidence",{}).get("supported",False):
        raise RuntimeError("Recovery cannot open gripper without visual support confirmation")
    if not task.move(task.open_gripper(arm,pos=1.)) or not task.plan_success:
        raise RuntimeError("Recovery gripper opening failed")
    audit["opened_after_visual_support_confirmation"] = True
    target = np.asarray(task.get_arm_pose(arm),dtype=float)
    target[2] += 0.08
    moved, result = move_ee_checked(task,arm,target)
    audit["retreat"] = result
    if not moved:
        raise RuntimeError("Recovery retreat failed")
