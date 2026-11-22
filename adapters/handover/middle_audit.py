"""Independent final-region evaluation; never imported by the carry planner."""
import numpy as np
from adapters.handover.pickup_audit import lowest_surface_z


def evaluate_middle(samples,initial_lowest_z,center_tolerance=.05,tilt_tolerance=15.):
    if len(samples)<5:
        return dict(success=False,reason="insufficient_final_hold_samples",feeds_back=False)
    recent=samples[-5:]
    relative=[]
    candidates=[]
    for sample in recent:
        block=np.asarray(sample["block_world"])
        hand=np.asarray(sample["left_ee_world"])
        local=np.linalg.solve(hand,block)[:3,3]
        relative.append(local)
        candidates.append(bool(0<=sample["left_command"]<.05 and np.linalg.norm(local)<=.20
                               and lowest_surface_z(block)-initial_lowest_z>=.03))
    drift=float(np.max(np.linalg.norm(np.asarray(relative)-relative[0],axis=1)))
    held=bool(all(candidates) and drift<=.02)
    block=np.asarray(recent[-1]["block_world"])
    center=block[:3,3]
    bottom=center+block[:3,:3]@np.array([0.,0.,-.1])
    center_error=float(np.linalg.norm(center-[0.,0.,1.]))
    bottom_error=float(np.linalg.norm(bottom-[0.,0.,.9]))
    tilt=float(np.degrees(np.arccos(np.clip(block[2,2],-1.,1.))))
    inside=bool(center_error<=center_tolerance)
    upright=bool(tilt<=tilt_tolerance)
    success=bool(inside and upright and held)
    reason=("middle_reached" if success else "held_geometry_not_verified" if not held
            else "block_tilt_exceeds_tolerance" if not upright else "block_center_outside_region")
    return dict(success=success,precise_3cm_success=bool(held and upright and center_error<=.03),
        reason=reason,center_xyz=center.tolist(),bottom_functional_xyz=bottom.tolist(),
        center_error_m=center_error,bottom_target_error_m=bottom_error,up_tilt_deg=tilt,
        held_geometry_verified=held,relative_translation_drift_m=drift,
        final_clearance_m=lowest_surface_z(block)-initial_lowest_z,
        center_tolerance_m=center_tolerance,tilt_tolerance_deg=tilt_tolerance,
        hold_samples=5,feeds_back=False,source="passive_simulation_after_fixed_carry_and_hold",
        evaluation_not_force_or_contact_proof=True)
