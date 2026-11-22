"""Minimum-rotation projection into a task's allowed support-axis cone.

No fixed object yaw or world-frame full orientation. Independent of perception,
robot framework and simulation; valid for both preserve/correct cases.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SupportConstraint:
    max_tilt_deg: float = 10.
    normal: tuple = (0.,0.,1.)


def project_orientation(rotation,constraint=SupportConstraint(),*,uncertainty_deg=0.):
    rotation=np.asarray(rotation,dtype=float)
    normal=np.asarray(constraint.normal,dtype=float)
    if rotation.shape!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<.999:
        raise ValueError("Expected a proper measured rotation")
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-9:
        raise ValueError("Expected a finite support normal")
    if not 0<=constraint.max_tilt_deg<90 or not 0<=uncertainty_deg<90:
        raise ValueError("Invalid task tilt limit or angular uncertainty")
    normal=normal/np.linalg.norm(normal)
    axis=rotation[:,2]
    angle=float(np.arccos(np.clip(axis@normal,-1,1)))
    allowed=float(np.radians(max(0.,constraint.max_tilt_deg-uncertainty_deg)))
    correction=max(0.,angle-allowed)
    output=rotation.copy()
    if correction>1e-9:
        turn=np.cross(axis,normal)
        if np.linalg.norm(turn)<1e-8:
            # Antiparallel axis: any perpendicular direction has equal cost;
            # use the current object x axis instead of inventing a world yaw.
            turn=rotation[:,0]
        turn=turn/np.linalg.norm(turn)
        x,y,z=turn
        skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        delta=np.eye(3)+np.sin(correction)*skew+(1-np.cos(correction))*(skew@skew)
        output=delta@rotation
    return output,dict(decision="preserve" if correction<1e-9 else "correct_required_tilt",
        measured_tilt_deg=float(np.degrees(angle)),angular_uncertainty_deg=float(uncertainty_deg),
        task_max_tilt_deg=constraint.max_tilt_deg,robust_target_tilt_deg=float(np.degrees(allowed)),
        rotation_needed_deg=float(np.degrees(correction)),yaw_constraint="free",
        source="minimum_rotation_projection_of_fresh_visual_object_axis",uses_object_truth=False)
