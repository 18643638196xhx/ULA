"""Map task-provided support geometry to generic feedback coordinates.

No object class, color, size, robot or simulator dependency. The caller supplies
its estimated support anchor, surface frame and task tolerances explicitly.
"""
from dataclasses import dataclass
import numpy as np
from adapters.constraint_feedback import BallRegion,IntervalRegion,CoordinateConstraint,decide_constraints


@dataclass(frozen=True)
class SupportPlacementGoal:
    support_anchor_local: tuple
    horizontal_tolerance_m: float
    gap_interval_m: tuple
    preferred_gap_m: float|None=None

    def basis(self,normal):
        normal=np.asarray(normal,dtype=float)
        if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-9:
            raise ValueError('A finite support-plane normal is required')
        normal=normal/np.linalg.norm(normal)
        reference=np.eye(3)[np.argmin(np.abs(normal))]
        x=reference-normal*(reference@normal);x/=np.linalg.norm(x)
        return np.column_stack((x,np.cross(normal,x),normal))

    def anchor_world(self,object_pose):
        pose=np.asarray(object_pose,dtype=float)
        anchor=np.asarray(self.support_anchor_local,dtype=float)
        if pose.shape!=(4,4) or anchor.shape!=(3,) or not np.isfinite(pose).all() or not np.isfinite(anchor).all():
            raise ValueError('A finite object pose and task-provided local support anchor are required')
        rotation=pose[:3,:3]
        if (not np.allclose(pose[3],[0.,0.,0.,1.],atol=1e-8) or
            not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or
            not np.isclose(np.linalg.det(rotation),1.,atol=1e-5)):
            raise ValueError('Object pose must be a proper rigid transform')
        return pose[:3,3]+pose[:3,:3]@anchor

    def decide(self,object_pose,support_origin,support_normal,*,horizontal_uncertainty_m,
               normal_uncertainty_m,observation_usable):
        origin=np.asarray(support_origin,dtype=float)
        if origin.shape!=(3,) or not np.isfinite(origin).all():
            raise ValueError('A finite target surface origin is required')
        frame=self.basis(support_normal)
        state=frame.T@(self.anchor_world(object_pose)-origin)
        if len(self.gap_interval_m)!=2:raise ValueError('A lower and upper release gap are required')
        preferred=None if self.preferred_gap_m is None else (self.preferred_gap_m,)
        constraints=(CoordinateConstraint('support_tangent',(0,1),BallRegion((0.,0.),self.horizontal_tolerance_m)),
            CoordinateConstraint('support_normal',(2,),IntervalRegion((self.gap_interval_m[0],),
                (self.gap_interval_m[1],),preferred)))
        decision=decide_constraints(state,constraints,
            dict(support_tangent=horizontal_uncertainty_m,support_normal=normal_uncertainty_m),
            observation_usable=observation_usable)
        target=None if decision.target is None else origin+frame@decision.target
        return decision,target
