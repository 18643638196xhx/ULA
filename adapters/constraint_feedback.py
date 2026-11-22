"""Task-independent feedback decisions for independent geometric constraints.

Works on abstract measured coordinates (lengths, angles, etc.), not robot APIs,
object classes, simulation scores or seeds. Uncertainty margins are supplied by
the estimator/task adapter; they are not guaranteed error bounds by this module.
Only products of non-overlapping balls/intervals are supported, not arbitrary
coupled constraints or collision avoidance. Motion still requires a planner.
"""
from dataclasses import dataclass, field
import numpy as np


def vector(value):
    result=np.asarray(value,dtype=float)
    if result.ndim!=1 or not len(result) or not np.isfinite(result).all():
        raise ValueError('Expected a nonempty finite coordinate vector')
    return result


@dataclass(frozen=True)
class BallRegion:
    center: tuple
    radius: float

    def assess(self,value,uncertainty):
        center=vector(self.center)
        margin=np.asarray(uncertainty,dtype=float)
        if value.shape!=center.shape or margin.ndim!=0 or not np.isfinite(margin) or margin<0:
            raise ValueError('Ball uncertainty must be a finite nonnegative radial margin')
        if not np.isfinite(self.radius) or self.radius<=0:
            raise ValueError('Ball radius must be positive')
        remaining=self.radius-float(margin)
        distance=float(np.linalg.norm(value-center))
        feasible=remaining>=0
        error=max(0.,distance-max(0.,remaining))/self.radius
        return dict(feasible=bool(feasible),satisfied=bool(feasible and distance<=remaining+1e-12),
            violation=error,target=center,margin=float(margin),robust_radius=remaining,
            measured_distance=distance)


@dataclass(frozen=True)
class IntervalRegion:
    lower: tuple
    upper: tuple
    preferred: tuple|None=None

    def assess(self,value,uncertainty):
        lower,upper=vector(self.lower),vector(self.upper)
        if lower.shape!=upper.shape or value.shape!=lower.shape or np.any(lower>=upper):
            raise ValueError('Interval bounds must match and have positive widths')
        margin=np.broadcast_to(np.asarray(uncertainty,dtype=float),lower.shape)
        if not np.isfinite(margin).all() or np.any(margin<0):
            raise ValueError('Interval margins must be finite and nonnegative')
        safe_lower,safe_upper=lower+margin,upper-margin
        feasible=bool(np.all(safe_lower<=safe_upper))
        preferred=(lower+upper)/2 if self.preferred is None else vector(self.preferred)
        if preferred.shape!=lower.shape or np.any(preferred<lower) or np.any(preferred>upper):
            raise ValueError('Preferred target must be inside the nominal interval')
        # A task preference must itself leave room for the supplied uncertainty.
        # Projection is dimensionless and does not introduce a fixed correction.
        target=np.clip(preferred,safe_lower,safe_upper) if feasible else preferred
        errors=np.maximum(np.maximum(safe_lower-value,value-safe_upper),0.)/(upper-lower)
        return dict(feasible=feasible,
            satisfied=bool(feasible and np.all(value>=safe_lower-1e-12) and np.all(value<=safe_upper+1e-12)),
            violation=float(np.linalg.norm(errors)),target=target,
            margin=margin.tolist(),robust_lower=safe_lower.tolist(),robust_upper=safe_upper.tolist())


@dataclass(frozen=True)
class CoordinateConstraint:
    name: str
    indices: tuple
    region: object


@dataclass
class FeedbackDecision:
    action: str
    reason: str
    target: np.ndarray|None
    violation: float
    constraints: list=field(default_factory=list)

    def audit(self):
        return dict(action=self.action,reason=self.reason,
            target=None if self.target is None else self.target.tolist(),
            normalized_violation=self.violation,constraints=self.constraints)


def decide_constraints(state,constraints,uncertainty,*,observation_usable=True):
    state=vector(state)
    if not constraints:raise ValueError('At least one task constraint is required')
    target=state.copy();records=[];used=set();names=set()
    for constraint in constraints:
        indices=constraint.indices
        if (not constraint.name or constraint.name in names or not indices or
            any(not isinstance(i,int) or i<0 or i>=len(state) or i in used for i in indices) or
            len(set(indices))!=len(indices)):
            raise ValueError('Constraints require unique names and disjoint valid coordinate indices')
        if constraint.name not in uncertainty:
            raise ValueError('Every task constraint requires an explicit uncertainty margin')
        used.update(indices);names.add(constraint.name)
        assessment=constraint.region.assess(state[list(indices)],uncertainty[constraint.name])
        if not assessment['satisfied']:
            target[list(indices)]=assessment['target']
        records.append(dict(name=constraint.name,indices=list(indices),
            **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in assessment.items()}))
    violation=float(np.linalg.norm([r['violation'] for r in records]))
    if not observation_usable:
        return FeedbackDecision('reobserve','observation_not_usable',None,violation,records)
    if not all(r['feasible'] for r in records):
        return FeedbackDecision('reobserve','uncertainty_leaves_no_feasible_target',None,violation,records)
    if all(r['satisfied'] for r in records):
        return FeedbackDecision('complete','all_robust_constraints_satisfied',state.copy(),0.,records)
    return FeedbackDecision('correct','measured_state_outside_robust_goal',target,violation,records)


@dataclass
class FeedbackBudget:
    max_stalled_steps: int=3
    max_reobservations: int=2
    relative_progress: float=.02
    absolute_progress: float=1e-4
    best: float|None=None
    stalled: int=0
    reobservations: int=0

    def check(self,decision):
        if self.max_stalled_steps<1 or self.max_reobservations<0 or min(self.relative_progress,self.absolute_progress)<0:
            raise ValueError('Invalid bounded feedback policy')
        if decision.action=='complete':return decision
        if decision.action=='reobserve':
            self.reobservations+=1
            if self.reobservations>self.max_reobservations:
                return FeedbackDecision('stop','reobservation_budget_exhausted',None,decision.violation,decision.constraints)
            return decision
        if decision.action!='correct':raise ValueError('Unknown feedback action')
        self.reobservations=0
        if self.best is None or self.best-decision.violation>max(self.absolute_progress,self.relative_progress*self.best):
            self.best=decision.violation;self.stalled=0
        else:self.stalled+=1
        if self.stalled>=self.max_stalled_steps:
            return FeedbackDecision('stop','measured_correction_not_improving',None,decision.violation,decision.constraints)
        return decision
