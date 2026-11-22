"""Task-independent joint-space ranking and bounded two-pass pose search.

Callers supply task-valid Cartesian candidates. No object identity, scene seed,
successful-task angle, simulation object pose, or outcome enters the ranker.
"""
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class PlanningBudget:
    name: str
    max_attempts: int
    timeout_s: float


QUICK = PlanningBudget("quick", 2, .5)
FULL = PlanningBudget("full_fallback", 10, 10.)


def joint_score(solution, current, limits):
    """Use actual joint coordinates, not angle wraparound across hard stops."""
    q, current, limits = (np.asarray(v, dtype=float) for v in (solution, current, limits))
    if (q.ndim != 1 or current.shape != q.shape or limits.shape != (2, len(q))
            or not all(np.isfinite(v).all() for v in (q, current, limits))):
        raise ValueError("Invalid finite joint solution/current/limits")
    span = limits[1]-limits[0]
    if np.any(span <= 0) or np.any(q < limits[0]) or np.any(q > limits[1]):
        raise ValueError("IK solution outside joint limits")
    motion = float(np.linalg.norm((q-current)/span))
    margin = float(np.min(np.minimum(q-limits[0], limits[1]-q)/span))
    # Dimensionless, bounded generic limit penalty; not an object/task angle prior.
    return dict(score=motion+.1*(1.-2.*margin),
                normalized_joint_motion=motion, normalized_joint_limit_margin=margin)


def rank_candidates(candidates):
    valid = [c for c in candidates if c.get("ik_feasible")]
    return sorted(valid, key=lambda c: (c["joint_rank"]["score"], c["id"]))


def search_candidates(candidates, plan, audit):
    """IK rejection is provisional: full fallback also revisits unsolved poses.

    Timeouts are solver stop criteria, not hard preemption of an in-flight GPU
    optimization. The original collision/velocity/accuracy tests stay enabled.
    """
    ranked = rank_candidates(candidates)
    audit.update(candidate_order=[c["id"] for c in ranked], search_attempts=[],
                 ik_is_not_path_proof=True, ik_failure_is_not_unreachability_proof=True)
    remaining = [c for c in candidates if not c.get("ik_feasible")]
    for budget, pool in ((QUICK, ranked), (FULL, ranked+remaining)):
        for candidate in pool:
            entry = dict(candidate_id=candidate["id"], budget=budget.name,
                         max_attempts=budget.max_attempts, timeout_s=budget.timeout_s)
            audit["search_attempts"].append(entry)
            start = time.perf_counter()
            trajectory = plan(candidate, budget)
            entry.update(wall_s=time.perf_counter()-start, status=trajectory["status"],
                         planner_status=trajectory.get("planner_status"),
                         solver=trajectory.get("solver_audit", {}))
            if trajectory["status"] == "Success":
                audit.update(selected_candidate_id=candidate["id"], selected_budget=budget.name)
                return candidate, trajectory
    raise RuntimeError("All geometry-ranked candidates failed bounded quick/full path search")
