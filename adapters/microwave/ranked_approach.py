"""Geometry-conditioned approach, IK ranking, then budgeted full path checks.

Microwave constraints remain explicit: vertical bar, near-horizontal closing
axis, a bounded approach tilt, and existing safety standoff distances. These
are not a learned universal grasp generator. Only the rank/search engine is
task independent. Original source files and other task controllers are intact.
"""
from contextlib import contextmanager
import time

import numpy as np

from adapters.microwave import control
from adapters.microwave.geometry import approach_target
from adapters.ranked_motion import FULL, QUICK, joint_score, search_candidates


@contextmanager
def replaced(obj, name, value):
    original, owned = getattr(obj, name), name in vars(obj)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, original)
        else:
            delattr(obj, name)


class QueryCaptured(Exception):
    def __init__(self, start, goal):
        self.start, self.goal = start, goal


def capture_query(robot, arm, matrix):
    """Reuse EXACT native world/EE/endlink/base transforms without planning.

    Only intercept the terminal solver call, synchronously in this process.
    No trajectories are returned/executed here. Unsupported IPC planners fail
    explicitly rather than silently ranking in a different coordinate frame.
    """
    from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz

    if robot.communication_flag:
        raise RuntimeError("Ranked approach requires an in-process CuRobo planner")
    planner = getattr(robot, f"{arm}_planner")
    motion_gen = planner.motion_gen

    def capture(start, goal, config):
        raise QueryCaptured(start, goal)

    with replaced(motion_gen, "plan_single", capture):
        try:
            getattr(robot, f"{arm}_plan_path")(matrix_to_pose_wxyz(matrix))
        except QueryCaptured as query:
            return motion_gen, query.start, query.goal
    raise RuntimeError("Native planner did not expose the expected single-pose query")


def geometric_candidates(hinge, handle, ee_to_tcp, current_tcp, *, clearance=.025, pitch_deg=0.):
    """Sample the allowed task cone, plus current tool direction projected on it.

    No 10-degree or successful-seed preference: the numeric sampling order is
    discarded by joint-state ranking. Preserve the old bounded candidates in
    the search pool; distance is a safety constraint, not a ranking preference.
    """
    normal = np.asarray(hinge["initial_normal"], dtype=float)
    forward = np.asarray(current_tcp)[:3, 0]
    projected = np.degrees(np.arctan2(-forward[2], -forward@normal))
    projected = float(np.clip(projected, -20., 20.))
    pitches = sorted(set(round(float(p), 6)
                         for p in [*np.linspace(-20., 20., 5), projected, pitch_deg]))
    standoff = max(.08, clearance)
    distances = sorted(set((standoff, max(.10, standoff), max(.12, standoff))))
    candidates = []
    for pitch in pitches:
        for distance in distances:
            candidates.append(dict(id=len(candidates), pitch_deg=pitch, standoff_m=distance,
                target_matrix=approach_target(hinge, handle, ee_to_tcp, distance, pitch)))
    return candidates


def prepare_candidate(robot, arm, candidate):
    mg, start, goal = capture_query(robot, arm, candidate["target_matrix"])
    limits = mg.kinematics.get_joint_limits().position.detach().cpu().numpy()
    current = start.position.detach().cpu().numpy().reshape(-1)
    begin = time.perf_counter()
    result = mg.solve_ik(goal, retract_config=start.position, return_seeds=4)
    success = result.success.detach().cpu().numpy().reshape(-1)
    solutions = result.solution.detach().cpu().numpy().reshape(-1, len(current))
    ranked = []
    for valid, solution in zip(success, solutions):
        if valid:
            try:
                ranked.append((joint_score(solution, current, limits), solution))
            except ValueError:
                continue
    candidate.update(ik_feasible=bool(ranked), ik_wall_s=time.perf_counter()-begin,
                     ik_successes=len(ranked), ik_is_not_path_proof=True,
                     collision_scope="native_robot_self_and_configured_table_not_full_scene")
    if ranked:
        score, solution = min(ranked, key=lambda item: item[0]["score"])
        candidate.update(joint_rank=score, ik_joint_solution=solution.tolist())
    return mg


def plan_target(robot, arm, matrix, budget):
    """One outer call; clone/change only attempt/time budgets, not constraints."""
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    mg, start, goal = capture_query(robot, arm, matrix)
    config = MotionGenPlanConfig(max_attempts=budget.max_attempts, timeout=budget.timeout_s)
    result = mg.plan_single(start, goal, config)
    audit = {key: float(getattr(result, key, 0.) or 0.) for key in
             ("attempts", "trajopt_attempts", "solve_time", "total_time", "ik_time", "graph_time", "trajopt_time")}
    plan = dict(status="Success" if bool(result.success.item()) else "Fail",
                planner_status=str(result.status), solver_audit=audit)
    if plan["status"] == "Success":
        plan.update(position=result.interpolated_plan.position.detach().cpu().numpy().copy(),
                    velocity=result.interpolated_plan.velocity.detach().cpu().numpy().copy())
    return plan


def execute_plan(task, arm, matrix, plan, audit, *, timer=None, slowdown=2):
    """Execute cached validated trajectory with ORIGINAL arrival/settling guard."""
    from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz

    expected = matrix_to_pose_wxyz(matrix)
    used = False

    def cached(target):
        nonlocal used
        if used or not np.allclose(target, expected, atol=1e-8, rtol=0):
            raise RuntimeError("Cached trajectory requested twice or for a different target")
        used = True
        return plan

    with replaced(task.robot, f"{arm}_plan_path", cached):
        return control.move_to(task, arm, matrix, audit, timer=timer, slowdown=slowdown)


def approach_handle(task, hinge, handle, result, *, arm="left", clearance=.08, timer=None,
                    gripper_opening=1., pitch_deg=0.):
    from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix

    if not np.isfinite(gripper_opening) or not .2 <= gripper_opening <= 1.:
        raise ValueError("Invalid initial gripper command")
    result.update(complete=False, uses_object_truth=False, arm=arm,
        initial_gripper_command=gripper_opening, near_approach_segments=[],
        approach_variant="geometry_ik_joint_rank_two_pass_budget", fixed_angle_priority=False,
        constraints_changed=False, extra_trial_motion=False)
    if not task.move(task.open_gripper(arm, pos=gripper_opening)) or not task.plan_success:
        raise RuntimeError("Failed to prepare gripper before handle approach")
    ee = pose_wxyz_to_matrix(task.get_arm_pose(arm))
    tcp = pose_wxyz_to_matrix(getattr(task.robot, f"get_{arm}_tcp_pose")())
    offset = np.linalg.solve(ee, tcp)
    with control._span(timer, "candidate_generation"):
        candidates = geometric_candidates(hinge, handle, offset, tcp,
                                           clearance=clearance, pitch_deg=pitch_deg)
    # Store only serializable task-level fields, not CUDA query objects.
    result["standoff_candidates"] = candidates
    with control._span(timer, "ik_candidate_screening"):
        for candidate in candidates:
            prepare_candidate(task.robot, arm, candidate)

    def plan(candidate, budget):
        with control._span(timer, "motion_planning"):
            return plan_target(task.robot, arm, candidate["target_matrix"], budget)

    candidate, trajectory = search_candidates(candidates, plan, result)
    result.update(approach_pitch_deg=candidate["pitch_deg"], standoff_m=candidate["standoff_m"],
                  selected_joint_rank=candidate.get("joint_rank"))
    result["arrival"] = execute_plan(task, arm, candidate["target_matrix"], trajectory,
                                     result, timer=timer)
    remaining = candidate["standoff_m"]-clearance
    if remaining > 1e-6:
        count = int(np.ceil(remaining/.02))
        for distance in np.linspace(candidate["standoff_m"], clearance, count+1)[1:]:
            target = approach_target(hinge, handle, offset, float(distance), candidate["pitch_deg"])
            segment = dict(clearance_m=float(distance), search_attempts=[])
            result["near_approach_segments"].append(segment)
            trajectory = None
            for budget in (QUICK, FULL):
                with control._span(timer, "motion_planning"):
                    trajectory = plan_target(task.robot, arm, target, budget)
                segment["search_attempts"].append(dict(budget=budget.name, status=trajectory["status"],
                    planner_status=trajectory.get("planner_status"), solver=trajectory.get("solver_audit")))
                if trajectory["status"] == "Success":
                    break
            if trajectory is None or trajectory["status"] != "Success":
                raise RuntimeError("Ranked approach near segment failed bounded quick/full planning")
            result["arrival"] = execute_plan(task, arm, target, trajectory, segment,
                                             timer=timer, slowdown=3)
    result["complete"] = True
