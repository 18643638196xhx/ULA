"""Cheap geometry ranking, bounded lazy planning, and exact trajectory reuse.

Only explicit CAD contacts, visual object pose and robot state are used. The
native grasp selector/planner and all other task sources remain untouched.
"""
from contextlib import contextmanager, ExitStack
import time

import numpy as np
import transforms3d as t3d


def quaternion_distance(first, second):
    """RoboTwin's angular distance / pi, robust to sign and roundoff."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (first.shape != (4,) or second.shape != (4,) or
            not np.isfinite([first, second]).all() or
            min(np.linalg.norm(first), np.linalg.norm(second)) < 1e-12):
        raise ValueError("Invalid orientation for candidate ranking")
    dot = abs(float(np.dot(first, second)/(np.linalg.norm(first)*np.linalg.norm(second))))
    return float(2*np.arccos(np.clip(dot, 0., 1.))/np.pi)


def geometric_candidates(task, actor, arm, contact_ids=None, *, pre_distance=.06):
    """Generate the same native contact/rotation pool WITHOUT any planning."""
    from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
    top = GRASP_DIRECTION_DIC["top_down_little_left" if str(arm) == "right" else "top_down_little_right"]
    side = GRASP_DIRECTION_DIC[task.robot.get_grasp_perfect_direction(arm)]
    current = np.asarray(task.get_arm_pose(arm), dtype=float)
    if current.shape != (7,) or not np.isfinite(current).all() or not .03 <= pre_distance <= .15:
        raise ValueError("Invalid pregrasp geometry")
    ids = list(range(len(actor.config["contact_points_pose"]))) if contact_ids is None else list(contact_ids)
    if not ids or len(ids) != len(set(ids)) or any(
            not isinstance(i, int) or not 0 <= i < len(actor.config["contact_points_pose"]) for i in ids):
        raise ValueError("Invalid pregrasp contact selection")
    contact_to_grasp = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    candidates = []
    for contact_id in ids:
        contact = np.asarray(actor.get_contact_point(contact_id, "matrix"), dtype=float)
        if contact.shape != (4, 4) or not np.isfinite(contact).all():
            raise ValueError("Invalid visually transformed CAD contact")
        grasp = contact @ contact_to_grasp
        origin = np.r_[grasp[:3, 3]-(.12+pre_distance)*grasp[:3, 0],
                       t3d.quaternions.mat2quat(grasp[:3, :3])]
        center = np.r_[contact[:3, 3], t3d.quaternions.mat2quat(contact[:3, :3])]
        poses = task.robot.create_target_pose_list(origin.tolist(), center.tolist(), arm)
        for rotation_id, pose in enumerate(poses):
            pose = np.asarray(pose, dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError("Invalid rotated pregrasp candidate")
            top_dis, side_dis = quaternion_distance(pose[3:], top), quaternion_distance(pose[3:], side)
            if top_dis < .15:
                category, score = 0, top_dis
            elif side_dis < .15:
                category, score = 1, side_dis
            else:
                category, score = 2, .7*top_dis+.3*side_dis
            distance = float(np.linalg.norm(pose[:3]-current[:3]))
            candidates.append(dict(contact_id=contact_id, rotation_id=rotation_id,
                target_ee_pose=pose.tolist(), category=category, orientation_score=score,
                translation_from_current_m=distance))
    return sorted(candidates, key=lambda c: (c["category"], c["orientation_score"],
        c["translation_from_current_m"], c["contact_id"], c["rotation_id"]))


def select_ranked_pregrasp(task, actor, arm, audit, contact_ids=None, *, shortlist=8, batch_size=10):
    from adapters.pillbottle_known_model import joint_limit_audit
    if not isinstance(shortlist, int) or shortlist < 1 or batch_size != 10:
        raise ValueError("Invalid bounded pregrasp search budget")
    start = time.perf_counter()
    candidates = geometric_candidates(task, actor, arm, contact_ids)
    audit.update(search_mode="ranked", candidate_count=len(candidates), shortlist_size=shortlist,
        ranking="native_topdown_then_arm_preference_then_translation",
        planning_attempts=[], planning_calls=dict(single=0, batch=0),
        planning_wall_s=dict(single=0., batch=0.), geometry_generation_s=time.perf_counter()-start,
        expanded_search=False, trajectory_reused=False, complete_candidate_pool_retained=True)
    plan_one = getattr(task.robot, f"{arm}_plan_path")
    plan_batch = getattr(task.robot, f"{arm}_plan_multi_path")

    def accepted(candidate, plan, rank, phase):
        attempt = dict(contact_id=candidate["contact_id"], rotation_id=candidate["rotation_id"],
            rank=rank, phase=phase, status=plan["status"], planner_status=plan.get("planner_status"))
        audit["planning_attempts"].append(attempt)
        if plan["status"] != "Success":
            if "INVALID_START" in str(plan.get("planner_status", "")):
                raise RuntimeError("Pregrasp planner rejected measured robot start state")
            return False
        limits = joint_limit_audit(task.robot, arm, plan)
        attempt["physical_joint_limit_check"] = limits
        velocity, position = np.asarray(plan.get("velocity")), np.asarray(plan["position"])
        if velocity.shape != position.shape or len(position) < 2 or not np.isfinite(velocity).all():
            raise ValueError("Invalid accepted pregrasp trajectory")
        if not limits["accepted"]:
            return False
        audit.update(chosen_candidate=dict(candidate, rank=rank), trajectory_reused=True)
        return True

    try:
        limit = min(shortlist, len(candidates))
        for rank, candidate in enumerate(candidates[:limit]):
            tick = time.perf_counter()
            plan = plan_one(candidate["target_ee_pose"])
            audit["planning_wall_s"]["single"] += time.perf_counter()-tick
            audit["planning_calls"]["single"] += 1
            if accepted(candidate, plan, rank, "shortlist"):
                return candidate["target_ee_pose"], plan
        # Native CuRobo's warmed batch size is 10. Pad the final partial batch,
        # but never count padding as new candidates or execute its trajectory.
        audit["expanded_search"] = limit < len(candidates)
        for offset in range(limit, len(candidates), batch_size):
            batch = candidates[offset:offset+batch_size]
            targets = [c["target_ee_pose"] for c in batch]
            targets += [targets[-1]]*(batch_size-len(targets))
            tick = time.perf_counter()
            plans = plan_batch(targets)
            audit["planning_wall_s"]["batch"] += time.perf_counter()-tick
            audit["planning_calls"]["batch"] += 1
            statuses = plans["status"]
            if len(statuses) != batch_size:
                raise ValueError("Unexpected native pregrasp batch size")
            for index, candidate in enumerate(batch):
                plan = dict(status=str(statuses[index]))
                if plan["status"] == "Success":
                    plan.update(position=np.asarray(plans["position"][index]).copy(),
                        velocity=np.asarray(plans["velocity"][index]).copy())
                if accepted(candidate, plan, offset+index, "expanded_batch"):
                    return candidate["target_ee_pose"], plan
        raise RuntimeError("No feasible annotated pregrasp after bounded full candidate search")
    finally:
        audit["selection_wall_s"] = time.perf_counter()-start


@contextmanager
def known_traditional_timing(original, timer, task, perception):
    """Cover legacy batch search too, without editing shared timing code."""
    from adapters.timing import timed_method
    with original(timer, task, perception), ExitStack() as stack:
        for obj, name, label in (
            (task, "choose_grasp_pose", "grasp_candidate_search"),
            (task.robot, "left_plan_multi_path", "batch_motion_planning"),
            (task.robot, "right_plan_multi_path", "batch_motion_planning"),
        ):
            if hasattr(obj, name):
                stack.enter_context(timed_method(timer, obj, name, label))
        yield
