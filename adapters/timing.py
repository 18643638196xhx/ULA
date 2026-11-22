"""Non-overlapping wall-time accounting; hooks never change callable results."""

from contextlib import contextmanager, ExitStack
from functools import wraps
from statistics import mean
import time


class EpisodeTiming:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.stages = {}
        self._stage = None
        self._stack = []

    def _charge(self, now):
        label = self._stack[-1]
        parts = self._stage["components_s"]
        parts[label] = parts.get(label, 0.0) + now - self._last
        self._last = now

    @contextmanager
    def phase(self, name):
        if self._stage is not None:
            raise RuntimeError("Timing phases must not overlap")
        self._stage = self.stages.setdefault(name, dict(
            wall_s=0.0, components_s={}, inclusive_s={}, calls={}))
        start = self.clock()
        self._last = start
        self._stack = ["other"]
        try:
            yield
        finally:
            end = self.clock()
            self._charge(end)
            self._stage["wall_s"] += end - start
            self._stage = None
            self._stack = []

    @contextmanager
    def span(self, label):
        if self._stage is None:
            yield
            return
        start = self.clock()
        self._charge(start)
        self._stack.append(label)
        try:
            yield
        finally:
            end = self.clock()
            self._charge(end)
            self._stack.pop()
            stage = self._stage
            stage["inclusive_s"][label] = stage["inclusive_s"].get(label, 0.0) + end - start
            stage["calls"][label] = stage["calls"].get(label, 0) + 1

    def snapshot(self, elapsed_s):
        accounted = sum(stage["wall_s"] for stage in self.stages.values())
        return dict(stages=self.stages, accounted_s=accounted,
                    unattributed_s=elapsed_s - accounted, episode_wall_s=elapsed_s)


@contextmanager
def timed_method(timer, obj, name, label):
    """Patch only this instance; restore inherited vs instance-owned attributes."""
    original = getattr(obj, name)
    owned = name in vars(obj)

    @wraps(original)
    def measured(*args, **kwargs):
        with timer.span(label):
            return original(*args, **kwargs)

    setattr(obj, name, measured)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, original)
        else:
            delattr(obj, name)


@contextmanager
def traditional_timing(timer, task, perception):
    from adapters import approach_geometry, pointcloud_placement, pointcloud_registration

    with ExitStack() as stack:
        if hasattr(perception, "supplementary"):
            stack.enter_context(timed_method(timer, perception, "supplementary", "supplementary_perception"))
        for obj, name, label in (
            (task, "get_camera_observation", "observation"),
            (perception, "estimate", "perception"),
            (task, "take_dense_action", "trajectory_execution"),
            (task, "set_gripper", "gripper_trajectory_generation"),
            (task, "grasp_actor", "grasp_target_generation"),
            (approach_geometry, "generate_geometry_candidates", "geometry_target_generation"),
            (approach_geometry, "generate_bounds_candidates", "bounds_target_generation"),
            (pointcloud_registration, "estimate_pointcloud_motion", "pointcloud_registration"),
            (pointcloud_placement, "placement_candidates", "pointcloud_placement_target_generation"),
            (task.robot, "left_plan_path", "motion_planning"),
            (task.robot, "right_plan_path", "motion_planning"),
        ):
            stack.enter_context(timed_method(timer, obj, name, label))
        yield


@contextmanager
def vla_timing(timer, task, model):
    # get_action returns NumPy actions, so its duration includes completion of
    # asynchronous JAX sampling and host transfer. No new GPU sync is inserted.
    # Nested TOPP calls are subtracted from action_execution, not double counted.
    with ExitStack() as stack:
        for obj, name, label in (
            (task, "get_obs", "observation"),
            (task, "take_action", "action_execution"),
            (task.robot.left_mplib_planner, "TOPP", "trajectory_time_parameterization"),
            (task.robot.right_mplib_planner, "TOPP", "trajectory_time_parameterization"),
            (model, "update_observation_window", "observation_packing"),
            (model, "get_action", "model_inference"),
            (model.policy, "_prepare_observation", "input_preprocessing"),
            (model.policy, "_output_transform", "output_postprocessing"),
        ):
            if not hasattr(obj, name):
                # Native pi0.5 policies expose infer() only; skip absent hooks.
                continue
            stack.enter_context(timed_method(timer, obj, name, label))
        yield


def summarize_timings(episodes):
    """Average equal-weight episode timings, never per-call or per-step means."""
    def summary(records):
        records = [record for record in records if "timing_profile" in record]
        if not records:
            return dict(samples=0, seeds=[], mean_episode_wall_s=None, stages={})
        profiles = [record["timing_profile"] for record in records]
        names = sorted({name for p in profiles for name in p["stages"]})
        stages = {}
        for name in names:
            items = [p["stages"].get(name, {}) for p in profiles]
            result = dict(wall_s=mean(item.get("wall_s", 0.0) for item in items))
            for field in ("components_s", "inclusive_s", "calls"):
                keys = sorted({key for item in items for key in item.get(field, {})})
                result[field] = {key: mean(item.get(field, {}).get(key, 0.0) for item in items)
                                 for key in keys}
            stages[name] = result
        return dict(samples=len(records), seeds=[r["seed"] for r in records], stages=stages,
                    mean_episode_wall_s=mean(p["episode_wall_s"] for p in profiles),
                    mean_unattributed_s=mean(p["unattributed_s"] for p in profiles))

    return dict(
        all_episodes=summary(episodes),
        excluding_first_episode=summary(episodes[1:]),
        successful_excluding_first_episode=summary([r for r in episodes[1:] if r["success"]]),
        definition="Host wall time, seconds. Stage wall times are disjoint; components_s "
                   "are exclusive and sum to their stage wall_s. inclusive_s are nested "
                   "diagnostics and MUST NOT be added to components_s. VLA model_inference "
                   "includes sampling, synchronization/host transfer and small dispatch costs, "
                   "but excludes separately measured input/output transforms. action_execution "
                   "includes interpolation, simulation and success checks, but excludes TOPP. "
                   "No extra GPU synchronization is introduced. First-episode warmup remains included.",
    )
