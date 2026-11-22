from types import SimpleNamespace
import unittest

from adapters.grasp import GraspComplete
from adapters.timing import EpisodeTiming, summarize_timings, timed_method, vla_timing


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TimingTests(unittest.TestCase):
    def test_nested_calls_are_exclusive_but_keep_inclusive_diagnostics(self):
        clock = Clock()
        timer = EpisodeTiming(clock)
        with timer.phase("vla"):
            clock.advance(1)
            with timer.span("action"):
                clock.advance(2)
                with timer.span("TOPP"):
                    clock.advance(3)
                clock.advance(4)
            clock.advance(5)
        result = timer.snapshot(15)["stages"]["vla"]
        self.assertEqual(result["wall_s"], 15)
        self.assertEqual(result["components_s"], dict(other=6, action=6, TOPP=3))
        self.assertEqual(result["inclusive_s"]["action"], 9)
        self.assertEqual(sum(result["components_s"].values()), result["wall_s"])

    def test_exception_records_last_call_and_restores_inherited_method(self):
        clock = Clock()
        timer = EpisodeTiming(clock)

        class Target:
            def run(self):
                clock.advance(2)
                raise GraspComplete()

        target = Target()
        with self.assertRaises(GraspComplete):
            with timer.phase("vla"), timed_method(timer, target, "run", "action"):
                target.run()
        self.assertNotIn("run", vars(target))
        self.assertEqual(timer.stages["vla"]["calls"]["action"], 1)
        self.assertEqual(timer.stages["vla"]["components_s"]["action"], 2)

    def test_hooks_preserve_args_results_and_restore_instance_callable(self):
        timer = EpisodeTiming()
        calls = []
        response = object()
        original = lambda *a, **kw: calls.append((a, kw)) or response
        target = SimpleNamespace(run=original)
        with timer.phase("test"), timed_method(timer, target, "run", "call"):
            self.assertIs(target.run(1, setting=2), response)
        self.assertIs(target.run, original)
        self.assertEqual(calls, [((1,), dict(setting=2))])

    def test_vla_final_interrupted_chunk_keeps_every_action_and_topp_call(self):
        clock = Clock()
        timer = EpisodeTiming(clock)
        planner = SimpleNamespace(TOPP=lambda: clock.advance(0.1))
        task = SimpleNamespace(robot=SimpleNamespace(left_mplib_planner=planner,
                                                     right_mplib_planner=SimpleNamespace(TOPP=planner.TOPP)),
                               get_obs=lambda: clock.advance(0.2))
        actions = []

        def action(value):
            actions.append(value)
            planner.TOPP()
            clock.advance(0.3)
            if value == 2:
                raise GraspComplete()

        task.take_action = action
        policy = SimpleNamespace(_prepare_observation=lambda: clock.advance(0.1),
                                 _output_transform=lambda: clock.advance(0.1))

        def get_action():
            policy._prepare_observation()
            clock.advance(0.5)  # Model + synchronization.
            policy._output_transform()

        model = SimpleNamespace(policy=policy, get_action=get_action,
                                update_observation_window=lambda: None)
        with self.assertRaises(GraspComplete):
            with timer.phase("vla_rollout"), vla_timing(timer, task, model):
                task.get_obs()
                model.get_action()
                for index in range(10):
                    task.take_action(index)
        stage = timer.stages["vla_rollout"]
        self.assertEqual(actions, [0, 1, 2])
        self.assertEqual(stage["calls"]["action_execution"], 3)
        self.assertEqual(stage["calls"]["trajectory_time_parameterization"], 3)
        self.assertAlmostEqual(stage["components_s"]["model_inference"], 0.5)
        self.assertAlmostEqual(sum(stage["components_s"].values()), stage["wall_s"])
        self.assertIs(task.take_action, action)
        self.assertIs(model.get_action, get_action)

    def test_summary_excludes_warmup_and_reports_all_attempts_separately(self):
        def episode(seed, success, duration):
            return dict(seed=seed, success=success, timing_profile=dict(
                stages=dict(vla_rollout=dict(wall_s=duration, components_s=dict(other=duration),
                                            inclusive_s={}, calls={})),
                episode_wall_s=duration + 1, unattributed_s=1))

        result = summarize_timings([episode(0, True, 100), episode(1, True, 4), episode(2, False, 2)])
        all_attempts = result["excluding_first_episode"]
        self.assertEqual(all_attempts["samples"], 2)
        self.assertEqual(all_attempts["stages"]["vla_rollout"]["wall_s"], 3)
        successful = result["successful_excluding_first_episode"]
        self.assertEqual(successful["seeds"], [1])
        self.assertEqual(successful["stages"]["vla_rollout"]["components_s"]["other"], 4)


if __name__ == "__main__":
    unittest.main()
