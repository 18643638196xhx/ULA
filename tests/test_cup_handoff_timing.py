"""Counterfactual controls must share their original prefix and only delay gate."""
from types import SimpleNamespace
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from adapters.cup_handoff_timing import (DelayedCommandMonitor, PairedObservationPolicy,
                                          action_hash, prefix_comparison)
from adapters.worldla import WorldLA


class HandoffTimingTests(unittest.TestCase):
    def test_zero_delay_is_original_three_closed_commands(self):
        context = {}
        monitor = DelayedCommandMonitor(context, extra_actions=0)
        self.assertFalse(monitor.update(.01,1)['ready'])
        self.assertFalse(monitor.update(.01,2)['ready'])
        self.assertTrue(monitor.update(.01,3)['ready'])
        self.assertEqual(context['first_gate_step'],3)

    def test_exactly_ten_extra_actions_and_no_reset_of_first_gate(self):
        context = {}
        monitor = DelayedCommandMonitor(context, extra_actions=10)
        for step in range(1,13):
            self.assertFalse(monitor.update(.01,step)['ready'])
        self.assertTrue(monitor.update(.01,13)['ready'])
        self.assertEqual(context['first_gate_step'],3)

    def test_reopened_delay_endpoint_does_not_claim_closed_grasp(self):
        monitor = DelayedCommandMonitor({}, extra_actions=1)
        for step in range(1,4):
            monitor.update(.01,step)
        decision = monitor.update(.8,4)
        self.assertTrue(decision['ready'])  # experiment stops at its fixed horizon
        self.assertFalse(decision['closed_stable_at_handoff'])
        self.assertFalse(decision['lifted_grasp_verified'])

    def test_observations_replayed_only_before_original_boundary(self):
        policy = PairedObservationPolicy.__new__(PairedObservationPolicy)
        policy.context = dict(reference=dict(first_gate_step=32,
            prefix_observations={30:{'source':'baseline'}}),replayed_observation_starts=[])
        task = SimpleNamespace(take_action_cnt=30)
        with patch.object(WorldLA,'step') as step:
            policy.step(task,{'source':'live'})
            self.assertEqual(step.call_args.args[1]['source'],'baseline')
            task.take_action_cnt=40
            policy.step(task,{'source':'live'})
            self.assertEqual(step.call_args.args[1]['source'],'live')

    def test_same_actions_with_different_physics_are_not_a_valid_pair(self):
        trace = [dict(actual_ee_pose=[0,0,1,1,0,0,0],cup_world_pose=np.eye(4).tolist())]
        reference = dict(first_gate_step=1,actions=[[0]*14],trace=trace)
        context = dict(reference=reference,first_gate_step=1,actions=[[0]*14],trace=trace)
        self.assertTrue(prefix_comparison(context)['physically_paired_within_1mm'])
        changed = dict(trace[0],actual_ee_pose=[.01,0,1,1,0,0,0])
        context['trace']=[changed]
        self.assertFalse(prefix_comparison(context)['physically_paired_within_1mm'])
        self.assertEqual(action_hash(reference['actions']),action_hash(context['actions']))

    def test_vision_gpu_override_is_process_local_and_rejects_multi_gpu(self):
        path=Path(__file__).resolve().parents[1]/'scripts/run_cup_handoff_timing.py'
        spec=importlib.util.spec_from_file_location('timing_split_test',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        worker=module.TimingVisionWorker.__new__(module.TimingVisionWorker)
        with patch.object(module.TrialCupWorker,'_launch_spec',return_value=(['python'],'/tmp',{'CUDA_VISIBLE_DEVICES':'7'})):
            with patch.dict(os.environ,{'ULA_TIMING_VISION_GPU':'6'}):
                self.assertEqual(worker._launch_spec()[2]['CUDA_VISIBLE_DEVICES'],'6')
            with patch.dict(os.environ,{'ULA_TIMING_VISION_GPU':'6,7'}):
                with self.assertRaises(ValueError):
                    worker._launch_spec()


if __name__ == '__main__':
    unittest.main()
