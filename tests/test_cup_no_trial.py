import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cup_no_trial_test', ROOT/'scripts/run_cup_no_trial_randomized.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Forbidden:
    def __getattr__(self, name):
        raise AssertionError('No-trial handoff must not interact with robot, vision, or object truth')


class NoTrialTests(unittest.TestCase):
    def test_bypass_never_moves_observes_or_claims_verified_grasp(self):
        result = {}
        decision = module.bypass_trial(Forbidden(), Forbidden(), 'left', result)
        self.assertTrue(decision['ready'])
        self.assertFalse(decision['lifted_grasp_verified'])
        self.assertFalse(result['trial_lift_executed'])
        self.assertFalse(result['verification_performed'])
        self.assertEqual(result['segments'], [])
        self.assertEqual(result['observations'], [])

    def test_setup_failures_are_not_method_failures(self):
        records = [dict(seed=0,success=False,failure_stage='setup',elapsed_s=1.),
            dict(seed=1,success=True,failure_stage=None,elapsed_s=20.),
            dict(seed=2,success=False,failure_stage='approach',elapsed_s=2.),
            dict(seed=3,success=True,failure_stage=None,elapsed_s=6.)]
        result = module.progress(records)
        self.assertEqual(result['raw_seed_attempts'],4)
        self.assertEqual(result['completed_valid_episodes'],3)
        self.assertEqual(result['successes'],2)
        self.assertAlmostEqual(result['success_rate'],2/3)
        self.assertEqual(result['initialization_rejection_seeds'],[0])
        self.assertEqual(result['statistics']['excluding_first_episode']['mean_successful_completion_s'],6.)


if __name__ == '__main__':
    unittest.main()
