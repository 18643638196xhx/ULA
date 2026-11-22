import unittest

from adapters.statistics import summarize_episodes


class StatisticsTests(unittest.TestCase):
    def test_warmup_and_failures_are_not_successful_completion_times(self):
        result = summarize_episodes([
            dict(seed=0, success=True, elapsed_s=100),
            dict(seed=1, success=False, elapsed_s=20),
            dict(seed=2, success=True, elapsed_s=40),
        ])
        self.assertEqual(result["all_episodes"]["success_rate"], 2 / 3)
        steady = result["excluding_first_episode"]
        self.assertEqual(steady["episodes"], 2)
        self.assertEqual(steady["mean_attempt_elapsed_s"], 30)
        self.assertEqual(steady["mean_successful_completion_s"], 40)

    def test_no_success_or_no_post_warmup_data_has_no_completion_mean(self):
        result = summarize_episodes([dict(seed=0, success=False, elapsed_s=10)])
        self.assertIsNone(result["all_episodes"]["mean_successful_completion_s"])
        self.assertIsNone(result["excluding_first_episode"]["mean_attempt_elapsed_s"])


if __name__ == "__main__":
    unittest.main()
