import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adapters.perception import ExternalPerception, PerceptionWorker

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/fake_vision_worker.py"


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.worker = PerceptionWorker(ROOT, self.output / "worker", timeout_s=10)
        self.addCleanup(self.worker.close)
        self.launch = patch.object(self.worker, "_launch_spec", return_value=(
            [sys.executable, "-u", str(FIXTURE)], ROOT,
            dict(os.environ, PYTHONDONTWRITEBYTECODE="1")))
        self.launch.start()
        self.addCleanup(self.launch.stop)

    def estimate(self, marker, directory="capture"):
        facade = ExternalPerception(ROOT, self.output / directory, worker=self.worker)
        return facade.estimate(np.full((2, 2, 3), marker), np.ones((2, 2)), np.eye(3), np.eye(4))

    def test_twenty_fresh_frames_across_facades_share_one_model_stack(self):
        self.assertEqual(self.worker.summary()["starts"], 0)
        pids = set()
        for index in range(20):
            value = self.estimate(index, f"episode{index // 2}/stage{index % 2}")
            pids.add(value.timings_s["vision_worker_pid"])
            self.assertEqual(value.cup_world_pose[0, 3], index)
            self.assertTrue(np.all(value.coaster_world_center == index))
            self.assertEqual(value.timings_s["vision_request_id"], index + 1)
            self.assertEqual(value.timings_s["vision_model_load"], 0.125 if index == 0 else 0)
        self.assertEqual(len(pids), 1)
        self.assertEqual(self.worker.summary()["starts"], 1)
        self.assertEqual(self.worker.summary()["completed_requests"], 20)
        log = (self.output / "worker/perception.log").read_text()
        self.assertEqual(log.count("Python model initialization log"), 1)
        self.assertEqual(log.count("Native model initialization log"), 1)
        self.worker.close()
        self.assertFalse(self.worker.summary()["running"])
        self.assertIsNotNone(self.worker.process.poll())
        self.worker.close()  # Idempotent cleanup.

    def test_frame_error_does_not_return_stale_estimate_or_reload(self):
        self.estimate(42)
        with self.assertRaisesRegex(RuntimeError, "fake frame failure"):
            self.estimate(-1)  # Old estimate.npz deliberately still exists.
        self.assertEqual(self.estimate(7).cup_world_pose[0, 3], 7)
        self.assertEqual(self.worker.summary()["starts"], 1)
        self.assertEqual(self.worker.summary()["completed_requests"], 2)

    def test_crash_fails_without_automatic_restart(self):
        self.estimate(1)
        self.worker.process.kill()
        self.worker.process.wait()
        with self.assertRaisesRegex(RuntimeError, "died"):
            self.estimate(2)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.estimate(3)
        self.assertEqual(self.worker.summary()["starts"], 1)

    def test_timeout_terminates_child_and_closes_worker(self):
        self.estimate(1)
        self.worker.timeout_s = 0.1
        with self.assertRaises(TimeoutError):
            self.estimate(-2)
        self.assertFalse(self.worker.summary()["running"])
        self.assertTrue(self.worker.summary()["closed"])

    def test_startup_error_cleans_up(self):
        command, cwd, env = self.worker._launch_spec()
        self.launch.target._launch_spec.return_value = (command + ["bad-load"], cwd, env)
        with self.assertRaisesRegex(RuntimeError, "fake initialization failure"):
            self.estimate(1)
        self.assertFalse(self.worker.summary()["running"])

    def test_rejects_inconsistent_model_parameters(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ExternalPerception(ROOT, self.output, fp_iterations=9, worker=self.worker)

    def test_pointcloud_payload_roundtrip_does_not_invent_a_pose(self):
        command, cwd, env = self.worker._launch_spec()
        self.launch.target._launch_spec.return_value = (command + ["pointcloud"], cwd, env)
        estimate = self.estimate(2)
        self.assertIsNone(estimate.cup_world_pose)
        np.testing.assert_allclose(estimate.cup_bounds_world, [[2,0,0],[2.1,0.1,0.1]])
        self.assertFalse(estimate.observation_audit["uses_prepared_object_model"])
        np.testing.assert_allclose(estimate.cup_points_world,np.full((40,3),2))

    def test_pointcloud_launch_does_not_add_foundationpose_paths(self):
        worker = PerceptionWorker(ROOT,self.output / "pointcloud_worker",backend="pointcloud")
        self.addCleanup(worker.close)
        command, cwd, environment = worker._launch_spec()
        self.assertEqual(command[command.index("--backend")+1],"pointcloud")
        self.assertNotIn("FoundationPose",environment["PYTHONPATH"])
        self.assertEqual(worker.summary()["backend"],"pointcloud")


if __name__ == "__main__":
    unittest.main()
