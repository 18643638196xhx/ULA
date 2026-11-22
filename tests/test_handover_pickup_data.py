from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.handover.pickup_data import slice_control, choose_splits, assert_disjoint_seeds, DenseRecorder


class PickupDataTests(unittest.TestCase):
    def test_episode_split_is_deterministic_and_disjoint(self):
        values = choose_splits(50)
        self.assertEqual(values.count("train"), 45)
        self.assertEqual(values.count("val"), 5)
        self.assertEqual(values, choose_splits(50))
        with self.assertRaises(ValueError):
            assert_disjoint_seeds([10, 11], [11, 12])
        assert_disjoint_seeds([10, 11], [12, 13])

    def test_slicing_preserves_velocity_and_gripper_timing(self):
        sequence = dict(left_arm=dict(position=np.arange(30).reshape(10,3), velocity=np.ones((10,3))),
            right_arm=None, left_gripper=dict(num_step=6, result=list(range(6)), per_step=.01), right_gripper=None)
        value = slice_control(sequence, 3, 8)
        np.testing.assert_array_equal(value["left_arm"]["position"], sequence["left_arm"]["position"][3:8])
        self.assertEqual(value["left_gripper"]["num_step"], 3)
        self.assertEqual(value["left_gripper"]["result"], [3,4,5])
        self.assertIsNone(slice_control(sequence, 8, 10)["left_gripper"])
        self.assertEqual(sequence["left_gripper"]["num_step"], 6)

    def test_hdf_pairs_keep_before_observation_and_future_command_separate(self):
        import h5py
        recorder = DenseRecorder(None, None, None)
        images = {c: np.zeros((8,8,3), np.uint8) for c in ("cam_high","cam_left_wrist","cam_right_wrist")}
        depths = {c: np.ones((8,8), np.float32) for c in images}
        recorder.frames = [dict(state=np.zeros(14), action=np.ones(14),
            actual_arm_feedback=np.zeros(12), images=images, depths=depths,
            stage="insert", physics_step_before=0, physics_step_after=15)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"sample.hdf5"
            recorder.save(path, dict(seed=123))
            with h5py.File(path) as data:
                self.assertTrue(np.all(data["observation.state"][()] == 0))
                self.assertTrue(np.all(data["action"][()] == 1))
                self.assertNotIn("block_pose", data)
                self.assertEqual(data["images/cam_high"].shape, (1,8,8,3))
            with self.assertRaises(FileExistsError):
                recorder.save(path, {})


if __name__ == "__main__":
    unittest.main()
