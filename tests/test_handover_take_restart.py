import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("take_restart_tests",ROOT/"scripts/restart_handover_take_training.py")
restart=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restart)


class TakeRestartTests(unittest.TestCase):
    def test_uses_existing_native_trainer_with_early_saves_and_four_cards(self):
        command=restart.training_command(Path("dataset"),Path("output"))
        self.assertTrue(command[1].endswith("train_pi05_handover_pickup.sh"))
        self.assertEqual(command[command.index("--save-interval")+1],"250")
        self.assertEqual(command[command.index("--fsdp-devices")+1],"4")
        self.assertEqual(command[command.index("--steps")+1],"10000")
        self.assertNotIn("--resume",command)

    def test_real_dataset_is_right_take_with_disjoint_scenes(self):
        metadata=restart.validate_dataset(ROOT/"datasets/handover_take50_20261107_v1")
        self.assertEqual(metadata["splits"],{"train":45,"val":5})
        self.assertEqual(metadata["frames"],6392)

    def test_atomic_status_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"status.json"
            restart.write_json(path,dict(status="running",completed_steps=250))
            self.assertEqual(json.loads(path.read_text())["completed_steps"],250)
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__=="__main__":
    unittest.main()
