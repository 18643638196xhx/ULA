import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("pi05_continuation_tests",ROOT/"scripts/continue_pi05_pickup_training.py")
continuation=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)


class ContinuationTests(unittest.TestCase):
    def protocol(self,directory):
        checkpoint=directory/"source"/"checkpoints"/"10000"
        for name in ("params","assets","train_state"):
            (checkpoint/name).mkdir(parents=True)
            (checkpoint/name/"fixture").write_text(name)
        return dict(architecture="native_pi0.5_LoRA_only",WAM=False,WorldLA=False,
            status="finished",completed_steps=10000,target_steps=10000,
            sources_unchanged=True,latest_checkpoint=str(checkpoint))

    def test_budget_is_total_not_extra_and_source_must_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=self.protocol(Path(temporary))
            continuation.validate_source(source,10000,100000)
            for changes in [dict(status="running"),dict(completed_steps=9999),
                            dict(sources_unchanged=False),dict(WAM=True)]:
                with self.assertRaises(ValueError):
                    continuation.validate_source(dict(source,**changes),10000,100000)
            with self.assertRaises(ValueError):
                continuation.validate_source(source,10000,10000)

    def test_source_checkpoint_survives_new_run_retention(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=self.protocol(root)
            output=root/"continued"
            output.mkdir()
            copied=continuation.prepare_checkpoint(source,output,10000,100000)
            original=Path(source["latest_checkpoint"])/"params/fixture"
            duplicate=copied/"params/fixture"
            self.assertEqual(original.stat().st_ino,duplicate.stat().st_ino)
            duplicate.unlink()
            self.assertEqual(original.read_text(),"params")
            self.assertEqual(json.loads((output/"protocol.json").read_text())["completed_steps"],10000)
            with self.assertRaises(FileExistsError):
                continuation.prepare_checkpoint(source,output,10000,100000)

    def test_own_process_is_live_and_missing_pid_is_not(self):
        self.assertTrue(continuation.process_alive(os.getpid()))
        self.assertFalse(continuation.process_alive(None))
        self.assertFalse(continuation.process_alive(999999999))


if __name__=="__main__":
    unittest.main()
