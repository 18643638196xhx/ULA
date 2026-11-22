import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("pi05_pickup_training_tests",ROOT/"scripts/train_pi05_handover_pickup.py")
training=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


class NativePickupTests(unittest.TestCase):
    def test_episode_ranges_do_not_leak(self):
        metadata=dict(episodes=[dict(length=2,split="train",seed=1),
                                dict(length=3,split="val",seed=2),
                                dict(length=1,split="train",seed=3)])
        self.assertEqual(training.episode_indices(metadata,"train"),([0,1,5],[1,3]))
        self.assertEqual(training.episode_indices(metadata,"val"),([2,3,4],[2]))

    def test_norm_is_train_only_and_gripper_stays_absolute(self):
        import h5py
        # Statistics helper imports only the local OpenPI normalization type.
        import sys
        sys.path.insert(0,str(ROOT/"external/pi05/src"))
        with tempfile.TemporaryDirectory() as directory:
            rows=[]
            for split,value in [("train",1.),("val",1000.)]:
                path=Path(directory)/(split+".hdf5")
                state=np.full((2,14),value,np.float32)
                action=state+.2
                with h5py.File(path,"w") as target:
                    target["observation.state"]=state
                    target["action"]=action
                rows.append(dict(path=str(path),length=2,split=split))
            stats=training.train_only_statistics(dict(episodes=rows),3)
            np.testing.assert_allclose(stats["state"].mean,1.)
            np.testing.assert_allclose(stats["actions"].mean[[0,1,7]],.2,atol=1e-6)
            np.testing.assert_allclose(stats["actions"].mean[[6,13]],1.2,atol=1e-6)

    def test_native_entry_does_not_import_WorldLA(self):
        import ast
        tree=ast.parse((ROOT/"scripts/train_pi05_handover_pickup.py").read_text())
        modules=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any("stage_router" in (m or "") or "worldla" in (m or "") for m in modules))


if __name__=="__main__":
    unittest.main()
