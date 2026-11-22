import unittest
import tempfile
from pathlib import Path
import numpy as np
from adapters.handover.take_data import TakeAudit, TakeRecorder, INSTRUCTION


def pose(x=0.,z=1.):
    value=np.eye(4)
    value[:3,3]=[x,0,z]
    return value


class TakeTests(unittest.TestCase):
    def audit(self,left=1.,right=0.,retreat=True,block_z=1.):
        audit=TakeAudit(.74,pose(x=-.12))
        for step in range(5):
            audit.observe(pose(z=block_z),pose(x=-.22 if retreat else -.12),
                          pose(x=.12),left,right,step)
        return audit

    def test_held_in_right_after_left_release_and_retreat(self):
        self.assertTrue(self.audit().report()["success"])

    def test_right_closed_alone_does_not_prove_handover(self):
        self.assertFalse(self.audit(left=0.).report()["success"])
        self.assertFalse(self.audit(retreat=False).report()["success"])

    def test_dropped_block_cannot_pass(self):
        self.assertFalse(self.audit(block_z=.84).report()["success"])
        self.assertFalse(self.audit(right=1.).report()["success"])

    def test_final_drop_clears_earlier_success(self):
        audit=self.audit()
        audit.observe(pose(z=.84),pose(x=-.22),pose(x=.12),1.,0.,6)
        self.assertFalse(audit.report()["success"])

    def test_relative_drift_rejected(self):
        audit=self.audit()
        audit.observe(pose(x=.025),pose(x=-.22),pose(x=.12),1.,0.,6)
        self.assertFalse(audit.report()["success"])

    def test_prompt_is_right_take_not_left_pickup(self):
        self.assertIn("right arm",INSTRUCTION)
        self.assertIn("left gripper",INSTRUCTION)

    def test_recorder_persists_right_stage_instruction_without_changing_old_recorder(self):
        import h5py
        from adapters.handover.pickup_data import CAMERAS, INSTRUCTION as LEFT_INSTRUCTION
        recorder=TakeRecorder(None,None,None)
        recorder.frames=[dict(state=np.zeros(14),action=np.ones(14),
            actual_arm_feedback=np.zeros(12),physics_step_before=0,physics_step_after=15,
            stage="right_close",images={name:np.zeros((2,2,3),np.uint8) for name in CAMERAS},
            depths={name:np.zeros((2,2),np.float32) for name in CAMERAS})]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"take.hdf5"
            recorder.save(path,{})
            with h5py.File(path) as source:
                self.assertEqual(source.attrs["instruction"],INSTRUCTION)
                self.assertEqual(source.attrs["schema"],"ula_handover_right_take_short_v1")
                self.assertEqual(source["action"].shape,(1,14))
                self.assertEqual(source["audit.physics_step_before"][0],0)
        self.assertIn("left arm",LEFT_INSTRUCTION)


if __name__=="__main__":
    unittest.main()
