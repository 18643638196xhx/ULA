from pathlib import Path
import sys
import unittest
import importlib.util
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin'),str(ROOT/'tests')]
from adapters.handover.held_vision import fit_cuboid
from adapters.handover.target_cloud import fit_target_cloud
from adapters.handover.temporal_vision import TemporalHeldTrack
from test_handover_target_cloud import surfaces


class RelaxedVisionTests(unittest.TestCase):
    def setUp(self):
        self.pose=np.eye(4);self.pose[:3,3]=[.1,.1,1.]
    def test_default_visible_span_gate_is_not_relaxed(self):
        p=surfaces(self.pose,partial=True)
        p=p[np.abs(p[:,2]-1.)<.025]
        with self.assertRaises(RuntimeError):fit_cuboid(p,self.pose)
        with self.assertRaises(RuntimeError):fit_target_cloud(p,self.pose)
    def test_opt_in_short_surface_can_fit_without_faking_position_observability(self):
        p=surfaces(self.pose,partial=True)
        p=p[np.abs(p[:,2]-1.)<.025]
        _,q,_=fit_target_cloud(p,self.pose,min_visible_span_m=.03)
        self.assertTrue(q['accepted'])
        self.assertFalse(q['position_observable'])
    def test_default_initialization_still_rejects_partial_position(self):
        t=TemporalHeldTrack()
        with self.assertRaises(RuntimeError):
            t.initialize(np.eye(4),self.pose,dict(accepted=True,position_observable=False))
    def test_provisional_prediction_is_not_a_release_anchor(self):
        t=TemporalHeldTrack()
        _,audit=t.initialize_partial(np.eye(4),self.pose,
            dict(accepted=True,position_observable=False,axis_uncertainty_deg=.5))
        self.assertFalse(audit['release_usable'])
        self.assertFalse(audit['full_visual_anchor_seen'])
        self.assertGreater(audit['bottom_z_95_m'],.025)
        _,q=t.update(surfaces(self.pose,partial=True))
        self.assertFalse(q['temporal_tracking']['release_usable'])
    def test_later_full_measurement_can_establish_release_anchor(self):
        t=TemporalHeldTrack()
        t.initialize_partial(np.eye(4),self.pose,
            dict(accepted=True,position_observable=False,axis_uncertainty_deg=.5))
        _,q=t.update(surfaces(self.pose))
        self.assertTrue(q['temporal_tracking']['full_visual_anchor_seen'])
        self.assertTrue(q['temporal_tracking']['release_usable'])
    def test_invalid_span_limits_are_rejected(self):
        for span in [0.,-.1,float('nan'),.3]:
            with self.assertRaises(ValueError):
                fit_target_cloud(surfaces(self.pose),self.pose,min_visible_span_m=span)
    def test_explicit_noncontiguous_seed_list_never_runs_other_seeds(self):
        spec=importlib.util.spec_from_file_location('relaxed_seed_mapping_test',
            ROOT/'scripts/eval_pi05_handover_full.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        selected=[40009,40024,40045,40051,40052,40053,40055]
        args=SimpleNamespace(seed=40000,seeds=selected)
        self.assertEqual([runner.evaluation_seed(args,i) for i in range(len(selected))],selected)
        with self.assertRaises(IndexError):runner.evaluation_seed(args,len(selected))
        self.assertEqual(runner.evaluation_seed(SimpleNamespace(seed=40000),3),40003)


if __name__=='__main__':unittest.main()
