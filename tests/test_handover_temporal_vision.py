from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin'),str(ROOT/'tests')]
from adapters.handover.temporal_vision import TemporalHeldTrack,surface_jacobian
from adapters.handover.held_vision import HALF
from test_handover_adaptive_place import Task,Observer,fake_move
from test_handover_target_cloud import surfaces
from adapters.handover.adaptive_place import adaptive_place
from types import SimpleNamespace


class TemporalVisionTests(unittest.TestCase):
    def setUp(self):
        self.pose=np.eye(4);self.pose[:3,3]=[.1,.1,1.]
        self.hand=np.eye(4);self.hand[:3,3]=[0.,0.,1.]
        self.quality=dict(accepted=True,position_observable=True,
            position_std_m=.001,axis_uncertainty_deg=.5)
    def track(self):
        track=TemporalHeldTrack()
        track.initialize(self.hand,self.pose,self.quality)
        return track
    def test_requires_full_post_handover_anchor(self):
        track=TemporalHeldTrack()
        quality=dict(self.quality,position_observable=False)
        with self.assertRaisesRegex(RuntimeError,'full post-handover'):
            track.initialize(self.hand,self.pose,quality)
    def test_partial_surfaces_retain_hidden_longitudinal_history(self):
        track=self.track()
        hand=self.hand.copy();hand[:3,3]+=[.025,.01,-.025]
        prediction=track.predict(hand)
        pose,quality=track.update(surfaces(prediction,partial=True))
        self.assertFalse(quality['position_observable'])
        self.assertTrue(quality['temporal_tracking']['release_usable'])
        self.assertLess(np.linalg.norm(pose[:3,3]-prediction[:3,3]),.001)
        self.assertGreater(quality['temporal_tracking']['bottom_z_95_m'],0)
    def test_single_face_does_not_erase_other_historical_coordinates(self):
        track=self.track()
        cloud=surfaces(self.pose,partial=True)[:1680]
        output,quality=track.update(cloud)
        self.assertFalse(quality['position_observable'])
        self.assertTrue(quality['accepted'])
        self.assertLess(np.linalg.norm(output[:3,3]-self.pose[:3,3]),.001)
    def test_hidden_direction_has_no_invented_measurement_information(self):
        points=surfaces(self.pose,partial=True)
        jac=surface_jacobian(points,self.pose)
        self.assertLess(np.linalg.norm(jac[:,5]),1e-5)
    def test_history_expires_and_uncertainty_grows_without_visibility(self):
        track=self.track();start=track.audit()['bottom_z_95_m']
        for _ in range(31):
            hand=track.hand.copy();hand[0,3]+=.02;track.predict(hand)
        audit=track.audit()
        self.assertFalse(audit['release_usable'])
        self.assertGreater(audit['bottom_z_95_m'],start)
    def test_abrupt_observed_motion_is_not_silently_rebound(self):
        track=self.track();moved=self.pose.copy();moved[0,3]+=.04
        with self.assertRaises(RuntimeError):track.update(surfaces(moved))
    def test_selection_excludes_distant_red_distractor(self):
        track=self.track();body=surfaces(self.pose)
        rng=np.random.default_rng(1)
        distractor=rng.uniform([.21,.11,.89],[.24,.13,.90],(1000,3))
        selected,audit=track.select(np.r_[body,distractor])
        self.assertEqual(len(selected),len(body))
        self.assertFalse(audit['components'][-1]['selected'])


@patch.dict(sys.modules,{'envs.utils':SimpleNamespace(ArmTag=lambda name:name)})
class TemporalControllerTests(unittest.TestCase):
    class HistoricalObserver(Observer):
        def observe(self,prior):
            pose,quality=super().observe(prior)
            quality['position_observable']=False
            quality['temporal_tracking']=dict(release_usable=True,
                bottom_xy_95_m=.001,bottom_z_95_m=.002)
            return pose,quality
    @patch('adapters.handover.adaptive_place.move_segment',fake_move)
    def test_bounded_history_can_authorize_occluded_endpoint(self):
        task=Task();result={}
        adaptive_place(task,task.binding,[.1,.1,.75],result,observer=self.HistoricalObserver(task))
        self.assertTrue(result['complete'])
        self.assertEqual(result['final_position_source'],'bounded_visual_history_and_current_surfaces')
    @patch('adapters.handover.adaptive_place.move_segment',fake_move)
    def test_expired_history_cannot_authorize_release(self):
        task=Task()
        class Expired(self.HistoricalObserver):
            def observe(self,prior):
                pose,quality=super().observe(prior)
                quality['temporal_tracking']['release_usable']=False
                return pose,quality
        with self.assertRaisesRegex(RuntimeError,'hidden bottom'):
            adaptive_place(task,task.binding,[.1,.1,.75],{},observer=Expired(task))
        self.assertNotIn('release',task.commands)
    @patch('adapters.handover.adaptive_place.move_segment',fake_move)
    def test_confidence_interval_not_only_mean_must_fit_release_bounds(self):
        task=Task()
        class Uncertain(self.HistoricalObserver):
            def observe(self,prior):
                pose,quality=super().observe(prior)
                quality['temporal_tracking']['bottom_z_95_m']=.02
                return pose,quality
        with self.assertRaisesRegex(RuntimeError,'confidence region'):
            adaptive_place(task,task.binding,[.1,.1,.75],{},observer=Uncertain(task))
        self.assertNotIn('release',task.commands)


if __name__=='__main__':unittest.main()
