from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin'),str(ROOT/'tests')]
from test_handover_adaptive_place import Task,Observer,fake_move
from adapters.handover.adaptive_place import adaptive_place
from adapters.support_placement_goal import SupportPlacementGoal


class QualityObserver(Observer):
    def observe(self,prior):
        pose,q=super().observe(prior)
        q['position_observable']=False
        q['temporal_tracking']=dict(release_usable=True,bottom_xy_95_m=.00395,bottom_z_95_m=.00391)
        return pose,q


@patch.dict(sys.modules,{'envs.utils':SimpleNamespace(ArmTag=lambda name:name)})
class ConstraintPlaceTests(unittest.TestCase):
    def goal(self):
        return SupportPlacementGoal((0.,0.,-.1),.008,(.003,.018),.01)
    def near_endpoint_task(self,xy,gap):
        task=Task(tilt=0.)
        task.hand[:3,3]=[xy,0.,.75+.1+gap]
        return task
    @patch('adapters.handover.adaptive_place.move_segment',fake_move)
    def test_horizontal_uncertainty_violation_is_corrected_before_release(self):
        task=self.near_endpoint_task(.00414,.01277);result={}
        adaptive_place(task,task.binding,[0,0,.75],result,observer=QualityObserver(task),feedback_goal=self.goal())
        self.assertTrue(result['complete'])
        corrections=[a for a in result['actions'] if a.get('constraint_feedback',{}).get('action')=='correct']
        self.assertGreater(len(corrections),0)
        self.assertEqual(result['final_constraint_feedback']['action'],'complete')
    @patch('adapters.handover.adaptive_place.move_segment',fake_move)
    def test_high_endpoint_is_lowered_by_computed_error_not_constant(self):
        task=self.near_endpoint_task(0.,.01519);result={}
        adaptive_place(task,task.binding,[0,0,.75],result,observer=QualityObserver(task),feedback_goal=self.goal())
        self.assertTrue(result['complete'])
        correction=next(a for a in result['actions'] if 'constraint_feedback' in a)
        self.assertAlmostEqual(correction['target_ee_pose'][2] if 'target_ee_pose' in correction else task.hand[2,3],.86)
    @patch('adapters.handover.adaptive_place.move_segment',fake_move)
    def test_uncertainty_too_large_reobserves_then_stops_without_moving(self):
        task=self.near_endpoint_task(0.,.01);result={}
        class Uncertain(QualityObserver):
            def observe(self,prior):
                pose,q=super().observe(prior);q['temporal_tracking']['bottom_z_95_m']=.02
                return pose,q
        with self.assertRaisesRegex(RuntimeError,'reobservation_budget_exhausted'):
            adaptive_place(task,task.binding,[0,0,.75],result,observer=Uncertain(task),feedback_goal=self.goal())
        self.assertEqual(task.commands,[])
        self.assertTrue(all(a['action']=='constraint_reobserve' for a in result['actions']))
    def test_motion_that_does_not_improve_stops_without_release(self):
        task=self.near_endpoint_task(.006,.01);result={}
        with patch('adapters.handover.adaptive_place.move_segment',lambda *args,**kwargs:None):
            with self.assertRaisesRegex(RuntimeError,'measured_correction_not_improving'):
                adaptive_place(task,task.binding,[0,0,.75],result,observer=QualityObserver(task),feedback_goal=self.goal())
        self.assertNotIn('release',task.commands)


if __name__=='__main__':unittest.main()
