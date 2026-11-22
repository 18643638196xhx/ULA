import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.constraint_feedback import (BallRegion,IntervalRegion,CoordinateConstraint,
    FeedbackBudget,decide_constraints)
from adapters.support_placement_goal import SupportPlacementGoal


class ConstraintFeedbackTests(unittest.TestCase):
    def ball(self,radius=.008):
        return (CoordinateConstraint('position',(0,1),BallRegion((0.,0.),radius)),)
    def test_nominally_inside_but_uncertainty_outside_requests_correction(self):
        decision=decide_constraints([.00414,0],self.ball(),{'position':.00395})
        self.assertEqual(decision.action,'correct')
        np.testing.assert_array_equal(decision.target,[0.,0.])
    def test_height_interval_projects_to_task_preference(self):
        c=(CoordinateConstraint('height',(0,),IntervalRegion((.003,),(.018,),(.01,))),)
        d=decide_constraints([.01519],c,{'height':.00391})
        self.assertEqual(d.action,'correct')
        self.assertAlmostEqual(d.target[0],.01)
    def test_complete_requires_entire_supplied_uncertainty_inside(self):
        self.assertEqual(decide_constraints([.002,0],self.ball(),{'position':.004}).action,'complete')
    def test_uncertainty_cannot_fit_requests_observation_not_motion(self):
        d=decide_constraints([0.,0.],self.ball(),{'position':.009})
        self.assertEqual(d.action,'reobserve')
        self.assertIsNone(d.target)
    def test_bad_observation_never_authorizes_completion(self):
        d=decide_constraints([0.,0.],self.ball(),{'position':0},observation_usable=False)
        self.assertEqual(d.action,'reobserve')
    def test_independent_satisfied_coordinates_are_preserved(self):
        constraints=self.ball()+(CoordinateConstraint('height',(2,),IntervalRegion((.003,),(.018,))),)
        d=decide_constraints([.006,0,.011],constraints,{'position':.004,'height':.002})
        np.testing.assert_array_equal(d.target,[0.,0.,.011])
    def test_same_engine_handles_door_angle_without_placement_logic(self):
        c=(CoordinateConstraint('hinge_angle',(0,),IntervalRegion((70.,),(110.,))),)
        state=np.array([20.]);budget=FeedbackBudget()
        actions=[]
        for _ in range(20):
            d=budget.check(decide_constraints(state,c,{'hinge_angle':5.}))
            actions.append(d.action)
            if d.action=='complete':break
            self.assertEqual(d.action,'correct')
            state+=np.clip(d.target-state,-15,15)
        self.assertEqual(actions[-1],'complete')
    def test_coordinate_unit_rescaling_keeps_decisions_and_normalized_error(self):
        original=decide_constraints([.006,.001],self.ball(),{'position':.004})
        scaled=decide_constraints([6.,1.],self.ball(8.),{'position':4.})
        self.assertEqual(original.action,scaled.action)
        self.assertAlmostEqual(original.violation,scaled.violation)
        np.testing.assert_allclose(original.target*1000,scaled.target)
    def test_progress_budget_stops_repeated_non_improvement(self):
        budget=FeedbackBudget(max_stalled_steps=2)
        d=decide_constraints([.02,0.],self.ball(),{'position':.004})
        self.assertEqual([budget.check(d).action for _ in range(3)],['correct','correct','stop'])
    def test_progressing_error_does_not_use_up_stall_budget(self):
        budget=FeedbackBudget(max_stalled_steps=2)
        for x in [.03,.02,.01,.006]:
            self.assertEqual(budget.check(decide_constraints([x,0],self.ball(),{'position':.004})).action,'correct')
    def test_reobservation_has_bounded_budget(self):
        budget=FeedbackBudget(max_reobservations=1)
        d=decide_constraints([0.,0.],self.ball(),{'position':.009})
        self.assertEqual(budget.check(d).action,'reobserve')
        self.assertEqual(budget.check(d).action,'stop')
    def test_conflicting_or_missing_constraints_fail_explicitly(self):
        with self.assertRaises(ValueError):decide_constraints([0,0],self.ball(),{})
        with self.assertRaises(ValueError):decide_constraints([0,0],self.ball()+self.ball(),{'position':0})
        with self.assertRaises(ValueError):decide_constraints([np.nan,0],self.ball(),{'position':0})
        with self.assertRaises(ValueError):decide_constraints([0,0],self.ball(),{'position':-.1})
    def test_audit_is_json_serializable(self):
        json.dumps(decide_constraints([.02,0],self.ball(),{'position':.001}).audit())
    def test_computed_targets_satisfy_randomized_robust_goals_across_scales(self):
        rng=np.random.default_rng(18)
        for _ in range(100):
            scale=10**rng.uniform(-3,3)
            center=rng.uniform(-2,2,2)*scale
            radius=rng.uniform(.1,1.)*scale
            lower=rng.uniform(-2,2)*scale
            width=rng.uniform(.1,2)*scale
            constraints=(CoordinateConstraint('offset',(0,1),BallRegion(tuple(center),radius)),
                CoordinateConstraint('height',(2,),IntervalRegion((lower,),(lower+width,))))
            margins={'offset':rng.uniform(0,.8)*radius,'height':rng.uniform(0,.4)*width}
            state=np.r_[center+rng.normal(0,2,2)*radius,lower+rng.uniform(-1,2)*width]
            decision=decide_constraints(state,constraints,margins)
            self.assertIn(decision.action,('correct','complete'))
            check=decide_constraints(decision.target,constraints,margins)
            self.assertEqual(check.action,'complete')
    def test_same_core_handles_multiple_independent_angle_coordinates(self):
        constraints=(CoordinateConstraint('hinge',(0,),IntervalRegion((.5,),(1.5,))),
            CoordinateConstraint('wrist',(1,),IntervalRegion((-.2,),(.2,))))
        decision=decide_constraints([.2,.1],constraints,{'hinge':.05,'wrist':.02})
        self.assertEqual(decision.action,'correct')
        self.assertAlmostEqual(decision.target[1],.1)
        self.assertEqual(decide_constraints(decision.target,constraints,
            {'hinge':.05,'wrist':.02}).action,'complete')


class SupportPlacementGoalTests(unittest.TestCase):
    def test_different_object_anchors_use_same_adapter(self):
        for half_height,tolerance in [(.1,.008),(.05,.015),(.03,.025)]:
            goal=SupportPlacementGoal((0.,0.,-half_height),tolerance,(.002,.02),.01)
            pose=np.eye(4);pose[:3,3]=[.2,.3,.7+half_height+.01]
            d,_=goal.decide(pose,[.2,.3,.7],[0,0,1],horizontal_uncertainty_m=.001,
                normal_uncertainty_m=.001,observation_usable=True)
            self.assertEqual(d.action,'complete')
    def test_tilted_support_frame_not_world_vertical(self):
        normal=np.array([0.,1.,1.])/np.sqrt(2)
        origin=np.array([.1,.2,.3]);goal=SupportPlacementGoal((0,0,0),.01,(.002,.02),.01)
        pose=np.eye(4);pose[:3,3]=origin+normal*.03
        d,target=goal.decide(pose,origin,normal,horizontal_uncertainty_m=.002,
            normal_uncertainty_m=.002,observation_usable=True)
        self.assertEqual(d.action,'correct')
        np.testing.assert_allclose(target,origin+normal*.01,atol=1e-12)
    def test_too_uncertain_support_goal_does_not_return_move_target(self):
        goal=SupportPlacementGoal((0,0,-.1),.008,(.003,.018),.01)
        d,target=goal.decide(np.eye(4),[0,0,0],[0,0,1],horizontal_uncertainty_m=.02,
            normal_uncertainty_m=.001,observation_usable=True)
        self.assertEqual(d.action,'reobserve');self.assertIsNone(target)
    def test_rigid_world_frame_change_preserves_support_decision(self):
        from scipy.spatial.transform import Rotation
        goal=SupportPlacementGoal((.01,-.02,-.05),.02,(.002,.025),.01)
        pose=np.eye(4);pose[:3,3]=[.1,.2,.8]
        origin=np.array([.08,.19,.72]);normal=np.array([0.,0.,1.])
        decision,target=goal.decide(pose,origin,normal,horizontal_uncertainty_m=.003,
            normal_uncertainty_m=.002,observation_usable=True)
        change=np.eye(4)
        change[:3,:3]=Rotation.from_euler('xyz',[29,17,-41],degrees=True).as_matrix()
        change[:3,3]=[.6,-.4,.3]
        moved,other=goal.decide(change@pose,change[:3,:3]@origin+change[:3,3],
            change[:3,:3]@normal,horizontal_uncertainty_m=.003,
            normal_uncertainty_m=.002,observation_usable=True)
        self.assertEqual(decision.action,moved.action)
        self.assertAlmostEqual(decision.violation,moved.violation)
        np.testing.assert_allclose(other,change[:3,:3]@target+change[:3,3],atol=1e-12)
    def test_nonrigid_object_pose_is_rejected(self):
        goal=SupportPlacementGoal((0.,0.,-.1),.02,(.002,.02))
        for bad in (np.diag([2.,1.,1.,1.]),np.diag([-1.,1.,1.,1.]),np.zeros((4,4))):
            with self.assertRaisesRegex(ValueError,'rigid transform'):
                goal.anchor_world(bad)


if __name__=='__main__':unittest.main()
