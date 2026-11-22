import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "tra_pipe/RoboTwin"))
from adapters.multiview import centred_wrist_target, move_ee_checked, supplement_after_approach
from adapters.recovery import GraspStalled, VisualStallCheck, retreat_for_retry, table_support_evidence
from adapters.timing import EpisodeTiming
from test_geometry_approach import runner


def estimate(border=False,z_shift=0.,table_z=0.74):
    bounds = np.array([[-0.28,-0.21,0.749],[-0.20,-0.13,0.829]])+[0,0,z_shift]
    return SimpleNamespace(cup_bounds_world=bounds,cup_points_world=np.zeros((40,3)),
        coaster_world_center=np.array([0.,0.,0.745]),timings_s={},
        observation_audit=dict(cup_mask_touches_border=border,table_height_m=table_z))


class MultiviewTests(unittest.TestCase):
    def test_camera_centring_uses_calibration_and_preserves_orientation(self):
        ee = np.array([0.,0.,1.,1.,0.,0.,0.])
        camera = dict(extrinsic_cv=np.eye(4))
        target = centred_wrist_target(ee,camera,[[-0.01,0.01,0.7],[0.03,0.03,0.8]])
        np.testing.assert_allclose(target[:3],[0.01,0.02,1.])
        np.testing.assert_allclose(target[3:],ee[3:])
        with self.assertRaises(ValueError):
            centred_wrist_target(ee,camera,[[1,1,0.7],[2,2,0.8]])

    def test_failed_plan_never_executes(self):
        task = SimpleNamespace(robot=SimpleNamespace(left_plan_path=lambda target:dict(status="Fail")))
        moved,audit = move_ee_checked(task,"left",np.array([0,0,1,1,0,0,0]))
        self.assertFalse(moved)

    def test_cup_only_wrist_never_supplies_coaster_target(self):
        head,wrist = estimate(True),estimate(False)
        wrist.coaster_world_center[:] = np.nan
        selected = []
        perception = SimpleNamespace(supplementary=lambda camera,label:wrist,
                                     select_reference=selected.append)
        task = SimpleNamespace(get_camera_observation=lambda name:{})
        result,audit = supplement_after_approach(task,perception,head,"left")
        self.assertTrue(audit["resolved"])
        self.assertEqual(len(selected),1)
        np.testing.assert_allclose(result.coaster_world_center,head.coaster_world_center)
        np.testing.assert_allclose(result.cup_points_world,head.cup_points_world)
        self.assertFalse(audit["robot_region_filtering"])

    def test_unresolved_view_is_not_invented_as_complete(self):
        task = SimpleNamespace(get_camera_observation=lambda name:{})
        perception = SimpleNamespace(supplementary=lambda *a:(_ for _ in ()).throw(RuntimeError("no cup")))
        head = estimate(True)
        result,audit = supplement_after_approach(task,perception,head,"left")
        self.assertIs(result,head)
        self.assertFalse(audit["resolved"])

    def test_new_mode_never_centres_wrist_using_invalid_geometry(self):
        wrist = estimate(True)
        wrist.cup_points_world = np.zeros((40, 3))
        # Genuinely invalid geometry (non-finite surface), not merely a tall
        # object: vertical aspect is only a diagnostic warning now.
        wrist.cup_points_world[0] = np.nan
        wrist.cup_bounds_world[1, 2] = 1.2
        task = SimpleNamespace(get_camera_observation=lambda name: {})
        p = SimpleNamespace(supplementary=lambda *a, **kw: wrist)
        out, audit = supplement_after_approach(task, p, estimate(True), "left", validate_geometry=True)
        self.assertFalse(audit["resolved"])
        self.assertEqual(audit["motions"], [])
        self.assertEqual(audit["views"][0]["rejected"], "unreliable_wrist_geometry")


class RecoveryTests(unittest.TestCase):
    def test_supported_cup_requires_table_height_top_and_complete_observation(self):
        reference = estimate()
        self.assertTrue(table_support_evidence(reference,reference.cup_bounds_world,0.74)["supported"])
        for observation in (estimate(True),estimate(z_shift=.04),estimate(table_z=.78)):
            self.assertFalse(table_support_evidence(observation,reference.cup_bounds_world,0.74)["supported"])

    def checker(self,observation):
        # Deliberately no cup actor, contact API, robot model or ground-truth pose.
        task = SimpleNamespace(take_action_cnt=120,get_camera_observation=lambda name:
                               dict(rgb=0,depth=0,intrinsic_cv=0,extrinsic_cv=0))
        perception = SimpleNamespace(estimate=lambda *a:observation,
                                     supplementary=lambda *a,**kw:observation)
        record = {}
        return VisualStallCheck(task,perception,"left",estimate().cup_bounds_world,0.74,record,0),record

    def test_two_supported_frames_authorize_stall_without_object_truth(self):
        checker,record = self.checker(estimate())
        with self.assertRaises(GraspStalled):
            checker()
        self.assertTrue(record["visual_stall_checks"][0]["supported"])
        self.assertEqual(len(record["visual_stall_checks"][0]["frames"]),2)

    def test_lifted_or_clipped_cup_does_not_authorize_opening(self):
        for observation in (estimate(True),estimate(z_shift=.04)):
            checker,record = self.checker(observation)
            checker()
            self.assertFalse(record["visual_stall_checks"][0]["supported"])

    def test_checker_waits_until_its_own_attempt_budget_window(self):
        checker,record = self.checker(estimate())
        checker.task.take_action_cnt = 119
        checker()
        self.assertEqual(record,{})

    def test_no_second_attempt_or_counter_reset_after_retry(self):
        handoff = dict(ready=True,arm="left",estimated_cup_bounds_world=estimate().cup_bounds_world.tolist(),
                       geometry_approach=dict(observation=dict(table_height_m=.74)))
        record = dict(handoff=handoff,grasp_handoff=None)
        task = SimpleNamespace(take_action_cnt=0,step_lim=500)
        args = SimpleNamespace(grasp_retries=1,grasp_lift=.03,grasp_stable_steps=3,
                               fp_iterations=5,grasp_stall_steps=120)
        starts = []

        def policy_rollout(task,policy,initial_z,record,monitor,arm,stall_check):
            starts.append(task.take_action_cnt)
            if len(starts)==1:
                task.take_action_cnt = 120
                record["visual_stall_checks"] = [dict(supported=True)]
                raise GraspStalled("visual support confirmed")
            task.take_action_cnt += 53
            record["grasp_handoff"] = dict(ready=True)

        import tempfile
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner,"ExternalPerception",return_value=SimpleNamespace()), \
                patch.object(runner,"vla_timing",side_effect=lambda *a:nullcontext()), \
                patch.object(runner,"traditional_timing",side_effect=lambda *a:nullcontext()), \
                patch.object(runner,"retreat_for_retry"), \
                patch.object(runner,"approach",return_value=copy.deepcopy(handoff)), \
                patch.object(runner,"snapshot"),patch.object(runner,"dump_json"), \
                patch.object(runner,"rollout",side_effect=policy_rollout):
            runner.rollout_with_recovery(task,SimpleNamespace(model=object()),.74,record,args,
                                         EpisodeTiming(),Path(tmp),None,SimpleNamespace())
        self.assertEqual(starts,[0,120])
        self.assertEqual(task.take_action_cnt,173)
        self.assertEqual(task.step_lim,500)
        self.assertEqual(len(record["recoveries"]),1)
        self.assertEqual([a["vla_steps"] for a in record["grasp_attempts"]],[120,53])

    def test_retreat_without_visual_authorization_cannot_open_gripper(self):
        with self.assertRaisesRegex(RuntimeError,"without visual support"):
            retreat_for_retry(SimpleNamespace(),"left",{})


if __name__ == "__main__":
    unittest.main()
