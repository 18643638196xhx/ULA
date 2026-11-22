import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "tra_pipe/RoboTwin"))
from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz, pose_wxyz_to_matrix
from adapters.pointcloud_registration import RegistrationConfig, estimate_pointcloud_motion
from adapters.pointcloud_placement import finish_pointcloud_placement, placement_candidates


def surface():
    rng = np.random.default_rng(123)
    return rng.uniform([-0.04,-0.03,0.76],[0.04,0.03,0.83],(1000,3))


class RegistrationTests(unittest.TestCase):
    def test_quality_limits_can_be_relaxed_without_changing_optimizer(self):
        source = surface()
        target = source+np.random.default_rng(17).normal(0,0.0005,source.shape)
        strict, strict_audit = estimate_pointcloud_motion(source,target,RegistrationConfig(max_rmse_m=1e-6))
        relaxed, relaxed_audit = estimate_pointcloud_motion(source,target,RegistrationConfig(max_rmse_m=0.012))
        self.assertIsNone(strict)
        self.assertIsNotNone(relaxed)
        for a,b in zip(strict_audit["attempts"],relaxed_audit["attempts"]):
            self.assertEqual(a["trimmed_rmse_m"],b["trimmed_rmse_m"])
            self.assertEqual(a["inferred_tilt_deg"],b["inferred_tilt_deg"])

    def test_recovers_small_rigid_motion_with_partial_overlap_and_outliers(self):
        source = surface()
        yaw, pitch = 0.12,0.07
        rotation = np.array([[np.cos(yaw),-np.sin(yaw),0], [np.sin(yaw),np.cos(yaw),0], [0,0,1.]])
        rotation = rotation@np.array([[np.cos(pitch),0,np.sin(pitch)], [0,1,0], [-np.sin(pitch),0,np.cos(pitch)]])
        shift = np.array([0.01,-0.02,0.04])
        target = source[:800]@rotation.T+shift
        outliers = np.random.default_rng(789).uniform([0.15,0.15,1.0],[0.25,0.25,1.1],(120,3))
        target = np.concatenate((target,outliers))
        motion, audit = estimate_pointcloud_motion(source,target,RegistrationConfig(voxel_m=0.001))
        self.assertTrue(audit["accepted"])
        np.testing.assert_allclose(motion[:3,:3],rotation,atol=0.015)
        np.testing.assert_allclose(motion[:3,3],shift,atol=0.003)
        self.assertFalse(audit["uses_prepared_object_model"])
        self.assertFalse(audit["semantic_6d_pose"])

    def test_unrelated_cloud_rejected_with_diagnostics(self):
        target = np.random.default_rng(456).uniform([-0.3,-0.3,0.8],[0.3,0.3,1.2],(500,3))
        motion, audit = estimate_pointcloud_motion(surface(),target)
        self.assertIsNone(motion)
        self.assertFalse(audit["accepted"])
        self.assertTrue(audit["attempts"])

    def test_invalid_or_degenerate_input_is_rejected(self):
        for cloud in (np.zeros((100,3)), np.zeros((2,3)),np.full((100,3),np.nan)):
            with self.subTest(shape=cloud.shape), self.assertRaises((ValueError,RuntimeError)):
                estimate_pointcloud_motion(surface(),cloud)
        with self.assertRaises(ValueError):
            RegistrationConfig(iterations=0)


class PlacementGeometryTests(unittest.TestCase):
    def test_hidden_bottom_uses_table_contact_not_lowest_visible_point(self):
        bounds = np.array([[-0.04,-0.03,0.77],[0.04,0.03,0.83]])
        motion = np.eye(4)
        angle = 0.12
        motion[:3,:3] = [[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]]
        motion[:3,3] = [0.01,-0.02,0.04]
        ee = np.eye(4)
        ee[:3,3] = [-0.1,0,0.95]
        target = np.array([0.2,0.1,0.745])
        goals, audit = placement_candidates(bounds,0.74,motion,ee,target,0.003)
        initial_anchor = np.array([0.,0.,0.74])
        held_anchor = motion[:3,:3]@initial_anchor+motion[:3,3]
        anchor_in_ee = np.linalg.solve(ee,np.r_[held_anchor,1])
        for yaw, goal in goals:
            np.testing.assert_allclose((goal@anchor_in_ee)[:3],target+[0,0,0.003],atol=1e-10)
            restored_up = goal[:3,:3]@ee[:3,:3].T@motion[:3,:3]@[0,0,1]
            np.testing.assert_allclose(restored_up,[0,0,1],atol=1e-10)
        self.assertEqual(audit["initial_support_anchor_world"][2],0.74)
        self.assertFalse(audit["hidden_bottom_reconstructed"])


class PlacementExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.reference = Path(self.temp.name) / "reference.npz"
        np.savez(self.reference,cup_points_world=surface(),
                 cup_bounds_world=np.array([[-0.04,-0.03,0.77],[0.04,0.03,0.83]]),
                 observation_audit_json=np.asarray(json.dumps(dict(table_height_m=0.74))))
        self.calls = []
        self.ee = np.eye(4)
        self.ee[:3,3] = [0,0,1]
        self.plan_fail = False
        self.miss_arrival = False
        self.execution_fail = False
        self.task = SimpleNamespace(
            plan_success=True,
            get_arm_pose=lambda arm:matrix_to_pose_wxyz(self.ee),
            get_camera_observation=lambda name:dict(rgb=0,depth=0,intrinsic_cv=0,extrinsic_cv=0),
            take_dense_action=self.execute,
            open_gripper=lambda *a,**k:"release",
            move_by_displacement=lambda *a,**k:"retreat",
            move=lambda action:self.calls.append(action) or True,
            robot=SimpleNamespace(left_plan_path=self.plan,left_arm_joints_name=["j"],
                                  left_entity=SimpleNamespace(get_active_joints=lambda:[SimpleNamespace(get_name=lambda:"j")],
                                                              get_qpos=lambda:np.zeros(1))))
        self.estimate = SimpleNamespace(cup_points_world=surface(),timings_s={},coaster_world_center=np.array([0.2,0.1,0.745]))
        self.perception = SimpleNamespace(estimate=lambda *a:self.estimate)

    def plan(self,target,**kwargs):
        self.calls.append("plan_place" if "last_qpos" in kwargs else "plan_pre")
        return dict(status="Fail") if self.plan_fail else dict(status="Success",position=np.zeros((1,1)),target=target)

    def execute(self,controls):
        self.calls.append("execute")
        if not self.miss_arrival:
            self.ee = pose_wxyz_to_matrix(controls["left_arm"]["target"])
        return not self.execution_fail

    def run_placement(self,motion=np.eye(4)):
        result = {}
        with patch("adapters.pointcloud_registration.estimate_pointcloud_motion",return_value=(motion,dict(accepted=motion is not None))), \
                patch.object(Path,"read_text",side_effect=AssertionError("No CAD/metadata reads allowed")):
            finish_pointcloud_placement(self.task,self.perception,"left",self.reference,result)
        return result

    def test_no_actor_truth_or_model_needed_and_release_is_last(self):
        result = self.run_placement()
        self.assertTrue(result["complete"])
        self.assertEqual(self.calls,["plan_pre","plan_place","execute","execute","release","retreat"])
        self.assertFalse(result["uses_functional_point_annotations"])
        self.assertFalse(result["uses_object_truth_for_planning"])
        self.assertFalse(result["uses_foundationpose"])

    def test_rejected_registration_never_plans_executes_or_releases(self):
        with self.assertRaisesRegex(RuntimeError,"unreliable surface registration"):
            self.run_placement(motion=None)
        self.assertEqual(self.calls,[])

    def test_exhausted_reobservation_never_plans_executes_or_releases(self):
        from adapters.reobservation import ReobservationRejected

        result = {}
        with patch("adapters.reobservation.observe_for_placement",
                   side_effect=ReobservationRejected("no credible observation", {})), \
                self.assertRaises(ReobservationRejected):
            finish_pointcloud_placement(self.task, self.perception, "left", self.reference,
                                        result, reobserve=True)
        self.assertEqual(self.calls, [])
        self.assertFalse(result["complete"])

    def test_all_plans_failed_never_execute_or_release(self):
        self.plan_fail = True
        with self.assertRaisesRegex(RuntimeError,"no feasible transport/place pair"):
            self.run_placement()
        self.assertEqual(self.calls,["plan_pre"]*21)

    def test_execution_failure_or_missing_arrival_never_releases_or_replans(self):
        for attribute in ("execution_fail","miss_arrival"):
            self.calls.clear()
            self.ee = np.eye(4)
            setattr(self,attribute,True)
            with self.subTest(attribute=attribute),self.assertRaisesRegex(RuntimeError,"move_pre_place"):
                self.run_placement()
            self.assertEqual(self.calls,["plan_pre","plan_place","execute"])
            setattr(self,attribute,False)


if __name__ == "__main__":
    unittest.main()
