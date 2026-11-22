import json
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.pillbottle_known_model import pose_fit, support_point, execute_checked, place, centered_camera_pose, direct_handoff, joint_limit_audit


class KnownModelTests(unittest.TestCase):
    def test_coarse_surface_does_not_call_or_read_foundationpose(self):
        from adapters.pillbottle_known_vision import KnownBottleVision
        vision = KnownBottleVision.__new__(KnownBottleVision)
        coarse = SimpleNamespace(observation_audit={}, timings_s={})
        with patch("adapters.pillbottle_known_vision.PillbottleVision.estimate", return_value=coarse):
            value = vision.estimate(None, None, None, None, coarse_only=np.asarray(True),
                cup_only=True, camera_name=np.asarray("left_camera"))
        self.assertIsNone(value.cup_world_pose)
        self.assertFalse(value.observation_audit["cad_pose_required"])
        self.assertFalse(hasattr(vision, "foundationpose"))
        self.assertFalse(hasattr(vision, "last_accepted_world_pose"))
        self.assertEqual(json.loads(json.dumps(value.observation_audit))["camera_name"], "left_camera")

    def test_close_requests_surface_only_without_changing_shared_adapter(self):
        from adapters.pillbottle_known_model import KnownPerception
        requests = []
        def estimate_files(source, destination):
            with np.load(source, allow_pickle=False) as data:
                requests.append({k: data[k].copy() for k in data.files})
            points = np.random.default_rng(1).uniform([-.03, -.03, .75], [.03, .03, .87], (100, 3))
            return SimpleNamespace(cup_world_pose=None, cup_points_world=points,
                cup_bounds_world=np.array([points.min(0), points.max(0)]), observation_audit={})
        worker = SimpleNamespace(fp_iterations=5, estimate_files=estimate_files)
        frame = dict(rgb=np.zeros((8, 8, 3), np.uint8), depth=np.ones((8, 8)),
            intrinsic_cv=np.eye(3), extrinsic_cv=np.eye(4)[:3])
        with tempfile.TemporaryDirectory() as directory:
            perception = KnownPerception(ROOT, directory, worker=worker)
            perception.supplementary(frame, "close_3_head_camera")
            perception.supplementary(frame, "held_head_camera", coarse_only=False)
        self.assertTrue(requests[0]["coarse_only"])
        self.assertFalse(requests[1]["coarse_only"])

    def test_precise_localization_retries_wrist_but_never_uses_coarse_as_cad_pose(self):
        from adapters.pillbottle_known_model import KnownPerception
        perception = KnownPerception.__new__(KnownPerception)
        perception.worker = SimpleNamespace(reference_points=None)
        perception.pad_only = lambda frame: SimpleNamespace(coaster_world_center=np.array([.2, .1, .74]), timings_s={})
        calls = []
        def supplementary(frame, label, **kw):
            calls.append(kw)
            self.assertFalse(kw["coarse_only"])
            if kw["camera_name"] == "head_camera":
                raise RuntimeError("bad CAD alignment")
            return SimpleNamespace(cup_world_pose=np.eye(4), observation_audit={}, timings_s={})
        perception.supplementary = supplementary
        task = SimpleNamespace(get_camera_observation=lambda name: {})
        audit = {}
        value = perception.localize_held(task, "left", audit)
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[1]["refine_cup"])
        np.testing.assert_allclose(value.coaster_world_center, [.2, .1, .74])
        self.assertTrue(audit["precision_observation_attempts"][-1]["accepted"])

    def test_coarse_clipped_mask_is_reprompted_with_initial_visual_roi(self):
        from adapters.pillbottle_known_model import KnownPerception
        points = np.random.default_rng(2).uniform([-.03, -.03, .94], [.03, .03, 1.06], (200, 3))
        calls = []
        def estimate_files(source, destination):
            with np.load(source, allow_pickle=False) as data:
                calls.append(dict(data))
            return SimpleNamespace(cup_points_world=points, cup_bounds_world=[points.min(0), points.max(0)],
                observation_audit=dict(cup_mask_touches_border=len(calls) == 1))
        worker = SimpleNamespace(fp_iterations=5, reference_points=points, estimate_files=estimate_files)
        frame = dict(rgb=np.zeros((240, 320, 3), np.uint8), depth=np.ones((240, 320)),
            intrinsic_cv=np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1]]), extrinsic_cv=np.eye(4)[:3])
        with tempfile.TemporaryDirectory() as directory:
            perception = KnownPerception(ROOT, directory, worker=worker)
            value = perception.supplementary(frame, "close_3_head_camera")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[1]["coarse_only"])
        self.assertTrue(calls[1]["refine_cup"])
        self.assertIn("object_roi_xyxy", calls[1])
        self.assertFalse(value.observation_audit["cup_mask_touches_border"])

    def test_arm_plan_is_rejected_when_physical_joint_limits_disagree(self):
        entity = SimpleNamespace(get_active_joints=lambda: [SimpleNamespace(get_name=lambda i=i: f"j{i}") for i in range(6)],
            get_qlimits=lambda: np.tile([-1., 1.], (6, 1)))
        robot = SimpleNamespace(left_entity=entity, left_arm_joints_name=[f"j{i}" for i in range(6)])
        position = np.zeros((3, 6))
        self.assertTrue(joint_limit_audit(robot, "left", dict(position=position))["accepted"])
        position[1, 3] = 1.1
        self.assertFalse(joint_limit_audit(robot, "left", dict(position=position))["accepted"])

    def test_low_clearance_pair_is_planned_before_motion_and_failed_descent_never_releases(self):
        from adapters.pillbottle_known_placement import finish_placement
        metadata = ROOT/"tra_pipe/RoboTwin/assets/objects/080_pillbottle/model_data1.json"
        for reject_descent in (False, True):
            events = []
            entity = SimpleNamespace(get_active_joints=lambda: [SimpleNamespace(get_name=lambda i=i: f"j{i}") for i in range(6)],
                get_qlimits=lambda: np.tile([-10., 10.], (6, 1)), get_qpos=lambda: np.zeros(6))
            def plan(target, last_qpos=None):
                events.append("plan")
                if target[2] > 1.00001 or (reject_descent and last_qpos is not None):
                    return dict(status="Fail", planner_status="IK_FAIL")
                return dict(status="Success", position=np.zeros((2, 6)), velocity=np.zeros((2, 6)))
            robot = SimpleNamespace(left_entity=entity, left_arm_joints_name=[f"j{i}" for i in range(6)], left_plan_path=plan)
            pose = np.eye(4)
            pose[2, 3] = .8
            perception = SimpleNamespace(estimate=lambda *a: SimpleNamespace(cup_world_pose=pose,
                coaster_world_center=np.array([0., 0., .78]), timings_s={}))
            def dense(control):
                events.append("dense")
                return True
            def move(action):
                events.append(action)
                return True
            task = SimpleNamespace(robot=robot, get_camera_observation=lambda name: dict(rgb=None, depth=None,
                intrinsic_cv=None, extrinsic_cv=None), get_arm_pose=lambda arm: [0., 0., 1., 1., 0., 0., 0.],
                take_dense_action=dense, open_gripper=lambda *a, **kw: "release",
                move_by_displacement=lambda *a, **kw: "retreat", move=move, plan_success=True)
            result = {}
            if reject_descent:
                with self.assertRaisesRegex(RuntimeError, "No feasible"):
                    finish_placement(task, perception, "left", metadata, result, upright_reference=np.eye(4))
                self.assertNotIn("dense", events)
                self.assertNotIn("release", events)
            else:
                finish_placement(task, perception, "left", metadata, result, upright_reference=np.eye(4))
                self.assertEqual(result["chosen_clearance_m"], .02)
                self.assertEqual(events[:5], ["plan"]*5)
                self.assertEqual(events[5:], ["dense", "dense", "release", "retreat"])
                self.assertTrue(result["complete"])

    def test_direct_handoff_does_not_observe_plan_or_move_and_does_not_claim_verified_grasp(self):
        class NoAccess:
            def __getattr__(self, name):
                raise AssertionError(f"Direct handoff must not access {name}")
        audit = {}
        decision = direct_handoff(NoAccess(), NoAccess(), "left", audit, NoAccess(), lift_m=None)
        self.assertTrue(decision["ready"])
        self.assertFalse(decision["lifted_grasp_verified"])
        self.assertTrue(audit["skipped"])
        self.assertFalse(audit["complete"])
        self.assertTrue(audit["handoff_authorized"])
        self.assertEqual(audit["segments"], [])

    def test_placement_rejects_missing_candidate_authorization(self):
        with self.assertRaisesRegex(RuntimeError, "close/near candidate"):
            place(None, None, "left", None, dict(complete=True), {})

    def test_tracking_accepts_3x4_camera_and_preserves_asset_center_offset(self):
        world, extrinsic, centered = np.eye(4), np.eye(4), np.eye(4)
        world[:3, 3] = [1, 2, 3]
        extrinsic[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        extrinsic[:3, 3] = [.1, .2, .3]
        centered[:3, 3] = [0, -.08, 0]
        camera = centered_camera_pose(world, extrinsic[:3], centered)
        self.assertEqual(camera.shape, (4, 4))
        np.testing.assert_allclose(camera @ centered, extrinsic @ world)

    def test_support_point_scales_local_offset_and_transforms_it(self):
        local = np.eye(4)
        local[:3, 3] = [1, 2, 3]
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[:3, 3] = [4, 5, 6]
        np.testing.assert_allclose(support_point(pose, dict(functional_matrix=[local], scale=[.1, .2, .3])),
                                   [3.6, 5.1, 6.9])

    def test_depth_fit_rejects_wrong_pose_without_truth(self):
        model = np.random.default_rng(0).uniform(-.03, .03, (1000, 3))
        pose = np.eye(4)
        pose[:3, 3] = [.1, .2, .8]
        visible = model[:500]+pose[:3, 3]
        self.assertTrue(pose_fit(visible, pose, model)["accepted"])
        pose[0, 3] += .1
        self.assertFalse(pose_fit(visible, pose, model)["accepted"])
        pose[:3, :3] = 0.
        with self.assertRaises(ValueError):
            pose_fit(visible, pose, model)

    def test_failed_arrival_stops_even_when_dense_action_returns_true(self):
        task = SimpleNamespace(take_dense_action=lambda control: True)
        plan = dict(status="Success", position=np.zeros((2, 6)), velocity=np.zeros((2, 6)))
        with patch("adapters.pillbottle_known_model.settle_at_target", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "did not reach"):
                execute_checked(task, "left", [0, 0, 1, 1, 0, 0, 0], plan, {})

    def test_worker_has_its_own_entry_and_declared_model_identity(self):
        from adapters.pillbottle_known_model import KnownWorker
        worker = KnownWorker(ROOT, ROOT/"results/unused_test", model_id=3)
        command, _, environment = worker._launch_spec()
        self.assertIn(str(ROOT/"scripts/perceive_pillbottle_known_model.py"), command)
        self.assertEqual(command[command.index("--model-id")+1], "3")
        self.assertIn("FoundationPose", environment["PYTHONPATH"])
        self.assertIsNone(worker.process)

    def test_place_wrapper_checks_transport_descent_not_retreat_and_restores(self):
        calls = []
        original_execute = lambda control: calls.append(control) or True
        task = SimpleNamespace(take_dense_action=original_execute)
        metadata = ROOT/"tra_pipe/RoboTwin/assets/objects/080_pillbottle/model_data1.json"
        initial = np.eye(4)
        held = initial.copy()
        # No rise: the bottle can still be resting on the table at handoff.
        held[2, 3] = 0.
        original_estimate = lambda *a, **kw: SimpleNamespace(cup_world_pose=held,
            coaster_world_center=np.array([.2, .1, 0.]))
        perception = SimpleNamespace(estimate=original_estimate,
            worker=SimpleNamespace(metadata_path=metadata))
        plan = dict(status="Success", position=np.zeros((2, 6)), velocity=np.zeros((2, 6)))
        result = {}
        def fake_finish(task, perception, arm, metadata, result, **kw):
            value = perception.estimate(None)
            self.assertAlmostEqual(value.coaster_world_center[2], .003)
            result.update(target_ee_pose=[0, 0, 1, 1, 0, 0, 0], chosen_clearance_m=.05, actions=[])
            for name in ("transport", "descent", "retreat"):
                self.assertTrue(task.take_dense_action(dict(left_arm=plan)))
                result["actions"].append(dict(action=name))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"reference.npz"
            np.savez(path, cup_world_pose=initial)
            with patch("adapters.pillbottle_known_placement.finish_placement", fake_finish), patch(
                    "adapters.pillbottle_known_model.settle_at_target", return_value=True) as settling:
                place(task, perception, "left", path, dict(handoff_authorized=True), result)
                self.assertEqual(settling.call_count, 2)
        self.assertEqual(result["initial_to_held_support_rise_m"], 0.)
        self.assertFalse(result["support_rise_required"])
        self.assertFalse(result["lifted_grasp_verified"])
        self.assertIs(task.take_dense_action, original_execute)
        self.assertIs(perception.estimate, original_estimate)
        self.assertEqual(len(calls[0]["left_arm"]["position"]), 3)
        self.assertIs(calls[2]["left_arm"], plan)


if __name__ == "__main__":
    unittest.main()
