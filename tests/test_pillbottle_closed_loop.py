from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.pillbottle_scene_geometry import table_clearance, gravity_upright
from adapters.pillbottle_closed_loop import finish_visual_placement, execute_with_local_retry


def upright(z=.74):
    p = np.eye(4)
    p[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    p[2, 3] = z
    return p


class ArrayTensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value
    def detach(self): return self
    def __getitem__(self, key): return ArrayTensor(self.value[key])


class ClosedLoopTests(unittest.TestCase):
    def test_table_guard_rejects_matching_surface_if_complete_model_is_below_table(self):
        model = np.array([[x, y, z] for x in (-.035, .035) for y in (0., .15) for z in (-.035, .035)])
        self.assertFalse(table_clearance(upright(.67), model, .74)["accepted"])
        self.assertTrue(table_clearance(upright(.74), model, .74)["accepted"])
        self.assertTrue(table_clearance(upright(.8), model, .74)["accepted"])

    def test_gravity_alignment_preserves_heading_and_does_not_mutate_reference(self):
        original = upright()
        a = np.radians(4.)
        original[:3, :3] = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a), np.cos(a)]])@original[:3, :3]
        saved = original.copy()
        corrected = gravity_upright(original)
        np.testing.assert_allclose(corrected[:3, 1], [0., 0., 1.])
        np.testing.assert_allclose(corrected[:3, :3].T@corrected[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_array_equal(saved, original)
        self.assertAlmostEqual(np.linalg.det(corrected[:3, :3]), 1.)

    def test_guarded_fp_checks_multiple_ranked_hypotheses_and_masks_background_depth(self):
        from adapters.pillbottle_known_vision import KnownBottleVision
        vision = KnownBottleVision.__new__(KnownBottleVision)
        vision.geometry_guard = True
        vision.table_height_world = .74
        vision.last_accepted_world_pose = upright(.67)
        vision.model_samples = np.array([[x, y, z] for x in (-.035, .035)
            for y in (0., .15) for z in (-.035, .035)])
        vision.fp_iterations, vision.device = 5, "cpu"
        vision.torch = SimpleNamespace(inference_mode=nullcontext, float32=np.float32,
            as_tensor=lambda value, **kw: ArrayTensor(value))
        depths = []
        def register(**kw):
            depths.append(kw["depth"])
            fp.poses = ArrayTensor(np.stack((upright(.67), upright(.74))))
            return upright(.67)
        fp = SimpleNamespace(get_tf_to_centered_mesh=lambda: ArrayTensor(np.eye(4)),
            track_one=lambda **kw: upright(.67), register=register)
        vision.foundationpose = fp
        observed = SimpleNamespace(cup_points_world=np.zeros((80, 3)), observation_audit=dict(table_height_m=.66),
            timings_s=dict(perception_total=0.))
        mask = np.zeros((12, 12), bool)
        mask[2:10, 2:10] = True
        with patch("adapters.pillbottle_known_vision.PillbottleVision.estimate", return_value=observed), patch(
                "adapters.pillbottle_known_vision.surface_mask", return_value=mask), patch(
                "adapters.pillbottle_known_vision.pose_fit", side_effect=lambda *a: dict(accepted=True)):
            value = vision.estimate(np.zeros((12, 12, 3), np.uint8), np.ones((12, 12)),
                np.eye(3), np.eye(4), cup_only=True)
        self.assertAlmostEqual(value.cup_world_pose[2, 3], .74)
        self.assertEqual(value.observation_audit["foundationpose_attempts"][-1]["selected_hypothesis_rank"], 1)
        self.assertTrue(np.all(depths[0][~mask] == 0))
        self.assertEqual(vision.table_height_world, .74)  # wrist local mode cannot overwrite head baseline

    def test_local_arrival_retry_replans_same_goal_without_weakening_joint_tolerance(self):
        plan = dict(status="Success", position=np.zeros((2, 6)), velocity=np.zeros((2, 6)))
        task = SimpleNamespace(get_arm_pose=lambda arm: [0., 0., 1., 1., 0., 0., 0.], plan_success=True,
            robot=SimpleNamespace(left_plan_path=lambda goal: plan))
        with patch("adapters.pillbottle_known_model.execute_checked", side_effect=[RuntimeError("not reached"), None]) as execute, patch(
                "adapters.pillbottle_known_model.joint_limit_audit", return_value=dict(accepted=True)):
            actual = execute_with_local_retry(task, "left", task.get_arm_pose("left"), plan, {})
        self.assertIs(actual, plan)
        self.assertEqual(execute.call_count, 2)
        self.assertTrue(all("allow_measured_handoff" not in call.kwargs for call in execute.call_args_list))

    def test_transport_hint_moves_image_search_region_instead_of_searching_old_location(self):
        from adapters.pillbottle_known_model import KnownPerception
        points = np.random.default_rng(3).uniform([-.02, -.02, .94], [.02, .02, 1.06], (300, 3))
        perception = KnownPerception.__new__(KnownPerception)
        perception.worker = SimpleNamespace(reference_points=points, reference_pose=np.eye(4))
        perception.pad_only = lambda frame: SimpleNamespace(coaster_world_center=np.array([.2, .1, .74]), timings_s={})
        seen = []
        hint = np.eye(4)
        hint[0, 3] = .2
        def supplementary(frame, label, **kw):
            seen.append((label, kw))
            return SimpleNamespace(cup_world_pose=hint.copy(), observation_audit={}, timings_s={})
        perception.supplementary = supplementary
        frame = dict(rgb=np.zeros((240, 320, 3), np.uint8),
            intrinsic_cv=np.array([[200., 0, 160], [0, 200., 120], [0, 0, 1]]), extrinsic_cv=np.eye(4))
        task = SimpleNamespace(get_camera_observation=lambda name: frame)
        perception.localize_held(task, "left", {}, pose_hint_world=hint)
        self.assertIn("projected_reference", seen[0][0])
        self.assertGreater(seen[0][1]["object_point_xy"][0], 190.)
        np.testing.assert_array_equal(seen[0][1]["pose_hint_world"], hint)

    def localization_fixture(self):
        from adapters.pillbottle_known_model import KnownPerception
        perception = KnownPerception.__new__(KnownPerception)
        perception.worker = SimpleNamespace(reference_points=None)
        perception.pad_only = lambda frame: SimpleNamespace(
            coaster_world_center=np.array([.1, .1, .74]), timings_s={})
        seen = []
        def supplementary(frame, label, **kw):
            seen.append(kw["camera_name"])
            return SimpleNamespace(cup_world_pose=upright(), observation_audit={},
                timings_s={}, camera_name=kw["camera_name"])
        perception.supplementary = supplementary
        task = SimpleNamespace(get_camera_observation=lambda name: {})
        return perception, task, seen

    def test_implausible_head_fit_falls_back_to_wrist_before_acceptance(self):
        perception, task, seen = self.localization_fixture()
        def validate(value):
            accepted = value.camera_name == "left_camera"
            return dict(accepted=accepted, reason="near" if accepted else "far_from_hand")
        audit = {}
        value = perception.localize_held(task, "left", audit, candidate_validator=validate)
        self.assertEqual(seen, ["head_camera", "left_camera"])
        self.assertEqual(value.camera_name, "left_camera")
        first, last = audit["precision_observation_attempts"]
        self.assertFalse(first["accepted"])
        self.assertEqual(first["next_action"], "try_next_camera")
        self.assertTrue(last["accepted"])
        np.testing.assert_allclose(value.coaster_world_center, [.1, .1, .74])

    def test_both_camera_fits_implausible_remain_rejected(self):
        perception, task, seen = self.localization_fixture()
        audit = {}
        with self.assertRaisesRegex(RuntimeError, "No credible held CAD pose"):
            perception.localize_held(task, "left", audit,
                candidate_validator=lambda value: dict(accepted=False, reason="far_from_hand"))
        self.assertEqual(seen, ["head_camera", "left_camera"])
        self.assertTrue(all(not a["accepted"] for a in audit["precision_observation_attempts"]))

    def test_credible_head_pose_does_not_add_unneeded_wrist_observation(self):
        perception, task, seen = self.localization_fixture()
        value = perception.localize_held(task, "left", {},
            candidate_validator=lambda value: dict(accepted=True, reason="near"))
        self.assertEqual(value.camera_name, "head_camera")
        self.assertEqual(seen, ["head_camera"])

    def test_legacy_localization_keeps_first_surface_fit_without_new_gate(self):
        perception, task, seen = self.localization_fixture()
        audit = {}
        perception.localize_held(task, "left", audit)
        self.assertEqual(seen, ["head_camera"])
        self.assertNotIn("candidate_validation", audit["precision_observation_attempts"][0])

    def test_large_arrival_miss_never_retries_or_releases(self):
        task = SimpleNamespace(get_arm_pose=lambda arm: [.2, 0., 1., 1., 0., 0., 0.], plan_success=True)
        with patch("adapters.pillbottle_known_model.execute_checked", side_effect=RuntimeError("not reached")) as execute:
            with self.assertRaises(RuntimeError):
                execute_with_local_retry(task, "left", [0., 0., 1., 1., 0., 0., 0.], {}, {})
        self.assertEqual(execute.call_count, 1)

    def simulate(self, *, fail_observation=False, correction=False, fail_descent=False, occluded=False):
        from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
        events, holds = [], []
        pose = np.array([0., 0., 1.02, 1., 0., 0., 0.])
        relation = np.linalg.solve(pose_wxyz_to_matrix(pose), upright(.85))
        names = [f"j{i}" for i in range(6)]
        entity = SimpleNamespace(get_active_joints=lambda: [SimpleNamespace(get_name=lambda n=n: n) for n in names],
            get_qlimits=lambda: np.tile([-10., 10.], (6, 1)), get_qpos=lambda: np.zeros(6))
        plan_calls = []
        def planner(goal, **kw):
            plan_calls.append(goal)
            if fail_descent and len(plan_calls) >= 3:
                return dict(status="Fail")
            return dict(status="Success", position=np.zeros((2, 6)), velocity=np.zeros((2, 6)))
        robot = SimpleNamespace(left_entity=entity, left_arm_joints_name=names, left_plan_path=planner,
            get_left_arm_real_jointState=lambda: [0.]*6+[0.],
            get_left_tcp_pose=lambda: [pose[0], pose[1], pose[2]-.12, *pose[3:]],
            set_arm_joints=lambda *a: holds.append(1), set_gripper=lambda *a: None)
        task = SimpleNamespace(robot=robot, plan_success=True, get_arm_pose=lambda arm: pose.copy(),
            scene=SimpleNamespace(step=lambda: None), open_gripper=lambda arm, pos: pos,
            move=lambda opening: events.append(("open", opening)) or True)
        count = [0]
        def localize(task, arm, audit, **kw):
            self.assertTrue(callable(kw.get("candidate_validator")))
            stage = audit["stage"]
            events.append(("observe", stage))
            if fail_observation and stage == "after_transport":
                raise RuntimeError("bad fresh pose")
            bottle = pose_wxyz_to_matrix(pose)@relation
            if correction and stage == "before_release_0":
                bottle[0, 3] += .02
            return SimpleNamespace(cup_world_pose=bottle,
                coaster_world_center=np.array([.12, .12, .74]), timings_s={},
                cup_bounds_world=np.array([bottle[:3, 3]+[-.035, -.035, 0.],
                    bottle[:3, 3]+[.035, .035, .01 if occluded else .15]]))
        perception = SimpleNamespace(worker=SimpleNamespace(reference_pad=np.array([.1, .1, .74])), localize_held=localize)
        def execute(task, arm, target, plan, audit):
            events.append(("move", audit.get("stage", "retreat")))
            pose[:] = target
            return plan
        result = {}
        with patch("adapters.pillbottle_closed_loop.execute_with_local_retry", execute):
            if fail_observation or fail_descent:
                with self.assertRaises(RuntimeError):
                    finish_visual_placement(task, perception, "left",
                        ROOT/"tra_pipe/RoboTwin/assets/objects/080_pillbottle/model_data1.json", result,
                        upright_reference=upright())
            else:
                finish_visual_placement(task, perception, "left",
                    ROOT/"tra_pipe/RoboTwin/assets/objects/080_pillbottle/model_data1.json", result,
                    upright_reference=upright())
        return result, events, holds

    def test_full_flow_reobserves_actual_transport_retains_static_pad_and_withdraws_locally(self):
        result, events, holds = self.simulate()
        self.assertTrue(result["complete"])
        np.testing.assert_allclose(result["coaster_world_center"], [.1, .1, .74])
        self.assertEqual([value for kind, value in events if kind == "open"], [1.])
        self.assertEqual(sum(kind == "move" and value == "retreat" for kind, value in events), 1)
        self.assertGreaterEqual(len(holds), 400)
        self.assertLess(events.index(("observe", "before_release_0")), events.index(("open", 1.)))

    def test_visual_goal_correction_occurs_before_any_release(self):
        result, events, holds = self.simulate(correction=True)
        self.assertTrue(result["complete"])
        self.assertIn(("move", "visual_correction_1"), events)
        self.assertLess(events.index(("move", "visual_correction_1")), events.index(("open", 1.)))

    def test_occluded_pinch_region_uses_depth_validated_cad_not_only_visible_lower_body(self):
        result, events, holds = self.simulate(occluded=True)
        self.assertTrue(result["complete"])
        self.assertTrue(any(o["tcp_to_visible_object_bounds_m"] > .03 for o in result["observations"]))
        self.assertTrue(all(o["tcp_to_observed_cad_bounds_m"] <= .03 for o in result["observations"]))

    def test_untrusted_fresh_observation_or_failed_updated_descent_never_releases(self):
        for options in (dict(fail_observation=True), dict(fail_descent=True)):
            result, events, holds = self.simulate(**options)
            self.assertFalse(result["complete"])
            self.assertFalse(any(kind == "open" for kind, value in events))


if __name__ == "__main__":
    unittest.main()
