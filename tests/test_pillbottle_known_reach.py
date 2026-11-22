from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]
from adapters.pillbottle_known_handoff import settle_measured_handoff
from adapters.pillbottle_known_placement import placement_candidates


class ReachTests(unittest.TestCase):
    def test_central_contacts_use_cad_geometry_and_respect_enabled_groups(self):
        import json
        from adapters.pillbottle_known_model import central_contact_group
        metadata = json.loads((ROOT/"tra_pipe/RoboTwin/assets/objects/080_pillbottle/model_data1.json").read_text())
        self.assertEqual(central_contact_group(metadata), list(range(8)))
        metadata["contact_points_mask"][0] = False
        self.assertEqual(central_contact_group(metadata), list(range(8, 16)))
        metadata["contact_points_mask"][1] = False
        with self.assertRaisesRegex(ValueError, "No enabled"):
            central_contact_group(metadata)

    def test_central_contacts_reject_bad_indices_and_nonfinite_geometry(self):
        from adapters.pillbottle_known_model import central_contact_group
        metadata = dict(contact_points_pose=[np.eye(4).tolist()], center=[0., 0., 0.],
            scale=[1., 1., 1.], contact_points_group=[[2]])
        with self.assertRaisesRegex(ValueError, "contact group"):
            central_contact_group(metadata)
        metadata.update(contact_points_group=[[0]], center=[np.nan, 0., 0.])
        with self.assertRaisesRegex(ValueError, "contact geometry"):
            central_contact_group(metadata)

    def test_measured_handoff_fallback_is_opt_in_for_approach_not_other_motion(self):
        from adapters.pillbottle_known_model import execute_checked
        task = SimpleNamespace(take_dense_action=lambda control: True,
            robot=SimpleNamespace(get_left_arm_real_jointState=lambda: [0.]*6+[1.]))
        plan = dict(status="Success", position=np.zeros((2, 6)), velocity=np.zeros((2, 6)))
        with patch("adapters.pillbottle_known_model.settle_at_target", return_value=False), patch(
                "adapters.pillbottle_known_handoff.settle_measured_handoff", return_value=True) as measured:
            with self.assertRaisesRegex(RuntimeError, "did not reach"):
                execute_checked(task, "left", [0, 0, 1, 1, 0, 0, 0], plan, {})
            measured.assert_not_called()
            audit = {}
            execute_checked(task, "left", [0, 0, 1, 1, 0, 0, 0], plan, audit, allow_measured_handoff=True)
            self.assertEqual(measured.call_count, 1)
            self.assertFalse(audit["strict_joint_target_reached"])

    def make_task(self, *, moving=False, outside=False):
        q = np.full(6, .1)
        commands, steps = [], []
        robot = SimpleNamespace(get_left_arm_real_jointState=lambda: [*q, 1.],
            set_arm_joints=lambda value, velocity, arm: commands.append(np.asarray(value).copy()))
        def step():
            steps.append(1)
            if moving:
                q[:] += .0003
        pose = [(.05 if outside else .005), 0., 1., 1., 0., 0., 0.]
        task = SimpleNamespace(robot=robot, scene=SimpleNamespace(step=step),
                               get_arm_pose=lambda arm: pose, plan_success=True)
        return task, q, commands, steps

    def test_stable_measured_joints_inside_original_region_are_held_and_handed_off(self):
        task, q, commands, steps = self.make_task()
        audit = {}
        self.assertTrue(settle_measured_handoff(task, "left", [0, 0, 1, 1, 0, 0, 0], audit))
        self.assertEqual(len(steps), 40)
        for target in commands:
            np.testing.assert_array_equal(target, q)
        self.assertFalse(audit["trial_lift"])
        self.assertEqual(audit["cartesian_position_tolerance_m"], .02)

    def test_outside_cartesian_region_is_rejected_before_any_control(self):
        task, _, commands, steps = self.make_task(outside=True)
        self.assertFalse(settle_measured_handoff(task, "left", [0, 0, 1, 1, 0, 0, 0], {}))
        self.assertFalse(commands)
        self.assertFalse(steps)

    def test_moving_joints_are_not_accepted_merely_because_tcp_is_near(self):
        task, _, _, _ = self.make_task(moving=True)
        self.assertFalse(settle_measured_handoff(task, "left", [0, 0, 1, 1, 0, 0, 0], {}))

    def test_small_settling_transient_must_converge_before_handoff(self):
        task, q, _, steps = self.make_task()
        def step():
            steps.append(1)
            q[:] = .1+.015*(1-np.exp(-len(steps)/20.))
        task.scene.step = step
        audit = {}
        self.assertTrue(settle_measured_handoff(task, "left", [0, 0, 1, 1, 0, 0, 0], audit))
        self.assertGreater(audit["physics_steps"], 80)
        self.assertEqual(audit["samples"][-1]["consecutive"], 3)
        self.assertLess(audit["samples"][-1]["delta_rad"], .001)

    def test_tilt_compensation_preserves_lowest_cad_height(self):
        vertices = np.array([[x, y, z] for x in (-.035, .035)
                             for y in (0., .15) for z in (-.035, .035)])
        upright = np.eye(4)
        upright[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        path = ROOT/"tra_pipe/RoboTwin/assets/objects/080_pillbottle/model_data1.json"
        with patch("adapters.pillbottle_known_placement.model_vertices", return_value=vertices):
            candidates = list(placement_candidates(upright, path))
        self.assertEqual(len(candidates), 60)
        self.assertTrue(all(tilt == 0 and height == 0 for tilt, _, _, height in candidates[:12]))
        for tilt, yaw, pose, compensation in candidates:
            self.assertLessEqual(abs(tilt), 10.)
            self.assertGreaterEqual(np.min(vertices @ pose[2, :3])+compensation, -1e-10)

    def test_tilted_placement_retraces_forward_route_only_after_both_forward_moves_pass(self):
        from adapters.pillbottle_known_placement import finish_placement
        metadata = ROOT/"tra_pipe/RoboTwin/assets/objects/080_pillbottle/model_data1.json"
        for reject_forward in (False, True):
            events = []
            entity = SimpleNamespace(get_active_joints=lambda: [SimpleNamespace(get_name=lambda i=i: f"j{i}") for i in range(6)],
                get_qlimits=lambda: np.tile([-10., 10.], (6, 1)), get_qpos=lambda: np.zeros(6))
            def plan(target, last_qpos=None):
                events.append("plan")
                return dict(status="Success", position=np.zeros((2, 6)), velocity=np.zeros((2, 6)))
            robot = SimpleNamespace(left_entity=entity, left_arm_joints_name=[f"j{i}" for i in range(6)], left_plan_path=plan)
            pose = np.eye(4)
            pose[2, 3] = .8
            perception = SimpleNamespace(estimate=lambda *a: SimpleNamespace(cup_world_pose=pose,
                coaster_world_center=np.array([0., 0., .78]), timings_s={}))
            def dense(control):
                events.append("dense")
                return not (reject_forward and events.count("dense") == 2)
            task = SimpleNamespace(robot=robot, get_camera_observation=lambda name: dict(rgb=None, depth=None,
                intrinsic_cv=None, extrinsic_cv=None), get_arm_pose=lambda arm: [0., 0., 1., 1., 0., 0., 0.],
                take_dense_action=dense,
                open_gripper=lambda *a, **kw: "release", move=lambda action: events.append(action) or True,
                plan_success=True)
            tilted = np.eye(4)
            a = np.radians(5.)
            tilted[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
            def candidates(*a):
                yield 5., 0., tilted, 0.
            with patch("adapters.pillbottle_known_placement.placement_candidates", candidates), patch(
                    "adapters.pillbottle_known_model.execute_checked", side_effect=lambda *a, **kw: events.append("exit")):
                if reject_forward:
                    with self.assertRaisesRegex(RuntimeError, "did not reach"):
                        finish_placement(task, perception, "left", metadata, {}, upright_reference=np.eye(4))
                    self.assertNotIn("release", events)
                    self.assertNotIn("exit", events)
                else:
                    result = {}
                    finish_placement(task, perception, "left", metadata, result, upright_reference=np.eye(4))
                    self.assertEqual(events[:3], ["plan"]*3)
                    self.assertEqual(events[3:], ["dense", "dense", "release", "exit", "exit"])
                    self.assertTrue(result["complete"])

    def test_reverse_trajectory_reverses_velocity_without_mutating_forward(self):
        from adapters.pillbottle_known_placement import reverse_trajectory
        positions = np.arange(18., dtype=float).reshape(3, 6)
        velocities = np.arange(18., 36.).reshape(3, 6)
        original = dict(position=positions.copy(), velocity=velocities.copy(), status="Success")
        result = reverse_trajectory(original)
        np.testing.assert_array_equal(result["position"], positions[::-1])
        np.testing.assert_array_equal(result["velocity"], -velocities[::-1])
        np.testing.assert_array_equal(original["position"], positions)
        np.testing.assert_array_equal(original["velocity"], velocities)


if __name__ == "__main__":
    unittest.main()
