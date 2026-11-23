"""Offline unit tests for P0 frame calibration, rule compiler, and runtime scheduler."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from adapters.primitive_executors import RuntimeContext
from adapters.handover.take_rollout import rollout_take, settle_receiver_gripper
from adapters.rule_compiler import load_and_compile_rule
from adapters.rule_runtime import CompiledRuleRunner
from adapters.runtime_frame_calibration import (
    FrameCalibration,
    estimate_arm_calibration,
    load_calibration,
    matrix_to_pose_wxyz,
    pose_wxyz_to_matrix,
    reconcile_recorded_ee_to_ee_link,
    reconcile_recorded_ee_to_plan_target,
    reconcile_recorded_ee_to_tcp,
    save_calibration,
)

ULA_ROOT = Path(__file__).resolve().parents[1]
HANDOVER_RULE_V3 = ULA_ROOT / "results/demo_primitives_handover_rgbd_geometry_v3/tasks/handover_block/rule.yaml"


def _make_pose(xyz, yaw_deg=0.0):
    rad = np.radians(yaw_deg)
    c, s = np.cos(rad), np.sin(rad)
    T = np.eye(4, dtype=float)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = xyz
    return T


def test_pose_roundtrip_and_calibration_recovery(tmp_path):
    T_plan_off = _make_pose([0.0, 0.0, 0.0], yaw_deg=0.0)
    T_link_off = _make_pose([-0.05, 0.01, 0.02], yaw_deg=15.0)
    T_tcp_off = _make_pose([0.12, 0.0, 0.0], yaw_deg=0.0)

    samples = []
    for i in range(6):
        T_rec = _make_pose([0.1 * i, -0.2 + 0.05 * i, 0.8 + 0.02 * i], yaw_deg=20.0 * i)
        samples.append({
            "recorded_ee": T_rec,
            "plan_target": T_rec @ T_plan_off,
            "ee_link": T_rec @ T_link_off,
            "tcp": T_rec @ T_tcp_off,
        })

    arm_cal = estimate_arm_calibration("left", "fl_link6", samples)
    assert arm_cal.max_plan_target_trans_err_m < 1e-6
    assert arm_cal.max_ee_link_rot_err_deg < 1e-4
    assert np.allclose(arm_cal.T_recorded_ee_to_tcp, T_tcp_off, atol=1e-6)

    cal = FrameCalibration(1, "aloha_agilex", "handover_block", {"left": arm_cal, "right": arm_cal})
    out = save_calibration(cal, tmp_path / "cal.json")
    with pytest.raises(FileExistsError):
        save_calibration(cal, out, overwrite=False)
    loaded = load_calibration(out)

    T_query = _make_pose([0.2, 0.3, 0.9], yaw_deg=45.0)
    assert np.allclose(reconcile_recorded_ee_to_plan_target(T_query, loaded, "left"), T_query @ T_plan_off, atol=1e-6)
    assert np.allclose(reconcile_recorded_ee_to_ee_link(T_query, loaded, "left"), T_query @ T_link_off, atol=1e-6)
    assert np.allclose(reconcile_recorded_ee_to_tcp(T_query, loaded, "left"), T_query @ T_tcp_off, atol=1e-6)
    assert np.allclose(pose_wxyz_to_matrix(matrix_to_pose_wxyz(T_query)), T_query, atol=1e-6)


def test_calibration_rejects_non_rigid_samples():
    samples = [
        {"recorded_ee": _make_pose([0, 0, 0]), "plan_target": _make_pose([0, 0, 0]),
         "ee_link": _make_pose([0.05, 0, 0]), "tcp": _make_pose([0.12, 0, 0])},
        {"recorded_ee": _make_pose([0.1, 0, 0]), "plan_target": _make_pose([0.1, 0, 0]),
         "ee_link": _make_pose([0.20, 0, 0]), "tcp": _make_pose([0.22, 0, 0])},
    ]
    with pytest.raises(ValueError, match="not rigid"):
        estimate_arm_calibration("left", "fl_link6", samples)


@pytest.mark.skipif(not HANDOVER_RULE_V3.exists(), reason="v3 handover rule not present")
def test_compile_real_handover_rule_v3():
    plan = load_and_compile_rule(HANDOVER_RULE_V3)
    ids = [s.stage_id for s in plan.stages]
    assert plan.task == "handover_block"
    assert len(ids) == 11

    smap = plan.stage_map()
    assert "handover_163_199" in smap["left_006_move"].depends_on
    assert "place_199_282" in smap["right_004_move"].depends_on
    assert ids.index("left_004_move") < ids.index("right_000_move") < ids.index("handover_163_199")
    assert ids.index("handover_163_199") < ids.index("place_199_282")
    assert "goal" in smap["place_199_282"].relations
    assert {"receiver_precontact", "receiver_attachment", "donor_attachment"} <= set(smap["handover_163_199"].relations)
    assert np.allclose(
        smap["handover_163_199"].reusable_object_goal,
        smap["left_004_move"].reusable_object_goal,
    )
    assert smap["left_006_move"].role == "retreat"


class _MockBackend:
    def __init__(self):
        self.poses = {
            "left": _make_pose([-0.2, 0.0, 0.9]),
            "right": _make_pose([0.2, 0.0, 0.9]),
        }
        self.grippers = {"left": 1.0, "right": 1.0}
        self.history = []

    def get_arm_pose(self, arm: str):
        return matrix_to_pose_wxyz(self.poses[arm])

    def move_to_pose(self, arm: str, target_pose_wxyz):
        self.poses[arm] = pose_wxyz_to_matrix(target_pose_wxyz)
        self.history.append(("move", arm, list(target_pose_wxyz)))
        return True

    def set_gripper(self, arm: str, open_value: float):
        self.grippers[arm] = float(open_value)
        self.history.append(("gripper", arm, float(open_value)))
        return True


@pytest.mark.skipif(not HANDOVER_RULE_V3.exists(), reason="v3 handover rule not present")
def test_runtime_end_to_end_mock_and_donor_safety():
    plan = load_and_compile_rule(HANDOVER_RULE_V3)
    shifted_box = _make_pose([-0.15, 0.08, 0.78], yaw_deg=25.0)
    shifted_target = _make_pose([0.18, -0.04, 0.74], yaw_deg=-15.0)

    backend = _MockBackend()
    live_poses = {"box": shifted_box.copy(), "target_box": shifted_target.copy()}

    def perception(entity: str, reason: str):
        return live_poses.get(entity)

    def vla_ok(stage, ctx):
        arm = stage.receiver_arm if stage.primitive == "handover" else stage.arm
        backend.grippers[arm] = 0.0
        live_poses["box"] = backend.poses[arm] @ _make_pose([0.11, 0.0, 0.01], yaw_deg=5.0)
        return {"success": True, "arm": arm}

    def confirm_handover(stage, ctx, vla_result):
        return {"accepted": True, "reason": "mock_receiver_capture_confirmed", "uses_object_truth": False}

    ctx = RuntimeContext(
        backend,
        FrameCalibration.identity(),
        perception,
        vla_ok,
        handover_confirmation_fn=confirm_handover,
        handover_vla_handles_receiver_approach=True,
    )
    report = CompiledRuleRunner(plan, ctx).run()
    assert report["controller_complete"] is True
    assert report["completed_stages"] == len(plan.stages)
    assert report["control_uses_object_truth"] is False
    place_rec = next(r for r in report["stages"] if r["stage_id"] == "place_199_282")
    assert place_rec["details"]["released"] is True
    handover_rec = next(r for r in report["stages"] if r["stage_id"] == "handover_163_199")
    assert handover_rec["details"]["donor_handover_alignment"]["reached"] is True
    assert handover_rec["details"]["donor_handover_alignment"]["source"] == (
        "demo_target_object_and_donor_attachment"
    )
    safe_pre = handover_rec["details"]["receiver_safe_preapproach"]
    assert safe_pre["reached"] is True
    assert safe_pre["source"] == "demo_precontact_minus_receiver_attachment"
    assert safe_pre["additional_standoff_m"] >= ctx.min_tcp_clearance_m

    backend2 = _MockBackend()
    live_poses2 = {"box": shifted_box.copy(), "target_box": shifted_target.copy()}

    def vla_fail_on_handover(stage, ctx):
        if stage.primitive == "handover":
            return {"success": False, "reason": "receiver_missed"}
        backend2.grippers[stage.arm] = 0.0
        live_poses2["box"] = backend2.poses[stage.arm] @ _make_pose([0.11, 0.0, 0.01])
        return {"success": True}

    ctx2 = RuntimeContext(backend2, FrameCalibration.identity(), lambda e, r: live_poses2.get(e), vla_fail_on_handover)
    report2 = CompiledRuleRunner(plan, ctx2).run()
    assert report2["controller_complete"] is False
    assert report2["failure_stage"] == "handover_163_199"
    assert backend2.grippers["left"] == 0.0

    backend3 = _MockBackend()
    live_poses3 = {"box": shifted_box.copy(), "target_box": shifted_target.copy()}

    def vla_reports_close(stage, ctx):
        arm = stage.receiver_arm if stage.primitive == "handover" else stage.arm
        backend3.grippers[arm] = 0.0
        live_poses3["box"] = backend3.poses[arm] @ _make_pose([0.11, 0.0, 0.01])
        return {"success": True}

    ctx3 = RuntimeContext(
        backend3,
        FrameCalibration.identity(),
        lambda e, r: live_poses3.get(e),
        vla_reports_close,
        handover_confirmation_fn=lambda stage, ctx, result: {
            "accepted": False,
            "reason": "mock_fresh_rgbd_rejected",
            "uses_object_truth": False,
        },
    )
    report3 = CompiledRuleRunner(plan, ctx3).run()
    assert report3["controller_complete"] is False
    assert report3["failure_stage"] == "handover_163_199"
    assert report3["failure_reason"] == "receiver_attachment_unconfirmed"
    assert backend3.grippers["left"] == 0.0
    assert ("gripper", "left", 1.0) not in backend3.history


class _FakeTakeRobot:
    def __init__(self):
        self.left_state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.0], dtype=float)
        self.right_state = np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 1.0], dtype=float)
        self.right_gripper = 1.0
        self.actual_right_opening = 1.0

    def get_left_arm_jointState(self):
        return self.left_state.tolist()

    def get_left_gripper_val(self):
        return float(self.left_state[6])

    def get_right_arm_jointState(self):
        return self.right_state.tolist()

    def get_right_gripper_val(self):
        return float(self.right_gripper)

    def get_actual_normalized_gripper_val(self, arm):
        assert arm == "right"
        return float(self.actual_right_opening)


class _FakeTakeTask:
    def __init__(self):
        self.robot = _FakeTakeRobot()
        self.take_action_cnt = 0
        self.step_lim = 50
        self.actions = []
        self.receiver_obstruction = 0.0

    def get_instruction(self):
        return "Take the block from the left hand."

    def get_obs(self):
        return {}

    def get_arm_pose(self, arm):
        x = -0.1 if arm == "left" else 0.1
        return [x, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]

    def take_action(self, values):
        action = np.asarray(values, dtype=float).copy()
        self.actions.append(action)
        self.robot.left_state = action[:7].copy()
        self.robot.right_state = action[7:14].copy()
        self.robot.right_gripper = float(action[13])
        physical_target = max(float(self.receiver_obstruction), float(action[13]))
        self.robot.actual_right_opening += 0.5 * (
            physical_target - self.robot.actual_right_opening
        )
        self.take_action_cnt += 1


class _FakeTakePolicy:
    chunk_steps = 10

    def __init__(self):
        self.actions = []
        for right_gripper in (1.0, 0.04, 0.03, 0.02, 0.01):
            action = np.full(14, 0.8, dtype=float)
            action[6] = 1.0  # Policy tries to release/move the donor.
            action[13] = right_gripper
            self.actions.append(action)

    def reset(self):
        pass

    def step(self, task, observation):
        for action in self.actions:
            if task.take_action_cnt >= task.step_lim:
                break
            task.take_action(action)


def test_take_rollout_protects_donor_and_stops_after_receiver_closure():
    # 1. Empty close: command-only streak is recorded, but VLA is NOT terminated early.
    empty_task = _FakeTakeTask()
    empty_task.receiver_obstruction = 0.0
    empty_policy = _FakeTakePolicy()
    empty_result = {}
    donor_before = empty_task.robot.left_state.copy()

    rollout_take(
        empty_task,
        empty_policy,
        empty_result,
        max_steps=20,
        protect_donor=True,
        receiver_closed_threshold=0.05,
        stable_receiver_close_steps=3,
    )

    assert empty_result["donor_protected"] is True
    assert empty_result["receiver_close_command_stable"] is True
    assert empty_result["receiver_close_stable"] is False
    assert empty_result["receiver_blocked_stable"] is False
    assert empty_result["stop_reason"] == "fixed_action_budget"
    assert empty_result["vla_steps"] == 20
    assert all(np.allclose(action[:7], donor_before) for action in empty_task.actions)
    assert empty_task.robot.get_left_gripper_val() == 0.0

    # 2. Physically blocked close: once finger encoder stabilizes above minimum opening,
    # rollout terminates early while keeping donor frozen.
    held_task = _FakeTakeTask()
    held_task.receiver_obstruction = 0.30
    held_policy = _FakeTakePolicy()
    held_policy.actions = []
    for right_gripper in (0.04, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
        action = np.full(14, 0.8, dtype=float)
        action[6] = 1.0  # Policy tries to release/move the donor.
        action[13] = right_gripper
        held_policy.actions.append(action)
    held_result = {}
    held_donor_before = held_task.robot.left_state.copy()

    rollout_take(
        held_task,
        held_policy,
        held_result,
        max_steps=30,
        protect_donor=True,
        receiver_closed_threshold=0.05,
        stable_receiver_close_steps=3,
        stable_blocked_steps=4,
    )

    assert held_result["donor_protected"] is True
    assert held_result["receiver_close_command_stable"] is True
    assert held_result["receiver_close_stable"] is True
    assert held_result["receiver_blocked_stable"] is True
    assert held_result["stop_reason"] == "receiver_blocked_stable_donor_protected"
    assert held_result["vla_steps"] < 30
    assert all(np.allclose(action[:7], held_donor_before) for action in held_task.actions)
    assert held_task.robot.get_left_gripper_val() == 0.0


def test_receiver_settle_distinguishes_blocked_from_empty_fingers():
    held = _FakeTakeTask()
    held.receiver_obstruction = 0.30
    held_result = {}
    held_audit = settle_receiver_gripper(held, held_result)
    assert held_audit["settled"] is True
    assert held_audit["receiver_aperture_blocked"] is True
    assert held_result["receiver_aperture_confirmed"] is True
    assert held_audit["final_measured_opening"] >= 0.30

    empty = _FakeTakeTask()
    empty.receiver_obstruction = 0.0
    empty_result = {}
    empty_audit = settle_receiver_gripper(empty, empty_result)
    assert empty_audit["settled"] is True
    assert empty_audit["receiver_aperture_blocked"] is False
    assert empty_result["receiver_aperture_confirmed"] is False


def test_closure_attachment_earliest_sustained_run_and_bounds():
    import numpy as np
    import pytest
    import sys
    rt = str(Path(__file__).resolve().parents[1] / 'tra_pipe' / 'RoboTwin')
    if rt not in sys.path:
        sys.path.insert(0, rt)
    from adapters.handover.middle_control import closure_attachment

    T_block = np.eye(4, dtype=float)
    T_block[:3, 3] = [0.0, 0.0, 0.84]
    # Pose near block (14 cm away) vs lifted pose (46 cm away)
    ee_near = [0.0, 0.0, 0.98, 1.0, 0.0, 0.0, 0.0]
    ee_far = [0.0, 0.0, 1.30, 1.0, 0.0, 0.0, 0.0]

    # Early 1-step glitch (far) -> open -> sustained real grasp (near) -> reopen -> late reclosure (far)
    cmds = [
        {"step": 1, "left": 1.0, "left_ee_pose": ee_far},
        {"step": 2, "left": 0.02, "left_ee_pose": ee_far},   # 1-step glitch: must be ignored
        {"step": 3, "left": 1.0, "left_ee_pose": ee_far},
        {"step": 23, "left": 0.01, "left_ee_pose": ee_near}, # earliest sustained run start
        {"step": 24, "left": 0.00, "left_ee_pose": ee_near},
        {"step": 25, "left": 0.00, "left_ee_pose": ee_near},
        {"step": 190, "left": 0.50, "left_ee_pose": ee_far}, # mid-air reopen
        {"step": 191, "left": 0.00, "left_ee_pose": ee_far}, # late reclosure: must be ignored
        {"step": 192, "left": 0.00, "left_ee_pose": ee_far},
        {"step": 193, "left": 0.00, "left_ee_pose": ee_far},
    ]
    att, info = closure_attachment(T_block, cmds, stable_steps=3)
    assert info["closure_step"] == 23
    assert abs(info["ee_to_block_distance_m"] - 0.14) < 1e-6
    assert abs(np.linalg.norm(att[:3, 3]) - 0.14) < 1e-6

    # Implausible distance (0.46 m) must raise instead of silently poisoning transport
    bad_cmds = [
        {"step": 10, "left": 0.0, "left_ee_pose": ee_far},
        {"step": 11, "left": 0.0, "left_ee_pose": ee_far},
        {"step": 12, "left": 0.0, "left_ee_pose": ee_far},
    ]
    with pytest.raises(RuntimeError, match="Implausible closure_attachment distance"):
        closure_attachment(T_block, bad_cmds, stable_steps=3)


def test_execute_rotate_right_multiplies_relative_rotation():
    import numpy as np
    from adapters.primitive_executors import RuntimeContext, execute_stage
    from adapters.rule_compiler import compile_demo_rule
    from adapters.runtime_frame_calibration import (
        ArmFrameCalibration,
        FrameCalibration,
        matrix_to_pose_wxyz,
        pose_wxyz_to_matrix,
        rotation_error_deg,
    )

    # 90-degree rotation around EE X-axis with local translation
    T_rel = np.array([
        [1.0, 0.0, 0.0, 0.02],
        [0.0, 0.0, -1.0, -0.05],
        [0.0, 1.0, 0.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    rule_data = {
        "schema": "ula_demo_primitives_v1",
        "task": "adjust_bottle",
        "active_arms": ["left"],
        "stages": [
            {
                "id": "left_003_rotate",
                "primitive": "rotate",
                "arm": "left",
                "controller": "traditional",
                "executor": "constrained_rotation",
                "demo_frames": [90, 130],
                "geometry": {
                    "relative_rotation": T_rel.tolist(),
                    "relative_rotation_role": "recorded_hand_motion_not_semantic_object_axis",
                },
            }
        ],
    }
    plan = compile_demo_rule(rule_data)
    assert len(plan.stages) == 1
    st = plan.stages[0]
    assert st.primitive == "rotate"
    assert st.executor == "constrained_rotation"
    assert st.relative_rotation is not None

    # Non-identity starting pose to verify right-multiplication T_cur @ T_rel (not left-multiplication)
    T_start = np.array([
        [0.0, -1.0, 0.0, -0.10],
        [1.0,  0.0, 0.0, -0.10],
        [0.0,  0.0, 1.0,  0.95],
        [0.0,  0.0, 0.0,  1.0],
    ], dtype=float)
    T_expected = T_start @ T_rel

    class _FakeBackend:
        def __init__(self, start_mat):
            self.pose = matrix_to_pose_wxyz(start_mat)
            self.calls = []

        def get_arm_pose(self, arm):
            return list(self.pose)

        def move_to_pose(self, arm, target_pose_wxyz, **kwargs):
            self.calls.append(list(target_pose_wxyz))
            self.pose = list(target_pose_wxyz)
            return True

    cal = FrameCalibration(
        schema_version=1,
        robot_name="aloha_agilex",
        source_task="adjust_bottle",
        arms={
            "left": ArmFrameCalibration(
                arm="left",
                ee_link="fl_link6",
                sample_count=1,
                T_recorded_ee_to_plan_target=np.eye(4).tolist(),
                T_recorded_ee_to_ee_link=np.eye(4).tolist(),
                T_recorded_ee_to_tcp=np.eye(4).tolist(),
                max_plan_target_trans_err_m=0.0,
                max_plan_target_rot_err_deg=0.0,
                max_ee_link_trans_err_m=0.0,
                max_ee_link_rot_err_deg=0.0,
                max_tcp_trans_err_m=0.0,
                max_tcp_rot_err_deg=0.0,
            )
        },
    )
    backend = _FakeBackend(T_start)
    ctx = RuntimeContext(
        robot_backend=backend,
        calibration=cal,
        perception_fn=lambda *a, **k: None,
        vla_fn=lambda *a, **k: {},
    )
    outcome = execute_stage(st, ctx)
    assert outcome.success is True
    assert outcome.reason == "ok"
    assert len(backend.calls) == 2  # 90 deg > 75 deg -> 2 SLERP waypoints
    T_final = pose_wxyz_to_matrix(backend.pose)
    assert np.linalg.norm(T_final[:3, 3] - T_expected[:3, 3]) < 1e-6
    assert rotation_error_deg(T_final[:3, :3], T_expected[:3, :3]) < 1e-4


def test_execute_move_bimanual_synchronized_lift():
    import numpy as np
    from adapters.primitive_executors import RuntimeContext, execute_stage
    from adapters.rule_compiler import CompiledStage
    from adapters.runtime_frame_calibration import (
        ArmFrameCalibration,
        FrameCalibration,
        matrix_to_pose_wxyz,
        pose_wxyz_to_matrix,
    )

    T_l0 = np.eye(4, dtype=float)
    T_l0[:3, 3] = [-0.15, 0.0, 0.78]
    T_r0 = np.eye(4, dtype=float)
    T_r0[:3, 3] = [0.15, 0.0, 0.78]

    class _FakeBimanualBackend:
        def __init__(self):
            self.poses = {"left": matrix_to_pose_wxyz(T_l0), "right": matrix_to_pose_wxyz(T_r0)}
            self.both_calls = []
            self.single_calls = []

        def get_arm_pose(self, arm):
            return list(self.poses[arm])

        def move_to_pose(self, arm, target_wxyz):
            self.single_calls.append((arm, list(target_wxyz)))
            self.poses[arm] = list(target_wxyz)
            return True

        def move_both_to_poses(self, l_wxyz, r_wxyz):
            self.both_calls.append((list(l_wxyz), list(r_wxyz)))
            self.poses["left"] = list(l_wxyz)
            self.poses["right"] = list(r_wxyz)
            return True

    def _arm_cal(arm, link):
        return ArmFrameCalibration(
            arm=arm, ee_link=link, sample_count=1,
            T_recorded_ee_to_plan_target=np.eye(4).tolist(),
            T_recorded_ee_to_ee_link=np.eye(4).tolist(),
            T_recorded_ee_to_tcp=np.eye(4).tolist(),
            max_plan_target_trans_err_m=0.0, max_plan_target_rot_err_deg=0.0,
            max_ee_link_trans_err_m=0.0, max_ee_link_rot_err_deg=0.0,
            max_tcp_trans_err_m=0.0, max_tcp_rot_err_deg=0.0,
        )

    cal = FrameCalibration(
        schema_version=1, robot_name="aloha_agilex", source_task="lift_pot",
        arms={"left": _arm_cal("left", "fl_link6"), "right": _arm_cal("right", "fr_link6")},
    )
    backend = _FakeBimanualBackend()
    ctx = RuntimeContext(
        robot_backend=backend, calibration=cal,
        perception_fn=lambda *a, **k: None, vla_fn=lambda *a, **k: {},
    )
    ctx.held_objects["left"] = "pot"
    ctx.held_objects["right"] = "pot"

    st_r = CompiledStage(
        stage_id="right_002_move", primitive="move", controller="traditional",
        executor="curobo", role="retreat", arm="right", donor_arm=None, receiver_arm=None,
        object_name="pot", reference_name=None, demo_frames=(92, 112), depends_on=(),
        motion_delta_xyz_m=(0.0, 0.0, 0.12),
    )
    st_l = CompiledStage(
        stage_id="left_002_move", primitive="move", controller="traditional",
        executor="curobo", role="retreat", arm="left", donor_arm=None, receiver_arm=None,
        object_name="pot", reference_name=None, demo_frames=(100, 112), depends_on=(),
        motion_delta_xyz_m=(0.0, 0.0, 0.12),
    )

    out1 = execute_stage(st_r, ctx)
    out2 = execute_stage(st_l, ctx)
    assert out1.success is True and out1.reason == "ok"
    assert out2.success is True and out2.reason == "bimanual_lift_already_done"
    assert len(backend.both_calls) == 1
    assert len(backend.single_calls) == 0
    assert abs(pose_wxyz_to_matrix(backend.poses["left"])[2, 3] - 0.90) < 1e-6
    assert abs(pose_wxyz_to_matrix(backend.poses["right"])[2, 3] - 0.90) < 1e-6


def test_estimate_object_z_yaw_adaptive_symmetry_modes(tmp_path):
    import numpy as np
    from adapters.unified_perception import estimate_object_z_yaw_adaptive

    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 300)
    r = rng.uniform(0.0, 0.04, 300)
    cyl = np.column_stack([r * np.cos(theta), r * np.sin(theta), rng.uniform(0.74, 0.85, 300)])
    Rz_cyl, info_cyl = estimate_object_z_yaw_adaptive(cyl, symmetric_180=True)
    assert info_cyl["method"] == "radially_symmetric_skip"
    assert np.allclose(Rz_cyl, np.eye(3), atol=1e-6)

    tpl = np.column_stack([
        rng.uniform(-0.15, 0.15, 400),
        rng.uniform(-0.03, 0.03, 400),
        rng.uniform(0.74, 0.80, 400),
    ])
    tpl_file = tmp_path / "bar.npz"
    np.savez(tpl_file, bar_points_world=tpl)
    ang = np.radians(32.0)
    R_true = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]])
    obs = tpl @ R_true.T + np.array([0.05, -0.04, 0.0])
    Rz_bar, info_bar = estimate_object_z_yaw_adaptive(obs, f"{tpl_file}#bar_points_world", symmetric_180=True)
    assert info_bar["method"] == "pca_axis_180"
    assert abs(info_bar["yaw_deg"] - 32.0) < 1.0


def test_execute_move_bimanual_adaptive_lift_and_scale_fallback():
    import numpy as np
    from adapters.primitive_executors import RuntimeContext, execute_stage
    from adapters.rule_compiler import CompiledStage
    from adapters.runtime_frame_calibration import (
        ArmFrameCalibration,
        FrameCalibration,
        matrix_to_pose_wxyz,
        pose_wxyz_to_matrix,
    )

    T_l0 = np.eye(4, dtype=float)
    T_l0[:3, 3] = [-0.15, 0.0, 0.90]
    T_r0 = np.eye(4, dtype=float)
    T_r0[:3, 3] = [0.15, 0.0, 0.90]

    class _CeilingBackend:
        def __init__(self, z_ceiling):
            self.poses = {"left": matrix_to_pose_wxyz(T_l0), "right": matrix_to_pose_wxyz(T_r0)}
            self.z_ceiling = z_ceiling
            self.calls = []

        def get_arm_pose(self, arm):
            return list(self.poses[arm])

        def move_both_to_poses(self, l_wxyz, r_wxyz):
            self.calls.append((l_wxyz[2], r_wxyz[2]))
            if max(l_wxyz[2], r_wxyz[2]) > self.z_ceiling:
                return False
            self.poses["left"] = list(l_wxyz)
            self.poses["right"] = list(r_wxyz)
            return True

    def _arm_cal(arm, link):
        return ArmFrameCalibration(
            arm=arm, ee_link=link, sample_count=1,
            T_recorded_ee_to_plan_target=np.eye(4).tolist(),
            T_recorded_ee_to_ee_link=np.eye(4).tolist(),
            T_recorded_ee_to_tcp=np.eye(4).tolist(),
            max_plan_target_trans_err_m=0.0, max_plan_target_rot_err_deg=0.0,
            max_ee_link_trans_err_m=0.0, max_ee_link_rot_err_deg=0.0,
            max_tcp_trans_err_m=0.0, max_tcp_rot_err_deg=0.0,
        )

    cal = FrameCalibration(
        schema_version=1, robot_name="aloha_agilex", source_task="grab_roller",
        arms={"left": _arm_cal("left", "fl_link6"), "right": _arm_cal("right", "fr_link6")},
    )
    backend = _CeilingBackend(z_ceiling=0.991)
    ctx = RuntimeContext(
        robot_backend=backend, calibration=cal,
        perception_fn=lambda *a, **k: None, vla_fn=lambda *a, **k: {},
    )
    ctx.held_objects["left"] = "roller"
    ctx.held_objects["right"] = "roller"
    st = CompiledStage(
        stage_id="left_002_move", primitive="move", controller="traditional",
        executor="curobo", role="retreat", arm="left", donor_arm=None, receiver_arm=None,
        object_name="roller", reference_name=None, demo_frames=(75, 87), depends_on=(),
        motion_delta_xyz_m=(0.0, 0.0, 0.12),
    )
    out = execute_stage(st, ctx)
    assert out.success is True
    assert out.details["used_scale"] == 0.75
    assert abs(pose_wxyz_to_matrix(backend.poses["left"])[2, 3] - 0.99) < 1e-6


def test_execute_place_levels_small_tilt_and_enforces_support_clearance():
    import numpy as np
    from adapters.primitive_executors import RuntimeContext, execute_stage
    from adapters.rule_compiler import CompiledStage
    from adapters.runtime_frame_calibration import (
        ArmFrameCalibration,
        FrameCalibration,
        matrix_to_pose_wxyz,
        pose_wxyz_to_matrix,
    )

    # 12-deg tilted goal (tracking noise) with too-low Z (would penetrate a 2cm plate)
    ang = np.radians(12.0)
    T_tilted_goal = np.array([
        [np.cos(ang), 0.0, -np.sin(ang), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [np.sin(ang), 0.0, np.cos(ang), 0.002],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)

    T_start = np.eye(4, dtype=float)
    T_start[:3, 3] = [-0.25, -0.10, 0.935]
    T_plate = np.eye(4, dtype=float)
    T_plate[:3, 3] = [-0.03, -0.10, 0.744]
    # Object is 15 cm below EE (z_obj = 0.785 when EE z = 0.935)
    T_att = np.eye(4, dtype=float)
    T_att[:3, 3] = [0.0, 0.0, -0.15]

    class _FakePlaceBackend:
        def __init__(self):
            self.pose = matrix_to_pose_wxyz(T_start)
            self.moves = []
            self.gripper_calls = []

        def get_arm_pose(self, arm):
            return list(self.pose)

        def move_to_pose(self, arm, target_wxyz):
            self.moves.append(list(target_wxyz))
            self.pose = list(target_wxyz)
            return True

        def set_gripper(self, arm, val):
            self.gripper_calls.append(val)
            return True

    cal = FrameCalibration(
        schema_version=1, robot_name="aloha_agilex", source_task="place_container_plate",
        arms={"left": ArmFrameCalibration(
            arm="left", ee_link="fl_link6", sample_count=1,
            T_recorded_ee_to_plan_target=np.eye(4).tolist(),
            T_recorded_ee_to_ee_link=np.eye(4).tolist(),
            T_recorded_ee_to_tcp=np.eye(4).tolist(),
            max_plan_target_trans_err_m=0.0, max_plan_target_rot_err_deg=0.0,
            max_ee_link_trans_err_m=0.0, max_ee_link_rot_err_deg=0.0,
            max_tcp_trans_err_m=0.0, max_tcp_rot_err_deg=0.0,
        )},
    )
    backend = _FakePlaceBackend()
    ctx = RuntimeContext(
        robot_backend=backend, calibration=cal,
        perception_fn=lambda name, reason: T_plate,
        vla_fn=lambda *a, **k: {},
        enable_approach_yaw_symmetry=False,
        object_init_z_min={"container": 0.741},
        support_z_top={"plate": 0.761},
        place_release_clearance_m=0.015,
    )
    ctx.attachments["left"] = T_att
    ctx.held_objects["left"] = "container"
    ctx.grasp_closure_ee_z["left"] = 0.935

    st = CompiledStage(
        stage_id="place_71_133", primitive="place", controller="traditional",
        executor="curobo_place", role="place", arm="left", donor_arm=None, receiver_arm=None,
        object_name="container", reference_name="plate", demo_frames=(71, 133), depends_on=(),
        reusable_object_goal=T_tilted_goal.tolist(),
    )
    out = execute_stage(st, ctx)
    assert out.success is True
    assert out.details["leveled_goal_tilt_deg"] > 10.0
    # Must have performed pre-transport vertical lift first, then preplace, then descend, then retreat
    assert "pre_transport_lift" in out.details
    # Descend Z must be clamped to >= 0.935 + (0.761 - 0.741) + 0.015 = 0.970
    descend_z = out.details["place_descend"]["target_pose_wxyz"][2]
    assert descend_z >= 0.9699
