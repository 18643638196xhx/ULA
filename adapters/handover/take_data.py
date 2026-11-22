"""Independent right-hand takeover demonstrations; pickup artifacts stay intact."""
from collections import deque
import numpy as np

from adapters.handover.pickup_audit import lowest_surface_z
from adapters.handover.pickup_data import DenseRecorder

INSTRUCTION = "Take the red block with the right arm, release it from the left gripper, and hold it in the right gripper."


class TakeAudit:
    """Passive geometry score AFTER control, never an online handoff gate."""
    def __init__(self, initial_lowest_z, left_at_start, stable_steps=5):
        self.initial_lowest_z = float(initial_lowest_z)
        self.left_at_start = np.asarray(left_at_start)[:3, 3].copy()
        self.window = deque(maxlen=stable_steps)
        self.stable_steps, self.samples = stable_steps, []

    def observe(self, block, left, right, left_command, right_command, step):
        block, left, right = map(np.asarray, (block, left, right))
        clearance = lowest_surface_z(block)-self.initial_lowest_z
        relative = np.linalg.solve(right, block)[:3, 3]
        retreat = float(np.linalg.norm(left[:3, 3]-self.left_at_start))
        left_distance = float(np.linalg.norm(left[:3, 3]-block[:3, 3]))
        candidate = bool(clearance >= .03 and left_command > .8
            and 0 <= right_command < .05 and np.linalg.norm(relative) <= .20
            and retreat >= .06 and left_distance >= .17)
        if candidate:
            self.window.append(relative.copy())
        else:
            self.window.clear()
        drift = float(np.max(np.linalg.norm(np.asarray(self.window)-self.window[0],axis=1))) if self.window else None
        sustained = bool(len(self.window) == self.stable_steps and drift <= .01)
        self.samples.append(dict(step=int(step), clearance_m=float(clearance),
            block_world=block.tolist(), left_command=float(left_command), right_command=float(right_command),
            left_retreat_m=retreat, left_to_block_m=left_distance,
            right_to_block_m=float(np.linalg.norm(relative)), relative_drift_m=drift,
            candidate=candidate, sustained=sustained))

    def report(self):
        return dict(success=bool(self.samples and self.samples[-1]["sustained"]),
            source="passive_simulation_geometry_success_filter_not_controller_input",
            feeds_back=False, stable_samples=self.stable_steps, clearance_threshold_m=.03,
            left_retreat_threshold_m=.06, left_distance_threshold_m=.17,
            right_distance_threshold_m=.20, relative_drift_threshold_m=.01,
            force_verified=False, samples=self.samples)


class TakeRecorder(DenseRecorder):
    def save(self, path, metadata):
        import h5py
        super().save(path, metadata)
        with h5py.File(path, "r+") as target:
            target.attrs["schema"] = "ula_handover_right_take_short_v1"
            target.attrs["instruction"] = INSTRUCTION
            target.attrs["state_semantics"] = "commanded_drive_targets_matching_native_pi05_runtime"


def expert(task, attachment, recorder, result):
    """Official lower grasp geometry, propagated from initial RGB-D + robot FK."""
    from envs.utils import ArmTag
    from adapters.handover.simple_control import block_metadata
    from adapters.handover.middle_control import hold_interval
    from traditional_place_empty_cup.controller import EstimatedActor
    from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
    left, right = ArmTag("left"), ArmTag("right")
    predicted = pose_wxyz_to_matrix(task.get_arm_pose("left"))@attachment
    actor = EstimatedActor(predicted, block_metadata())
    arm, actions = task.grasp_actor(actor, right, pre_grasp_dis=.07,
                                    grasp_dis=0., contact_point_id=[4,5,6,7])
    if len(actions) != 3:
        raise RuntimeError("Expected pregrasp, insert and close actions")
    result.update(predicted_block_pose=predicted.tolist(), uses_object_truth=False,
        known_primitive_size_m=[.06,.06,.20], lower_contact_ids=[4,5,6,7],
        targets=[list(action.target_pose) for action in actions[:2]],
        start="actual_pi05_left_pickup_and_traditional_middle_endpoint",
        trial_lift=False, stages=[])
    sequence = [("right_open", lambda: task.open_gripper(right)),
        ("right_pregrasp", lambda: (arm,[actions[0]])),
        ("right_insert", lambda: (arm,[actions[1]])),
        ("right_close", lambda: (arm,[actions[2]])),
        ("left_release", lambda: task.open_gripper(left)),
        ("left_retreat", lambda: task.move_by_displacement(left,z=.10,move_axis="arm"))]
    with recorder.recording():
        for stage, command in sequence:
            recorder.stage = stage
            if not task.move(command()) or not task.plan_success:
                raise RuntimeError(f"Right-take expert failed during {stage}")
            result["stages"].append(stage)
        recorder.stage = "right_hold"
        for _ in range(10):
            hold_interval(task, recorder.stride)
    result["complete"] = True
