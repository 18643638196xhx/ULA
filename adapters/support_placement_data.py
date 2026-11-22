"""Dense local demonstrations recorded after the ULA pre-grasp handoff."""

from contextlib import contextmanager
from pathlib import Path
import json

import numpy as np


CAMERAS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
MOTORS = [
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper", "right_waist",
    "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle",
    "right_wrist_rotate", "right_gripper",
]


def commanded_state(task):
    value = np.asarray(
        task.robot.get_left_arm_jointState() + task.robot.get_right_arm_jointState(),
        dtype=np.float32,
    )
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError("Expected a finite 14-D commanded robot state")
    return value


def _slice_control(sequence, start, end):
    result = {}
    for name in ("left_arm", "right_arm", "left_gripper", "right_gripper"):
        item = sequence[name]
        if item is None:
            result[name] = None
        elif name.endswith("arm"):
            size = len(item["position"])
            result[name] = (dict(item, position=item["position"][start:end],
                                      velocity=item["velocity"][start:end])
                            if start < size else None)
        else:
            stop = min(end, int(item["num_step"]))
            result[name] = (dict(item, result=item["result"][start:stop], num_step=stop-start)
                            if start < stop else None)
    return result


class LocalTaskRecorder:
    """Pairs the image/state before an interval with its final command.

    Recording starts only after unified vision and the traditional controller
    have reached the task's ULA pre-grasp handoff.
    """

    def __init__(self, task, instruction, stride=15):
        self.task = task
        self.instruction = instruction
        self.stride = int(stride)
        self.frames = []
        self.stage = "teacher_move_000"
        self.physics_steps = 0
        self.move_index = 0

    def _capture(self):
        observation = self.task.get_obs()
        return dict(
            state=np.asarray(observation["joint_action"]["vector"], np.float32).copy(),
            actual_arm_feedback=np.asarray(
                self.task.robot.get_left_arm_real_jointState()[:6]
                + self.task.robot.get_right_arm_real_jointState()[:6], np.float32),
            images={name: np.asarray(observation["observation"][camera]["rgb"], np.uint8).copy()
                    for name, camera in CAMERAS.items()},
            stage=self.stage,
            physics_step_before=self.physics_steps,
        )

    @contextmanager
    def recording(self):
        original_dense = self.task.take_dense_action
        original_move = self.task.move

        def dense(sequence, save_freq=-1):
            sizes = [len(value["position"]) if name.endswith("arm") else int(value["num_step"])
                     for name, value in sequence.items() if value is not None]
            if not sizes or max(sizes) < 1:
                raise ValueError("Empty expert control sequence")
            for start in range(0, max(sizes), self.stride):
                end = min(start + self.stride, max(sizes))
                frame = self._capture()
                original_dense(_slice_control(sequence, start, end), save_freq=None)
                self.physics_steps += end-start
                frame.update(action=commanded_state(self.task),
                             physics_step_after=self.physics_steps)
                self.frames.append(frame)
            return True

        def move(action):
            self.stage = f"teacher_move_{self.move_index:03d}"
            self.move_index += 1
            return original_move(action)

        self.task.take_dense_action = dense
        self.task.move = move
        try:
            yield self
        finally:
            self.task.take_dense_action = original_dense
            self.task.move = original_move

    def save(self, path, metadata):
        import h5py

        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        if not self.frames:
            raise ValueError("Empty local demonstration")
        with h5py.File(path, "x") as target:
            target.attrs["schema"] = "ula_support_placement_local_v1"
            target.attrs["metadata_json"] = json.dumps(metadata)
            target.attrs["instruction"] = self.instruction
            target.attrs["sample_stride_physics_steps"] = self.stride
            target.attrs["recording_boundary"] = "after_visual_localization_and_traditional_pregrasp"
            target.attrs["teacher_truth_not_in_model_inputs"] = True
            for key, source in (
                ("observation.state", "state"), ("action", "action"),
                ("audit.actual_arm_feedback", "actual_arm_feedback"),
                ("audit.physics_step_before", "physics_step_before"),
                ("audit.physics_step_after", "physics_step_after"),
            ):
                target.create_dataset(key, data=np.asarray([frame[source] for frame in self.frames]))
            target.create_dataset("stage", data=[frame["stage"] for frame in self.frames],
                                  dtype=h5py.string_dtype())
            for name in CAMERAS:
                target.create_dataset(
                    f"images/{name}",
                    data=np.stack([frame["images"][name] for frame in self.frames]),
                    compression="gzip", compression_opts=2,
                )


def choose_splits(count, seed=20261113, validation=5):
    if not 1 <= validation < count:
        raise ValueError("Need nonempty train and validation episode sets")
    selected = set(np.random.default_rng(seed).permutation(count)[:validation].tolist())
    return ["val" if index in selected else "train" for index in range(count)]
