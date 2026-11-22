"""Local short demonstrations; commanded-state semantics match deployed WorldLA.

Each sample is observation BEFORE a dense-control interval and its commanded
joint target at the END of that interval, never image/action after-the-fact.
Ground-truth object fields belong only to a separate audit, not model inputs.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np

INSTRUCTION = "Pick up the red block with the left arm and hold it above the table."
CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_camera",
           "cam_right_wrist": "right_camera"}
MOTORS = ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
          "left_wrist_angle", "left_wrist_rotate", "left_gripper", "right_waist",
          "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle",
          "right_wrist_rotate", "right_gripper"]


def commanded_state(task):
    value = np.asarray(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState(), np.float32)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError("Expected finite 14-D commanded robot state")
    return value


def slice_control(sequence, start, end):
    sliced = {}
    for name in ("left_arm", "right_arm", "left_gripper", "right_gripper"):
        item = sequence[name]
        if item is None:
            sliced[name] = None
        elif name.endswith("arm"):
            size = len(item["position"])
            sliced[name] = (dict(item, position=item["position"][start:end],
                velocity=item["velocity"][start:end]) if start < size else None)
        else:
            stop = min(end, int(item["num_step"]))
            sliced[name] = (dict(item, result=item["result"][start:stop],
                num_step=stop-start) if start < stop else None)
    return sliced


class DenseRecorder:
    def __init__(self, task, audit, pose_to_matrix, stride=15):
        self.task, self.audit, self.pose_to_matrix, self.stride = task, audit, pose_to_matrix, stride
        self.frames = []
        self.stage = "open"
        self.physics_steps = 0

    def capture_before(self):
        observation = self.task.get_obs()
        return dict(state=np.asarray(observation["joint_action"]["vector"], np.float32).copy(),
            actual_arm_feedback=np.asarray(self.task.robot.get_left_arm_real_jointState()[:6]
                +self.task.robot.get_right_arm_real_jointState()[:6], np.float32),
            images={name: np.asarray(observation["observation"][camera]["rgb"], np.uint8).copy()
                    for name, camera in CAMERAS.items()},
            depths={name: np.asarray(observation["observation"][camera]["depth"]).copy()
                    for name, camera in CAMERAS.items()}, stage=self.stage,
            physics_step_before=self.physics_steps)

    @contextmanager
    def recording(self):
        original = self.task.take_dense_action
        def dense(sequence, save_freq=-1):
            sizes = [len(v["position"]) if k.endswith("arm") else int(v["num_step"])
                     for k, v in sequence.items() if v is not None]
            if not sizes or max(sizes) == 0:
                raise ValueError("Empty expert control sequence")
            for start in range(0, max(sizes), self.stride):
                end = min(start+self.stride, max(sizes))
                frame = self.capture_before()
                original(slice_control(sequence, start, end), save_freq=None)
                self.physics_steps += end-start
                frame.update(action=commanded_state(self.task), physics_step_after=self.physics_steps)
                self.frames.append(frame)
                # Passive scorer only; does not choose targets or alter execution.
                self.audit.observe(self.task.box.get_pose().to_transformation_matrix(),
                    self.pose_to_matrix(self.task.get_arm_pose("left")),
                    float(self.task.robot.get_left_gripper_val()), len(self.frames))
            return True
        self.task.take_dense_action = dense
        try:
            yield self
        finally:
            self.task.take_dense_action = original

    def save(self, path, metadata):
        import h5py
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        if not self.frames:
            raise ValueError("Empty demonstration")
        with h5py.File(path, "x") as target:
            target.attrs["schema"] = "ula_handover_pickup_short_v1"
            target.attrs["metadata_json"] = json.dumps(metadata)
            target.attrs["instruction"] = INSTRUCTION
            target.attrs["sample_stride_physics_steps"] = self.stride
            target.attrs["state_semantics"] = "commanded_drive_targets_matching_WorldLA_runtime"
            target.attrs["action_semantics"] = "commanded_drive_targets_after_interval"
            for key, source in [("observation.state", "state"), ("action", "action"),
                    ("audit.actual_arm_feedback", "actual_arm_feedback"),
                    ("audit.physics_step_before", "physics_step_before"),
                    ("audit.physics_step_after", "physics_step_after")]:
                target.create_dataset(key, data=np.asarray([f[source] for f in self.frames]))
            target.create_dataset("stage", data=[f["stage"] for f in self.frames], dtype=h5py.string_dtype())
            for name in CAMERAS:
                target.create_dataset(f"images/{name}", data=np.stack([f["images"][name] for f in self.frames]),
                                      compression="gzip", compression_opts=2)
                target.create_dataset(f"depths/{name}", data=np.stack([f["depths"][name] for f in self.frames]),
                                      compression="gzip", compression_opts=2)


def choose_splits(count, seed=20261107, validation=5):
    if not 1 <= validation < count:
        raise ValueError("Need separate nonempty train/validation episode sets")
    indices = set(np.random.default_rng(seed).permutation(count)[:validation].tolist())
    return ["val" if i in indices else "train" for i in range(count)]


def assert_disjoint_seeds(train_seeds, evaluation_seeds):
    overlap = set(train_seeds) & set(evaluation_seeds)
    if overlap:
        raise ValueError(f"Training/evaluation seed leakage: {sorted(overlap)}")
