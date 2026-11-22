"""CAD-only swept collision check for an object rigidly held by a robot hand.

The stock RoboTwin CuRobo model contains the arm and gripper but not the object
inside the gripper.  This module checks a planned joint trajectory afterwards
using the visually fitted object/support CAD poses.  It never reads simulator
actors, contacts, segmentation IDs, or task success.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _mesh_box(path, scale):
    import trimesh

    mesh = trimesh.load(Path(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if scale.ndim == 0:
        scale = np.repeat(scale, 3)
    vertices = vertices * scale.reshape(1, 3)
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError(f"Invalid CAD mesh for collision checking: {path}")
    scaled = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)
    to_box, extents = trimesh.bounds.oriented_bounds(scaled)
    return np.linalg.inv(to_box), np.asarray(extents, dtype=float) / 2.0


def _matrix(position, quaternion_wxyz):
    result = np.eye(4)
    result[:3, 3] = np.asarray(position, dtype=float)
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=float)
    result[:3, :3] = Rotation.from_quat(
        [quaternion_wxyz[1], quaternion_wxyz[2],
         quaternion_wxyz[3], quaternion_wxyz[0]]).as_matrix()
    return result


def _transform(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def _obb_penetrates(a_pose, a_half, b_pose, b_half, penetration_m):
    """Separating-axis test for penetration deeper than ``penetration_m``."""
    rotation = a_pose[:3, :3].T @ b_pose[:3, :3]
    translation = a_pose[:3, :3].T @ (b_pose[:3, 3] - a_pose[:3, 3])
    absolute = np.abs(rotation) + 1e-10
    tolerance = float(penetration_m)
    for axis in range(3):
        radius_a = a_half[axis]
        radius_b = float(b_half @ absolute[axis])
        if radius_a + radius_b - abs(translation[axis]) <= tolerance:
            return False
    for axis in range(3):
        radius_a = float(a_half @ absolute[:, axis])
        radius_b = b_half[axis]
        if radius_a + radius_b - abs(float(translation @ rotation[:, axis])) <= tolerance:
            return False
    # The nine cross-product axes are needed when neither box has a separating
    # face normal.  Scale the metric tolerance by the cross-axis norm.
    for a_axis in range(3):
        for b_axis in range(3):
            norm = float(np.sqrt(max(0.0, 1.0 - rotation[a_axis, b_axis] ** 2)))
            if norm < 1e-8:
                continue
            radius_a = (a_half[(a_axis + 1) % 3] * absolute[(a_axis + 2) % 3, b_axis]
                        + a_half[(a_axis + 2) % 3] * absolute[(a_axis + 1) % 3, b_axis])
            radius_b = (b_half[(b_axis + 1) % 3] * absolute[a_axis, (b_axis + 2) % 3]
                        + b_half[(b_axis + 2) % 3] * absolute[a_axis, (b_axis + 1) % 3])
            distance = abs(translation[(a_axis + 2) % 3] * rotation[(a_axis + 1) % 3, b_axis]
                           - translation[(a_axis + 1) % 3] * rotation[(a_axis + 2) % 3, b_axis])
            if radius_a + radius_b - distance <= tolerance * norm:
                return False
    return True


class HeldObjectCollisionGuard:
    """Check held-object/support penetration along CuRobo joint paths."""

    def __init__(self, task, arm, *, object_mesh, object_scale,
                 support_mesh, support_scale, object_pose_world,
                 support_pose_world, penetration_m=0.002):
        from adapters.handover.rgbd_obstacles import world_to_planner
        from curobo.types.robot import JointState

        self.task = task
        self.arm = str(arm)
        self.planner = getattr(task.robot, f"{self.arm}_planner")
        self.motion = self.planner.motion_gen
        self.JointState = JointState
        self.object_box_pose, self.object_half = _mesh_box(
            object_mesh, object_scale)
        self.support_box_pose, self.support_half = _mesh_box(
            support_mesh, support_scale)
        self.penetration_m = float(penetration_m)

        world_planner = world_to_planner(self.planner, self.arm)
        self.support_pose = world_planner @ np.asarray(support_pose_world, dtype=float)
        self.support_box_world = self.support_pose @ self.support_box_pose
        object_pose = world_planner @ np.asarray(object_pose_world, dtype=float)

        entity = getattr(task.robot, f"{self.arm}_entity")
        all_names = [joint.get_name() for joint in entity.get_active_joints()]
        self.joint_names = list(self.planner.active_joints_name)
        indices = [all_names.index(name) for name in self.joint_names]
        self.current_q = np.asarray(entity.get_qpos(), dtype=float)[indices]
        current_ee = self._ee_matrices(self.current_q[None])[0]
        self.ee_to_object = np.linalg.solve(current_ee, object_pose)
        self.support_inverse = np.linalg.inv(self.support_pose)

    def _ee_matrices(self, positions):
        state = self.JointState.from_position(
            self.motion.tensor_args.to_device(np.asarray(positions, dtype=float)),
            joint_names=self.joint_names)
        kinematics = self.motion.compute_kinematics(state)
        positions = kinematics.ee_pose.position.detach().cpu().numpy()
        quaternions = kinematics.ee_pose.quaternion.detach().cpu().numpy()
        return [_matrix(p, q) for p, q in zip(positions, quaternions)]

    def check(self, trajectories, max_joint_step_rad=0.02):
        from adapters.handover.rgbd_obstacles import dense_joint_path

        current = self.current_q.copy()
        checked = 0
        for segment_index, trajectory in enumerate(trajectories):
            positions = np.asarray(trajectory["position"], dtype=float)
            path = dense_joint_path(current, positions, max_step_rad=max_joint_step_rad)
            # Batch kinematics avoids one GPU launch per sample.
            for begin in range(0, len(path), 256):
                matrices = self._ee_matrices(path[begin:begin + 256])
                for local_index, ee_pose in enumerate(matrices):
                    object_pose = ee_pose @ self.ee_to_object @ self.object_box_pose
                    if _obb_penetrates(
                            object_pose, self.object_half,
                            self.support_box_world, self.support_half,
                            self.penetration_m):
                        return dict(
                            safe=False,
                            segment_index=int(segment_index),
                            sample_index=int(checked + local_index),
                            penetration_tolerance_m=self.penetration_m,
                            uses_simulator_truth=False,
                            geometry_source="visual_pose_plus_known_CAD")
                checked += len(matrices)
            current = positions[-1].copy()
        return dict(
            safe=True,
            samples_checked=int(checked),
            penetration_tolerance_m=self.penetration_m,
            uses_simulator_truth=False,
            geometry_source="visual_pose_plus_known_CAD")
