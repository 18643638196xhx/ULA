import numpy as np
from scipy.spatial.transform import Rotation

from adapters.placement_relations import placement_orientation


def _quat_abs(rotation):
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return np.abs(np.asarray([w, x, y, z]))


def test_mouse_demo_prior_satisfies_official_orientation_clause():
    rotation, source = placement_orientation("place_mouse_pad", np.eye(3))
    quat = _quat_abs(rotation)
    assert source == "demo_world_rotation_medoid"
    assert min(abs(quat[2] * quat[3] - 0.49),
               abs(quat[0] * quat[1] - 0.49)) < 0.015


def test_stapler_demo_prior_satisfies_official_orientation_clause():
    rotation, source = placement_orientation("move_stapler_pad", np.eye(3))
    quat = _quat_abs(rotation)
    assert source == "demo_world_rotation_medoid"
    assert quat.max() - quat.min() < 0.02


def test_fan_demo_prior_faces_robot():
    rotation, source = placement_orientation("place_fan", np.eye(3))
    quat = _quat_abs(rotation)
    assert source == "demo_world_rotation_medoid"
    assert np.all(np.abs(quat - np.asarray([0.707, 0.707, 0.0, 0.0])) < 0.05)


def test_world_prior_does_not_follow_ambiguous_pad_yaw():
    support = Rotation.from_euler("z", 1.2).as_matrix()
    first, _ = placement_orientation("place_fan", np.eye(3))
    rotated_support, _ = placement_orientation("place_fan", support)
    assert np.allclose(first, rotated_support)
