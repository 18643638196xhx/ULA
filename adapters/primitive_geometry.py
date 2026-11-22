"""Shared geometry contract for demonstration-derived ULA primitives.

The extractor stores relations, never reusable demonstration world targets.
At runtime the same relations are composed with freshly observed object and
target poses.  This module deliberately contains no task-name branches.
"""
from __future__ import annotations

import numpy as np


SCHEMA_VERSION = 1

# Each primitive declares which geometric relation makes it reusable.  The
# concrete entity names (cup, block, pad, door, ...) are supplied by role
# binding, not by primitive-specific code.
GEOMETRY_TEMPLATES = {
    "move": {
        "approach": ("object", "recorded_ee"),
        "held_transport": ("target", "object"),
    },
    "grasp": {
        "precontact": ("object", "recorded_ee"),
        "attachment": ("recorded_ee", "object"),
    },
    "release": {
        "prerelease_object_goal": ("target", "object"),
    },
    "push": {"contact_path": ("object", "recorded_ee")},
    "pull": {"contact_path": ("object", "recorded_ee")},
    "rotate": {"contact_path": ("object", "recorded_ee")},
    "press": {"contact_path": ("object", "recorded_ee")},
    "strike": {"contact_path": ("object", "recorded_ee")},
    "place": {
        "goal": ("target", "object"),
        "runtime_attachment": ("recorded_ee", "object"),
    },
    "handover": {
        "receiver_precontact": ("object", "receiver_recorded_ee"),
        "receiver_attachment": ("receiver_recorded_ee", "object"),
        "donor_attachment": ("donor_recorded_ee", "object"),
    },
    "open": {"contact_path": ("object", "recorded_ee")},
}


def _transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-4)):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return matrix


def relative_pose(T_world_reference, T_world_controlled):
    """Return ``T_reference_controlled`` from two world-frame poses."""
    reference = _transform(T_world_reference, "T_world_reference")
    controlled = _transform(T_world_controlled, "T_world_controlled")
    return np.linalg.inv(reference) @ controlled


def apply_relative_pose(T_world_reference, T_reference_controlled):
    """Move a demonstrated relation to a freshly observed reference pose."""
    reference = _transform(T_world_reference, "T_world_reference")
    relation = _transform(T_reference_controlled, "T_reference_controlled")
    return reference @ relation


def hand_goal_for_object_goal(T_world_target, T_target_object_goal,
                              T_ee_object_runtime):
    """Resolve the EE pose that realizes an object-relative placement goal.

    ``T_ee_object_runtime`` must be measured after the current grasp.  It must
    not be copied from the demonstration because VLA may grasp differently.
    """
    world_object = apply_relative_pose(T_world_target, T_target_object_goal)
    attachment = _transform(T_ee_object_runtime, "T_ee_object_runtime")
    return world_object @ np.linalg.inv(attachment)


def relation_record(name, reference, controlled, frame,
                    T_world_reference, T_world_controlled, source):
    """Create one serializable, self-describing demonstrated relation."""
    return {
        "name": name,
        "reference": reference,
        "controlled": controlled,
        "frame": int(frame),
        "T_reference_controlled": relative_pose(
            T_world_reference, T_world_controlled),
        "source": source,
    }
