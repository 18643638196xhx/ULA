"""Demonstration-relative approach candidates with geometric fallback support.

The online controller consumes only a compiled prior and the current visual
object estimate.  Demonstration world positions are never replayed directly.
Two explicitly different contracts are supported:

``object_6d``
    The offline compiler supplied a true object-to-EE transform.  A reliable
    current visual object pose is required.

``visual_bounds_top_anchor``
    The demonstration contains robot poses but no object pose (the public
    RoboTwin HDF5 case).  Its approach displacement is transferred to the top
    centre of the currently observed bounds.  Only the demonstrated approach
    displacement is transferred.  TCP orientation comes from the robot's
    already-validated geometric template, because an absolute demo EE rotation
    is not object-relative and previously caused large execution residuals.
"""

import json
from pathlib import Path

import numpy as np


def _rigid(value, name):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0., 0., 0., 1.], atol=1e-6)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-4)):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return value


def load_demo_approach_prior(path):
    """Load and validate a small, auditable compiled demonstration prior."""
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "ula_demo_relative_approach_v1":
        raise ValueError("Unsupported demonstration approach prior schema")
    mode = value.get("mode")
    if mode == "object_6d":
        _rigid(value["object_to_ee"], "object_to_ee")
    elif mode == "visual_bounds_top_anchor":
        delta = np.asarray(value.get("interaction_to_approach_world_m"), float)
        rotation = np.asarray(value.get("approach_ee_rotation_world"), float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("Demonstration approach displacement must be a finite 3-vector")
        check = np.eye(4); check[:3, :3] = rotation
        _rigid(check, "demonstration approach rotation")
    else:
        raise ValueError(f"Unsupported demonstration approach mode: {mode}")
    return value


def _nominal_transforms(prior, bounds_world, arm, ee_to_tcp,
                        object_pose_world=None, safe_tcp_rotation_world=None):
    source_arm = str(prior.get("arm", arm))
    if source_arm not in ("left", "right") or arm not in ("left", "right"):
        raise ValueError("Demonstration and runtime arms must be left or right")
    mode = prior["mode"]
    if mode == "object_6d":
        if object_pose_world is None:
            raise RuntimeError("Full demonstration prior requires a current visual object pose")
        ee = _rigid(object_pose_world, "current visual object pose") @ _rigid(
            prior["object_to_ee"], "object_to_ee")
        # An opposite-arm prior must be compiled explicitly in object
        # coordinates; mirroring an arbitrary object frame would be ambiguous.
        if source_arm != arm:
            raise RuntimeError("Full 6D demonstration prior is not compiled for the selected arm")
        return ee, ee @ ee_to_tcp

    bounds = np.asarray(bounds_world, float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[1] < bounds[0])):
        raise ValueError("Current visual bounds must be finite and ordered")
    delta = np.asarray(prior["interaction_to_approach_world_m"], float)
    if source_arm != arm:
        reflection = np.diag([-1., 1., 1.])
        delta = reflection @ delta
    anchor = (bounds[0] + bounds[1]) / 2.
    anchor[2] = bounds[1, 2]
    if safe_tcp_rotation_world is None:
        raise ValueError("Bounds-anchor demo transfer requires a safe TCP orientation template")
    rotation_check = np.eye(4)
    rotation_check[:3, :3] = np.asarray(safe_tcp_rotation_world, float)
    safe_rotation = _rigid(rotation_check, "safe TCP orientation template")[:3, :3]
    # The visual anchor is a point on the object surface and therefore has TCP
    # semantics.  Assigning it to EE and then applying ee_to_tcp introduced an
    # extra kinematic offset (12 cm on Aloha) in the previous implementation.
    tcp = np.eye(4)
    tcp[:3, :3] = safe_rotation
    tcp[:3, 3] = anchor + delta
    ee = tcp @ np.linalg.inv(ee_to_tcp)
    return ee, tcp


def generate_demo_relative_candidates(prior, bounds_world, arm, ee_to_tcp,
                                      *, object_pose_world=None, uncertainty_m,
                                      safe_tcp_rotation_world=None):
    """Return exact demonstration transfer followed by measured-error neighbours.

    The first candidate is always the unmodified transferred demonstration.
    Neighbours translate it in the plane normal to the demonstrated approach
    direction.  Their radius is the supplied visual uncertainty, rather than a
    task-specific distance.
    """
    ee_to_tcp = _rigid(ee_to_tcp, "ee_to_tcp")
    nominal_ee, nominal_tcp = _nominal_transforms(
        prior, bounds_world, str(arm), ee_to_tcp, object_pose_world,
        safe_tcp_rotation_world)
    uncertainty = float(uncertainty_m)
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        raise ValueError("Visual approach uncertainty must be finite and positive")

    if prior["mode"] == "visual_bounds_top_anchor":
        approach = -np.asarray(prior["interaction_to_approach_world_m"], float)
        if str(prior.get("arm", arm)) != str(arm):
            approach[0] *= -1.
    else:
        # Local EE +Z is a stable neighbourhood normal when the prior is a full
        # object-relative transform.  It does not alter the nominal candidate.
        approach = nominal_tcp[:3, 2]
    norm = float(np.linalg.norm(approach))
    approach = approach / norm if norm > 1e-9 else np.array([0., 0., -1.])
    seed = np.array([1., 0., 0.]) if abs(approach[0]) < .8 else np.array([0., 1., 0.])
    tangent_a = np.cross(approach, seed); tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(approach, tangent_a); tangent_b /= np.linalg.norm(tangent_b)
    offsets = [np.zeros(3), uncertainty*tangent_a, -uncertainty*tangent_a,
               uncertainty*tangent_b, -uncertainty*tangent_b]
    candidates = []
    for index, offset in enumerate(offsets):
        tcp = nominal_tcp.copy(); tcp[:3, 3] += offset
        ee = tcp @ np.linalg.inv(ee_to_tcp)
        candidates.append(dict(
            source="demo_exact" if index == 0 else "demo_visual_uncertainty_neighbour",
            demo_candidate_index=index, position_offset_world_m=offset.tolist(),
            yaw_offset_rad=0., approach_pitch_deg=0., ee_world=ee,
            tcp_world=tcp,
        ))
    audit = dict(
        source="compiled_demonstration_relative_approach",
        prior_mode=prior["mode"], prior_task=prior.get("task"),
        demonstration_arm=prior.get("arm"), runtime_arm=str(arm),
        exact_candidate_index=0, neighbourhood_radius_m=uncertainty,
        neighbourhood_radius_source="current_RGBD_point_spacing",
        demonstration_world_position_replayed=False,
        anchor_semantics="TCP_on_current_visual_object_surface",
        orientation_source=("full_object_relative_pose" if prior["mode"] == "object_6d"
                            else "validated_geometry_TCP_template"),
        stored_demo_absolute_rotation_applied=bool(prior["mode"] == "object_6d"),
        ee_tcp_offset_applied_exactly_once=True,
        uses_simulator_truth_online=False,
    )
    return candidates, audit


def prepend_demo_candidates(fallback_candidates, fallback_audit, prior, bounds_world,
                            arm, ee_to_tcp, *, object_pose_world=None, uncertainty_m):
    """Order candidates as demo exact -> demo neighbours -> geometry fallback."""
    if not fallback_candidates:
        raise ValueError("Demo-relative approach requires geometry fallback candidates")
    candidates, demo_audit = generate_demo_relative_candidates(
        prior, bounds_world, arm, ee_to_tcp, object_pose_world=object_pose_world,
        uncertainty_m=uncertainty_m,
        safe_tcp_rotation_world=fallback_candidates[0]["tcp_world"][:3, :3])
    # Reuse only orientations already validated by the generic geometry
    # template.  These variants keep the demonstrated TCP position and avoid
    # copying an absolute wrist pose from a different scene/arm configuration.
    orientation_variants = []
    nominal_tcp = candidates[0]["tcp_world"]
    for fallback in fallback_candidates[1:]:
        rotation = np.asarray(fallback["tcp_world"][:3, :3], float)
        if np.allclose(rotation, nominal_tcp[:3, :3], atol=1e-6):
            continue
        if any(np.allclose(rotation, item["tcp_world"][:3, :3], atol=1e-6)
               for item in orientation_variants):
            continue
        tcp = nominal_tcp.copy(); tcp[:3, :3] = rotation
        orientation_variants.append(dict(
            source="demo_safe_orientation_variant",
            demo_candidate_index=len(candidates)+len(orientation_variants),
            position_offset_world_m=[0., 0., 0.],
            yaw_offset_rad=float(fallback.get("yaw_offset_rad", 0.)),
            approach_pitch_deg=float(fallback.get("approach_pitch_deg", 0.)),
            tcp_world=tcp, ee_world=tcp @ np.linalg.inv(ee_to_tcp),
        ))
        if len(orientation_variants) == 2:
            break
    candidates.extend(orientation_variants)
    for index, candidate in enumerate(fallback_candidates):
        copied = dict(candidate)
        copied.setdefault("source", "geometry_bounds_fallback")
        copied["geometry_fallback_index"] = index
        copied.setdefault("approach_pitch_deg", 0.)
        candidates.append(copied)
    audit = dict(fallback_audit)
    audit.update(
        source="demo_relative_then_geometry_bounds_fallback",
        candidate_order=["demo_exact", "demo_visual_uncertainty_neighbours",
                         "demo_safe_orientation_variants",
                         "geometry_bounds_fallback"],
        demonstration=demo_audit,
        demo_candidate_count=len(candidates)-len(fallback_candidates),
        safe_orientation_variant_count=len(orientation_variants),
        geometry_fallback_count=len(fallback_candidates),
    )
    return candidates, audit
