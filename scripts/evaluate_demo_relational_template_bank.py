#!/usr/bin/env python3
"""Evaluate relational clean-demo templates with set-valued instance output.

This is an isolated companion to ``evaluate_demo_model_template_bank.py``.
The latter jointly assigns named roles but still treats a permutation of two
interchangeable roles as a competing answer.  For collections such as bowls,
cans, breads and bottles that is not a real task ambiguity: the *set* of
measured regions matters, not which arbitrary role number receives each one.

For every ``collection_group`` declared interchangeable in the schema-v2
manifest this adapter:

1. unions and deduplicates semantic RGB-D candidates from all role queries;
2. evaluates the shared clean-demo family bank only once;
3. clusters repeated SAM masks of the same spatial instance;
4. selects the required number of distinct instances as a set;
5. scores that set modulo role permutation.

Singleton targets still use the original evaluator unchanged.  No evaluation
label, simulator pose, CAD or runtime model id is used for prediction.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "demo_model_template_bank_base",
    ROOT / "scripts/evaluate_demo_model_template_bank.py",
)
BASE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(BASE)

_ORIGINAL_PREDICT = BASE._predict
_ORIGINAL_SCORE = BASE.OLD.SCORING.score_prediction
_ACTIVE_INTERCHANGEABLE: list[dict] = []


def _cluster_spatial_instances(hypotheses: list[dict]) -> list[dict]:
    """Keep the lowest-cost interpretation of each measured spatial region."""
    clusters: list[dict] = []
    for value in sorted(
        (item for item in hypotheses if item["accepted"]),
        key=lambda item: item["selection_cost"],
    ):
        matching = next(
            (index for index, old in enumerate(clusters)
             if BASE._same_region(value, old)),
            None,
        )
        if matching is None:
            clusters.append(value)
        elif value["selection_cost"] < clusters[matching]["selection_cost"]:
            clusters[matching] = value
    return sorted(clusters, key=lambda item: item["selection_cost"])


def _select_interchangeable_set(
    hypotheses: list[dict], cardinality: int, ambiguity_gap: float,
) -> tuple[list[dict] | None, dict]:
    instances = _cluster_spatial_instances(hypotheses)
    if len(instances) < cardinality:
        return None, {
            "reason": "too_few_distinct_family_consistent_instances",
            "required_cardinality": cardinality,
            "observed_instance_count": len(instances),
        }
    selected = instances[:cardinality]
    boundary_gap = None
    if len(instances) > cardinality:
        # The cheapest competing set replaces only the weakest selected member.
        boundary_gap = (
            instances[cardinality]["selection_cost"]
            - selected[-1]["selection_cost"]
        )
        if boundary_gap < ambiguity_gap:
            return None, {
                "reason": "ambiguous_collection_membership",
                "required_cardinality": cardinality,
                "observed_instance_count": len(instances),
                "boundary_gap": boundary_gap,
            }
    return selected, {
        "reason": None,
        "required_cardinality": cardinality,
        "observed_instance_count": len(instances),
        "boundary_gap": boundary_gap,
        "selected_set_cost": float(sum(v["selection_cost"] for v in selected)),
    }


def _merged_archived_target(archived: dict, roles: list[str]) -> dict:
    candidates = []
    for role in roles:
        target = archived.get("targets", {}).get(role)
        if target is None:
            raise KeyError(f"Target {role!r} absent from archived candidates")
        candidates.extend(target.get("candidates", []))
    return {"candidates": candidates}


def _shared_templates(banks: dict[str, list[dict]], roles: list[str]) -> list[dict]:
    unique = {}
    for role in roles:
        for template in banks[role]:
            unique.setdefault(template["model_key"], template)
    return list(unique.values())


def _robust_bounds(points: np.ndarray) -> np.ndarray:
    return np.quantile(np.asarray(points, dtype=float), (0.01, 0.99), axis=0)


def _container_opening_candidate(template: dict, archived_target: dict,
                                 ambiguity_gap: float) -> tuple[dict | None, dict]:
    """Find a measured opening/rim when a full container template is partial.

    The gates scale from the demonstrated container footprint.  They are not
    task names or fixed dustbin dimensions.  A conservative planar semantic
    mask is sufficient because the full visible-centre position is recovered
    from the demonstrated opening-to-centre offset.
    """
    reference_extent = np.asarray(template["extent_m"], dtype=float)
    reference_planar = np.sort(reference_extent[:2])
    evaluated = []
    for candidate in BASE._candidate_regions(archived_target):
        bounds = _robust_bounds(candidate["points"])
        extent = np.maximum(bounds[1] - bounds[0], 1e-6)
        planar = np.sort(extent[:2])
        ratio = planar / np.maximum(reference_planar, 1e-6)
        planarity = float(extent[2] / max(planar[-1], 1e-6))
        accepted = bool(
            ratio.min() >= 0.55
            and ratio.max() <= 2.0
            and planarity <= 0.20
        )
        semantic = candidate.get("semantic_score")
        semantic_penalty = 1.0 - float(np.clip(
            0.0 if semantic is None else semantic, 0.0, 1.0
        ))
        cost = float(
            np.mean(np.abs(np.log(np.maximum(ratio, 1e-6))))
            + 0.5 * planarity
            + 0.1 * semantic_penalty
        )
        opening = (bounds[0] + bounds[1]) / 2.0
        evaluated.append({
            **candidate,
            "accepted": accepted,
            "bounds_world": bounds,
            "extent_m": extent,
            "planar_ratio_to_demo": ratio,
            "planarity": planarity,
            "selection_cost": cost,
            "opening_center_world": opening,
        })
    accepted = sorted(
        (item for item in evaluated if item["accepted"]),
        key=lambda item: item["selection_cost"],
    )
    def compact(values):
        return [{
            "candidate_id": item["candidate_id"],
            "measured_mask_path": str(item["path"]),
            "source": item["source"],
            "semantic_score": item.get("semantic_score"),
            "accepted": item["accepted"],
            "extent_m": item["extent_m"],
            "planar_ratio_to_demo": item["planar_ratio_to_demo"],
            "planarity": item["planarity"],
            "selection_cost": item["selection_cost"],
            "opening_center_world": item["opening_center_world"],
        } for item in values]
    if not accepted:
        return None, {
            "reason": "no_demo_scaled_planar_opening_candidate",
            "candidates": compact(evaluated),
        }
    selected = accepted[0]
    competing_gap = None
    for other in accepted[1:]:
        # Alternative masks on the same opening (solid top, rim, eroded rim)
        # are repeated explanations, not different container identities.
        distance = np.linalg.norm(
            selected["opening_center_world"][:2]
            - other["opening_center_world"][:2]
        )
        if distance <= 0.25 * np.linalg.norm(reference_planar):
            continue
        competing_gap = other["selection_cost"] - selected["selection_cost"]
        if competing_gap < ambiguity_gap:
            return None, {
                "reason": "ambiguous_demo_scaled_container_openings",
                "competing_gap": competing_gap,
                "candidates": compact(evaluated),
            }
        break
    return selected, {
        "reason": None,
        "competing_gap": competing_gap,
        "candidates": compact(evaluated),
    }


def _opening_output(target: str, selected: dict, template: dict) -> dict:
    opening = np.asarray(selected["opening_center_world"], dtype=float)
    extent = np.asarray(template["extent_m"], dtype=float)
    # In the clean template the upper visible surface is approximately half a
    # demonstrated visible height above its robust centre.  Reusing that
    # relative relation recovers a whole-object centre for legacy consumers,
    # while control receives the actually measured opening geometry.
    visible_centre = opening.copy()
    visible_centre[2] -= 0.5 * extent[2]
    return {
        "key": target,
        "status": "ok",
        "required_available": True,
        "fresh": True,
        "geometry": {
            "position_world": visible_centre,
            "visible_surface_position_world": visible_centre,
            "position_reference": "demo_opening_to_visible_centre_relation",
            "opening_center_world": opening,
            "opening_bounds_world": selected["bounds_world"],
            "opening_extent_xy_m": selected["extent_m"][:2],
            "opening_normal_world": [0.0, 0.0, 1.0],
            "functional_regions": ["opening", "interior_volume", "rim"],
        },
        "usable": {"position": True, "bounds": True},
        "evidence": [{
            "measured_mask_path": str(selected["path"]),
            "candidate_id": selected["candidate_id"],
            "source": selected["source"],
            "functional_region": "container_opening",
        }],
        "reasons": [],
    }


def _relational_predict(task_spec: dict, banks: dict[str, list[dict]], archived: dict,
                        ambiguity_gap: float, semantic_weight: float,
                        completeness_weight: float) -> tuple[dict, dict]:
    global _ACTIVE_INTERCHANGEABLE
    collections = [
        group for group in task_spec.get("collection_groups", [])
        if group.get("interchangeable")
    ]
    _ACTIVE_INTERCHANGEABLE = [dict(group) for group in collections]
    opening_fallback_enabled = (
        task_spec.get("functional_region_inference")
        == "demo_scaled_opening_with_RGBD_body_support"
    )
    has_opening_target = opening_fallback_enabled and any(
        "opening" in regions
        for regions in task_spec.get("functional_regions", {}).values()
    )
    if not collections and not has_opening_target:
        return _ORIGINAL_PREDICT(
            task_spec, banks, archived, ambiguity_gap,
            semantic_weight, completeness_weight,
        )

    started = time.perf_counter()
    selections: dict[str, dict | None] = {}
    reasons: dict[str, str | None] = {}
    direct_outputs: dict[str, dict] = {}
    target_audit: dict[str, dict] = {}
    assignment_audit: list[dict] = []
    grouped: set[str] = set()

    for collection in collections:
        roles = list(collection["roles"])
        cardinality = int(collection.get("cardinality", len(roles)))
        if cardinality != len(roles):
            raise ValueError("Current relational output requires one slot per role")
        grouped.update(roles)
        merged = _merged_archived_target(archived, roles)
        hypotheses = BASE._candidate_hypotheses(
            _shared_templates(banks, roles), merged,
            semantic_weight, completeness_weight,
        )
        selected_set, group_audit = _select_interchangeable_set(
            hypotheses, cardinality, ambiguity_gap,
        )
        assignment_audit.append({
            "roles": roles,
            "family": collection.get("family"),
            "selection_semantics": "unordered_distinct_instance_set",
            **group_audit,
        })
        if selected_set is None:
            for role in roles:
                selections[role] = None
                reasons[role] = group_audit["reason"]
        else:
            # Role numbering is deliberately arbitrary for an interchangeable
            # family.  Spatial sorting makes output deterministic; evaluation
            # and downstream collection primitives are permutation invariant.
            selected_set = sorted(
                selected_set,
                key=lambda item: tuple(item["best"]["match"]["position_world"]),
            )
            for role, selected in zip(roles, selected_set):
                selections[role] = selected
                reasons[role] = None
        compact = [BASE._compact_hypothesis(item) for item in hypotheses]
        for index, role in enumerate(roles):
            target_audit[role] = {
                "template_count": len(banks[role]),
                "candidate_region_count": len(hypotheses),
                "accepted_region_count": sum(item["accepted"] for item in hypotheses),
                "selected_candidate_id": None if selections[role] is None
                                         else selections[role]["candidate_id"],
                "selected_model_key": None if selections[role] is None
                                      else selections[role]["best"]["model_key"],
                "reason": reasons[role],
                "shared_collection_hypotheses": roles[0],
                # Avoid multiplying a potentially large audit by cardinality.
                "candidates": compact if index == 0 else [],
            }

    for target, templates in banks.items():
        if target in grouped:
            continue
        if target not in archived.get("targets", {}):
            raise KeyError(f"Target {target!r} absent from archived candidates")
        hypotheses = BASE._candidate_hypotheses(
            templates, archived["targets"][target],
            semantic_weight, completeness_weight,
        )
        selections[target], selection_audit = BASE._single_select(
            hypotheses, ambiguity_gap,
        )
        reasons[target] = selection_audit["reason"]
        regions = task_spec.get("functional_regions", {}).get(target, [])
        opening_audit = None
        if (opening_fallback_enabled and selections[target] is None
                and "opening" in regions):
            opening, opening_audit = _container_opening_candidate(
                templates[0], archived["targets"][target], ambiguity_gap,
            )
            if opening is not None:
                direct_outputs[target] = _opening_output(target, opening, templates[0])
                reasons[target] = None
        assignment_audit.append({"roles": [target], **selection_audit})
        target_audit[target] = {
            "template_count": len(templates),
            "candidate_region_count": len(hypotheses),
            "accepted_region_count": sum(item["accepted"] for item in hypotheses),
            "selected_candidate_id": None if selections[target] is None
                                     else selections[target]["candidate_id"],
            "selected_model_key": None if selections[target] is None
                                  else selections[target]["best"]["model_key"],
            "reason": reasons[target],
            "functional_opening_fallback": opening_audit,
            "candidates": [BASE._compact_hypothesis(item) for item in hypotheses],
        }

    outputs = {
        target: direct_outputs.get(target) or BASE._target_output(
            target, selections[target], reasons[target]
        )
        for target in banks
    }
    return outputs, {
        "timing_s": {"relational_prediction": time.perf_counter() - started},
        "assignments": assignment_audit,
        "targets": target_audit,
    }


def _relational_score(targets, truth, frame, groups):
    covered = {
        role for collection in _ACTIVE_INTERCHANGEABLE
        for role in collection["roles"]
    }
    rewritten = {}
    for key, members in groups.items():
        remaining = [member for member in members if member not in covered]
        if remaining:
            rewritten[key] = remaining
    for index, collection in enumerate(_ACTIVE_INTERCHANGEABLE):
        roles = [role for role in collection["roles"] if role in targets and role in truth]
        if roles:
            rewritten[f"relational_collection_{index}"] = roles
    return _ORIGINAL_SCORE(targets, truth, frame, rewritten)


BASE._predict = _relational_predict
BASE.OLD.SCORING.score_prediction = _relational_score


if __name__ == "__main__":
    BASE.main()
