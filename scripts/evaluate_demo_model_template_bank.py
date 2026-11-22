#!/usr/bin/env python3
"""Evaluate a multi-model clean-demo point-cloud template bank.

The input manifest is produced by ``build_demo_model_template_bank.py``.  For
each frozen semantic DINO/SAM candidate, this adapter evaluates every model
template in the target's family and first collapses them to the best model for
that *same measured region*.  Therefore model ambiguity is not mistaken for
object-instance ambiguity.

Targets in a ``distinct_group`` are assigned jointly and may not reuse the
same image region.  Ordered roles (for example left bottle then right bottle)
are resolved from measured world coordinates rather than simulator object
identities.  Evaluation-only renderer masks are opened only after prediction
has been saved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_OLD_SPEC = importlib.util.spec_from_file_location(
    "single_demo_template_evaluator",
    ROOT / "scripts/evaluate_demo_template_localization.py",
)
OLD = importlib.util.module_from_spec(_OLD_SPEC)
assert _OLD_SPEC.loader is not None
_OLD_SPEC.loader.exec_module(OLD)

from adapters.unified_vision import RGBDFrame  # noqa: E402
from adapters.unified_vision.contracts import serializable  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(serializable(value), ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _load_manifest(path: Path, task: str) -> tuple[dict, dict[str, list[dict]]]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Expected schema_version 2 model-template-bank manifest")
    task_spec = manifest.get("tasks", {}).get(task)
    if not task_spec:
        raise KeyError(f"Task {task!r} is absent from {path}")
    banks = {}
    for target, target_spec in task_spec.get("targets", {}).items():
        records = []
        seen_models = set()
        for item in target_spec.get("templates", []):
            model_key = item["model_key"]
            if model_key in seen_models:
                raise ValueError(f"Duplicate model template {target}:{model_key}")
            seen_models.add(model_key)
            source = _resolve(item["path"])
            template = OLD._template_record(target, source, item["array"])
            template.update(
                model_key=model_key,
                model_id=item.get("model_id"),
                family=item.get("family", target_spec.get("family")),
                demonstrated_role=item.get("demonstrated_role"),
            )
            records.append(template)
        if not records:
            raise ValueError(f"Empty template bank for {task}:{target}")
        banks[target] = records
    return task_spec, banks


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.logical_and(left, right).sum() /
                 max(1, np.logical_or(left, right).sum()))


def _pixel_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.where(mask)
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def _interval_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, max(a0, b0) - min(a1, b1))


def _overlap_fraction(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0) + 1.0)
    return overlap / max(1.0, min(a1 - a0 + 1.0, b1 - b0 + 1.0))


def _adjacent_fragments(left: dict, right: dict) -> bool:
    """Detect split masks using image adjacency plus metric continuity."""
    if _iou(left["mask"], right["mask"]) > 0.05:
        return False
    ax0, ay0, ax1, ay1 = _pixel_box(left["mask"])
    bx0, by0, bx1, by1 = _pixel_box(right["mask"])
    x_gap = _interval_gap(ax0, ax1, bx0, bx1)
    y_gap = _interval_gap(ay0, ay1, by0, by1)
    image_continuous = (
        x_gap <= 6 and _overlap_fraction(ay0, ay1, by0, by1) >= 0.25
    ) or (
        y_gap <= 6 and _overlap_fraction(ax0, ax1, bx0, bx1) >= 0.25
    )
    if not image_continuous:
        return False
    left_bounds = np.quantile(left["points"], (0.01, 0.99), axis=0)
    right_bounds = np.quantile(right["points"], (0.01, 0.99), axis=0)
    gaps = np.asarray([
        _interval_gap(left_bounds[0, i], left_bounds[1, i],
                      right_bounds[0, i], right_bounds[1, i])
        for i in range(3)
    ])
    # Fragmented surfaces may be separated along one dimension, but should
    # overlap in the other two and remain within 3 cm in metric space.
    return bool(gaps.max() <= 0.03 and np.count_nonzero(gaps <= 0.003) >= 2)


def _candidate_regions(archived_target: dict) -> list[dict]:
    originals = OLD._candidate_records(archived_target)
    merged = []
    for index, left in enumerate(originals):
        for right in originals[index + 1:]:
            if not _adjacent_fragments(left, right):
                continue
            semantic_scores = [
                value for value in (left["semantic_score"], right["semantic_score"])
                if value is not None
            ]
            merged.append({
                "path": left["path"],
                "component_paths": [left["path"], right["path"]],
                "mask": np.logical_or(left["mask"], right["mask"]),
                "points": np.concatenate((left["points"], right["points"]), axis=0),
                "candidate_id": f"merge({left['candidate_id']}+{right['candidate_id']})",
                "source": "adjacent_RGBD_fragment_union",
                "prompt": left["prompt"] or right["prompt"],
                "semantic_score": max(semantic_scores) if semantic_scores else None,
            })
    return originals + merged


def _candidate_hypotheses(templates: list[dict], archived_target: dict,
                          semantic_weight: float,
                          completeness_weight: float) -> list[dict]:
    """Return one spatial hypothesis per mask, after collapsing model matches."""
    hypotheses = []
    _cand_times = []
    _match_times = []
    for candidate in _candidate_regions(archived_target):
        _c0 = time.perf_counter()
        model_matches = []
        for template in templates:
            _m0 = time.perf_counter()
            match = OLD._match(template, candidate["points"])
            _match_times.append(time.perf_counter() - _m0)
            model_matches.append({
                "model_key": template["model_key"],
                "model_id": template["model_id"],
                "family": template["family"],
                "demonstrated_role": template["demonstrated_role"],
                "match": match,
            })
        accepted = sorted(
            (item for item in model_matches if item["match"].get("accepted")),
            key=lambda item: item["match"]["cost"],
        )
        best = accepted[0] if accepted else None
        _cand_times.append(time.perf_counter() - _c0)
        semantic_score = candidate.get("semantic_score")
        semantic_penalty = 1.0 - float(np.clip(
            0.0 if semantic_score is None else semantic_score, 0.0, 1.0
        ))
        reference_overlap = None
        completeness_penalty = None
        if best is not None:
            registration = best["match"]["registration"]
            attempt = registration["attempts"][registration["selected_attempt"]]
            reference_overlap = float(attempt["reference_overlap"])
            completeness_penalty = 1.0 - reference_overlap
        selection_cost = None if best is None else float(
            best["match"]["cost"]
            + semantic_weight * semantic_penalty
            + completeness_weight * completeness_penalty
        )
        model_gap = None
        if len(accepted) > 1:
            model_gap = accepted[1]["match"]["cost"] - accepted[0]["match"]["cost"]
        hypotheses.append({
            **candidate,
            "accepted": best is not None,
            "best": best,
            "selection_cost": selection_cost,
            "semantic_penalty": semantic_penalty,
            "reference_overlap": reference_overlap,
            "completeness_penalty": completeness_penalty,
            "model_gap": model_gap,
            "accepted_model_count": len(accepted),
            "model_matches": model_matches,
        })
    if hypotheses:
        hypotheses[-1]["timing_s"] = {
            "candidates": len(_cand_times),
            "match_calls": len(_match_times),
            "candidate_total_s": float(sum(_cand_times)),
            "match_total_s": float(sum(_match_times)),
            "match_mean_ms": float(1000 * sum(_match_times) / max(1, len(_match_times))),
        }
    return hypotheses


def _same_region(left: dict, right: dict) -> bool:
    # Candidate IDs differ between target queries, so the measured masks are
    # the authoritative cross-target identity.
    return _iou(left["mask"], right["mask"]) > 0.50


def _spatially_different(left: tuple[dict, ...], right: tuple[dict, ...]) -> bool:
    return any(not _same_region(a, b) for a, b in zip(left, right))


def _order_valid(assignment: dict[str, dict], role_orders: list[dict],
                 role_partitions: list[dict]) -> bool:
    axes = {"x": 0, "y": 1, "z": 2}
    for relation in role_orders:
        roles = relation["roles"]
        if not all(role in assignment for role in roles):
            continue
        axis = axes[relation["axis"]]
        values = [assignment[role]["best"]["match"]["position_world"][axis]
                  for role in roles]
        if relation.get("ascending", True):
            if any(left > right for left, right in zip(values, values[1:])):
                return False
        elif any(left < right for left, right in zip(values, values[1:])):
            return False
    for partition in role_partitions:
        roles = partition["roles"]
        if not all(role in assignment for role in roles):
            continue
        axis = axes[partition["axis"]]
        split = partition["split_world"]
        for role, side in zip(roles, partition["sides"]):
            value = assignment[role]["best"]["match"]["position_world"][axis]
            if side == "lower" and not value < split:
                return False
            if side == "upper" and not value > split:
                return False
    return True


def _joint_select(
    roles: list[str],
    hypotheses: dict[str, list[dict]],
    role_orders: list[dict],
    role_partitions: list[dict],
    ambiguity_gap: float,
) -> tuple[dict[str, dict] | None, dict]:
    options = [
        [item for item in hypotheses[role] if item["accepted"]]
        for role in roles
    ]
    if any(not values for values in options):
        return None, {
            "reason": "one_or_more_roles_have_no_model_consistent_candidate",
            "valid_assignment_count": 0,
        }
    valid = []
    for combination in itertools.product(*options):
        if any(_same_region(combination[i], combination[j])
               for i in range(len(combination))
               for j in range(i + 1, len(combination))):
            continue
        assignment = dict(zip(roles, combination))
        if not _order_valid(assignment, role_orders, role_partitions):
            continue
        valid.append((
            float(sum(item["selection_cost"] for item in combination)),
            combination,
        ))
    valid.sort(key=lambda item: item[0])
    if not valid:
        return None, {
            "reason": "no_distinct_order_consistent_joint_assignment",
            "valid_assignment_count": 0,
        }
    selected_cost, selected = valid[0]
    competing_gap = None
    for other_cost, other in valid[1:]:
        if _spatially_different(selected, other):
            competing_gap = other_cost - selected_cost
            if competing_gap < ambiguity_gap:
                return None, {
                    "reason": "ambiguous_distinct_joint_assignments",
                    "valid_assignment_count": len(valid),
                    "best_cost": selected_cost,
                    "competing_gap": competing_gap,
                }
            break
    return dict(zip(roles, selected)), {
        "reason": None,
        "valid_assignment_count": len(valid),
        "best_cost": selected_cost,
        "competing_gap": competing_gap,
    }


def _single_select(values: list[dict], ambiguity_gap: float) -> tuple[dict | None, dict]:
    accepted = sorted(
        (value for value in values if value["accepted"]),
        key=lambda value: value["selection_cost"],
    )
    if not accepted:
        return None, {"reason": "no_model_consistent_semantic_candidate"}
    selected = accepted[0]
    spatial_gap = None
    for other in accepted[1:]:
        if _same_region(selected, other):
            continue
        spatial_gap = other["selection_cost"] - selected["selection_cost"]
        if spatial_gap < ambiguity_gap:
            return None, {
                "reason": "ambiguous_model_bank_consistent_regions",
                "spatial_gap": spatial_gap,
            }
        break
    return selected, {"reason": None, "spatial_gap": spatial_gap}


def _target_output(target: str, selected: dict | None, reason: str | None) -> dict:
    if selected is None:
        status = "ambiguous" if reason and reason.startswith("ambiguous") else "not_observed"
        return {
            "key": target,
            "status": status,
            "required_available": False,
            "fresh": True,
            "geometry": {},
            "usable": {"position": False, "bounds": False},
            "evidence": [],
            "reasons": [reason],
        }
    model = selected["best"]
    match = model["match"]
    return {
        "key": target,
        "status": "ok",
        "required_available": True,
        "fresh": True,
        "geometry": {
            "position_world": match["position_world"],
            "bounds_world": match["bounds_world"],
            "demo_template_to_current_world": match["demo_template_to_current_world"],
            "matched_model_key": model["model_key"],
            "matched_model_id": model["model_id"],
            "position_reference": "measured_candidate_visible_surface_bounds_centre",
            "pose_kind": "relative_to_selected_clean_demo_model_template",
        },
        "usable": {"position": True, "bounds": True},
        "evidence": [{
            "measured_mask_path": str(selected["path"]),
            "candidate_id": selected["candidate_id"],
            "source": selected["source"],
            "component_paths": [str(path) for path in selected.get(
                "component_paths", [selected["path"]]
            )],
            "matched_model_key": model["model_key"],
            "model_match_cost": match["cost"],
            "selection_cost": selected["selection_cost"],
            "reference_overlap": selected["reference_overlap"],
            "accepted_model_count_for_same_region": selected["accepted_model_count"],
            "model_cost_gap": selected["model_gap"],
        }],
        "reasons": [],
    }


def _compact_hypothesis(value: dict) -> dict:
    return {
        "candidate_id": value["candidate_id"],
        "measured_mask_path": str(value["path"]),
        "source": value["source"],
        "component_paths": [str(path) for path in value.get(
            "component_paths", [value["path"]]
        )],
        "prompt": value["prompt"],
        "semantic_score": value["semantic_score"],
        "accepted": value["accepted"],
        "selected_model": None if value["best"] is None else value["best"]["model_key"],
        "selected_model_cost": None if value["best"] is None else value["best"]["match"]["cost"],
        "selection_cost": value["selection_cost"],
        "semantic_penalty": value["semantic_penalty"],
        "reference_overlap": value["reference_overlap"],
        "completeness_penalty": value["completeness_penalty"],
        "accepted_model_count": value["accepted_model_count"],
        "model_gap": value["model_gap"],
        "model_matches": value["model_matches"],
    }


def _predict(task_spec: dict, banks: dict[str, list[dict]], archived: dict,
             ambiguity_gap: float, semantic_weight: float,
             completeness_weight: float) -> tuple[dict, dict]:
    _p0 = time.perf_counter()
    hypotheses = {}
    _per_target = {}
    for target, templates in banks.items():
        if target not in archived.get("targets", {}):
            raise KeyError(f"Target {target!r} absent from archived candidates")
        _q0 = time.perf_counter()
        hypotheses[target] = _candidate_hypotheses(
            templates, archived["targets"][target], semantic_weight,
            completeness_weight,
        )
        _per_target[target] = time.perf_counter() - _q0
    _p1 = time.perf_counter()

    selections: dict[str, dict | None] = {}
    reasons: dict[str, str | None] = {}
    assignment_audit = []
    grouped = set()
    role_orders = task_spec.get("role_orders", [])
    role_partitions = task_spec.get("role_partitions", [])
    for roles in task_spec.get("distinct_groups", []):
        grouped.update(roles)
        selected, audit = _joint_select(
            roles, hypotheses, role_orders, role_partitions, ambiguity_gap
        )
        assignment_audit.append({"roles": roles, **audit})
        for role in roles:
            selections[role] = None if selected is None else selected[role]
            reasons[role] = audit["reason"]
    for target in banks:
        if target in grouped:
            continue
        selections[target], audit = _single_select(hypotheses[target], ambiguity_gap)
        reasons[target] = audit["reason"]
        assignment_audit.append({"roles": [target], **audit})

    outputs = {
        target: _target_output(target, selections[target], reasons[target])
        for target in banks
    }
    _p2 = time.perf_counter()
    audit = {
        "timing_s": {
            "candidate_hypotheses": _p1 - _p0,
            "hypotheses_per_target": _per_target,
            "selection": _p2 - _p1,
        },
        "assignments": assignment_audit,
        "targets": {
            target: {
                "template_count": len(banks[target]),
                "candidate_region_count": len(hypotheses[target]),
                "accepted_region_count": sum(item["accepted"] for item in hypotheses[target]),
                "selected_candidate_id": None if selections[target] is None
                                         else selections[target]["candidate_id"],
                "selected_model_key": None if selections[target] is None
                                      else selections[target]["best"]["model_key"],
                "reason": reasons[target],
                "candidates": [_compact_hypothesis(item)
                               for item in hypotheses[target]],
            }
            for target in banks
        },
    }
    return outputs, audit


def evaluate(args) -> None:
    output = args.output
    if output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError("--output must be a new directory under ULA")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    task_spec, banks = _load_manifest(args.manifest, args.task)
    source_report_path = args.source / "tasks" / args.task / "report.json"
    if not source_report_path.is_file():
        raise FileNotFoundError(source_report_path)
    source_report = OLD.SCORING.read(source_report_path)
    rows = [row for row in source_report["rows"] if row.get("evaluation")][:args.episodes]
    if not rows:
        raise ValueError("No evaluated archived RGB-D rows found")

    output.mkdir(parents=True)
    input_hashes = {
        str(args.manifest): _digest(args.manifest),
        str(source_report_path): _digest(source_report_path),
    }
    for templates in banks.values():
        for template in templates:
            input_hashes[str(template["path"])] = template["sha256"]
    report = {
        "status": "running",
        "task": args.task,
        "planned": len(rows),
        "source": str(args.source),
        "manifest": str(args.manifest),
        "robot_actions": 0,
        "uses_CAD_for_selection": False,
        "uses_runtime_model_id": False,
        "uses_simulator_labels_for_inference": False,
        "candidate_sources": sorted(OLD.ALLOWED_CANDIDATE_SOURCES),
        "ambiguity_gap": args.ambiguity_gap,
        "template_counts": {target: len(values) for target, values in banks.items()},
        "distinct_groups": task_spec.get("distinct_groups", []),
        "role_orders": task_spec.get("role_orders", []),
        "role_partitions": task_spec.get("role_partitions", []),
        "semantic_weight": args.semantic_weight,
        "completeness_weight": args.completeness_weight,
        "rows": [],
        "limitations": [
            "This evaluates model-bank selection on frozen semantic DINO/SAM candidates, not fresh detector recall.",
            "Each model template is a single clean head-camera visible-surface point cloud.",
            "A matched rotation is relative to that model's clean demonstration view, not a CAD semantic frame.",
        ],
    }

    def save() -> None:
        per_target = {}
        for target in banks:
            values = [row["evaluation"][target] for row in report["rows"]]
            per_target[target] = {
                "scenes": len(values),
                "localized": sum(bool(value["localization_verified"]) for value in values),
                "outputs": sum(bool(value["output"]) for value in values),
                "identity_matches": sum(bool(value["identity_match"]) for value in values),
            }
        report["summary"] = {
            "evaluated_scenes": len(report["rows"]),
            "all_requested_targets_localized": sum(
                all(value["localization_verified"] for value in row["evaluation"].values())
                for row in report["rows"]
            ),
            "per_target": per_target,
        }
        _dump(output / "report.json", report)

    save()
    try:
        for index, row in enumerate(rows):
            started = time.perf_counter()
            case = Path(row["directory"])
            sensor = case / "rgbd.npz"
            labels = case / "evaluation_only_masks.npz"
            provenance = case / "known_model_provenance.json"
            prediction_file = OLD._find_prediction(row, args.source)
            for path in (sensor, labels, provenance, prediction_file):
                if not path.is_file():
                    raise FileNotFoundError(path)
            input_hashes[str(sensor)] = _digest(sensor)
            input_hashes[str(prediction_file)] = _digest(prediction_file)
            _t0 = time.perf_counter()
            frame = RGBDFrame.from_npz(
                sensor, camera="head", capture_id="initial", depth_unit="mm"
            )
            _t1 = time.perf_counter()
            archived = OLD.SCORING.read(prediction_file)
            _t2 = time.perf_counter()
            targets, audit = _predict(
                task_spec, banks, archived, args.ambiguity_gap,
                args.semantic_weight, args.completeness_weight,
            )
            _t3 = time.perf_counter()
            case_output = output / f"case_{index:04d}_seed{row['seed']}"
            case_output.mkdir()
            prediction = {
                "task": args.task,
                "seed": row["seed"],
                "targets": targets,
                "audit": audit,
                "inference_inputs": {
                    "sensor": str(sensor),
                    "frozen_semantic_candidates": str(prediction_file),
                    "model_template_bank_manifest": str(args.manifest),
                },
            }
            output_prediction = case_output / "prediction.json"
            _dump(output_prediction, prediction)
            locked_hash = _digest(output_prediction)

            # Truth is deliberately opened only after the prediction is saved.
            with np.load(labels, allow_pickle=False) as values:
                truth = {key: values[key].copy() for key in values.files}
            groups = {}
            for item in OLD.SCORING.read(provenance):
                if item["key"] in targets:
                    groups.setdefault(item["equivalence_group"], []).append(item["key"])
            _t4 = time.perf_counter()
            metrics = OLD.SCORING.score_prediction(targets, truth, frame, groups)
            _t5 = time.perf_counter()
            if _digest(output_prediction) != locked_hash:
                raise RuntimeError("Prediction changed after evaluation")
            report["rows"].append({
                "seed": row["seed"],
                "directory": str(case),
                "prediction": str(output_prediction),
                "prediction_sha256": locked_hash,
                "evaluation": metrics,
                "all_requested_targets_localized": all(
                    value["localization_verified"] for value in metrics.values()
                ),
                "inference_wall_s": time.perf_counter() - started,
                # Split so I/O and hashing are not mistaken for inference cost.
                "timing_s": {
                    "frame_load": _t1 - _t0,
                    "archived_prediction_read": _t2 - _t1,
                    "predict": _t3 - _t2,
                    "write_and_hash": _t4 - _t3,
                    "scoring": _t5 - _t4,
                },
                "labels_feed_back_to_inference": False,
            })
            save()
            current = report["summary"]["all_requested_targets_localized"]
            print(args.task, row["seed"], f"{current}/{len(report['rows'])}", flush=True)
        report["status"] = "finished"
        report["inputs_unchanged"] = all(
            _digest(Path(path)) == digest for path, digest in input_hashes.items()
        )
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _dump(output / "input_hashes.json", input_hashes)
        save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", type=Path,
                        default=OLD.DEFAULT_SOURCE)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--ambiguity-gap", type=float, default=0.005)
    parser.add_argument("--semantic-weight", type=float, default=0.10)
    parser.add_argument("--completeness-weight", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.manifest = _resolve(args.manifest)
    args.source = _resolve(args.source)
    args.output = _resolve(args.output)
    if not args.manifest.is_file():
        parser.error("manifest does not exist")
    if not 1 <= args.episodes <= 50:
        parser.error("episodes must be in [1, 50]")
    if not 0 <= args.ambiguity_gap <= 1:
        parser.error("ambiguity gap must be in [0, 1]")
    if not 0 <= args.semantic_weight <= 1:
        parser.error("semantic weight must be in [0, 1]")
    if not 0 <= args.completeness_weight <= 1:
        parser.error("completeness weight must be in [0, 1]")
    evaluate(args)


if __name__ == "__main__":
    main()
