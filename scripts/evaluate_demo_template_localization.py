#!/usr/bin/env python3
"""Evaluate clean-demo point-cloud templates on archived randomized RGB-D.

This is an intentionally isolated experiment adapter.  It does not import a
CAD mesh, a CAD score, a CAD pose, simulator object poses, or renderer masks
during inference.  It reuses only the saved semantic DINO/SAM candidate masks
and their measured RGB-D point clouds, then replaces CAD selection with a
clean-demo point-cloud registration score.

The official ``demo_clean`` HDF5 files contain RGB and camera calibration but
not depth.  A template must therefore come from a synchronized clean RGB-D
capture (or an already saved clean perception ``estimate.npz``).  Renderer
labels are opened only after a prediction has been written, for offline
scoring, and never feed back into selection.

Example using the existing clean cup smoke capture::

  python scripts/evaluate_demo_template_localization.py evaluate \
    --task place_empty_cup --episodes 50 \
    --template cup=results/place_empty_cup_adaptive_smoke_v2_20261113/episode_000_seed0/perception/estimate.npz#cup_points_world \
    --template coaster=results/place_empty_cup_adaptive_smoke_v2_20261113/episode_000_seed0/perception/estimate.npz#coaster_points_world \
    --output results/demo_template_place_empty_cup_50
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.pointcloud_registration import (  # noqa: E402
    RegistrationConfig,
    estimate_pointcloud_motion,
)
from adapters.unified_vision import RGBDFrame  # noqa: E402
from adapters.unified_vision.contracts import serializable  # noqa: E402


_SCORING_SPEC = importlib.util.spec_from_file_location(
    "demo_template_scoring", ROOT / "scripts/replay_vision_evidence_pair.py"
)
SCORING = importlib.util.module_from_spec(_SCORING_SPEC)
assert _SCORING_SPEC.loader is not None
_SCORING_SPEC.loader.exec_module(SCORING)

DEFAULT_SOURCE = ROOT / "results/frustum_vision50_2500_20261112_v1"
ALLOWED_CANDIDATE_SOURCES = {
    "semantic_box_SAM",
    "measured_foreground_ROI_multimask_SAM",
}


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


def _parse_template(value: str) -> tuple[str, Path, str]:
    """Parse TARGET=FILE#ARRAY without making ':' in paths special."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("template must be TARGET=FILE#ARRAY")
    target, source = value.split("=", 1)
    path_text, separator, array = source.partition("#")
    target = target.strip()
    if not target or not path_text:
        raise argparse.ArgumentTypeError("template target and file are required")
    return target, _resolve(path_text), array if separator else "points_world"


def _voxelize(points: np.ndarray, voxel_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Template/candidate point cloud must be Nx3")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 40:
        raise ValueError("Point cloud has fewer than 40 finite points")
    _, indices = np.unique(
        np.floor(points / voxel_m).astype(np.int64), axis=0, return_index=True
    )
    return points[np.sort(indices)]


def _robust_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.quantile(points, (0.01, 0.99), axis=0))


def _extent(points: np.ndarray) -> np.ndarray:
    lower, upper = _robust_bounds(points)
    return np.maximum(upper - lower, 1e-6)


def _template_record(target: str, path: Path, array: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Template file does not exist: {path}")
    with np.load(path, allow_pickle=False) as values:
        if array not in values:
            raise KeyError(f"{path} has no point array {array!r}; keys={values.files}")
        raw = np.asarray(values[array], dtype=float)
    preliminary = _voxelize(raw, 0.002)
    extent = _extent(preliminary)
    diameter = float(np.linalg.norm(extent))
    if not 0.01 <= diameter <= 1.5:
        raise ValueError(f"Implausible template diameter {diameter:.4f} m for {target}")
    voxel = float(np.clip(diameter / 60.0, 0.0015, 0.004))
    points = _voxelize(raw, voxel)
    centre = np.median(points, axis=0)
    return {
        "target": target,
        "path": path,
        "array": array,
        "sha256": _digest(path),
        "points": points - centre,
        "source_centre_world": centre,
        "extent_m": _extent(points),
        "diameter_m": diameter,
        "voxel_m": voxel,
        "point_count": len(points),
    }


def _registration_config(template: dict) -> RegistrationConfig:
    """Scale all geometric gates from the demonstration, not a task name."""
    diameter = template["diameter_m"]
    voxel = template["voxel_m"]
    return RegistrationConfig(
        voxel_m=voxel,
        correspondence_m=float(np.clip(0.25 * diameter, 0.018, 0.06)),
        inlier_m=float(np.clip(0.10 * diameter, 0.007, 0.02)),
        max_rmse_m=float(np.clip(0.075 * diameter, 0.006, 0.015)),
        min_observation_overlap=0.35,
        min_reference_overlap=0.25,
        max_tilt_deg=30.0,
        iterations=45,
        min_points=40,
        yaw_starts_rad=tuple(np.linspace(-np.pi, np.pi, 16, endpoint=False)),
    )


def _shape_penalty(reference_extent: np.ndarray, candidate_extent: np.ndarray) -> float:
    """Yaw-invariant XY extent plus explicit vertical extent disagreement."""
    reference = np.r_[np.sort(reference_extent[:2]), reference_extent[2]]
    candidate = np.r_[np.sort(candidate_extent[:2]), candidate_extent[2]]
    return float(np.mean(np.abs(np.log(np.maximum(candidate, 1e-6) /
                                           np.maximum(reference, 1e-6)))))


def _match(template: dict, candidate_points: np.ndarray) -> dict:
    config = _registration_config(template)
    try:
        points = _voxelize(candidate_points, config.voxel_m)
    except ValueError as error:
        return {"accepted": False, "reason": str(error)}
    candidate_extent = _extent(points)
    size_penalty = _shape_penalty(template["extent_m"], candidate_extent)
    # A category instance may differ in scale, but a many-fold discrepancy is
    # not a plausible instance of the same demonstrated object.
    if size_penalty > 0.75:
        return {
            "accepted": False,
            "reason": "demo_candidate_extent_mismatch",
            "shape_penalty": size_penalty,
            "candidate_extent_m": candidate_extent,
        }
    motion, audit = estimate_pointcloud_motion(template["points"], points, config)
    # Correct the generic same-episode registration provenance: this adapter
    # deliberately registers a different episode's clean demonstration.
    audit.update(
        reference_source="separate_clean_demo_measured_pointcloud_template",
        cross_episode_template=True,
        uses_prepared_object_model=False,
        uses_CAD=False,
    )
    if motion is None:
        return {
            "accepted": False,
            "reason": audit.get("failure_reason", "demo_registration_failed"),
            "shape_penalty": size_penalty,
            "candidate_extent_m": candidate_extent,
            "registration": audit,
        }
    attempt = audit["attempts"][audit["selected_attempt"]]
    normalized_rmse = attempt["trimmed_rmse_m"] / template["diameter_m"]
    cost = (
        normalized_rmse
        + 0.30 * (1.0 - attempt["observation_overlap"])
        + 0.20 * (1.0 - attempt["reference_overlap"])
        + 0.25 * size_penalty
    )
    lower, upper = _robust_bounds(points)
    return {
        "accepted": True,
        "cost": float(cost),
        "shape_penalty": size_penalty,
        "candidate_extent_m": candidate_extent,
        "position_world": ((lower + upper) / 2.0),
        "bounds_world": np.stack((lower, upper)),
        "demo_template_to_current_world": motion,
        "registration": audit,
    }


def _candidate_records(target: dict) -> list[dict]:
    """Load only semantic-mask RGB-D evidence; never read CAD verification."""
    candidates = []
    seen_paths = set()
    for group in target.get("candidates", []):
        for summary in group.get("candidates", []):
            if summary.get("source") not in ALLOWED_CANDIDATE_SOURCES:
                continue
            path_text = summary.get("measured_mask_path")
            if not path_text:
                continue
            path = Path(path_text).resolve()
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)
            with np.load(path, allow_pickle=False) as values:
                mask = values["mask"].astype(bool)
                points = np.asarray(values["points_world"], dtype=float)
            candidates.append({
                "path": path,
                "mask": mask,
                "points": points,
                "candidate_id": summary.get("candidate_id", path.stem),
                "source": summary.get("source"),
                "prompt": summary.get("prompt"),
                "semantic_score": summary.get("semantic_score"),
            })
    # Remove repeated SAM interpretations of the same measured region before
    # applying an ambiguity gate.
    unique = []
    for candidate in candidates:
        if any(
            np.logical_and(candidate["mask"], old["mask"]).sum()
            / max(1, np.logical_or(candidate["mask"], old["mask"]).sum()) > 0.90
            for old in unique
        ):
            continue
        unique.append(candidate)
    return unique


def _prediction(template: dict, target: dict, ambiguity_gap: float) -> tuple[dict, dict]:
    evaluated = []
    for candidate in _candidate_records(target):
        result = _match(template, candidate["points"])
        evaluated.append({**candidate, "match": result})
    accepted = sorted(
        (value for value in evaluated if value["match"]["accepted"]),
        key=lambda value: value["match"]["cost"],
    )
    reason = None
    selected = accepted[0] if accepted else None
    if selected is None:
        reason = "no_demo_template_consistent_semantic_candidate"
    elif len(accepted) > 1:
        gap = accepted[1]["match"]["cost"] - accepted[0]["match"]["cost"]
        overlap = np.logical_and(accepted[0]["mask"], accepted[1]["mask"]).sum() / max(
            1, np.logical_or(accepted[0]["mask"], accepted[1]["mask"]).sum()
        )
        if gap < ambiguity_gap and overlap < 0.50:
            selected = None
            reason = "ambiguous_demo_template_consistent_candidates"
    compact = []
    for value in evaluated:
        compact.append({
            "candidate_id": value["candidate_id"],
            "measured_mask_path": str(value["path"]),
            "source": value["source"],
            "prompt": value["prompt"],
            "semantic_score": value["semantic_score"],
            "match": value["match"],
        })
    audit = {
        "target": template["target"],
        "candidate_count": len(evaluated),
        "accepted_count": len(accepted),
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "reason": reason,
        "candidates": compact,
    }
    if selected is None:
        status = "ambiguous" if reason and reason.startswith("ambiguous") else "not_observed"
        return {
            "key": template["target"],
            "status": status,
            "required_available": False,
            "fresh": True,
            "geometry": {},
            "usable": {"position": False, "bounds": False},
            "evidence": [],
            "reasons": [reason],
        }, audit
    match = selected["match"]
    return {
        "key": template["target"],
        "status": "ok",
        "required_available": True,
        "fresh": True,
        "geometry": {
            "position_world": match["position_world"],
            "bounds_world": match["bounds_world"],
            "demo_template_to_current_world": match["demo_template_to_current_world"],
            "position_reference": "measured_candidate_visible_surface_bounds_centre",
            "pose_kind": "relative_to_clean_demo_template_not_semantic_CAD_frame",
        },
        "usable": {"position": True, "bounds": True},
        "evidence": [{
            "measured_mask_path": str(selected["path"]),
            "candidate_id": selected["candidate_id"],
            "source": selected["source"],
        }],
        "reasons": [],
    }, audit


def _find_prediction(row: dict, source: Path) -> Path:
    explicit = row.get("prediction_directory")
    if explicit and (Path(explicit) / "prediction.json").is_file():
        return Path(explicit) / "prediction.json"
    # Source reports can move as a directory.  Resolve the request index from
    # the row order only as a documented fallback.
    raise FileNotFoundError(
        "Source row does not reference an available prediction directory; "
        f"rerun scripts/replay_frustum_vision50.py or use its output: {source}"
    )


def evaluate(args) -> None:
    output = args.output.resolve()
    source = args.source.resolve()
    if output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError("--output must be a new directory under the ULA workspace")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    report_path = source / "tasks" / args.task / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"No archived task report: {report_path}")
    parsed = [_parse_template(value) for value in args.template]
    if len({item[0] for item in parsed}) != len(parsed):
        raise ValueError("Each target may have only one template")
    templates = {key: _template_record(key, path, array) for key, path, array in parsed}
    source_report = SCORING.read(report_path)
    rows = [row for row in source_report["rows"] if row.get("evaluation")][:args.episodes]
    if not rows:
        raise ValueError("No evaluated archived RGB-D rows found")
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "scope": "fixed_semantic_DINO_SAM_candidates_clean_demo_pointcloud_selection_only",
        "task": args.task,
        "planned": len(rows),
        "source": str(source),
        "robot_actions": 0,
        "uses_CAD_for_selection": False,
        "uses_CAD_scores": False,
        "uses_CAD_poses": False,
        "uses_simulator_labels_for_inference": False,
        "candidate_sources": sorted(ALLOWED_CANDIDATE_SOURCES),
        "ambiguity_gap": args.ambiguity_gap,
        "templates": {
            key: {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in value.items() if k != "points"
            }
            for key, value in templates.items()
        },
        "rows": [],
        "limitations": [
            "This tests template selection on frozen semantic DINO/SAM candidates, not detector recall from a fresh run.",
            "The official demo_clean HDF5 stores RGB but no depth; template points require a synchronized clean RGB-D sidecar.",
            "A returned rotation is relative to the clean demonstration template, not a globally resolved semantic CAD frame.",
        ],
    }

    def save() -> None:
        per_target = {}
        for key in templates:
            values = [row["evaluation"][key] for row in report["rows"] if key in row["evaluation"]]
            per_target[key] = {
                "scenes": len(values),
                "localized": sum(bool(v["localization_verified"]) for v in values),
                "outputs": sum(bool(v["output"]) for v in values),
                "identity_matches": sum(bool(v["identity_match"]) for v in values),
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
    input_hashes = {str(report_path): _digest(report_path)}
    input_hashes.update({str(value["path"]): value["sha256"] for value in templates.values()})
    try:
        for index, row in enumerate(rows):
            started = time.perf_counter()
            case = Path(row["directory"])
            sensor = case / "rgbd.npz"
            label_file = case / "evaluation_only_masks.npz"
            provenance = case / "known_model_provenance.json"
            prediction_file = _find_prediction(row, source)
            for path in (sensor, prediction_file, label_file, provenance):
                if not path.is_file():
                    raise FileNotFoundError(path)
            input_hashes[str(sensor)] = _digest(sensor)
            input_hashes[str(prediction_file)] = _digest(prediction_file)
            frame = RGBDFrame.from_npz(sensor, camera="head", capture_id="initial", depth_unit="mm")
            archived = SCORING.read(prediction_file)
            targets = {}
            audits = {}
            for key, template in templates.items():
                if key not in archived["targets"]:
                    raise KeyError(f"Target {key!r} absent from archived candidates")
                targets[key], audits[key] = _prediction(
                    template, archived["targets"][key], args.ambiguity_gap
                )
            case_output = output / f"case_{index:04d}_seed{row['seed']}"
            case_output.mkdir()
            prediction = {
                "task": args.task,
                "seed": row["seed"],
                "targets": targets,
                "audit": audits,
                "inference_inputs": {
                    "sensor": str(sensor),
                    "frozen_semantic_candidates": str(prediction_file),
                    "templates": {key: str(value["path"]) for key, value in templates.items()},
                },
            }
            output_prediction = case_output / "prediction.json"
            _dump(output_prediction, prediction)
            prediction_hash = _digest(output_prediction)

            # Evaluation-only truth is opened for the first time after the
            # prediction is persisted and locked.  It cannot affect selection.
            with np.load(label_file, allow_pickle=False) as values:
                truth = {key: values[key].copy() for key in values.files}
            groups = {}
            for item in SCORING.read(provenance):
                if item["key"] in targets:
                    groups.setdefault(item["equivalence_group"], []).append(item["key"])
            metrics = SCORING.score_prediction(targets, truth, frame, groups)
            if _digest(output_prediction) != prediction_hash:
                raise RuntimeError("Prediction changed after evaluation")
            report["rows"].append({
                "seed": row["seed"],
                "directory": str(case),
                "prediction": str(output_prediction),
                "prediction_sha256": prediction_hash,
                "evaluation": metrics,
                "all_requested_targets_localized": all(
                    value["localization_verified"] for value in metrics.values()
                ),
                "inference_wall_s": time.perf_counter() - started,
                "labels_feed_back_to_inference": False,
            })
            save()
            current = report["summary"]["all_requested_targets_localized"]
            print(args.task, row["seed"], f"{current}/{len(report['rows'])}", flush=True)
        report["status"] = "finished"
        report["inputs_unchanged"] = all(_digest(Path(path)) == digest
                                           for path, digest in input_hashes.items())
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _dump(output / "input_hashes.json", input_hashes)
        save()


def inspect_demo(args) -> None:
    """Make the missing-depth limitation explicit before a long experiment."""
    import h5py

    path = args.demo_hdf5.resolve()
    with h5py.File(path, "r") as handle:
        datasets = []
        handle.visititems(lambda name, value: datasets.append(name)
                          if isinstance(value, h5py.Dataset) else None)
    depth = [name for name in datasets if "depth" in name.lower()]
    value = {
        "demo_hdf5": str(path),
        "datasets": datasets,
        "depth_datasets": depth,
        "can_build_metric_pointcloud_directly": bool(depth),
        "required_sidecar_if_false": "synchronized depth or pre-segmented points_world NPZ",
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("evaluate", help="run the fixed-candidate localization experiment")
    run.add_argument("--task", required=True)
    run.add_argument("--template", action="append", required=True,
                     help="TARGET=FILE#POINT_ARRAY; repeat for object/support")
    run.add_argument("--episodes", type=int, default=50)
    run.add_argument("--ambiguity-gap", type=float, default=0.03)
    run.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    run.add_argument("--output", type=Path, required=True)
    inspect = subparsers.add_parser("inspect-demo", help="report whether a demo HDF5 contains depth")
    inspect.add_argument("--demo-hdf5", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        if not 1 <= args.episodes <= 50:
            parser.error("--episodes must be in [1, 50]")
        if not 0 <= args.ambiguity_gap <= 1:
            parser.error("--ambiguity-gap must be in [0, 1]")
        args.source = _resolve(args.source)
        args.output = _resolve(args.output)
        evaluate(args)
    else:
        args.demo_hdf5 = _resolve(args.demo_hdf5)
        inspect_demo(args)


if __name__ == "__main__":
    main()
