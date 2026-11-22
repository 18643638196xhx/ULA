#!/usr/bin/env python3
"""Upgrade clean-demo templates to a relational, multi-instance manifest.

The original schema-v1 manifest stores one independently evaluated template
for every semantic role.  That is the wrong inference model for collections:
three bottles (or bowls) are three *distinct instances* drawn from one visual
family, not three unrelated single-object queries which may all select the
same measured mask.

This isolated adapter keeps every measured clean RGB-D template unchanged and
only changes their organization:

* roles in a collection share the union of the demonstrated family templates;
* collection roles are assigned jointly and may not reuse one image region;
* task-level relation and functional-region metadata is carried forward for
  the later rule/geometry layer.

No CAD, simulator pose, runtime model id or evaluation mask is added.  The
output is schema version 2 and can be consumed by
``evaluate_demo_model_template_bank.py``.  Existing manifests are never
modified or overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


# The roles are derived from the concrete targets already present in the
# clean-demo manifest.  Relations describe task semantics; they do not expose
# randomized-scene object identity to inference.
RELATIONAL_TASKS: dict[str, dict[str, Any]] = {
    "dump_bin_bigbin": {
        "collections": [],
        "relations": [
            {"type": "contents_into", "source": "deskbin", "target": "dustbin"},
        ],
        "functional_regions": {
            "dustbin": ["opening", "interior_volume", "rim"],
        },
    },
    "put_bottles_dustbin": {
        "collections": [
            {
                "family": "bottle_collection",
                "roles": ["bottles_0", "bottles_1", "bottles_2"],
                "interchangeable": True,
            },
        ],
        "relations": [
            {"type": "inside", "source": "bottles.*", "target": "dustbin"},
        ],
        "functional_regions": {
            "dustbin": ["opening", "interior_volume", "rim"],
        },
    },
    "stack_bowls_three": {
        "collections": [
            {
                "family": "bowl_collection",
                "roles": ["bowl1", "bowl2", "bowl3"],
                "interchangeable": True,
            },
        ],
        "relations": [
            {
                "type": "stack_chain",
                "members": "bowl*",
                "order_source": "demonstrated_manipulation_events",
            },
        ],
        "functional_regions": {},
    },
    "stack_bowls_two": {
        "collections": [
            {
                "family": "bowl_collection",
                "roles": ["bowl1", "bowl2"],
                "interchangeable": True,
            },
        ],
        "relations": [
            {
                "type": "stack_chain",
                "members": "bowl*",
                "order_source": "demonstrated_manipulation_events",
            },
        ],
        "functional_regions": {},
    },
    "place_bread_basket": {
        "collections": [
            {
                "family": "bread_collection",
                "roles": ["bread_0", "bread_1"],
                "interchangeable": True,
            },
        ],
        "relations": [
            {"type": "inside", "source": "bread_*", "target": "breadbasket"},
        ],
        "functional_regions": {
            "breadbasket": ["opening", "interior_volume", "rim"],
        },
    },
    "place_cans_plasticbox": {
        "collections": [
            {
                "family": "can_collection",
                "roles": ["object1", "object2"],
                "interchangeable": True,
            },
        ],
        "relations": [
            {
                "type": "assigned_to_functional_slots",
                "source": "object*",
                "target": "plasticbox",
            },
        ],
        "functional_regions": {
            "plasticbox": ["opening", "interior_volume", "slots"],
        },
    },
    "hanging_mug": {
        "collections": [],
        "relations": [
            {"type": "hang", "source": "mug", "target": "rack"},
        ],
        "functional_regions": {
            "mug": ["handle"],
            "rack": ["hook", "support_axis"],
        },
    },
    "open_microwave": {
        "collections": [],
        "relations": [
            {
                "type": "articulated_open",
                "source": "microwave.handle",
                "target": "microwave.door_joint",
            },
        ],
        "functional_regions": {
            "microwave": ["body", "door", "handle", "hinge_axis"],
        },
    },
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _template(target: str, record: dict[str, Any], family: str) -> dict[str, Any]:
    source = _resolve(record["path"])
    array = record["array"]
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as values:
        if array not in values:
            raise KeyError(f"{source} has no array {array!r}")
        points = np.asarray(values[array])
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 40:
        raise ValueError(f"Invalid measured point cloud {target}:{source}#{array}")
    return {
        "model_key": f"{family}/{target}",
        "model_id": target,
        "family": family,
        "demonstrated_role": target,
        "path": str(source),
        "array": array,
        "point_count": int(len(points)),
    }


def build_manifest(source: dict[str, Any], selected: list[str]) -> dict[str, Any]:
    if source.get("schema_version") != 1 or not isinstance(source.get("tasks"), dict):
        raise ValueError("Expected the schema-v1 clean-demo template manifest")
    unknown = sorted(set(selected) - set(RELATIONAL_TASKS))
    if unknown:
        raise KeyError(f"Unsupported relational tasks: {unknown}")

    output: dict[str, Any] = {
        "schema_version": 2,
        "kind": "clean_RGBD_relational_multi_instance_template_bank",
        "source": "schema_v1_clean_demo_measured_pointcloud_templates",
        "uses_CAD": False,
        "uses_runtime_model_id": False,
        "uses_simulator_pose_for_inference": False,
        "runtime_instance_assignment": "joint_distinct_measured_regions",
        "tasks": {},
    }
    for task in selected:
        old_targets = source["tasks"].get(task)
        if not isinstance(old_targets, dict) or not old_targets:
            raise KeyError(f"Task {task!r} is absent from source manifest")
        specification = RELATIONAL_TASKS[task]
        targets: dict[str, Any] = {}
        grouped: set[str] = set()
        distinct_groups: list[list[str]] = []
        collection_groups: list[dict[str, Any]] = []

        for collection in specification["collections"]:
            roles = list(collection["roles"])
            missing = [role for role in roles if role not in old_targets]
            if missing:
                raise KeyError(f"{task} collection has missing roles: {missing}")
            family = collection["family"]
            shared_bank = [_template(role, old_targets[role], family) for role in roles]
            for role in roles:
                # Copy dictionaries so downstream audit augmentation for one
                # role can never mutate another role's bank.
                targets[role] = {
                    "family": family,
                    "templates": [dict(item) for item in shared_bank],
                }
            grouped.update(roles)
            distinct_groups.append(roles)
            collection_groups.append({
                "family": family,
                "roles": roles,
                "cardinality": len(roles),
                "interchangeable": bool(collection.get("interchangeable", False)),
                "selection": "joint_without_region_reuse",
            })

        for target, record in old_targets.items():
            if target in grouped:
                continue
            family = f"{task}/{target}"
            targets[target] = {
                "family": family,
                "templates": [_template(target, record, family)],
            }

        output["tasks"][task] = {
            "targets": targets,
            "distinct_groups": distinct_groups,
            "role_orders": [],
            "role_partitions": [],
            "collection_groups": collection_groups,
            "relations": specification["relations"],
            "functional_regions": specification["functional_regions"],
            # Region names are task semantics only.  Never activate a planar
            # opening merely because it has a compatible size: the randomized
            # frame may contain a same-sized table patch.  A later fresh RGB-D
            # stage must first establish supporting container-body evidence.
            "functional_region_inference": "metadata_only_pending_RGBD_body_support",
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="schema-v1 clean-demo template manifest")
    parser.add_argument("--output", type=Path, required=True,
                        help="new schema-v2 relational manifest")
    parser.add_argument("--tasks", nargs="+", choices=tuple(RELATIONAL_TASKS),
                        default=list(RELATIONAL_TASKS))
    args = parser.parse_args()
    source_path = _resolve(args.source)
    output_path = _resolve(args.output)
    if not source_path.is_file():
        parser.error(f"source does not exist: {source_path}")
    if output_path == ROOT or not output_path.is_relative_to(ROOT):
        parser.error("--output must be a new JSON file under the ULA workspace")
    if output_path.exists():
        parser.error("refusing to overwrite existing output")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("duplicate task")
    manifest = build_manifest(json.loads(source_path.read_text()), list(args.tasks))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
