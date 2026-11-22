import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_demo_relational_template_manifest",
    ROOT / "scripts/build_demo_relational_template_manifest.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(path, key):
    return {"path": str(path), "array": f"{key}_points_world"}


def test_collection_roles_share_family_bank_and_require_distinct_regions(tmp_path):
    source_file = tmp_path / "templates.npz"
    np.savez(
        source_file,
        object1_points_world=np.zeros((40, 3)),
        object2_points_world=np.ones((41, 3)),
        plasticbox_points_world=np.full((42, 3), 2.0),
    )
    source = {
        "schema_version": 1,
        "tasks": {
            "place_cans_plasticbox": {
                key: _record(source_file, key)
                for key in ("object1", "object2", "plasticbox")
            },
        },
    }

    manifest = MODULE.build_manifest(source, ["place_cans_plasticbox"])
    task = manifest["tasks"]["place_cans_plasticbox"]

    assert task["distinct_groups"] == [["object1", "object2"]]
    assert task["collection_groups"][0]["cardinality"] == 2
    assert task["collection_groups"][0]["interchangeable"] is True
    left = task["targets"]["object1"]["templates"]
    right = task["targets"]["object2"]["templates"]
    assert {item["model_key"] for item in left} == {
        "can_collection/object1",
        "can_collection/object2",
    }
    assert left == right
    assert len(task["targets"]["plasticbox"]["templates"]) == 1
    assert task["functional_regions"]["plasticbox"] == [
        "opening", "interior_volume", "slots"
    ]
    assert task["functional_region_inference"] == (
        "metadata_only_pending_RGBD_body_support"
    )


def test_missing_collection_role_is_rejected(tmp_path):
    source_file = tmp_path / "templates.npz"
    np.savez(source_file, bowl1_points_world=np.zeros((40, 3)))
    source = {
        "schema_version": 1,
        "tasks": {
            "stack_bowls_two": {
                "bowl1": _record(source_file, "bowl1"),
            },
        },
    }

    try:
        MODULE.build_manifest(source, ["stack_bowls_two"])
    except KeyError as error:
        assert "bowl2" in str(error)
    else:
        raise AssertionError("missing collection role must fail preflight")
