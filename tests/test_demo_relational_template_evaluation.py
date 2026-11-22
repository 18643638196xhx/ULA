import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_demo_relational_template_bank",
    ROOT / "scripts/evaluate_demo_relational_template_bank.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _hypothesis(x, cost, *, accepted=True, width=3):
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, x:x + width] = True
    return {
        "accepted": accepted,
        "selection_cost": cost,
        "mask": mask,
        "best": {"match": {"position_world": np.array([x, 0.0, 0.0])}},
    }


def test_permutations_of_same_selected_set_are_not_ambiguous():
    hypotheses = [
        _hypothesis(0, 0.10),
        _hypothesis(5, 0.12),
    ]
    selected, audit = MODULE._select_interchangeable_set(hypotheses, 2, 0.05)
    assert selected is not None
    assert len(selected) == 2
    assert audit["reason"] is None


def test_uncertain_collection_membership_is_still_rejected():
    hypotheses = [
        _hypothesis(0, 0.10, width=2),
        _hypothesis(3, 0.12, width=2),
        _hypothesis(7, 0.13, width=2),
    ]
    selected, audit = MODULE._select_interchangeable_set(hypotheses, 2, 0.05)
    assert selected is None
    assert audit["reason"] == "ambiguous_collection_membership"


def test_duplicate_masks_count_as_one_spatial_instance():
    hypotheses = [
        _hypothesis(0, 0.10),
        _hypothesis(0, 0.11),
        _hypothesis(5, 0.12),
    ]
    selected, audit = MODULE._select_interchangeable_set(hypotheses, 2, 0.01)
    assert selected is not None
    assert audit["observed_instance_count"] == 2


def test_container_opening_uses_demo_scaled_planar_region():
    reference = {
        "extent_m": np.array([0.24, 0.32, 0.52]),
    }
    small = _hypothesis(0, 0.0)["mask"]
    opening = np.zeros((10, 10), dtype=bool)
    opening[1:9, 1:9] = True
    archived = {
        "candidates": [{
            "candidates": [
                {
                    "source": "semantic_box_SAM",
                    "measured_mask_path": "unused-small.npz",
                },
                {
                    "source": "semantic_box_SAM",
                    "measured_mask_path": "unused-opening.npz",
                },
            ],
        }],
    }
    # Keep this test independent of filesystem candidate archives.
    old = MODULE.BASE._candidate_regions
    MODULE.BASE._candidate_regions = lambda _target: [
        {
            "path": Path("small.npz"), "mask": small,
            "points": np.array([[0, 0, 0], [0.05, 0.05, 0.01]] * 30),
            "candidate_id": "small", "source": "semantic_box_SAM",
            "semantic_score": 0.8,
        },
        {
            "path": Path("opening.npz"), "mask": opening,
            "points": np.array([[0, 0, 0], [0.30, 0.36, 0.01]] * 30),
            "candidate_id": "opening", "source": "semantic_box_SAM",
            "semantic_score": 0.7,
        },
    ]
    try:
        selected, audit = MODULE._container_opening_candidate(reference, archived, 0.01)
    finally:
        MODULE.BASE._candidate_regions = old
    assert selected is not None
    assert selected["candidate_id"] == "opening"
    assert audit["reason"] is None
