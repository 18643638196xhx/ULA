import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('vision800audit',Path(__file__).resolve().parents[1]/'scripts/audit_vision_800_funnel.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_two_queries_cannot_reuse_one_reference_instance():
    assert not audit.can_assign(['a','b'],{('a','a'),('b','a')})


def test_equivalent_references_may_be_permuted():
    assert audit.can_assign(['a','b'],{('a','b'),('b','a')})


def test_mask_identity_requires_precision_and_minimum_overlap():
    t=np.zeros((20,20),bool);t[:10,:10]=True
    assert audit.identity(t,t)[0]
    assert not audit.identity(np.ones_like(t),t)[0]
    small=np.zeros_like(t);small[:2,:2]=True
    assert not audit.identity(small,t)[0]
