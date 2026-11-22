import numpy as np
import pytest
from adapters.fixed_candidate_selection_v1 import projection_visibility,frustum_cost


def projection(x):
    vertices=np.array([[-.2,-.2,0],[.2,-.2,0],[.2,.2,0],[-.2,.2,0]])
    pose=np.eye(4);pose[:3,3]=[x,0,1]
    K=np.array([[100,0,50],[0,100,50],[0,0,1]])
    return projection_visibility(vertices,pose,K,np.eye(4),(100,100))


def test_full_projection_no_penalty():
    v=projection(0);assert np.isclose(v['fraction'],1)
    assert np.isclose(frustum_cost(.4,v),.4)


def test_partial_projection_soft_penalty_not_rejection():
    v=projection(.5);assert 0<v['fraction']<1
    assert np.isfinite(frustum_cost(.4,v)) and frustum_cost(.4,v)>.4


def test_offscreen_projection_and_zero_weight():
    v=projection(3);assert v['fraction']==0
    assert frustum_cost(.4,v)==2.4
    assert frustum_cost(.4,v,weight=0)==.4


def test_weight_fixed_for_all_objects_and_no_identity_proof():
    v=projection(.5);assert not v['is_identity_evidence']
    assert np.isclose(frustum_cost(.7,v)-.7,frustum_cost(.2,v)-.2)


@pytest.mark.parametrize('fraction',[-.1,1.1,float('nan')])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):frustum_cost(.4,dict(fraction=fraction))


def test_runtime_adapter_does_not_override_rejection(monkeypatch):
    from adapters.frustum_selection_v1 import FrustumCadBackend
    from adapters.unified_vision.backends import DinoSamBackend
    original=dict(accepted=False,cost=None)
    monkeypatch.setattr(DinoSamBackend,'verify_cad',lambda *args:original)
    backend=object.__new__(FrustumCadBackend)
    assert backend.verify_cad({},None) is original


def test_runtime_cost_preserves_original_result(monkeypatch):
    from types import SimpleNamespace
    from adapters.frustum_selection_v1 import FrustumCadBackend
    from adapters.unified_vision.backends import DinoSamBackend
    import adapters.frustum_selection_v1 as module
    original=dict(accepted=True,cost=.4,object_to_world=np.eye(4))
    monkeypatch.setattr(DinoSamBackend,'verify_cad',lambda *args:original)
    monkeypatch.setattr(module,'projection_visibility',lambda *args:dict(fraction=.5))
    backend=object.__new__(FrustumCadBackend);spec=SimpleNamespace()
    backend.frustum_vertices={repr(spec):np.zeros((4,3))}
    frame=SimpleNamespace(intrinsic=np.eye(3),world_to_camera=np.eye(4),depth=np.ones((10,10)))
    out=backend.verify_cad(dict(frame=frame),spec)
    assert out['accepted'] and np.isclose(out['cost'],1.4)
    assert out['geometric_cost']==.4 and original['cost']==.4
