from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
from adapters.cad_prompted_sam_v1.prompts import cad_prompts
from adapters.cad_prompted_sam_v1.backend import CadPromptedSamBackend
from adapters.unified_vision.backends import DinoSamBackend
from adapters.unified_vision import VisionConfig
from test_unified_vision_cad import frame,candidate,accepted


def test_prompts_only_use_measured_supported_pixels():
    d=np.ones((60,80));r=np.zeros_like(d);r[10:50,20:60]=1
    d[20:30,30:40]=0;d[30:40,30:40]=.7
    kw,a=cad_prompts(d,r)
    assert a['accepted'] and not a['hidden_pixels_generated']
    for (x,y),label in zip(kw['point_coords'].astype(int),kw['point_labels']):
        if label:assert d[y,x]>0 and abs(d[y,x]-r[y,x])<=.008
        else:assert r[y,x]==0


def test_alternative_pose_region_never_becomes_negative():
    d=np.ones((60,80));r=np.zeros_like(d);r[10:50,20:60]=1
    union=r>0;union[:,60:70]=True
    kw,_=cad_prompts(d,r,union)
    for (x,y),label in zip(kw['point_coords'].astype(int),kw['point_labels']):
        if label==0:assert not union[y,x]


def test_missing_or_occluded_depth_cannot_seed_positive():
    r=np.ones((40,50))
    for d in (np.zeros_like(r),r*.7):
        kw,a=cad_prompts(d,r);assert kw is None and not a['accepted']


def test_bounded_point_budget():
    kw,a=cad_prompts(np.ones((50,50)),np.ones((50,50)),max_points=3)
    assert len(a['positive_points'])<=3 and len(a['negative_points'])<=3


def test_original_candidates_preserved_and_new_mask_not_preaccepted(monkeypatch):
    f=frame();original=candidate();new=original.mask.copy();new[20:60,35:45]=True
    monkeypatch.setattr(DinoSamBackend,'propose',lambda *a,**k:[original])
    b=object.__new__(CadPromptedSamBackend);b.config=VisionConfig();b.use_cad_prompts=True
    b.guidance_frame=None;b.guidance_cache={};b.image_key='old_crop'
    b.guidance_stats=dict(coarse_checks=0,guided_sam_calls=0,added_candidates=0,errors=[])
    def verify(record,model):
        rd=np.zeros(f.depth.shape);rd[new]=1;record['cad_render_depth_m']=rd;return accepted()
    b.verify_cad=verify
    sam=SimpleNamespace(set_image=lambda rgb:None,predict=lambda **kw:(new[None],np.array([.9]),None))
    torch=SimpleNamespace(inference_mode=nullcontext,autocast=lambda *a,**k:nullcontext(),bfloat16=None)
    b.model=SimpleNamespace(sam=sam,torch=torch)
    q=SimpleNamespace(cad=object(),hints=None,min_extent_m=None,max_extent_m=None)
    proposals=b.propose(f,q,refine=True)
    assert proposals[0] is original and len(proposals)==2
    assert proposals[1].identity_evidence['cad_prompt_audit']['new_mask_requires_fresh_CAD_verification']
    assert b.image_key is None
    assert b.propose(f,q,refine=True)[1] is proposals[1]
    assert b.guidance_stats['guided_sam_calls']==1


def test_no_CAD_or_baseline_does_not_run_extra_registration(monkeypatch):
    original=candidate();monkeypatch.setattr(DinoSamBackend,'propose',lambda *a,**k:[original])
    b=object.__new__(CadPromptedSamBackend);b.use_cad_prompts=False
    assert b.propose(frame(),SimpleNamespace(cad=object()),refine=True)==[original]
