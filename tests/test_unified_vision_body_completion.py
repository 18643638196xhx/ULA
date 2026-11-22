import numpy as np
import pytest
from dataclasses import replace
from adapters.unified_vision import VisionConfig,TargetQuery,CadModel,UnifiedVision
from adapters.unified_vision.cad_geometry import complete_measured_mask
from test_unified_vision_cad import frame,candidate,accepted,Backend


def test_growth_keeps_thin_connected_part_not_disconnected_region():
    f=frame();m=np.zeros(f.depth.shape,bool);m[20:60,10:40]=True
    render=np.zeros(f.depth.shape);render[m]=1
    render[38:42,40:80]=1  # thin extension
    render[65:75,90:110]=1  # detached lookalike
    out,audit=complete_measured_mask(f.depth_m,m,render,frame=f)
    assert audit['accepted']
    assert out[39,75] and not out[68,100]
    assert audit['body_connected_measured_growth']


def test_nonmatching_depth_never_added():
    f=frame();m=candidate().mask;render=np.zeros(f.depth.shape);render[m]=1
    render[35:45,35:60]=.9
    out,_=complete_measured_mask(f.depth_m,m,render,frame=f)
    assert not out[40,50]


def test_engine_preserves_verified_thin_extension():
    f=frame();a=candidate();render=np.zeros(f.depth.shape);render[a.mask]=1
    render[38:42,35:80]=1
    class Complete(Backend):
        def verify_cad(self,record,model):
            record['cad_render_depth_m']=render
            return dict(accepted(),visible_completion=dict(accepted=True))
    q=TargetQuery('x',('entity',),cad=CadModel('static.glb'))
    target=UnifiedVision(Complete([a],None),VisionConfig(complete_visible_surface=True,
        body_guided_completion=True)).observe([f],[q],episode_id='e').targets['x']
    assert target.required_available
    assert target.views['head']['mask'].mask[39,75]
    assert target.evidence[0]['foreground_quality']['verified_body_completion_preserved']


def test_opt_in_requires_completion():
    assert not VisionConfig().body_guided_completion
    with pytest.raises(ValueError):VisionConfig(body_guided_completion=True)


def gap_scene(width=2,occluder=False):
    f=frame();m=np.zeros(f.depth.shape,bool);m[20:60,10:40]=True
    render=np.zeros(f.depth.shape);render[m]=1;render[38:42,40:70]=1
    depth=f.depth.copy();depth[38:42,40:40+width]=.7 if occluder else 0
    return replace(f,depth=depth),m,render


def test_small_missing_gap_associated_without_fabricating_points():
    f,m,r=gap_scene()
    old,_=complete_measured_mask(f.depth_m,m,r,frame=f)
    new,a=complete_measured_mask(f.depth_m,m,r,frame=f,max_gap_m=.02)
    assert not old[39,60] and new[39,60]
    assert not new[39,40] and not new[39,41]
    assert a['bounded_gap_recovered_pixels']>0


def test_large_gap_and_foreground_occluder_block_bridge():
    for width,occluder in ((6,False),(2,True)):
        f,m,r=gap_scene(width,occluder)
        new,_=complete_measured_mask(f.depth_m,m,r,frame=f,max_gap_m=.02)
        assert not new[39,60]


def test_cad_projection_gap_blocks_bridge():
    f,m,r=gap_scene();r[38:42,40:42]=0
    new,_=complete_measured_mask(f.depth_m,m,r,frame=f,max_gap_m=.02)
    assert not new[39,60]


def test_invalid_gap_budget_rejected():
    for v in (-.01,.06,float('nan')):
        with pytest.raises(ValueError):VisionConfig(completion_max_gap_m=v)
