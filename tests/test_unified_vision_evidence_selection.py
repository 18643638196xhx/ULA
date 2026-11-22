from dataclasses import replace
import numpy as np
import pytest
from adapters.unified_vision import VisionConfig,TargetQuery,CadModel,UnifiedVision
from adapters.unified_vision.identity import semantic_evidence,identity_ranking,selection_cost
from adapters.unified_vision.cad_geometry import structure_depth_evidence,render_agreement
from test_unified_vision_cad import frame,candidate,accepted,Backend


def record(c,score=.5):
    return dict(mask=c,cad_verification=accepted(score),quality={})


def test_geometry_never_increases_identity_score():
    c=candidate();c.identity_evidence.update(semantic_evidence(c.mask,[(.8,c.box,'entity')]))
    r=record(c);v=identity_ranking(r);r['cad_verification']['cost']=0
    assert identity_ranking(r)==v
    assert not v['geometry_contributes'] and not v['identity_proven']


def test_missing_semantic_is_not_positive_identity():
    assert identity_ranking(record(candidate()))['score']==0


def test_box_background_is_penalized_without_reading_labels():
    a=candidate();b=candidate()
    a.identity_evidence.update(semantic_evidence(a.mask,[(.8,a.box,'entity')]))
    b.identity_evidence.update(semantic_evidence(b.mask,[(.8,[0,0,120,80],'entity')]))
    assert identity_ranking(record(a))['score']>identity_ranking(record(b))['score']


def test_selection_cost_opt_in_and_no_probability():
    r=record(candidate());r['identity_ranking']=dict(score=.8)
    assert selection_cost(r,VisionConfig())==.5
    assert np.isclose(selection_cost(r,VisionConfig(evidence_ranked_cad=True)),.7)


def test_local_empty_space_not_hidden_by_large_good_surface():
    render=np.ones((80,80));depth=render.copy();depth[:20,:20]=1.1
    assert render_agreement(depth,np.ones_like(depth,bool),render)['accepted']
    v=structure_depth_evidence(depth,render)
    assert not v['accepted'] and v['state']=='contradicted'


def test_occluded_or_missing_part_not_counted_as_support():
    render=np.ones((80,80));depth=render.copy();depth[:20,:20]=.8
    assert structure_depth_evidence(depth,render)['accepted']
    v=structure_depth_evidence(np.zeros_like(depth),render)
    assert v['state']=='unobservable' and not v['occlusion_or_missing_depth_is_support']


def test_single_target_finalists_saved():
    q=TargetQuery('x',('entity',),cad=CadModel('static.glb'))
    t=UnifiedVision(Backend([candidate()],lambda r:accepted()),
        VisionConfig(audit_candidate_selection=True)).observe([frame()],[q],episode_id='e').targets['x']
    assert t.assignment['final_competition']
    assert t.assignment['selected_candidate_ids']


def test_default_off():
    c=VisionConfig()
    assert not c.evidence_ranked_cad and not c.verify_scene_structure


def test_stronger_identity_can_outrank_slightly_better_geometry():
    weak=candidate(10,.2);strong=candidate(75,.9)
    for c in (weak,strong):
        c.identity_evidence.update(semantic_evidence(c.mask,[(c.score,c.box,'entity')]))
    q=TargetQuery('x',('entity',),cad=CadModel('static.glb'))
    backend=Backend([weak,strong],lambda r:accepted(.1 if r['mask'].box[0]==10 else .2))
    t=UnifiedVision(backend,VisionConfig(evidence_ranked_cad=True)).observe([frame()],[q],episode_id='e').targets['x']
    assert t.required_available
    assert t.views['head']['mask'].box[0]==75


@pytest.mark.parametrize('ranked',[False,True])
def test_final_audit_does_not_read_stale_preparation_options(ranked):
    class StalePreparation(UnifiedVision):
        def _prepare_query(self,*args,**kwargs):
            item=super()._prepare_query(*args,**kwargs)
            # Reproduce a preparation option invalidated before final choose.
            item['options'].append([dict(candidate_id='stale_rejected',
                cad_verification=dict(accepted=False,cost=None))])
            return item
    q=TargetQuery('x',('entity',),cad=CadModel('static.glb'))
    config=VisionConfig(audit_candidate_selection=True,evidence_ranked_cad=ranked)
    t=StalePreparation(Backend([candidate()],lambda r:accepted()),config).observe(
        [frame()],[q],episode_id='e').targets['x']
    assert t.required_available
    audit=t.assignment['final_competition']
    assert audit and all('stale_rejected' not in a['candidate_ids'] for a in audit)
    assert all(np.isfinite(a['selection_cost']) for a in audit)


def test_early_rejection_clears_previous_competition_audit():
    q=TargetQuery('x',('entity',),cad=CadModel('static.glb'))
    backend=Backend([candidate()],lambda r:accepted())
    engine=UnifiedVision(backend,VisionConfig(audit_candidate_selection=True))
    first=engine.observe([frame()],[q],episode_id='first').targets['x']
    assert first.assignment['final_competition']
    backend.verify=lambda r:dict(accepted=False,cost=None,object_to_world=None)
    second=engine.observe([frame()],[q],episode_id='second').targets['x']
    assert not second.required_available
    assert second.assignment['final_competition']==[]


def test_audit_flag_does_not_change_selection():
    q=TargetQuery('x',('entity',),cad=CadModel('static.glb'))
    def run(audit):
        return UnifiedVision(Backend([candidate()],lambda r:accepted()),
            VisionConfig(audit_candidate_selection=audit)).observe([frame()],[q],episode_id='e').targets['x']
    a,b=run(False),run(True)
    assert a.status==b.status and a.required_available==b.required_available
    np.testing.assert_array_equal(a.geometry['position_world'],b.geometry['position_world'])
