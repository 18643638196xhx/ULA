"""Explicit position semantics, conservative identity and measured completion."""
from dataclasses import asdict,replace
import json
import unittest
import numpy as np

from adapters.unified_vision import CadModel,TargetQuery,VisionConfig,UnifiedVision
from adapters.unified_vision.contracts import MaskCandidate
from adapters.unified_vision.identity import semantic_evidence
from adapters.unified_vision.cad_geometry import complete_measured_mask
from adapters.unified_vision.assignment import same_instance
from diagnostics.unified_vision_evaluation import evaluate_target,visible_reference
from test_unified_vision_cad import frame,candidate,accepted,Backend
from adapters.unified_vision.backends import DinoSamBackend


class PositionTests(unittest.TestCase):
    def test_explicit_cad_position_never_overwrites_visible_reference(self):
        q=TargetQuery('x',('entity',),cad=CadModel('static.glb'),position_reference='cad_origin')
        value=UnifiedVision(Backend([candidate()],lambda r:accepted(x=.3))).observe([frame()],[q],episode_id='e').targets['x']
        np.testing.assert_allclose(value.geometry['position_world'],[.3,0,1])
        self.assertTrue(value.geometry['position_is_model_inferred'])
        self.assertFalse(value.geometry['hidden_support_anchor_observed'])
        self.assertFalse(np.allclose(value.geometry['visible_surface_position_world'],[.3,0,1]))
        json.dumps(value.to_dict(),allow_nan=False)

    def test_cad_anchor_is_rotated_in_model_coordinates(self):
        pose=np.eye(4);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pose[:3,3]=[.1,.2,1.]
        q=TargetQuery('x',('entity',),cad=CadModel('static.glb'),position_reference='cad_anchor',anchor_object_m=(.1,0,0))
        result=UnifiedVision(Backend([candidate()],lambda r:dict(accepted=True,cost=.2,object_to_world=pose)))
        value=result.observe([frame()],[q],episode_id='e').targets['x']
        np.testing.assert_allclose(value.geometry['position_world'],[.1,.3,1.])
        self.assertEqual(TargetQuery(**asdict(q)),q)

    def test_invalid_reference_combinations_rejected(self):
        for kwargs in (dict(position_reference='cad_origin'),dict(position_reference='bad'),
                       dict(cad=CadModel('x'),position_reference='cad_anchor'),
                       dict(anchor_object_m=(0,0,0)),
                       dict(cad=CadModel('x'),position_reference='cad_anchor',anchor_object_m=(0,np.nan,0))):
            with self.assertRaises(ValueError):TargetQuery('x',('entity',),**kwargs)

    def test_old_metric_scores_visible_position_not_inferred_anchor(self):
        f=frame();m=candidate().mask;reference,_=visible_reference(f,m)
        target=dict(required_available=True,fresh=True,usable=dict(position=True),
            geometry=dict(position_world=[999,999,999],position_is_model_inferred=True,
                          visible_surface_position_world=reference.tolist()))
        score=evaluate_target(target,m,m,f)
        self.assertTrue(score['localization_verified'])
        self.assertFalse(score['semantic_CAD_origin_or_hidden_bottom_error'])
        target['geometry'].pop('visible_surface_position_world')
        self.assertFalse(evaluate_target(target,m,m,f)['localization_verified'])


class IdentityTests(unittest.TestCase):
    def test_recovery_changes_proposal_threshold_only_and_restores_on_error(self):
        from types import SimpleNamespace
        backend=object.__new__(DinoSamBackend);backend.config=VisionConfig()
        def detect(*args):
            self.assertEqual(backend.model.box_threshold,.15)
            raise RuntimeError('detector failure')
        backend.model=SimpleNamespace(box_threshold=.3,_run_dino=detect)
        with self.assertRaises(RuntimeError):backend._detect(None,'entity',True)
        self.assertEqual(backend.model.box_threshold,.3)
    def test_geometry_without_semantics_cannot_name_target_in_strict_mode(self):
        q=TargetQuery('x',('entity',),cad=CadModel('static.glb'))
        b=Backend([candidate()],lambda r:accepted())
        value=UnifiedVision(b,VisionConfig(require_semantic_identity=True)).observe([frame()],[q],episode_id='e').targets['x']
        self.assertFalse(value.required_available)

    def test_supported_lower_score_candidate_survives_unsupported_best_fit(self):
        a=candidate(10,.99);b=candidate(75,.4)
        b.identity_evidence=semantic_evidence(b.mask,[(.4,b.box,'entity')])
        backend=Backend([a,b],lambda r:accepted(.1 if r['mask'].box[0]==10 else .2))
        q=TargetQuery('x',('entity',),cad=CadModel('static.glb'))
        result=UnifiedVision(backend,VisionConfig(require_semantic_identity=True)).observe([frame()],[q],episode_id='e')
        self.assertEqual(result.targets['x'].evidence[0]['box_xyxy'][0],75)

    def test_semantic_support_is_spatial_and_not_identity_proof(self):
        a=candidate();b=candidate(75)
        self.assertFalse(semantic_evidence(a.mask,[(.99,b.box,'entity')])['supported'])
        self.assertFalse(semantic_evidence(a.mask,[(0.,a.box,'unlabelled')])['supported'])
        v=semantic_evidence(a.mask,[(.4,a.box,'entity')])
        self.assertTrue(v['supported']);self.assertFalse(v['identity_proven'])
        self.assertFalse(v['score_is_probability'])


class CompletionTests(unittest.TestCase):
    def test_missing_measured_pixels_recovered_without_filling_occlusion(self):
        depth=np.ones((40,50));rendered=np.zeros_like(depth);rendered[5:35,10:40]=1.
        original=rendered>0;original[:12]=False
        depth[15:20,15:20]=.8;depth[20:25,25:30]=0.
        out,audit=complete_measured_mask(depth,original,rendered)
        self.assertTrue(audit['accepted']);self.assertGreater(audit['added_pixels'],0)
        self.assertTrue(out[7,20]);self.assertFalse(audit['hidden_pixels_generated'])
        # Existing mask is preserved; only NEW pixels require valid depth.
        original[15:25,15:30]=False
        out,audit=complete_measured_mask(depth,original,rendered)
        self.assertFalse(out[16,16]);self.assertFalse(out[22,27])
        self.assertFalse(out[0,0])

    def test_wrong_depth_cannot_be_used_to_complete_model(self):
        depth=np.ones((40,50));rendered=np.full_like(depth,.9);mask=np.zeros_like(depth,bool);mask[10:20,10:20]=True
        out,audit=complete_measured_mask(depth,mask,rendered)
        self.assertFalse(audit['accepted']);np.testing.assert_array_equal(out,mask)

    def test_same_object_fragments_are_deduplicated_only_with_matching_render(self):
        f=frame();a=candidate(10);b=candidate(75)
        rendered=np.ones(f.depth.shape)
        records=[dict(frame=f,mask=c,points=np.zeros((40,3)),cad_verification=accepted(),
                      cad_render_depth_m=rendered.copy()) for c in (a,b)]
        self.assertTrue(same_instance([records[0]],[records[1]],VisionConfig()))
        records[1]['cad_render_depth_m']=rendered+.1
        self.assertFalse(same_instance([records[0]],[records[1]],VisionConfig()))

    def test_completion_rebuilds_actual_saved_geometry(self):
        f=frame();a=candidate();render=np.zeros(f.depth.shape)
        render[20:60,10:45]=1.
        class CompleteBackend(Backend):
            def verify_cad(self,record,model):
                record['cad_render_depth_m']=render
                result=accepted();result['visible_completion']=dict(accepted=True)
                return result
        q=TargetQuery('x',('entity',),cad=CadModel('static.glb'))
        result=UnifiedVision(CompleteBackend([a],None),VisionConfig(complete_visible_surface=True)).observe([f],[q],episode_id='e')
        target=result.targets['x']
        self.assertTrue(target.required_available)
        self.assertGreater(target.views['head']['mask'].mask.sum(),a.mask.sum())


if __name__=='__main__':unittest.main()
