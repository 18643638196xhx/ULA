"""Optional, target-independent CAD selection. No GPU or simulator in tests."""
from dataclasses import asdict,replace
import json
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np

from adapters.unified_vision import (CadModel,CadLimits,RGBDFrame,TargetQuery,
    VisionConfig,UnifiedVision)
from adapters.unified_vision.contracts import MaskCandidate
from adapters.unified_vision.cad_geometry import TriangleSurface,point_agreement,render_agreement


ROOT=Path(__file__).resolve().parents[1]


def frame(camera='head',translation=0.,capture='initial'):
    transform=np.eye(4);transform[0,3]=-translation
    return RGBDFrame(camera,capture,np.zeros((80,120,3),np.uint8),np.ones((80,120)),
        np.array([[200.,0,60],[0,200,40],[0,0,1]]),transform,'m')


def candidate(left=10,score=.8):
    mask=np.zeros((80,120),bool);mask[20:60,left:left+25]=True
    return MaskCandidate(mask,np.array([left,20,left+25,60]),score,'arbitrary entity')


def accepted(cost=.2,x=0.):
    pose=np.eye(4);pose[:3,3]=[x,0,1]
    return dict(accepted=True,cost=cost,object_to_world=pose)


class Backend:
    def __init__(self,proposals,verify):
        self.proposals=proposals;self.verify=verify;self.checked=[]
    def propose(self,frame,query,*,roi=None,refine=False):
        return self.proposals(frame,refine) if callable(self.proposals) else self.proposals
    def verify_cad(self,record,model):
        self.checked.append((record['frame'].camera,record['mask'].box[0],model))
        return self.verify(record)


class CadContractTests(unittest.TestCase):
    def test_native_stdout_cannot_corrupt_json_protocol(self):
        script=ROOT/'scripts/run_unified_vision.py'
        code=("import importlib.util,os,json; "
              f"s=importlib.util.spec_from_file_location('replay',{str(script)!r}); "
              "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
              "p=m.isolate_protocol_output();os.write(1,b'native diagnostic\\n'); "
              "print('python diagnostic');print(json.dumps({'status':'ok'}),file=p,flush=True)")
        value=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(value.stdout),{'status':'ok'})
        self.assertIn('native diagnostic',value.stderr)
        self.assertIn('python diagnostic',value.stderr)

    def test_optional_specification_and_shared_limits_roundtrip(self):
        query=TargetQuery('any_key',('any name',),cad=dict(mesh_path='static.glb',scale=[.1,.2,.3]))
        loaded=TargetQuery(**json.loads(json.dumps(asdict(query))))
        self.assertEqual(loaded.cad,query.cad)
        config=VisionConfig(**json.loads(json.dumps(asdict(VisionConfig()))))
        self.assertIsInstance(config.cad,CadLimits)
        self.assertIsNone(TargetQuery('x',('something',)).cad)

    def test_invalid_model_and_limits_rejected(self):
        for kwargs in (dict(mesh_path=''),dict(mesh_path='x',scale=[0,1,1]),
                       dict(mesh_path='x',scale=[1,1]),dict(mesh_path='x',scale=[1,np.nan,1]),
                       dict(mesh_path='x',mesh_to_object=np.ones((4,4)))):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):CadModel(**kwargs)
        for kwargs in (dict(iterations=0),dict(max_models=True),dict(max_hypotheses=2.5),
                       dict(min_mask_precision=1.1),dict(min_cost_gap=0),dict(depth_tolerance_m=np.inf)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):CadLimits(**kwargs)

    def test_static_model_transform_is_fixed_not_live_pose(self):
        transform=np.eye(4);transform[0,3]=.1
        spec=CadModel('x',mesh_to_object=transform)
        transform[0,3]=9
        self.assertEqual(spec.mesh_to_object[0][3],.1)
        self.assertIsInstance(spec.mesh_to_object,tuple)

    def test_example_changes_only_explicit_optional_cad(self):
        plain=json.loads((ROOT/'configs/unified_vision/bottle_pad.json').read_text())
        known=json.loads((ROOT/'configs/unified_vision/bottle_pad_cad.json').read_text())
        known['queries'][0].pop('cad')
        self.assertEqual(plain,known)

    def test_native_backend_is_lazy_and_missing_mesh_does_not_load_weights(self):
        from adapters.unified_vision.cad_backend import FoundationPoseVerifier
        verifier=FoundationPoseVerifier(ROOT,ROOT/'.runtime/tmp/cad_lazy_test',VisionConfig())
        self.assertEqual(verifier.summary()['weight_loads'],0)
        with patch.object(verifier,'_load_shared',side_effect=AssertionError('must stay lazy')):
            with self.assertRaises(FileNotFoundError):verifier._model(CadModel('does_not_exist.glb'))

    def test_initialization_failure_is_cached_not_reloaded_per_candidate(self):
        from adapters.unified_vision.cad_backend import FoundationPoseVerifier
        verifier=FoundationPoseVerifier(ROOT,ROOT/'.runtime/tmp/cad_lazy_test',VisionConfig())
        with patch.object(verifier,'_initialize_shared',side_effect=ImportError('missing renderer')) as initialize:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError,'missing renderer'):verifier._load_shared()
            self.assertEqual(initialize.call_count,1)


class CadGeometryTests(unittest.TestCase):
    def setUp(self):
        self.depth=np.ones((40,50));self.mask=np.zeros((40,50),bool)
        self.mask[10:30,10:40]=True
        self.render=np.where(self.mask,1.,0.)

    def test_matching_render_is_accepted_and_json_finite(self):
        value=render_agreement(self.depth,self.mask,self.render)
        self.assertTrue(value['accepted']);self.assertEqual(value['cost'],0)
        json.dumps(value,allow_nan=False)

    def test_returned_pose_without_matching_depth_is_not_a_match(self):
        rendered=self.render.copy();rendered[self.mask]=.95
        value=render_agreement(self.depth,self.mask,rendered)
        self.assertFalse(value['accepted'])
        self.assertIn('CAD_occupies_observed_empty_space',value['reasons'])

    def test_matching_tiny_patch_cannot_explain_whole_visible_model(self):
        mask=self.mask.copy();mask[15:30]=False
        value=render_agreement(self.depth,mask,self.render)
        self.assertFalse(value['accepted'])
        self.assertIn('visible_CAD_not_covered_by_candidate',value['reasons'])

    def test_occlusion_may_hide_some_but_not_almost_all_of_model(self):
        depth=self.depth.copy();depth[10:14,10:40]=.8
        mask=self.mask&(depth==1.)
        self.assertTrue(render_agreement(depth,mask,self.render)['accepted'])
        depth[10:28,10:40]=.8
        self.assertFalse(render_agreement(depth,self.mask&(depth==1.),self.render)['accepted'])

    def test_large_candidate_not_explained_by_small_render_is_rejected(self):
        value=render_agreement(self.depth,np.ones_like(self.mask),self.render)
        self.assertIn('candidate_surface_not_explained_by_CAD',value['reasons'])

    def test_missing_render_or_depth_fails_without_nan_cost(self):
        for observed,rendered in ((self.depth,np.zeros_like(self.depth)),
                                  (self.depth,np.full_like(self.depth,np.nan)),
                                  (np.zeros_like(self.depth),self.render)):
            value=render_agreement(observed,self.mask,rendered)
            self.assertFalse(value['accepted']);json.dumps(value,allow_nan=False)

    def test_observed_surface_is_checked_in_world_coordinates(self):
        model=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]])
        surface=TriangleSurface(model,np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]]))
        pose=np.eye(4);pose[:3,3]=[.2,.3,1.]
        self.assertTrue(point_agreement(model+pose[:3,3],pose,surface)['accepted'])
        self.assertFalse(point_agreement(model+[1,1,1],pose,surface)['accepted'])

    def test_malformed_geometry_is_not_registered(self):
        with self.assertRaises(ValueError):render_agreement(self.depth,self.mask,self.render[:2])
        with self.assertRaises(ValueError):point_agreement(np.zeros((2,3)),np.eye(4),np.zeros((3,3)))


class CadSelectionTests(unittest.TestCase):
    def setUp(self):self.query=TargetQuery('object',('generic entity',),cad=CadModel('static.glb'))

    def observe(self,backend,frames=None,query=None):
        return UnifiedVision(backend).observe(frames or [frame()],[query or self.query],episode_id='test')

    def test_all_unique_candidates_checked_before_ambiguity_rejection(self):
        backend=Backend([candidate(10,.8),candidate(75,.79)],
            lambda r:accepted() if r['mask'].box[0]==75 else dict(accepted=False))
        scene=self.observe(backend);value=scene.targets['object']
        self.assertEqual(len(backend.checked),2)
        self.assertTrue(value.required_available)
        self.assertEqual(value.evidence[0]['box_xyxy'][0],75)
        self.assertTrue(scene.audit['uses_prepared_object_models'])
        self.assertFalse(scene.audit['uses_object_truth'])
        self.assertFalse(np.allclose(value.geometry['position_world'],[0,0,1]))
        np.testing.assert_allclose(value.geometry['full_object_pose_world'],accepted()['object_to_world'])
        self.assertFalse(value.geometry['hidden_support_anchor_observed'])

    def test_cross_view_mode_supplies_other_current_masks_to_verifier(self):
        seen=[]
        def verify(record):
            seen.append((record.get('require_cross_view_cad'),
                         [p.camera for p in record.get('cross_view_frames',[])]))
            return accepted()
        backend=Backend([candidate()],verify)
        config=VisionConfig(require_cross_view_cad=True)
        UnifiedVision(backend,config).observe([frame(),frame('wrist')],[self.query],episode_id='e')
        self.assertTrue(all(flag for flag,peers in seen))
        self.assertTrue(all(peers for flag,peers in seen))
        self.assertEqual({tuple(peers) for flag,peers in seen},{('head',),('wrist',)})

    def test_cross_view_frame_is_available_even_if_other_detector_returns_nothing(self):
        seen=[]
        class PeerBackend(Backend):
            def propose(self,frame,query,**kwargs):return [candidate()] if frame.camera=='head' else []
        def verify(record):
            seen.extend(p.camera for p in record['cross_view_frames'])
            return dict(accepted=False,reasons=['same_CAD_pose_not_supported_in_another_view'])
        engine=UnifiedVision(PeerBackend([],verify),VisionConfig(require_cross_view_cad=True))
        scene=engine.observe([frame(),frame('wrist')],[self.query],episode_id='e')
        self.assertEqual(seen,['wrist'])
        self.assertFalse(scene.targets['object'].required_available)

    def test_cad_can_override_a_unique_but_incorrect_semantic_first_place(self):
        backend=Backend([candidate(10,.95),candidate(75,.4)],
            lambda r:accepted() if r['mask'].box[0]==75 else dict(accepted=False))
        self.assertEqual(self.observe(backend).targets['object'].evidence[0]['box_xyxy'][0],75)

    def test_duplicate_masks_are_registered_only_once(self):
        backend=Backend([candidate(10,.8),candidate(10,.79)],lambda r:accepted())
        self.assertTrue(self.observe(backend).targets['object'].required_available)
        self.assertEqual(len(backend.checked),1)

    def test_two_close_cad_fits_remain_ambiguous_regardless_of_detector_gap(self):
        backend=Backend([candidate(10,.99),candidate(75,.3)],lambda r:accepted())
        value=self.observe(backend).targets['object']
        self.assertFalse(value.required_available)
        self.assertIn('ambiguous_CAD_consistent_candidates',value.reasons)

    def test_clear_cad_quality_gap_selects_best_not_highest_detector_score(self):
        backend=Backend([candidate(10,.95),candidate(75,.6)],
            lambda r:accepted(.2 if r['mask'].box[0]==75 else .7))
        self.assertEqual(self.observe(backend).targets['object'].evidence[0]['box_xyxy'][0],75)

    def test_failed_cad_has_no_silent_model_free_fallback(self):
        backend=Backend([candidate()],lambda r:dict(accepted=False,reasons=['wrong shape']))
        value=self.observe(backend).targets['object']
        self.assertFalse(value.required_available)
        self.assertIn('no_CAD_consistent_current_candidate',value.reasons)
        self.assertEqual(len(value.candidate_records),1)

    def test_bad_or_unavailable_backend_fails_closed(self):
        for response in (None,accepted(np.nan),accepted(-.1),dict(accepted=True,cost=0),
                         dict(accepted=True,cost=0,object_to_world=np.ones((4,4)))):
            with self.subTest(response=response):
                backend=Backend([candidate()],lambda r:response)
                self.assertFalse(self.observe(backend).targets['object'].required_available)
        backend=type('NoCad',(),dict(propose=lambda *a,**kw:[candidate()]))()
        self.assertFalse(self.observe(backend).targets['object'].required_available)

    def test_registration_runtime_error_is_a_candidate_failure(self):
        def verify(record):raise RuntimeError('registration failed')
        backend=Backend([candidate()],verify)
        value=self.observe(backend).targets['object']
        self.assertFalse(value.required_available)
        self.assertIn('registration failed',str(value.candidates))

    def test_common_cad_origin_associates_different_visible_surfaces(self):
        backend=Backend([candidate()],lambda r:accepted())
        value=self.observe(backend,[frame(),frame('wrist',translation=.15)]).targets['object']
        self.assertTrue(value.required_available);self.assertEqual(len(value.evidence),2)

    def test_disagreeing_cad_origins_in_current_cameras_are_not_erased(self):
        backend=Backend([candidate()],lambda r:accepted(x=0 if r['frame'].camera=='head' else .2))
        value=self.observe(backend,[frame(),frame('wrist')]).targets['object']
        self.assertFalse(value.required_available)
        self.assertIn('multi_view_disagreement_no_consistent_3D_target',value.reasons)

    def test_no_cad_query_keeps_old_selection_and_never_calls_verifier(self):
        backend=Backend([candidate(10,.8),candidate(75,.79)],lambda r:accepted())
        scene=self.observe(backend,query=replace(self.query,cad=None))
        self.assertEqual(scene.targets['object'].status,'ambiguous')
        self.assertFalse(scene.audit['uses_prepared_object_models']);self.assertFalse(backend.checked)

    def test_cad_checks_initial_and_refined_masks_before_choosing(self):
        backend=Backend(lambda f,refine:[candidate(75 if refine else 10)],
            lambda r:accepted() if r['mask'].box[0]==75 else dict(accepted=False))
        value=self.observe(backend).targets['object']
        self.assertTrue(value.required_available)
        self.assertEqual(sorted(check[1] for check in backend.checked),[10,75])

    def test_on_demand_refinement_skips_recovery_after_unique_initial_cad_pose(self):
        calls=[]
        def proposals(_frame,refine):
            calls.append(refine)
            return [candidate(75 if refine else 10)]
        backend=Backend(proposals,lambda _record:accepted())
        scene=UnifiedVision(backend,VisionConfig(cad_refine_on_demand=True)).observe(
            [frame()],[self.query],episode_id='test')
        self.assertTrue(scene.targets['object'].required_available)
        self.assertEqual(calls,[False])
        self.assertEqual([check[1] for check in backend.checked],[10])

    def test_on_demand_refinement_preserves_recovery_after_initial_cad_rejection(self):
        calls=[]
        def proposals(_frame,refine):
            calls.append(refine)
            return [candidate(75 if refine else 10)]
        backend=Backend(proposals,
            lambda record:accepted() if record['mask'].box[0]==75 else dict(accepted=False))
        scene=UnifiedVision(backend,VisionConfig(cad_refine_on_demand=True)).observe(
            [frame()],[self.query],episode_id='test')
        self.assertTrue(scene.targets['object'].required_available)
        self.assertEqual(calls,[False,True])
        self.assertEqual(sorted(check[1] for check in backend.checked),[10,75])

    def test_current_history_uses_cad_origin_not_visible_centre(self):
        backend=Backend([candidate()],lambda r:accepted())
        engine=UnifiedVision(backend)
        engine.observe([frame()],[self.query],episode_id='test')
        scene=engine.observe([frame(translation=.15,capture='next')],[self.query],episode_id='test')
        self.assertTrue(scene.targets['object'].required_available)

    def test_candidate_masks_saved_even_when_no_target_selected(self):
        spec=importlib.util.spec_from_file_location('ula_cad_replay',ROOT/'scripts/run_unified_vision.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        save=module.save
        backend=Backend([candidate(10,.8),candidate(75,.79)],lambda r:dict(accepted=False))
        scene=self.observe(backend)
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as directory:
            output=Path(directory)/'prediction';save(output,scene)
            raw=json.loads((output/'prediction.json').read_text())
            target=raw['targets']['object']
            self.assertNotIn('candidate_records',target)
            self.assertEqual(len(list(output.glob('object_candidate_*.npz'))),2)
            for group in target['candidates']:
                for item in group['candidates']:
                    with np.load(item['measured_mask_path']) as values:
                        self.assertEqual(values['mask'].dtype,bool)
                        self.assertEqual(values['mask'].shape,(80,120))
            json.dumps(raw,allow_nan=False)


if __name__=='__main__':unittest.main()
