"""Regression tests for depth, exclusive instances and bounded proposal recall.

Synthetic sensor arrays and stub model calls only; no renderer labels, simulator,
CUDA context, detector weights or CAD registration are used by these unit tests.
"""
import ast
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from adapters.unified_vision import RGBDFrame,TargetQuery,CadModel,UnifiedVision,VisionConfig
from adapters.unified_vision.contracts import MaskCandidate
from adapters.unified_vision.backends import DinoSamBackend,distinct_boxes
from adapters.unified_vision.evidence import context,depth_region_proposals


ROOT=Path(__file__).resolve().parents[1]


def frame(capture='initial'):
    return RGBDFrame('head',capture,np.full((80,120,3),100,np.uint8),np.ones((80,120)),
        np.array([[200.,0,60],[0,200,40],[0,0,1]]),np.eye(4),'m')


def candidate(left=8,score=.8,width=22):
    mask=np.zeros((80,120),bool);mask[20:60,left:left+width]=True
    return MaskCandidate(mask,np.array([left,20,left+width,60]),score,'generic entity')


def query(key,model='same.glb'):
    return TargetQuery(key,('generic entity',),cad=None if model is None else CadModel(model))


class Backend:
    def __init__(self,proposals,fit=None):
        self.proposals=proposals;self.fit=fit;self.checked=[]
    def propose(self,frame,query,*,roi=None,refine=False):
        return self.proposals(frame,query,refine) if callable(self.proposals) else self.proposals
    def verify_cad(self,record,model):
        self.checked.append((model.mesh_path,record['mask'].box[0]))
        if self.fit is not None:return self.fit(record,model)
        pose=np.eye(4);pose[:3,3]=record['geometry']['position_world']
        return dict(accepted=True,cost=.2,object_to_world=pose)


class DepthReadbackTests(unittest.TestCase):
    def method(self,name='get_rgbd'):
        # Execute the real method without importing SAPIEN/open3d/torch.
        tree=ast.parse((ROOT/'tra_pipe/RoboTwin/envs/camera/camera.py').read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Camera')
        fn=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
        helper=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='geometric_depth_mm')
        namespace={'np':np}
        exec(compile(ast.Module(body=[helper,fn],type_ignores=[]),'camera.depth','exec'),namespace)
        return namespace[name]

    def test_color_alpha_never_rescales_or_invalidates_geometric_distance(self):
        rgba=np.ones((2,3,4));rgba[...,3]=[[1.,.4,0.],[.1,.7,1.]]
        positions=np.zeros((2,3,4));positions[...,2]=-1.
        calls=[]
        camera=SimpleNamespace(take_picture=lambda:calls.append('capture'),
            get_picture=lambda name:{'Color':rgba,'Position':positions}[name],
            get_intrinsic_matrix=lambda:np.eye(3),get_extrinsic_matrix=lambda:np.eye(4),
            get_model_matrix=lambda:np.eye(4))
        owner=SimpleNamespace(collect_wrist_camera=False,static_camera_list=[camera],
            static_camera_name=['head_camera'])
        result=self.method()(owner)
        np.testing.assert_array_equal(result['depth'],np.full((2,3),1000.))
        self.assertEqual(calls,['capture'])
        self.assertEqual(result['rgb'].shape,(2,3,3))

    def test_invalid_geometry_is_zero_even_when_color_is_opaque(self):
        rgba=np.ones((1,5,4));positions=np.zeros((1,5,4));positions[...,2]=-1.
        positions[0,0,3]=1;positions[0,1,0]=np.nan;positions[0,2,2]=np.inf
        positions[0,3,2]=1
        camera=SimpleNamespace(take_picture=lambda:None,
            get_picture=lambda name:{'Color':rgba,'Position':positions}[name],
            get_intrinsic_matrix=lambda:np.eye(3),get_extrinsic_matrix=lambda:np.eye(4),
            get_model_matrix=lambda:np.eye(4))
        owner=SimpleNamespace(collect_wrist_camera=False,static_camera_list=[camera],
            static_camera_name=['head_camera'])
        np.testing.assert_array_equal(self.method()(owner)['depth'],[[0,0,0,0,1000]])

    def test_batch_depth_uses_same_corrected_geometry_for_head_and_wrists(self):
        positions=np.zeros((2,3,4));positions[...,2]=-.8
        camera=SimpleNamespace(get_picture=lambda name:positions)
        owner=SimpleNamespace(collect_wrist_camera=True,collect_head_camera=True,
            left_camera=camera,right_camera=camera,static_camera_list=[camera,camera],
            static_camera_name=['head_camera','other_camera'],
            get_rgba=lambda:(_ for _ in ()).throw(AssertionError('color must not affect depth')))
        result=self.method('get_depth')(owner)
        self.assertEqual(set(result),{'head_camera','other_camera','left_camera','right_camera'})
        for value in result.values():np.testing.assert_array_equal(value['depth'],np.full((2,3),800.))


class JointInstanceTests(unittest.TestCase):
    def observe(self,backend,queries):
        return UnifiedVision(backend).observe([frame()],queries,episode_id='test')

    def test_two_identical_objects_get_two_distinct_slots_not_two_ambiguities(self):
        backend=Backend([candidate(),candidate(80,.79)])
        scene=self.observe(backend,[query('first'),query('second')])
        a,b=scene.targets.values()
        self.assertTrue(a.required_available and b.required_available)
        self.assertNotEqual(a.assignment['instance_id'],b.assignment['instance_id'])
        self.assertFalse(a.assignment['slot_is_semantic_identity'])
        self.assertEqual(a.assignment['interchangeable_keys'],['first','second'])
        self.assertGreater(np.linalg.norm(a.geometry['position_world']-b.geometry['position_world']),.2)
        self.assertEqual(len(backend.checked),2)  # Identical CAD+mask registered once.
        json.dumps(scene.to_dict(),allow_nan=False)

    def test_missing_instance_cannot_duplicate_the_only_object(self):
        scene=self.observe(Backend([candidate()]),[query('a'),query('b')])
        self.assertEqual(sum(v.required_available for v in scene.targets.values()),1)
        missing=next(v for v in scene.targets.values() if not v.required_available)
        self.assertIn('no_distinct_instance_available',missing.reasons)

    def test_extra_equally_good_instance_is_not_silently_chosen(self):
        scene=self.observe(Backend([candidate(4),candidate(45),candidate(88)]),[query('a'),query('b')])
        self.assertTrue(all(v.status=='ambiguous' for v in scene.targets.values()))

    def test_distinct_models_use_joint_fit_not_greedy_first_query(self):
        def fit(record,model):
            left=record['mask'].box[0]<40;pose=np.eye(4)
            pose[:3,3]=record['geometry']['position_world']
            cost=(.1 if left else .12) if model.mesh_path=='a.glb' else (.1 if left else 1.)
            return dict(accepted=True,cost=cost,object_to_world=pose)
        scene=self.observe(Backend([candidate(),candidate(80)],fit),[query('a','a.glb'),query('b','b.glb')])
        self.assertTrue(all(v.required_available for v in scene.targets.values()))
        self.assertEqual(scene.targets['a'].evidence[0]['box_xyxy'][0],80)
        self.assertEqual(scene.targets['b'].evidence[0]['box_xyxy'][0],8)

    def test_distinct_models_with_tied_assignments_remain_ambiguous(self):
        scene=self.observe(Backend([candidate(),candidate(80)]),[query('a','a.glb'),query('b','b.glb')])
        self.assertTrue(all(v.status=='ambiguous' for v in scene.targets.values()))
        self.assertTrue(all(not v.required_available for v in scene.targets.values()))

    def test_distinct_models_cannot_arbitrarily_claim_single_shared_object(self):
        scene=self.observe(Backend([candidate()]),[query('a','a.glb'),query('b','b.glb')])
        self.assertTrue(all(v.status=='ambiguous' for v in scene.targets.values()))

    def test_single_query_keeps_unique_target_ambiguity_gate(self):
        value=self.observe(Backend([candidate(),candidate(80)]),[query('a')]).targets['a']
        self.assertFalse(value.required_available)
        self.assertIn('ambiguous_CAD_consistent_candidates',value.reasons)

    def test_tracking_preserves_measured_instances_when_proposal_order_changes(self):
        backend=Backend(lambda f,q,r:([candidate(),candidate(80)] if f.capture_id=='initial'
                                      else [candidate(80),candidate()]))
        engine=UnifiedVision(backend);queries=[query('a'),query('b')]
        old=engine.observe([frame()],queries,episode_id='test')
        current=engine.observe([frame('next')],queries,episode_id='test')
        for key in ('a','b'):
            self.assertTrue(current.targets[key].required_available)
            np.testing.assert_allclose(old.targets[key].geometry['position_world'],
                                       current.targets[key].geometry['position_world'])

    def test_model_free_equivalent_queries_also_receive_distinct_instances(self):
        scene=self.observe(Backend([candidate(),candidate(80)]),[query('a',None),query('b',None)])
        self.assertTrue(all(v.required_available for v in scene.targets.values()))


class RetainedCandidateTests(unittest.TestCase):
    def test_failed_refinement_cannot_erase_the_only_correct_initial_mask(self):
        def fit(record,model):
            if record['mask'].box[0]!=8:return dict(accepted=False)
            pose=np.eye(4);return dict(accepted=True,cost=.2,object_to_world=pose)
        backend=Backend(lambda f,q,r:[candidate(80 if r else 8)],fit)
        value=UnifiedVision(backend).observe([frame()],[query('a')],episode_id='e').targets['a']
        self.assertTrue(value.required_available)
        self.assertEqual(value.evidence[0]['box_xyxy'][0],8)
        self.assertEqual(len(backend.checked),2)

    def test_overlapping_sam_alternative_is_verified_before_mask_nms(self):
        big=candidate(width=30);small=candidate(width=22)
        def fit(record,model):
            if record['mask'].mask.sum()!=small.mask.sum():return dict(accepted=False)
            return dict(accepted=True,cost=.2,object_to_world=np.eye(4))
        backend=Backend(lambda f,q,r:[big,small] if r else [big],fit)
        value=UnifiedVision(backend).observe([frame()],[query('a')],episode_id='e').targets['a']
        self.assertTrue(value.required_available)
        self.assertEqual(len(backend.checked),2)
        self.assertEqual(value.views['head']['mask'].mask.sum(),small.mask.sum())

    def test_correct_fifth_candidate_is_no_longer_cut_off_at_four(self):
        proposals=[candidate(left,1-i*.1,width=14) for i,left in enumerate((3,25,47,69,94))]
        def fit(record,model):
            return (dict(accepted=True,cost=.2,object_to_world=np.eye(4))
                    if record['mask'].box[0]==94 else dict(accepted=False))
        backend=Backend(lambda f,q,r:proposals if r else proposals[:4],fit)
        value=UnifiedVision(backend).observe([frame()],[query('a')],episode_id='e').targets['a']
        self.assertTrue(value.required_available)
        self.assertEqual(value.evidence[0]['box_xyxy'][0],94)
        self.assertEqual(len(backend.checked),5)


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


class Sam:
    def __init__(self):self.calls=[]
    def set_image(self,image):self.shape=image.shape[:2]
    def predict(self,**kwargs):
        self.calls.append(kwargs);x0,y0,x1,y1=np.rint(kwargs['box']).astype(int)
        x0=max(0,x0);y0=max(0,y0);x1=min(self.shape[1],x1);y1=min(self.shape[0],y1)
        values=[]
        for shrink in ((0,2,4) if kwargs['multimask_output'] else (0,)):
            mask=np.zeros(self.shape,bool);mask[y0:y1,x0:max(x0,x1-shrink)]=True;values.append(mask)
        return np.asarray(values),np.array([.99,.9,.8])[:len(values)],None


class NativeProposalTests(unittest.TestCase):
    def backend(self,detect,config=None):
        calls=[];sam=Sam()
        def run(rgb,prompt):
            calls.append((rgb.shape,prompt));boxes,scores=detect(rgb,prompt)
            return Tensor(boxes),Tensor(scores),[]
        model=SimpleNamespace(torch=SimpleNamespace(inference_mode=nullcontext,
            autocast=lambda *a,**kw:nullcontext(),bfloat16=None),sam=sam,_run_dino=run,
            _xyxy=lambda box,w,h:np.asarray(box)*[w,h,w,h],load_timings_s={})
        with patch.dict('sys.modules',{'adapters.rgbd_perception':SimpleNamespace(
                GroundedPointCloudPerception=lambda root:model)}):
            backend=DinoSamBackend(ROOT/'tra_pipe/RoboTwin',config=config or VisionConfig(),project_root=ROOT)
        backend._diameter=lambda specification:None
        return backend,calls,sam

    def test_multiple_sam_masks_survive_until_cad_and_equivalent_query_is_cached(self):
        backend,calls,sam=self.backend(lambda rgb,p:([[.15,.2,.45,.7]],[.7]))
        with patch('adapters.unified_vision.backends.depth_region_proposals',return_value=[]):
            a=backend.propose(frame(),query('a'),refine=True)
            # Use the exact same snapshot, as the engine does.
            f=frame('cached');backend.propose(f,query('a'),refine=True)
            previous=(len(calls),len(sam.calls));b=backend.propose(f,query('b'),refine=True)
        self.assertGreaterEqual(len(a),3)
        self.assertEqual(previous,(len(calls),len(sam.calls)))
        self.assertGreaterEqual(len(b),3)

    def test_empty_full_image_detection_can_recover_from_a_tile(self):
        backend,calls,_=self.backend(lambda rgb,p:([],[]) if rgb.shape[:2]==(80,120)
                                     else ([[.2,.2,.7,.8]],[.4]))
        with patch('adapters.unified_vision.backends.depth_region_proposals',return_value=[]):
            proposals=backend.propose(frame(),query('a'),refine=True)
        self.assertTrue(proposals)
        self.assertEqual(len(calls),1+VisionConfig().max_recovery_tiles)
        self.assertLessEqual(len(proposals),VisionConfig().max_refined_candidates)
        self.assertTrue(all(np.asarray(p.mask).shape==(80,120) for p in proposals))

    def test_geometry_reserve_does_not_crowd_out_lower_scoring_semantic_regions(self):
        boxes=[[x,.25,x+.10,.75] for x in (.02,.18,.34,.50,.66,.82)]
        backend,_,_=self.backend(lambda rgb,p:(boxes,[.9,.8,.7,.6,.5,.4]))
        regions=[(0.,np.array([x,2,x+10,15]),'unlabelled') for x in (2,30,58,86)]
        with patch('adapters.unified_vision.backends.depth_region_proposals',return_value=regions):
            proposals=backend.propose(frame(),query('a'),refine=True)
        self.assertTrue(any(p.score==.4 for p in proposals))
        self.assertTrue(any(p.source=='CAD_required_RGBD_region_SAM' for p in proposals))

    def test_rgbd_regions_recover_without_any_text_detection_but_require_cad(self):
        f=frame();rgb=f.rgb.copy();depth=f.depth.copy()
        rgb[25:55,30:60]=[20,120,240];depth[25:55,30:60]=.8;f=replace(f,rgb=rgb,depth=depth)
        config=VisionConfig();q=query('a');ctx=context(f,q,config)
        regions=depth_region_proposals(ctx,q,f,config,diameter_m=.4)
        self.assertTrue(regions)
        self.assertEqual(depth_region_proposals(ctx,query('a',None),f,config),[])
        backend,_,_=self.backend(lambda rgb,p:([],[]))
        result=backend.propose(f,q,refine=True)
        self.assertTrue(result)
        self.assertTrue(all(p.source=='CAD_required_RGBD_region_SAM' for p in result))
        self.assertFalse(backend.propose(f,query('other',None),refine=True))

    def test_budget_validation_and_empty_selection(self):
        self.assertEqual(distinct_boxes([(.9,np.array([0,0,10,10]),'x')],0),[])
        for kwargs in (dict(max_refined_candidates=0),dict(max_masks_per_box=True),dict(max_recovery_tiles=1.2)):
            with self.assertRaises(ValueError):VisionConfig(**kwargs)


if __name__=='__main__':unittest.main()
