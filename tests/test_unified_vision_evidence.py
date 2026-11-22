from dataclasses import asdict,replace
import unittest
import numpy as np

from adapters.unified_vision import RGBDFrame,TargetQuery,VisualHints,VisionConfig,UnifiedVision
from adapters.unified_vision.contracts import MaskCandidate
from adapters.unified_vision.evidence import context,clean,seed_point,region_proposals


def scene():
    rgb=np.zeros((96,128,3),np.uint8);rgb[:]=[100,100,100]
    depth=np.ones((96,128));k=np.array([[200,0,64],[0,200,48],[0,0,1.]])
    return RGBDFrame('head','first',rgb,depth,k,np.eye(4),'m')


def rectangle(x0=20,x1=55,y0=25,y1=70):
    value=np.zeros((96,128),bool);value[y0:y1,x0:x1]=True;return value


def proposal(mask,score=.8):
    y,x=np.nonzero(mask)
    return MaskCandidate(mask,np.array([x.min(),y.min(),x.max()+1,y.max()+1]),score,'arbitrary target')


class Backend:
    def __init__(self,callback):self.callback=callback;self.calls=[]
    def propose(self,frame,query,*,roi=None,refine=False):
        self.calls.append((frame.camera,refine));return self.callback(frame,query,refine)


class EvidenceTests(unittest.TestCase):
    def test_optional_hints_leave_existing_query_valid_and_roundtrip(self):
        self.assertIsNone(TargetQuery('target',('thing',)).hints)
        q=TargetQuery('target',('thing',),hints=dict(colors=['orange','white'],all_colors=True,
            support_relation='above',up_world=[0,0,-2]))
        back=TargetQuery(**asdict(q))
        self.assertEqual(back.hints,q.hints)
        self.assertEqual(back.hints.up_world,(0,0,-1))

    def test_support_direction_and_unknown_hints_cannot_be_silently_guessed(self):
        with self.assertRaises(ValueError):VisualHints(support_relation='above')
        with self.assertRaises(ValueError):VisualHints(colors=('invisible_color',))
        with self.assertRaises(ValueError):VisualHints(up_world=(0,0,0))
        with self.assertRaises(ValueError):VisualHints(vertical_aspect_range=(3,1),up_world=(0,0,1))

    def test_plain_query_does_not_require_color_or_horizontal_support(self):
        f=scene();mask=rectangle();q=TargetQuery('anything',('unfamiliar object',))
        selected,points,audit=clean(mask,f,q,VisionConfig())
        self.assertGreater(len(points),40)
        self.assertIsNone(audit['support'])
        self.assertTrue(selected.any())

    def test_declared_appearance_removes_wrong_high_scoring_candidate(self):
        good=rectangle();bad=rectangle(80,115);f=scene();rgb=f.rgb.copy()
        rgb[good]=[30,120,220];rgb[bad]=[220,30,30];f=replace(f,rgb=rgb)
        backend=Backend(lambda f,q,r:[proposal(bad,.95),proposal(good,.5)])
        q=TargetQuery('whatever',('unseen name',),hints=VisualHints(colors=('blue',),appearance_required=True))
        result=UnifiedVision(backend).observe([f],[q],episode_id='episode').targets[q.key]
        self.assertTrue(result.required_available)
        self.assertEqual(result.evidence[0]['semantic_score'],.5)

    def test_same_hints_work_with_arbitrary_entity_keys_and_names(self):
        f=scene();rgb=f.rgb.copy();mask=rectangle();rgb[mask]=[30,120,220];f=replace(f,rgb=rgb)
        values=[]
        for key,name in [('cup','cup'),('basket','basket'),('unseen','unfamiliar structure')]:
            q=TargetQuery(key,(name,),hints=dict(colors=['blue']))
            backend=Backend(lambda f,q,r:[proposal(mask)])
            values.append(UnifiedVision(backend).observe([f],[q],episode_id='e').targets[key].geometry['position_world'])
        np.testing.assert_allclose(values[0],values[1]);np.testing.assert_allclose(values[1],values[2])

    def test_support_cleanup_does_not_use_known_table_height(self):
        f=scene();mask=rectangle();depth=f.depth.copy();depth[mask]=.8
        f=replace(f,depth=depth);q=TargetQuery('target',('thing',),
            hints=dict(support_relation='above',up_world=[0,0,-1]))
        mask[72:78,20:55]=True
        selected,points,audit=clean(mask,f,q,VisionConfig())
        self.assertLess(points[:,2].max(),.9)
        self.assertGreater(audit['removed_pixels'],0)
        self.assertAlmostEqual(audit['support']['offset_m'],-1.)

    def test_small_disconnected_depth_island_does_not_move_target_bounds(self):
        f=scene();mask=rectangle();mask[5:12,95:110]=True;depth=f.depth.copy();depth[5:12,95:110]=2.
        selected,points,audit=clean(mask,replace(f,depth=depth),TargetQuery('t',('thing',)),VisionConfig())
        self.assertFalse(selected[5:12,95:110].any())
        self.assertLess(points[:,2].max(),1.1)

    def test_thin_surface_is_not_eroded_away(self):
        mask=rectangle(30,33,20,75)
        _,points,audit=clean(mask,scene(),TargetQuery('t',('thin structure',)),VisionConfig())
        self.assertFalse(audit['mask_eroded_for_geometry'])
        self.assertGreater(len(points),40)

    def test_cleanup_is_invariant_to_world_origin_translation(self):
        f=scene();mask=rectangle();q=TargetQuery('t',('thing',));a=clean(mask,f,q,VisionConfig())[0]
        transform=f.world_to_camera.copy();transform[:3,3]=[20,30,40]
        b=clean(mask,replace(f,world_to_camera=transform),q,VisionConfig())[0]
        np.testing.assert_array_equal(a,b)

    def test_ambiguous_selection_triggers_a_real_second_segmentation(self):
        a,b=rectangle(),rectangle(80,115)
        backend=Backend(lambda f,q,r:[proposal(a)] if r else [proposal(a,.8),proposal(b,.79)])
        value=UnifiedVision(backend).observe([scene()],[TargetQuery('t',('thing',))],episode_id='e').targets['t']
        self.assertTrue(value.required_available)
        self.assertIn(('head',True),backend.calls)
        self.assertTrue(any(isinstance(v,dict) and v.get('operation')=='bounded_resegmentation_after_candidate_validation'
                            for v in value.reasons))

    def test_failed_resegmentation_cannot_erase_a_contradictory_camera(self):
        mask=rectangle();backend=Backend(lambda f,q,r:[] if r and f.camera=='wrist' else [proposal(mask)])
        transform=np.eye(4);transform[0,3]=-.4
        value=UnifiedVision(backend).observe([scene(),replace(scene(),camera='wrist',world_to_camera=transform)],
            [TargetQuery('t',('thing',))],episode_id='e').targets['t']
        self.assertFalse(value.required_available)

    def test_seed_is_a_measured_foreground_pixel_not_an_invalid_box_centre(self):
        f=scene();rgb=f.rgb.copy();depth=np.zeros_like(f.depth);mask=rectangle(20,35,25,65)
        rgb[mask]=[30,120,220];depth[mask]=.8;f=replace(f,rgb=rgb,depth=depth)
        q=TargetQuery('t',('thing',),hints=dict(colors=['blue']))
        point=seed_point(np.array([10,10,100,80]),context(f,q,VisionConfig()),q)
        self.assertTrue(mask[int(point[1]),int(point[0])])

    def test_color_regions_require_matching_semantic_evidence(self):
        f=scene();rgb=f.rgb.copy();mask=rectangle();rgb[mask]=[30,120,220];f=replace(f,rgb=rgb)
        q=TargetQuery('t',('arbitrary target',),hints=dict(colors=['blue']))
        ctx=context(f,q,VisionConfig())
        self.assertEqual(region_proposals(ctx,q,[],4),[])
        self.assertTrue(region_proposals(ctx,q,[(.6,np.array([20,25,55,70]),'arbitrary target')],4))

    def test_local_structures_cannot_bypass_explicit_appearance(self):
        f=scene();body=rectangle(20,110,15,85);part=rectangle(75,81,25,75)
        depth=f.depth.copy();depth[part]=.98;rgb=f.rgb.copy();rgb[part]=[220,30,30]
        f=replace(f,depth=depth,rgb=rgb)
        backend=Backend(lambda f,q,r:[proposal(body)] if q.key=='body' else [proposal(part)])
        parent=TargetQuery('body',('arbitrary body',))
        for color,expected in [('blue',False),('red',True)]:
            child=TargetQuery('part',('arbitrary part',),parent='body',search_protrusions=True,
                hints=dict(colors=[color],appearance_required=True))
            result=UnifiedVision(backend).observe([f],[parent,child],episode_id='e').targets['part']
            self.assertEqual(result.required_available,expected)


if __name__=='__main__':unittest.main()
