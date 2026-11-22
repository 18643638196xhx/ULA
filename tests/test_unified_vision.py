import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from adapters.unified_vision import RGBDFrame,TargetQuery,VisionConfig,UnifiedVision
from adapters.unified_vision.contracts import MaskCandidate
from adapters.unified_vision.engine import ordered_queries
from adapters.unified_vision.geometry import world_grid,plane,geometry,protrusions,surface_patches
from adapters.unified_vision.tracking import surface_motion


def frame(camera='head',capture='c0',*,translation=0.,timestamp=None):
    rgb=np.zeros((96,128,3),np.uint8);depth=np.ones((96,128))
    k=np.array([[200,0,64],[0,200,48],[0,0,1.]])
    t=np.eye(4);t[0,3]=-translation
    return RGBDFrame(camera,capture,rgb,depth,k,t,'m',timestamp)


def maskbox(x0=20,x1=60,y0=20,y1=70):
    mask=np.zeros((96,128),bool);mask[y0:y1,x0:x1]=True
    return mask


def candidate(mask,score=.8,prompt='generic entity'):
    y,x=np.nonzero(mask)
    return MaskCandidate(mask,np.array([x.min(),y.min(),x.max()+1,y.max()+1]),score,prompt)


class FakeBackend:
    def __init__(self,values):self.values=values;self.calls=[]
    def propose(self,frame,query,*,roi=None,refine=False):
        self.calls.append((frame.camera,query.key,refine,roi))
        if callable(self.values):return self.values(frame,query,refine)
        return self.values.get((frame.camera,query.key),[])


class ContractTests(unittest.TestCase):
    def test_sensor_scale_and_world_frame_are_explicit(self):
        a=frame();b=replace(a,depth=a.depth*1000,depth_unit='mm')
        np.testing.assert_allclose(world_grid(a)[0],world_grid(b)[0])
        translated=world_grid(frame(translation=.2))[0]
        np.testing.assert_allclose(translated[48,64],[.2,0,1])
        with self.assertRaises(ValueError):replace(a,depth_unit='auto')

    def test_snapshots_do_not_change_when_caller_reuses_sensor_buffer(self):
        a=frame();raw=a.rgb.copy();b=replace(a,rgb=raw)
        raw[:]=255
        self.assertEqual(b.rgb.max(),0)
        with self.assertRaises(ValueError):b.rgb[0,0]=255

    def test_bad_calibration_rejected(self):
        with self.assertRaises(ValueError):replace(frame(),intrinsic=np.zeros((3,3)))
        transform=np.eye(4);transform[0,0]=2.
        with self.assertRaises(ValueError):replace(frame(),world_to_camera=transform)

    def test_npz_loader_whitelists_sensor_arrays(self):
        a=frame()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'sensor.npz'
            np.savez_compressed(path,rgb=a.rgb,depth=a.depth,intrinsic_cv=a.intrinsic,
                extrinsic_cv=a.world_to_camera[:3],actor_pose=np.full((4,4),999),
                evaluation_only_mask=np.ones((96,128)))
            b=RGBDFrame.from_npz(path,camera='head',capture_id='c0',depth_unit='m')
            self.assertFalse(hasattr(b,'actor_pose'))
            np.testing.assert_allclose(world_grid(a)[0],world_grid(b)[0])

    def test_query_graph_sorted_and_invalid_graphs_rejected(self):
        parent=TargetQuery('body',('object',))
        child=TargetQuery('part',(),parent='body',search_protrusions=True)
        self.assertEqual([q.key for q in ordered_queries([child,parent])],['body','part'])
        with self.assertRaises(ValueError):ordered_queries([child])
        with self.assertRaises(ValueError):ordered_queries([parent,parent])
        with self.assertRaises(ValueError):ordered_queries([
            TargetQuery('a',('a',),parent='b'),TargetQuery('b',('b',),parent='a')])
        with self.assertRaises(ValueError):TargetQuery('a',())


class GeometryTests(unittest.TestCase):
    def setUp(self):self.config=VisionConfig()

    def plane_points(self):
        x,y=np.meshgrid(np.linspace(-.12,.12,50),np.linspace(-.12,.12,50))
        return np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]

    def structure(self):
        x,y,z=np.meshgrid(np.linspace(-.01,.01,6),np.linspace(-.05,.05,30),np.linspace(.015,.045,8))
        return np.c_[x.ravel(),y.ravel(),z.ravel()]

    def test_plane_has_no_vertical_or_world_axis_prior(self):
        points=self.plane_points()
        rotation=Rotation.from_euler('xyz',[.7,.3,-.2]).as_matrix()
        points=points@rotation.T+[.1,.2,.3]
        normal=rotation[:,2];camera=np.array([.1,.2,.3])+normal
        value=plane(points,camera,self.config)
        self.assertIsNotNone(value)
        self.assertGreater(value['normal_world']@normal,.999)
        self.assertFalse(value['hidden_boundary_observed'])
        self.assertFalse(value['placement_region_certified'])

    def test_line_and_invalid_depth_do_not_make_a_plane(self):
        line=np.c_[np.linspace(-.1,.1,100),np.zeros((100,2))]
        self.assertIsNone(plane(line,[0,0,1],self.config))
        a=frame();a=replace(a,depth=np.full(a.depth.shape,np.nan))
        grid,valid=world_grid(a)
        self.assertFalse(valid.any())
        with self.assertRaises(RuntimeError):geometry(grid[valid],[0,0,0],self.config,('position',))

    def test_visible_centre_is_not_hidden_bottom_or_six_dof_pose(self):
        shape,usable,_=geometry(self.plane_points(),[0,0,1],self.config,('position','bounds','axis'))
        self.assertFalse(usable['axis'])
        self.assertIsNone(shape['full_object_pose_world'])
        self.assertFalse(shape['hidden_support_anchor_observed'])
        self.assertIsNone(shape['position_error_bound_m'])

    def test_surface_relative_protrusion_not_camera_depth_order(self):
        body=self.plane_points();part=self.structure()
        values,audit=protrusions(body,np.r_[body,part],[0,0,1],self.config)
        self.assertGreater(len(values),0)
        self.assertGreater(values[0]['position_world'][2],.012)
        self.assertIsNone(values[0]['graspable'])
        self.assertFalse(values[0]['physical_attachment_proven'])
        self.assertEqual(audit['patch_sources'],['supported_global_surface_patch'])
        rotation=Rotation.from_euler('xyz',[.7,-.4,.2]).as_matrix()
        rotated,_=protrusions(body@rotation.T,np.r_[body,part]@rotation.T,
                             rotation@np.array([0,0,1.]),self.config)
        self.assertGreater(len(rotated),0)

    def test_nearer_but_unrelated_neighbour_is_not_a_parent_part(self):
        body=self.plane_points()
        values,_=protrusions(body,self.structure()+[.4,0,0],[0,0,1],self.config)
        self.assertEqual(values,[])

    def test_curved_body_uses_local_surfaces_without_class_branch(self):
        phi,y=np.meshgrid(np.linspace(-1.2,1.2,110),np.linspace(-.12,.12,50))
        radius=.2
        body=np.c_[radius*np.sin(phi.ravel()),y.ravel(),radius*np.cos(phi.ravel())]
        patches=surface_patches(body,[0,0,1],self.config)
        self.assertTrue(any(p['source']=='local_tangent_patch_of_measured_surface' for p in patches))
        x,y,z=np.meshgrid(np.linspace(-.008,.008,6),np.linspace(-.04,.04,25),np.linspace(.218,.245,9))
        part=np.c_[x.ravel(),y.ravel(),z.ravel()]
        values,audit=protrusions(body,np.r_[body,part],[0,0,1],self.config)
        self.assertGreater(len(values),0)
        self.assertIn('local_tangent_patch_of_measured_surface',audit['patch_sources'])

    def test_multiple_structures_remain_multiple_candidates(self):
        body=self.plane_points()
        search=np.r_[body,self.structure()+[-.06,0,0],self.structure()+[.06,0,0]]
        values,_=protrusions(body,search,[0,0,1],self.config)
        self.assertGreaterEqual(len(values),2)


class EngineTests(unittest.TestCase):
    def setUp(self):self.query=TargetQuery('target',('anything',),attributes=('position','bounds','plane'))

    def test_one_pipeline_for_arbitrary_target_names(self):
        backend=FakeBackend({('head','target'):[candidate(maskbox())]})
        engine=UnifiedVision(backend)
        for i,prompt in enumerate(('cup','bottle','basket','door','arbitrary visual entity')):
            query=replace(self.query,prompts=(prompt,))
            result=engine.observe([frame(capture=str(i))],[query],episode_id=str(i))
            self.assertTrue(result.targets['target'].required_available)
            self.assertTrue(result.targets['target'].usable['plane'])
            self.assertFalse(result.audit['grasp_or_release_authorized'])
            json.dumps(result.to_dict(),allow_nan=False)

    def test_low_score_geometric_candidate_can_survive_high_score_invalid_depth(self):
        bad=maskbox(80,110);good=maskbox()
        depth=frame().depth.copy();depth[bad]=0
        backend=FakeBackend({('head','target'):[candidate(bad,.95),candidate(good,.6)]})
        result=UnifiedVision(backend).observe([replace(frame(),depth=depth)],[self.query],episode_id='e')
        self.assertTrue(result.targets['target'].required_available)
        self.assertEqual(result.targets['target'].evidence[0]['semantic_score'],.6)

    def test_required_plane_selects_flat_candidate_over_higher_score_curved_object(self):
        curved=maskbox(20,60);flat=maskbox(80,115)
        depth=frame().depth.copy()
        x=np.indices(depth.shape)[1]
        depth[curved]=1.+.06*np.cos((x[curved]-40)*np.pi/40)
        backend=FakeBackend({('head','target'):[candidate(curved,.8),candidate(flat,.75)]})
        query=replace(self.query,required=('position','plane'))
        value=UnifiedVision(backend).observe([replace(frame(),depth=depth)],[query],episode_id='e').targets['target']
        self.assertTrue(value.required_available)
        self.assertEqual(value.evidence[0]['semantic_score'],.75)

    def test_two_similar_candidates_are_ambiguous_not_first_detection_wins(self):
        backend=FakeBackend({('head','target'):[candidate(maskbox(),.8),candidate(maskbox(80,115),.79)]})
        value=UnifiedVision(backend).observe([frame()],[self.query],episode_id='e').targets['target']
        self.assertEqual(value.status,'ambiguous')
        self.assertFalse(value.required_available)
        self.assertEqual(len(value.candidates),2)

    def test_duplicate_masks_across_prompt_variants_are_not_two_objects(self):
        backend=FakeBackend({('head','target'):[candidate(maskbox(),.8),candidate(maskbox(),.79)]})
        value=UnifiedVision(backend).observe([frame()],[self.query],episode_id='e').targets['target']
        self.assertTrue(value.required_available)

    def test_fallback_view_recovers_no_head_depth(self):
        backend=FakeBackend({('head','target'):[candidate(maskbox())],('wrist','target'):[candidate(maskbox())]})
        head=replace(frame(),depth=np.zeros((96,128)))
        value=UnifiedVision(backend).observe([head,frame('wrist')],[self.query],episode_id='e').targets['target']
        self.assertTrue(value.required_available)
        self.assertEqual([x['camera'] for x in value.evidence],['wrist'])
        self.assertLessEqual(len(backend.calls),4)

    def test_consistent_views_fuse_and_contradictory_views_do_not(self):
        backend=FakeBackend({('head','target'):[candidate(maskbox(),.95)],('wrist','target'):[candidate(maskbox(),.5)]})
        engine=UnifiedVision(backend)
        good=engine.observe([frame(),frame('wrist')],[self.query],episode_id='e').targets['target']
        self.assertEqual(len(good.evidence),2)
        bad=engine.observe([frame(capture='c1'),frame('wrist','c1',translation=.3)],
                           [self.query],episode_id='e').targets['target']
        self.assertFalse(bad.required_available)

    def test_unsynchronized_repeated_or_old_frames_cannot_fake_fresh_observation(self):
        backend=FakeBackend({('head','target'):[candidate(maskbox())]})
        engine=UnifiedVision(backend)
        with self.assertRaises(ValueError):engine.observe([frame(timestamp=0),frame('wrist',timestamp=1)],
            [self.query],episode_id='e')
        engine.observe([frame(timestamp=1.)],[self.query],episode_id='e')
        with self.assertRaises(ValueError):engine.observe([frame(timestamp=1.)],[self.query],episode_id='e')
        with self.assertRaises(ValueError):engine.observe([frame(capture='old',timestamp=.5)],[self.query],episode_id='e')

    def test_missing_target_returns_nonusable_last_known_not_fake_current_geometry(self):
        backend=FakeBackend(lambda f,q,r:[candidate(maskbox())] if f.capture_id=='c0' else [])
        engine=UnifiedVision(backend)
        engine.observe([frame()],[self.query],episode_id='e')
        value=engine.observe([frame(capture='c1')],[self.query],episode_id='e').targets['target']
        self.assertFalse(value.fresh)
        self.assertFalse(value.required_available)
        self.assertEqual(value.geometry,{})
        self.assertFalse(value.last_known['usable_for_current_control'])
        other=engine.observe([frame(capture='c2')],[self.query],episode_id='new_episode').targets['target']
        self.assertIsNone(other.last_known)

    def test_expired_track_is_not_reported_as_current(self):
        backend=FakeBackend(lambda f,q,r:[candidate(maskbox())] if f.capture_id=='c0' else [])
        engine=UnifiedVision(backend)
        engine.observe([frame(timestamp=0.)],[self.query],episode_id='e')
        value=engine.observe([frame(capture='c1',timestamp=10.)],[self.query],episode_id='e').targets['target']
        self.assertIsNone(value.last_known)

    def test_parent_then_part_is_generic_and_requires_current_parent(self):
        body=maskbox(20,110,15,85);part=maskbox(75,81,25,75)
        backend=FakeBackend({('head','body'):[candidate(body)],('head','part'):[candidate(part)]})
        depth=frame().depth.copy();depth[part]=.98
        parent=TargetQuery('body',('arbitrary body',))
        child=TargetQuery('part',('arbitrary part',),parent='body',search_protrusions=True)
        result=UnifiedVision(backend).observe([replace(frame(),depth=depth)],[child,parent],episode_id='e')
        self.assertTrue(result.targets['part'].required_available)
        self.assertTrue(result.targets['part'].evidence[0]['relation']['part_semantics_supported'])
        self.assertFalse(result.targets['part'].evidence[0]['relation']['physical_attachment_proven'])
        empty=UnifiedVision(FakeBackend({('head','part'):[candidate(part)]}))
        value=empty.observe([frame()],[parent,child],episode_id='e').targets['part']
        self.assertFalse(value.required_available)

    def test_query_change_does_not_reuse_another_objects_track(self):
        engine=UnifiedVision(FakeBackend({('head','target'):[candidate(maskbox())]}))
        engine.observe([frame()],[self.query],episode_id='e')
        with self.assertRaises(ValueError):engine.observe([frame(capture='c1')],
            [replace(self.query,prompts=('different thing',))],episode_id='e')

    def test_unknown_protrusions_are_not_renamed_to_requested_semantics(self):
        body=maskbox(20,110,15,85);part=maskbox(75,81,25,75)
        depth=frame().depth.copy();depth[part]=.98
        parent=TargetQuery('body',('body',))
        child=TargetQuery('part',('specific part',),parent='body',search_protrusions=True)
        backend=FakeBackend({('head','body'):[candidate(body)]})
        result=UnifiedVision(backend).observe([replace(frame(),depth=depth)],[parent,child],episode_id='e')
        self.assertFalse(result.targets['part'].required_available)
        unnamed=replace(child,prompts=())
        result=UnifiedVision(backend).observe([replace(frame(),depth=depth)],[parent,unnamed],episode_id='e')
        self.assertTrue(result.targets['part'].required_available)

    def test_part_search_does_not_bypass_explicit_geometry_priors(self):
        body=maskbox(20,110,15,85);part=maskbox(75,81,25,75)
        depth=frame().depth.copy();depth[part]=.98
        parent=TargetQuery('body',('body',))
        child=TargetQuery('part',(),parent='body',search_protrusions=True,max_extent_m=(.2,.01,.2))
        backend=FakeBackend({('head','body'):[candidate(body)]})
        value=UnifiedVision(backend).observe([replace(frame(),depth=depth)],[parent,child],episode_id='e')
        self.assertFalse(value.targets['part'].required_available)

    def test_fused_surface_does_not_bypass_explicit_geometry_priors(self):
        backend=FakeBackend({('head','target'):[candidate(maskbox())],('wrist','target'):[candidate(maskbox())]})
        query=TargetQuery('target',('thing',),max_extent_m=(.20,.3,.1))
        value=UnifiedVision(backend).observe([frame(),frame('wrist',translation=.04)],[query],episode_id='e')
        self.assertEqual(value.targets['target'].status,'rejected')
        self.assertFalse(value.targets['target'].required_available)


class TrackingTests(unittest.TestCase):
    def test_flat_surface_does_not_prove_full_six_dof_motion(self):
        x,y=np.meshgrid(np.linspace(-.1,.1,35),np.linspace(-.1,.1,35))
        points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        motion,audit=surface_motion(points,points+[.005,0,0],VisionConfig())
        self.assertIsNone(motion)
        self.assertLess(audit['observable_rank'],6)
        self.assertFalse(audit['accepted'])

    def test_rich_surface_rigid_motion_has_no_upright_or_world_z_restriction(self):
        rng=np.random.default_rng(4)
        points=rng.uniform(-.1,.1,(1600,3))
        # Three intersecting uneven faces with distinct extents.
        for i in range(3):points[i::3,i]=.1
        rotation=Rotation.from_rotvec([.015,-.01,.01]).as_matrix()
        actual=points@rotation.T+[.003,-.002,.004]
        motion,audit=surface_motion(points,actual,VisionConfig())
        self.assertIsNotNone(motion)
        np.testing.assert_allclose(motion[:3,3],[.003,-.002,.004],atol=.002)
        self.assertFalse(audit['symmetry_globally_resolved'])


class WorkerTests(unittest.TestCase):
    def test_worker_launches_existing_vision_environment_and_data1_caches(self):
        from adapters.unified_vision.worker import UnifiedVisionWorker
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'worker';output.mkdir()
            worker=UnifiedVisionWorker(root,output,gpu=3)
            command,cwd,environment=worker._launch_spec()
            self.assertIn('.conda_envs/RoboTwin_yzy/bin/python',command[0])
            self.assertTrue(command[1].endswith('run_unified_vision.py'))
            self.assertIn('--serve',command)
            self.assertNotIn('FoundationPose',environment['PYTHONPATH'])
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'],'3')
            self.assertTrue(environment['TMPDIR'].startswith(str(root)))
            worker.close()

    def test_wire_protocol_preserves_sensor_units_episode_and_single_process(self):
        from adapters.unified_vision.worker import UnifiedVisionWorker
        spec=importlib.util.spec_from_file_location('unified_vision_server_test',
            Path(__file__).resolve().parents[1]/'scripts/run_unified_vision.py')
        server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            backend=FakeBackend({('head','target'):[candidate(maskbox())]})
            engine=UnifiedVision(backend)
            class FakeWorker(UnifiedVisionWorker):
                def _start(self):
                    self.output.mkdir(parents=True,exist_ok=True)
                    if self.process is None:
                        self.process=SimpleNamespace(pid=1234)
                        self._stats['starts']+=1
                def _send(self,message):self.request=message
                def _receive(self):
                    self.last_wire_depth=np.load(self.request['frames'][0]['path'])['depth']
                    server.execute(engine,self.request)
                    return dict(status='ok',id=self.request['id'])
                def close(self):self._closed=True
            worker=FakeWorker(root,root/'worker')
            with patch.object(server,'ROOT',root):
                query=TargetQuery('target',('thing',))
                first=worker.observe([replace(frame(),depth=frame().depth*1000,depth_unit='mm')],
                                     [query],episode_id='one')
                np.testing.assert_allclose(worker.last_wire_depth,1.)
                second=worker.observe([frame(capture='c1')],[query],episode_id='one')
            self.assertTrue(first.targets['target'].required_available)
            self.assertEqual(first.audit['worker_pid'],second.audit['worker_pid'])
            self.assertEqual(worker._stats['starts'],1)
            self.assertEqual(worker._stats['completed_requests'],2)
            self.assertTrue(Path(second.targets['target'].evidence[0]['measured_mask_path']).exists())
            worker.close()

    def test_worker_rejects_artifact_directory_outside_project(self):
        from adapters.unified_vision.worker import UnifiedVisionWorker
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(ValueError):UnifiedVisionWorker(root,root.parent/'other_project')


if __name__=='__main__':unittest.main()
