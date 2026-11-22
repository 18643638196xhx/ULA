import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from adapters.unified_vision import RGBDFrame,TargetQuery
from diagnostics.vision50_catalog import TASK_TARGETS,targets,captions
from diagnostics.vision50_assets import AssetPreparer,assemble_urdf,record_creations

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ula_vision50_runner',ROOT/'scripts/evaluate_unified_vision50.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class CatalogTests(unittest.TestCase):
    def test_arm_planner_import_is_blocked_but_not_replaced_with_motion(self):
        import sys
        with patch.dict(sys.modules):
            sys.modules.pop('envs.robot.planner',None)
            runner.block_planner_import()
            module=sys.modules['envs.robot.planner']
            with self.assertRaisesRegex(RuntimeError,'forbidden'):module.CuroboPlanner()

    def test_exactly_fifty_existing_tasks_and_explicit_collection_expansion(self):
        self.assertEqual(len(TASK_TARGETS),50)
        self.assertTrue(all((ROOT/'tra_pipe/RoboTwin/envs'/(name+'.py')).exists() for name in TASK_TARGETS))
        task=SimpleNamespace(bread=[object(),object()],breadbasket=object())
        self.assertEqual([key for key,value in targets(task,'place_bread_basket')],['bread_0','bread_1','breadbasket'])
        with self.assertRaises(ValueError):targets(task,'arbitrary_unknown_task')
        with self.assertRaises(ValueError):targets(SimpleNamespace(bread=[]),'place_bread_basket')

    def test_coarse_color_never_guesses_random_color_as_a_hard_prior(self):
        self.assertEqual(runner.coarse_color([1,0,0]),'red')
        self.assertIsNone(runner.coarse_color([.3,.7,.4]))
        self.assertIsNone(runner.coarse_color(None))
        self.assertEqual(captions('080_pillbottle'),('pill bottle',))

    def test_queries_supply_shape_not_pose_contact_or_labels(self):
        class Actor:
            def get_pose(self):raise AssertionError('live pose is forbidden')
        actor=Actor();task=SimpleNamespace(box=actor,target_box=Actor())
        creates={id(value):dict(kind='create_box',half_size=[.03,.03,.1],color=[1,0,0]) for value in (task.box,task.target_box)}
        preparer=SimpleNamespace(prepare=lambda *a,**kw:dict(cad=dict(mesh_path='static.glb'),audit={}))
        public,private=runner.prepare_queries(task,'handover_block',creates,preparer)
        for value in public:
            self.assertIsNotNone(TargetQuery(**value).cad)
            self.assertFalse(set(value)&{'pose','position','mask','contacts','roi','entity_id'})
        self.assertEqual(private[0]['equivalence_group'],private[1]['equivalence_group'])

    def test_snapshot_detects_parameter_changes_instead_of_retuning(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.py';source.write_text('original')
            freeze=dict(source_sha256={'source.py':runner.digest(source)})
            (root/'freeze.json').write_text(json.dumps(freeze))
            with patch.object(runner,'ROOT',root):
                runner.check_freeze(root)
                source.write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'Frozen'):runner.check_freeze(root)


class AssetTests(unittest.TestCase):
    def test_rigid_scale_whitelist_ignores_contact_metadata(self):
        import trimesh
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);asset=root/'assets/objects/test';(asset/'visual').mkdir(parents=True)
            trimesh.creation.box().export(asset/'visual/base0.glb')
            (asset/'model_data0.json').write_text(json.dumps(dict(scale=[.1,.2,.3],contact_points_pose='not geometry',target_pose='not geometry')))
            preparer=AssetPreparer(root,root/'generated')
            value=preparer.prepare(dict(kind='create_actor',modelname='test',model_id=0,scale=[1,1,1]))
            self.assertEqual(value['cad']['scale'],[.1,.2,.3])
            self.assertNotIn('contact_points_pose',json.dumps(value))

    def test_primitive_full_extent_and_fixed_color_are_supplied_model(self):
        import trimesh
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);preparer=AssetPreparer(root,root/'generated')
            description=dict(kind='create_box',half_size=[.03,.04,.1],color=[1,0,0])
            first=preparer.prepare(description);second=preparer.prepare(description)
            self.assertIs(first,second)
            mesh=trimesh.load(first['cad']['mesh_path'],force='mesh')
            np.testing.assert_allclose(mesh.extents,[.06,.08,.2],atol=1e-6)
            self.assertFalse(first['audit']['world_pose_read'])

    def test_urdf_static_fk_origin_scale_and_joint_fraction_no_live_state(self):
        import trimesh
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);trimesh.creation.box(extents=[1,1,1]).export(root/'box.obj')
            (root/'model_data.json').write_text(json.dumps(dict(scale=.1,init_qpos=[])))
            (root/'mobility.urdf').write_text('''<robot name="test"><link name="root"/><link name="moving"><visual><origin xyz="0 0 1"/><geometry><mesh filename="box.obj"/></geometry></visual></link><joint name="slide" type="prismatic"><parent link="root"/><child link="moving"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="0" upper="2"/></joint></robot>''')
            mesh,audit=assemble_urdf(root,initial_joint_fraction=.5)
            np.testing.assert_allclose(mesh.bounds.mean(0),[.1,0,.2],atol=1e-6)
            self.assertFalse(audit['live_joint_state_read'])

    def test_creation_recorder_discards_pose_and_restores_every_alias(self):
        import sys
        module=SimpleNamespace(__name__='envs.utils.create_actor')
        result=object()
        def creator(scene=None,pose=None,modelname='test',model_id=0,scale=(1,1,1),half_size=(1,1,1),color=(1,0,0)):return result
        for name in ('create_actor','create_box','create_visual_box','create_sapien_urdf_obj'):setattr(module,name,creator)
        aliases=SimpleNamespace(__name__='envs.fake',create_actor=creator)
        with patch.dict(sys.modules,{'envs.utils.create_actor':module,'envs.fake':aliases}):
            with record_creations() as records:
                aliases.create_actor(scene='forbidden',pose='forbidden')
                self.assertNotIn('pose',records[id(result)])
                self.assertNotIn('scene',records[id(result)])
            self.assertIs(aliases.create_actor,creator)


class EvaluationTests(unittest.TestCase):
    def test_duplicate_predictions_cannot_pass_two_interchangeable_instances(self):
        from diagnostics.unified_vision_evaluation import visible_reference
        rgb=np.zeros((40,80,3),np.uint8);depth=np.ones((40,80))
        frame=RGBDFrame('head','initial',rgb,depth,np.array([[200,0,40],[0,200,20],[0,0,1.]]),np.eye(4),'m')
        left=np.zeros_like(depth,bool);left[10:30,5:25]=True
        right=np.zeros_like(left);right[10:30,55:75]=True
        segmentation=np.zeros((*depth.shape,4),np.uint32);segmentation[left,1]=1;segmentation[right,1]=2
        camera=SimpleNamespace(get_picture=lambda name:segmentation)
        task=SimpleNamespace(cameras=SimpleNamespace(static_camera_list=[camera],static_camera_name=['head_camera']))
        private=[dict(key='a',actor=SimpleNamespace(per_scene_id=1),equivalence_group='same'),dict(key='b',actor=SimpleNamespace(per_scene_id=2),equivalence_group='same')]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);maskfile=root/'mask.npz';np.savez_compressed(maskfile,mask=left)
            target=dict(required_available=True,fresh=True,status='ok',usable=dict(position=True),
                geometry=dict(position_world=visible_reference(frame,left)[0].tolist()),
                evidence=[dict(measured_mask_path=str(maskfile),camera='head')])
            scene=SimpleNamespace(to_dict=lambda:dict(targets=dict(a=target,b=target)))
            score=runner.evaluate_scene(task,private,scene,frame,root)
            self.assertFalse(score['all_targets_localized'])
            self.assertEqual(sum(v['localization_verified'] for v in score['targets'].values()),1)

    def test_initialization_rejections_are_not_localization_failures(self):
        rows=[dict(seed=0,initialized=False,initialization_rejected=True),
            dict(seed=1,initialized=True,evaluation=dict(all_targets_localized=True,targets={'a':dict(localization_verified=True,identity_match=True,output=True,mask_metrics=dict(target_visible=True))}))]
        score=runner.task_summary(rows)
        self.assertEqual(score['localization_rate'],1.)
        self.assertEqual(score['initialization_rejected_seeds'],[0])

    def test_infrastructure_failure_is_unknown_not_a_zero_localization_rate(self):
        score=runner.task_summary([dict(seed=1,initialized=True,infrastructure_error=True)])
        self.assertIsNone(score['localization_rate'])


if __name__=='__main__':unittest.main()
