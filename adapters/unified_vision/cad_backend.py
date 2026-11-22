"""Lazy, shared FoundationPose weights; bounded per-model CAD verification.

Reads only supplied static meshes + sensor/candidate records. No object truth,
task-specific class, contact annotations, previous pose or robot transport prior.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

from .contracts import rigid
from .identity import cad_color_evidence
from .cad_geometry import structure_depth_evidence
from .cad_geometry import TriangleSurface,point_agreement,render_agreement,scene_depth_evidence,complete_measured_mask


class FoundationPoseVerifier:
    def __init__(self,project_root,output_dir,config):
        self.root=Path(project_root).resolve();self.output=Path(output_dir).resolve()
        if not self.output.is_relative_to(self.root):raise ValueError('CAD outputs must stay under ULA')
        self.config=config;self.models={};self.shared=None;self.initialization_error=None
        self.stats=dict(weight_loads=0,model_loads=0,model_evictions=0,
                        candidate_checks=0,registrations=0,load_s=0.,
                        model_lookup_s=0.,registration_s=0.,surface_distance_s=0.,
                        rendering_s=0.,render_agreement_s=0.,completion_s=0.)

    def _load_shared(self):
        if self.shared is not None:return
        if self.initialization_error is not None:raise RuntimeError(self.initialization_error)
        try:self._initialize_shared()
        except Exception as error:
            self.initialization_error=f'CAD_backend_initialization_failed: {type(error).__name__}: {error}'
            raise RuntimeError(self.initialization_error) from error

    def _initialize_shared(self):
        started=time.perf_counter()
        fp=self.root/'tra_pipe/RoboTwin/third_party/FoundationPose'
        for path in (fp,fp/'mycpp/build'):
            if str(path) not in sys.path:sys.path.insert(0,str(path))
        import torch
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose,ScorePredictor,PoseRefinePredictor
        from Utils import nvdiffrast_render
        self.shared=dict(torch=torch,trimesh=trimesh,estimator=FoundationPose,
            scorer=ScorePredictor(),refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),render=nvdiffrast_render)
        self.stats['weight_loads']+=1;self.stats['load_s']+=time.perf_counter()-started

    def _model(self,specification):
        path=Path(specification.mesh_path)
        if not path.is_absolute():path=self.root/path
        path=path.resolve()
        if not path.is_file():raise FileNotFoundError('Explicit CAD mesh missing: '+str(path))
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        key=(str(path),specification.scale,specification.mesh_to_object)
        if key in self.models:
            value=self.models.pop(key)
            if value['sha256']!=sha:raise RuntimeError('Loaded CAD mesh changed during run')
            # Reinsert on access: regular dict insertion order is the LRU order.
            self.models[key]=value
            return value
        if len(self.models)>=self.config.cad.max_models:
            oldest=next(iter(self.models));evicted=self.models.pop(oldest)
            del evicted
            self.stats['model_evictions']+=1
            # FoundationPose estimators own model-specific CUDA tensors. Shared
            # scorer/refiner weights remain loaded; only the old mesh is freed.
            import gc
            gc.collect()
            if self.shared is not None:self.shared['torch'].cuda.empty_cache()
        self._load_shared();started=time.perf_counter();trimesh=self.shared['trimesh']
        mesh=trimesh.load(str(path),force='scene')
        if isinstance(mesh,trimesh.Scene):mesh=trimesh.util.concatenate(tuple(mesh.dump()))
        if len(mesh.vertices)<3 or not np.isfinite(mesh.vertices).all():raise ValueError('Invalid CAD mesh')
        visual=mesh.visual
        material=getattr(visual,'material',None)
        has_texture=bool(getattr(visual,'uv',None) is not None and material is not None and
            (getattr(material,'image',None) is not None or getattr(material,'baseColorTexture',None) is not None))
        if isinstance(mesh.visual,trimesh.visual.texture.TextureVisuals):
            material=mesh.visual.material
            if not hasattr(material,'image'):
                from PIL import Image
                texture=getattr(material,'baseColorTexture',None)
                if texture is None:
                    color=np.asarray(material.main_color,dtype=np.uint8)[:3]
                    texture=Image.fromarray(np.tile(color,(2,2,1)),mode='RGB')
                mesh.visual.material=trimesh.visual.material.SimpleMaterial(image=texture)
        mesh.apply_scale(specification.scale)
        if specification.mesh_to_object is not None:mesh.apply_transform(specification.mesh_to_object)
        surface=TriangleSurface(np.asarray(mesh.vertices),np.asarray(mesh.faces))
        directory=self.output/f"model_{int(self.stats['model_loads']):03d}"
        directory.mkdir(parents=True,exist_ok=False)
        estimator=self.shared['estimator'](model_pts=np.asarray(mesh.vertices),
            model_normals=np.asarray(mesh.vertex_normals),mesh=mesh,
            scorer=self.shared['scorer'],refiner=self.shared['refiner'],glctx=self.shared['glctx'],
            debug=0,debug_dir=str(directory))
        value=dict(estimator=estimator,surface=surface,sha256=sha,path=str(path),specification=asdict(specification),has_texture=has_texture)
        self.models[key]=value;self.stats['model_loads']+=1
        self.stats['load_s']+=time.perf_counter()-started
        return value

    def verify(self,record,specification):
        started=time.perf_counter();self.stats['candidate_checks']+=1
        phase=dict(model_lookup_s=0.,registration_s=0.,surface_distance_s=0.,
                   rendering_s=0.,render_agreement_s=0.,completion_s=0.)
        phase_started=time.perf_counter();model=self._model(specification)
        phase['model_lookup_s']+=time.perf_counter()-phase_started
        estimator=model['estimator'];torch=self.shared['torch']
        frame=record['frame'];mask=record['mask'].mask&np.isfinite(frame.depth_m)&(frame.depth_m>0)
        if mask.sum()<self.config.min_points:raise RuntimeError('insufficient_CAD_candidate_depth')
        # Unknown depth outside the candidate; do not fit to neighbouring objects.
        depth=np.where(mask,frame.depth_m,0.).astype(np.float32)
        estimator.pose_last=None;estimator.poses=None;estimator.gt_pose=None
        self.stats['registrations']+=1
        phase_started=time.perf_counter()
        with torch.inference_mode():
            camera_pose=estimator.register(K=np.asarray(frame.intrinsic,np.float32),
                rgb=np.array(frame.rgb,copy=True,order='C'),depth=depth,ob_mask=mask,
                iteration=self.config.cad.iterations)
        phase['registration_s']+=time.perf_counter()-phase_started
        if camera_pose is None:raise RuntimeError('FoundationPose_returned_no_pose')
        centering=estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
        hypotheses=[np.asarray(camera_pose)]
        centred=None
        if estimator.poses is not None:
            centred=estimator.poses[:self.config.cad.max_hypotheses]
            hypotheses=[p@centering for p in centred.detach().cpu().numpy()]
        else:
            centred=torch.as_tensor(np.asarray(camera_pose)[None]@np.linalg.inv(centering),
                dtype=torch.float32,device='cuda')
        camera_to_world=np.linalg.inv(frame.world_to_camera)
        checks=[];passed=[]
        for index,hypothesis in enumerate(hypotheses):
            try:
                world=rigid(camera_to_world@rigid(hypothesis))
                phase_started=time.perf_counter()
                surface=point_agreement(record['points'],world,model['surface'],self.config.cad)
                phase['surface_distance_s']+=time.perf_counter()-phase_started
                check=dict(hypothesis=index,object_to_world=world,point_agreement=surface,
                           accepted=False,reasons=[])
                checks.append(check)
                if not surface['accepted']:
                    check['reasons']=['measured_surface_disagrees_with_CAD'];continue
                phase_started=time.perf_counter()
                with torch.inference_mode():
                    rendered_rgb,rendered,_=self.shared['render'](K=np.asarray(frame.intrinsic,np.float32),
                        H=mask.shape[0],W=mask.shape[1],ob_in_cams=centred[index:index+1],
                        glctx=self.shared['glctx'],mesh_tensors=estimator.mesh_tensors)
                render_depth=rendered[0].detach().cpu().numpy()
                phase['rendering_s']+=time.perf_counter()-phase_started
                phase_started=time.perf_counter()
                fit=render_agreement(frame.depth_m,mask,render_depth,self.config.cad,self.config.min_points)
                phase['render_agreement_s']+=time.perf_counter()-phase_started
                check.update(accepted=fit['accepted'],render_agreement=fit,reasons=fit['reasons'])
                if fit['accepted'] and self.config.verify_scene_structure:
                    structure=structure_depth_evidence(frame.depth_m,render_depth,self.config.cad,self.config.min_points)
                    check['scene_structure_evidence']=structure
                    if not structure['accepted']:
                        check.update(accepted=False,reasons=['local_CAD_surface_occupies_measured_free_space'])
                        continue
                if fit['accepted'] and self.config.verify_cad_color:
                    color=dict(state='unobservable',accepted=True,reason='no_supplied_texture')
                    if model.get('has_texture',False):
                        selected=mask&(render_depth>0)&(np.abs(frame.depth_m-render_depth)<=self.config.cad.depth_tolerance_m)
                        color=cad_color_evidence(frame.rgb.astype(float)/255.,
                            rendered_rgb[0].detach().cpu().numpy(),selected,self.config.min_points)
                    check['color_evidence']=color
                    if not color['accepted']:
                        check.update(accepted=False,reasons=['CAD_texture_color_contradicts_observation'])
                        continue
                if fit['accepted'] and record.get('require_cross_view_cad'):
                    peer_checks=[];supported=False;contradicted=False
                    # SAME pose, not another independent best fit, must explain
                    # a different camera's measured candidate and scene depth.
                    for pf in record.get('cross_view_frames',[]):
                        # A second segmentation is not required to measure
                        # whether this SAME pose contradicts that scene's depth.
                        audit=dict(camera=pf.camera,accepted=False,
                            source='independent_scene_depth_at_projected_CAD_silhouette',
                            semantic_identity_proven=False)
                        peer_checks.append(audit)
                        ppose=pf.world_to_camera@world
                        centered=ppose@np.linalg.inv(centering)
                        with torch.inference_mode():
                            _,pd,_=self.shared['render'](K=np.asarray(pf.intrinsic,np.float32),
                                H=pf.depth.shape[0],W=pf.depth.shape[1],
                                ob_in_cams=torch.as_tensor(centered[None],dtype=torch.float32,device='cuda'),
                                glctx=self.shared['glctx'],mesh_tensors=estimator.mesh_tensors)
                        peer_depth=pd[0].detach().cpu().numpy()
                        evidence=scene_depth_evidence(pf.depth_m,peer_depth,self.config.cad,self.config.min_points)
                        audit.update(evidence,accepted=evidence['state']=='supported')
                        if evidence['state']=='supported':supported=True
                        if evidence['state']=='contradicted':contradicted=True
                    footprint=np.isfinite(render_depth)&(render_depth>0)
                    clipped=bool(footprint[0].any() or footprint[-1].any() or
                                 footprint[:,0].any() or footprint[:,-1].any())
                    unknown_only=bool(peer_checks) and all(p.get('state')=='unobservable' for p in peer_checks)
                    supported=(supported or (unknown_only and not clipped and not record.get('clipped',True))) and not contradicted
                    check['cross_view_status']=('contradicted' if contradicted else
                        'unobservable_primary_unclipped' if unknown_only and supported else
                        'supported' if supported else 'insufficient_cross_view_evidence')
                    check['cross_view_verification']=peer_checks
                    check['accepted']=supported
                    if not supported:check['reasons']=['same_CAD_pose_not_supported_in_another_view']
                else:supported=True
                if fit['accepted'] and supported:
                    cost=float(fit['cost']+surface['median_point_to_model_m']/self.config.cad.max_median_error_m+
                               1-surface['observed_surface_inliers'])
                    check['cost']=cost;passed.append((cost,world,render_depth,index))
            except (ValueError,RuntimeError) as error:
                checks.append(dict(hypothesis=index,accepted=False,reasons=[str(error)]))
        best=min(passed,key=lambda value:value[0]) if passed else None
        result=dict(accepted=best is not None,cost=None if best is None else best[0],
            object_to_world=None if best is None else best[1],selected_hypothesis=None if best is None else best[3],
            mesh_path=model['path'],mesh_sha256=model['sha256'],fixed_model_scale=specification.scale,
            hypotheses=checks,elapsed_s=time.perf_counter()-started,
            uses_object_pose_truth=False,uses_contact_annotations=False,scale_optimized=False,
            symmetry_globally_resolved=False,object_identity_proven=False,
            surface_distance_method=model['surface'].distance_method,
            timing_s=phase)
        if best is not None:
            record['cad_render_depth_m']=best[2]
            if self.config.complete_visible_surface:
                phase_started=time.perf_counter()
                completed,audit=complete_measured_mask(frame.depth_m,mask,best[2],self.config.cad,self.config.min_points,
                    frame=frame if self.config.body_guided_completion else None,
                    max_gap_m=self.config.completion_max_gap_m)
                # New pixels must also meet the unchanged exact surface gate.
                from .geometry import world_grid
                grid,valid=world_grid(frame)
                agreement=point_agreement(grid[completed&valid],best[1],model['surface'],self.config.cad)
                phase['completion_s']+=time.perf_counter()-phase_started
                audit['point_agreement']=agreement
                audit['accepted']=audit['accepted'] and agreement['accepted']
                result['visible_completion']=audit
                if audit['accepted']:record['completed_mask']=completed
        phase['total_s']=time.perf_counter()-started
        for key,value in phase.items():
            if key in self.stats:self.stats[key]+=value
        return result

    def summary(self):
        return dict(self.stats,initialized=self.shared is not None,cached_models=len(self.models),
                    initialization_error=self.initialization_error)
