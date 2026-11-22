"""Shared semantic backend. Only generic prompts/masks cross this boundary."""
from pathlib import Path
from typing import Protocol
import numpy as np

from .contracts import MaskCandidate,VisionConfig
from .evidence import context,region_proposals,seed_point,clean,foreground,appearance,depth_region_proposals
from .assignment import query_signature
from .identity import semantic_evidence


def box_iou(a,b):
    area=float(np.prod(np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0)))
    union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area)
    return area/max(1.,union)


def distinct_boxes(detections,limit):
    """Bound work without spending it on overlapping prompt variants."""
    if limit<=0:return []
    unique=[]
    for value in detections:
        box=value[1]
        if any(box_iou(box,other[1])>.8 for other in unique):continue
        unique.append(value)
        if len(unique)>=limit:break
    return unique


class MaskBackend(Protocol):
    def propose(self,frame,query,*,roi=None,refine=False):
        """Return bounded MaskCandidate instances, not selected object identity."""


class DinoSamBackend:
    """Load local DINO/SAM once; optional CAD loads FP lazily, once.

    The legacy loader is reused only for initialization and _run_dino/_xyxy.
    No cup/door/pad detectors, phrase matching or task-specific estimate method
    are called. Each query is handled by the same code below.
    """
    def __init__(self,robotwin_root,*,max_candidates=None,config=None,project_root=None,cad_output=None):
        from adapters.rgbd_perception import GroundedPointCloudPerception
        self.model=GroundedPointCloudPerception(Path(robotwin_root))
        self.image_key=None
        self.config=VisionConfig() if config is None else config
        self.max_candidates=self.config.max_candidates if max_candidates is None else max_candidates
        self.detection_cache={};self.context_cache={};self.frame_key=None
        self.proposal_cache={};self.model_diameters={}
        self.frame_rgb=None
        self.project_root=Path(project_root) if project_root is not None else Path(robotwin_root).absolute().parents[1]
        self.cad_output=Path(cad_output) if cad_output is not None else self.project_root/'.runtime_cache/unified_cad'
        self.cad_verifier=None

    def _diameter(self,specification):
        if specification is None:return None
        key=repr(specification)
        if key not in self.model_diameters:
            try:
                import trimesh
                path=Path(specification.mesh_path)
                if not path.is_absolute():path=self.project_root/path
                mesh=trimesh.load(str(path),force='scene')
                if isinstance(mesh,trimesh.Scene):mesh=trimesh.util.concatenate(tuple(mesh.dump()))
                mesh.apply_scale(specification.scale)
                if specification.mesh_to_object is not None:mesh.apply_transform(specification.mesh_to_object)
                self.model_diameters[key]=float(np.linalg.norm(mesh.extents))
            except (OSError,ValueError,ImportError):self.model_diameters[key]=None
        return self.model_diameters[key]

    def _detect(self,rgb,prompt,recovery=False):
        # Lower proposal threshold only, never CAD acceptance thresholds.
        # Restore the shared detector even when a recovery call raises.
        old=getattr(self.model,'box_threshold',None)
        try:
            if recovery and old is not None:
                self.model.box_threshold=min(old,self.config.recovery_detection_threshold)
            return self.model._run_dino(rgb,prompt)
        finally:
            if old is not None:self.model.box_threshold=old

    def propose(self,frame,query,*,roi=None,refine=False):
        model=self.model;torch=model.torch
        rgb=np.array(frame.rgb,copy=True,order='C');h,w=rgb.shape[:2]
        frame_key=(frame.capture_id,frame.camera,id(frame.rgb))
        if self.frame_key!=frame_key:
            self.detection_cache.clear();self.context_cache.clear();self.proposal_cache.clear();self.frame_key=frame_key
            self.frame_rgb=frame.rgb  # Keep ownership: an array ID cannot be recycled into a cache hit.
        context_key=query_signature(query)
        if context_key not in self.context_cache:self.context_cache[context_key]=context(frame,query,self.config)
        ctx=self.context_cache[context_key]
        proposal_key=(context_key,None if roi is None else tuple(np.asarray(roi)),refine)
        if proposal_key in self.proposal_cache:return list(self.proposal_cache[proposal_key])
        offset=np.array([0,0,0,0])
        if roi is not None:
            lower=np.maximum(np.floor(roi[:2]).astype(int),[0,0])
            upper=np.minimum(np.ceil(roi[2:]).astype(int)+1,[w,h])
            if np.any(upper-lower<3):return []
            x0,y0=lower;x1,y1=upper
            rgb=np.ascontiguousarray(rgb[y0:y1,x0:x1]);offset=np.array([x0,y0,x0,y0])
        ch,cw=rgb.shape[:2]
        recovery=bool(refine and query.cad is not None and self.config.require_semantic_identity)
        cache_key=(tuple(query.prompts),tuple(offset),rgb.shape,recovery)
        if cache_key not in self.detection_cache:
            found=[]
            for prompt in query.prompts:
                boxes,scores,_=self._detect(rgb,prompt.strip().rstrip('.')+' .',recovery)
                boxes=boxes.detach().cpu().numpy();scores=scores.detach().cpu().numpy().reshape(-1)
                for box,score in zip(boxes,scores):
                    found.append((float(score),model._xyxy(box,cw,ch)+offset,prompt))
            self.detection_cache[cache_key]=found
        detections=list(self.detection_cache[cache_key])
        budget=self.config.recovery_max_boxes if refine else self.max_candidates
        if refine and roi is None:
            detections+=region_proposals(ctx,query,detections,budget)
        geometric=[]
        if refine and query.cad is not None:
            # Independent region seeds can recover a complete textual miss.
            geometric=depth_region_proposals(ctx,query,frame,self.config,
                diameter_m=self._diameter(query.cad),roi=roi)
            # Bounded overlapping image tiles increase small-object resolution.
            # They never move the robot, and every result still needs CAD.
            if len(distinct_boxes(detections,budget))<self.max_candidates:
                for index,(fx,fy) in enumerate(((0.,0.),(.4,0.),(0.,.4),(.4,.4))):
                    if index>=self.config.max_recovery_tiles:break
                    tx0=int(fx*cw);ty0=int(fy*ch);tx1=min(cw,tx0+int(np.ceil(.6*cw)))
                    ty1=min(ch,ty0+int(np.ceil(.6*ch)))
                    tile=rgb[ty0:ty1,tx0:tx1]
                    if min(tile.shape[:2])<3:continue
                    tile_origin=offset+np.array([tx0,ty0,tx0,ty0])
                    tile_key=(tuple(query.prompts),tuple(tile_origin),tile.shape,recovery)
                    if tile_key not in self.detection_cache:
                        found=[]
                        for prompt in query.prompts:
                            boxes,scores,_=self._detect(tile,prompt.strip().rstrip('.')+' .',recovery)
                            for box,score in zip(boxes.detach().cpu().numpy(),scores.detach().cpu().numpy().reshape(-1)):
                                found.append((float(score),model._xyxy(box,tile.shape[1],tile.shape[0])+tile_origin,prompt))
                        self.detection_cache[tile_key]=found
                    detections.extend(self.detection_cache[tile_key])
        def priority(value):
            score,box,_=value;yy,xx=np.indices((h,w))
            selected=foreground(ctx,query)&(xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])
            _,color_score,supported=appearance(selected,ctx,query)
            penalty=.25 if query.hints is not None and query.hints.appearance_required and not supported else 0.
            return score+.25*color_score-penalty
        detections.sort(key=priority,reverse=True)
        # Reserve some recovery work for CAD-backed, non-semantic hypotheses.
        reserve=min(len(geometric),max(1,budget//4)) if geometric else 0
        semantic=distinct_boxes(detections,budget-reserve)
        detections=distinct_boxes(semantic+geometric,budget)
        result=[];per_box=[]
        for score,box,prompt in detections:
            if refine:
                pad=max(4.,.1*max(box[2:]-box[:2]))
                lower=np.maximum(np.floor(box[:2]-pad).astype(int),[0,0])
                upper=np.minimum(np.ceil(box[2:]+pad).astype(int)+1,[w,h])
                x0,y0=lower;x1,y1=upper
                if min(upper-lower)<3:continue
                crop=np.array(frame.rgb[y0:y1,x0:x1],copy=True,order='C');origin=np.array([x0,y0,x0,y0])
                key=(frame_key,tuple(origin),crop.shape)
            else:
                crop=rgb;origin=offset;key=(frame_key,tuple(offset),rgb.shape)
            if self.image_key!=key:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model.sam.set_image(crop)
                self.image_key=key
            kwargs=dict(box=np.asarray(box)-origin,multimask_output=refine)
            if refine:
                kwargs.update(point_coords=(seed_point(box,ctx,query)-origin[:2])[None],
                              point_labels=np.ones(1,np.int32))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                masks,sam_scores,_=model.sam.predict(**kwargs)
            choices=[]
            for mask,sam_score in zip(np.asarray(masks).reshape(-1,*crop.shape[:2]),np.asarray(sam_scores).reshape(-1)):
                full=np.zeros((h,w),bool);x0,y0=origin[:2]
                full[y0:y0+crop.shape[0],x0:x0+crop.shape[1]]=mask.astype(bool)
                if refine:
                    try:
                        _,_,quality=clean(full,frame,query,self.config,ctx)
                    except RuntimeError:continue
                    rank=float(sam_score)+.25*quality['appearance_score']+.15*quality['shape_score']
                else:rank=float(sam_score)
                choices.append((rank,full))
            if not choices:continue
            choices.sort(key=lambda value:value[0],reverse=True)
            keep=self.config.max_masks_per_box if refine and query.cad is not None else 1
            per_box.append([MaskCandidate(full,box,score,prompt,
                source=('CAD_required_RGBD_region_SAM' if score==0. else
                        'measured_foreground_ROI_multimask_SAM' if refine else 'semantic_box_SAM'),
                identity_evidence=semantic_evidence(full,detections))
                for _,full in choices[:keep]])
        # Round robin: retain multiple regions before extra masks of one box.
        limit=self.config.max_refined_candidates if refine else self.max_candidates
        for index in range(max((len(v) for v in per_box),default=0)):
            for choices in per_box:
                if index<len(choices):result.append(choices[index])
                if len(result)>=limit:break
            if len(result)>=limit:break
        self.proposal_cache[proposal_key]=tuple(result)
        return result

    def summary(self):
        return dict(name='shared_local_GroundingDINO_SAM2',
            load_timings_s=self.model.load_timings_s,
            loads_CAD=self.cad_verifier is not None and bool(self.cad_verifier.models),
            loads_FoundationPose=self.cad_verifier is not None and self.cad_verifier.shared is not None,
            cad=self.cad_summary())

    def verify_cad(self,record,specification):
        if self.cad_verifier is None:
            from .cad_backend import FoundationPoseVerifier
            self.cad_verifier=FoundationPoseVerifier(self.project_root,self.cad_output,self.config)
        return self.cad_verifier.verify(record,specification)

    def cad_summary(self):
        return dict(initialized=False,weight_loads=0,cached_models=0) if self.cad_verifier is None else self.cad_verifier.summary()
