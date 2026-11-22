"""Bounded coarse registration -> depth-backed prompts -> new SAM candidates."""
from dataclasses import replace
import numpy as np
from adapters.unified_vision.backends import DinoSamBackend
from adapters.unified_vision.contracts import MaskCandidate
from adapters.unified_vision.evidence import clean,context
from adapters.unified_vision.identity import semantic_evidence
from .prompts import cad_prompts


class CadPromptedSamBackend(DinoSamBackend):
    def __init__(self,*args,use_cad_prompts=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.use_cad_prompts=use_cad_prompts;self.guidance_cache={};self.guidance_frame=None
        self.guidance_stats=dict(coarse_checks=0,guided_sam_calls=0,added_candidates=0,errors=[])

    def propose(self,frame,query,*,roi=None,refine=False):
        original=super().propose(frame,query,roi=roi,refine=refine)
        if not self.use_cad_prompts or not refine or query.cad is None:return original
        frame_key=(frame.camera,frame.capture_id,id(frame.rgb))
        if self.guidance_frame!=frame_key:
            self.guidance_cache.clear();self.guidance_frame=frame_key
        key=(repr(query),None if roi is None else tuple(np.asarray(roi,float)))
        if key in self.guidance_cache:return list(self.guidance_cache[key])
        ctx=context(frame,query,self.config);coarse=[];audits=[]
        # All original candidates remain available to the downstream selector.
        # Only three coarse registrations and two SAM masks/pose are added.
        for seed in sorted(original,key=lambda c:c.score,reverse=True)[:3]:
            self.guidance_stats['coarse_checks']+=1
            try:
                mask,points,quality=clean(seed.mask,frame,query,self.config,ctx)
                c=replace(seed,mask=mask)
                record=dict(frame=frame,mask=c,points=points,quality=quality)
                result=self.verify_cad(record,query.cad)
                rd=record.get('cad_render_depth_m')
                if not result.get('accepted') or rd is None:
                    audits.append(dict(source='CAD_prompt_seed',accepted=False,reason='coarse_CAD_not_verified'));continue
                coarse.append((seed,result,np.array(rd,copy=True)))
            except (RuntimeError,ValueError,OSError,ImportError) as error:
                audits.append(dict(source='CAD_prompt_seed',accepted=False,reason=str(error)))
        added=[]
        union=np.logical_or.reduce([rd>0 for _,_,rd in coarse]) if coarse else None
        detections=[(c.score,c.box,c.prompt) for c in original if c.score>0]
        for seed,result,rd in coarse:
            kwargs,audit=cad_prompts(frame.depth_m,rd,union,
                tolerance_m=self.config.cad.max_median_error_m,min_pixels=self.config.min_points)
            audits.append(audit)
            if kwargs is None:continue
            torch=self.model.torch
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    self.model.sam.set_image(np.array(frame.rgb,copy=True,order='C'))
                    masks,scores,_=self.model.sam.predict(**kwargs)
                self.guidance_stats['guided_sam_calls']+=1
                self.image_key=None  # Existing crop embedding cache is invalid.
                masks=np.asarray(masks).reshape(-1,*frame.depth.shape);scores=np.asarray(scores).reshape(-1)
                for i in np.argsort(-scores)[:2]:
                    mask=masks[i].astype(bool)
                    if any(np.array_equal(mask,c.mask) for c in original+added):continue
                    evidence=semantic_evidence(mask,detections)
                    evidence['cad_prompt_audit']=dict(audit,coarse_pose=result['object_to_world'],
                        sam_score=float(scores[i]),coarse_geometry_is_identity=False,
                        new_mask_requires_fresh_CAD_verification=True)
                    added.append(MaskCandidate(mask,kwargs['box'],seed.score,seed.prompt,
                        source='experimental_CAD_depth_prompted_SAM',identity_evidence=evidence))
            except (RuntimeError,ValueError) as error:
                self.image_key=None
                self.guidance_stats['errors'].append(str(error))
        self.guidance_stats['added_candidates']+=len(added)
        self.guidance_cache[key]=tuple(original+added)
        self.last_prompt_audits=audits
        return list(self.guidance_cache[key])

    def summary(self):
        return dict(super().summary(),cad_prompt_experiment=dict(enabled=self.use_cad_prompts,
            **self.guidance_stats))
