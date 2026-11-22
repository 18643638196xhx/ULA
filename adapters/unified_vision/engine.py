"""Single target-agnostic localization pipeline and bounded sensor fallback.

    vision.observe(frames, queries, episode_id='...')

Queries name visual entities and relationships; no task names or actor state.
The module returns observability, never a motion/release authorization.
"""
from dataclasses import asdict
import hashlib
import time
import numpy as np
from scipy.spatial import cKDTree

from .contracts import VisionConfig,MaskCandidate,TargetObservation,SceneObservation,rigid
from .geometry import world_grid,geometry,voxelize,protrusions
from .tracking import surface_motion
from .evidence import clean,context
from .assignment import assign_instances,same_instance
from .identity import identity_supported,identity_ranking,selection_cost,semantic_evidence


def ordered_queries(queries):
    queries=list(queries)
    if len({q.key for q in queries})!=len(queries):raise ValueError('Duplicate query keys')
    if any(q.parent is not None and q.parent not in {p.key for p in queries} for q in queries):
        raise ValueError('A part query requires its parent query in the same request')
    result=[];pending=queries.copy()
    while pending:
        ready=[q for q in pending if q.parent is None or q.parent in {p.key for p in result}]
        if not ready:raise ValueError('Cyclic parent relationships')
        result.extend(ready)
        pending=[q for q in pending if q not in ready]
    return result


def check_extent(shape,query):
    extent=np.diff(shape['bounds_world'],axis=0)[0]
    if query.min_extent_m is not None and np.any(extent<query.min_extent_m):
        raise RuntimeError('visible_extent_below_explicit_prior')
    if query.max_extent_m is not None and np.any(extent>query.max_extent_m):
        raise RuntimeError('visible_extent_above_explicit_prior')


class UnifiedVision:
    def __init__(self,backend,config=VisionConfig()):
        self.backend,self.config=backend,config
        self.reset()

    def reset(self,episode_id=None):
        self.episode_id=episode_id;self.history={};self.capture_index=0
        self.seen_captures=set();self.last_timestamp=None

    def _last(self,key,timestamp):
        value=self.history.get(key)
        if value is None:return None
        if self.capture_index-value['index']>self.config.max_missed_captures+1:return None
        if (timestamp is not None and value['timestamp'] is not None and
            timestamp-value['timestamp']>self.config.max_history_age_s):return None
        return value

    def _summary(self,record):
        return dict(camera=record['frame'].camera,prompt=record['mask'].prompt,
            capture_id=record['frame'].capture_id,box_xyxy=record['mask'].box,
            semantic_score=record['mask'].score,semantic_score_is_probability=False,
            evidence_ranking_score=record.get('rank',record['mask'].score),
            candidate_id=record.get('candidate_id'),cad_verification=record.get('cad_verification'),
            identity_gate=record.get('identity_gate'),
            identity_ranking=record.get('identity_ranking'),
            selection_cost=(selection_cost(record,self.config) if
                (record.get('cad_verification') or {}).get('accepted') else None),
            foreground_quality=record.get('quality'),
            identity_evidence=record['mask'].identity_evidence,
            source=record['mask'].source,position_world=record['geometry']['position_world'],
            bounds_world=record['geometry']['bounds_world'],point_count=len(record['points']),
            mask_touches_image_border=record['clipped'],relation=record.get('relation'))

    def _record(self,frame,candidate,query,grid,valid,ctx=None):
        mask=np.asarray(candidate.mask)
        if mask.dtype!=bool or mask.shape!=valid.shape or not np.isfinite(candidate.score):
            raise ValueError('Backend returned an invalid mask or score')
        box=np.asarray(candidate.box,float)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):
            raise ValueError('Backend returned an invalid image box')
        mask,points,quality=clean(mask,frame,query,self.config,ctx)
        candidate=MaskCandidate(mask,candidate.box,candidate.score,candidate.prompt,candidate.source,candidate.identity_evidence)
        camera=np.linalg.inv(frame.world_to_camera)[:3,3]
        shape,usable,points=geometry(points,camera,self.config,query.attributes)
        check_extent(shape,query)
        y,x=np.nonzero(mask)
        clipped=bool(len(y) and (y.min()==0 or y.max()==mask.shape[0]-1 or
                                  x.min()==0 or x.max()==mask.shape[1]-1))
        rank=float(candidate.score+.25*quality['appearance_score']+.15*quality['shape_score'])
        return dict(frame=frame,mask=candidate,points=points,geometry=shape,usable=usable,
                    clipped=clipped,rank=rank,quality=quality)

    def _roi(self,record):
        box=np.asarray(record['mask'].box,float).copy()
        pad=max(4.,.1*max(box[2:]-box[:2]))
        return box+[-pad,-pad,pad,pad]

    def _structure_records(self,query,frame,parent,semantic,grid,valid,attempts):
        # A parent-specific local search is the SAME operator for every named
        # entity. It never checks strings such as cup/door/basket/handle.
        roi=self._roi(parent);yy,xx=np.indices(valid.shape)
        search=valid&(xx>=roi[0])&(xx<=roi[2])&(yy>=roi[1])&(yy<=roi[3])
        reference=parent['mask'].mask&valid
        for candidate in semantic:reference&=~candidate.mask
        if reference.sum()<self.config.min_points:reference=parent['mask'].mask&valid
        camera=np.linalg.inv(frame.world_to_camera)[:3,3]
        structures,audit=protrusions(grid[reference],grid[search],camera,self.config)
        attempts.append(dict(camera=frame.camera,operation='generic_parent_surface_protrusions',**audit))
        records=[]
        for structure in structures:
            association=[]
            for candidate in semantic:
                selected=grid[candidate.mask&valid]
                if len(selected)<self.config.min_points:continue
                distances=cKDTree(selected).query(structure['points'])[0]
                fraction=float(np.mean(distances<=.012))
                if fraction>=.4:association.append((candidate.score,candidate,fraction))
            # With a named part query, an unmatched geometric protrusion is
            # reported as a candidate, NOT silently called the requested part.
            if query.prompts and not association:
                attempts.append(dict(camera=frame.camera,operation='part_semantic_association',
                    accepted=False,reason='geometry_without_matching_part_semantics',
                    position_world=structure['position_world'].tolist()))
                continue
            points=structure['points']
            camera_points=points@frame.world_to_camera[:3,:3].T+frame.world_to_camera[:3,3]
            pixels=camera_points@frame.intrinsic.T
            pixels=pixels[:,:2]/pixels[:,2:]
            low,high=pixels.min(0),pixels.max(0)
            mask=np.zeros(valid.shape,bool)
            for x,y in np.rint(pixels).astype(int):
                if 0<=y<mask.shape[0] and 0<=x<mask.shape[1]:mask[y,x]=True
            selected=max(association,key=lambda value:value[0]) if association else None
            quality=None
            if query.hints is not None:
                try:
                    mask,points,quality=clean(mask,frame,query,self.config)
                    if cKDTree(parent['points']).query(points)[0].min()>self.config.attachment_distance_m:
                        raise RuntimeError('hint_filtered_structure_not_near_measured_parent')
                except RuntimeError as error:
                    attempts.append(dict(camera=frame.camera,operation='part_explicit_evidence',reason=str(error)))
                    continue
            candidate=MaskCandidate(mask,np.r_[low,high],selected[0] if selected else 0.,
                selected[1].prompt if selected else 'unnamed protruding structure',source='parent_surface_depth_structure')
            shape,usable,points=geometry(points,camera,self.config,query.attributes)
            try:check_extent(shape,query)
            except RuntimeError as error:
                attempts.append(dict(camera=frame.camera,operation='part_extent_prior',reason=str(error)))
                continue
            record=dict(frame=frame,mask=candidate,points=points,geometry=shape,usable=usable,clipped=False,
                relation=dict(parent=query.parent,physical_attachment_proven=False,
                    source='parent_ROI_surface_protrusion_and_proximity',
                    part_semantics_supported=selected is not None,
                    semantic_surface_overlap=None if selected is None else selected[2]))
            if quality is not None:
                record.update(quality=quality,rank=float(candidate.score+
                    .25*quality['appearance_score']+.15*quality['shape_score']))
            records.append(record)
        return records

    def _groups(self,records,*,cad=False):
        groups=[]
        position=lambda r:(np.asarray(r['cad_verification']['object_to_world'])[:3,3]
                           if cad else r['geometry']['position_world'])
        tolerance=self.config.cad.cross_view_translation_m if cad else self.config.association_distance_m
        for record in records:
            for group in groups:
                if any(r['frame'].camera==record['frame'].camera for r in group):continue
                if all(np.linalg.norm(position(record)-position(r))<=tolerance for r in group):
                    group.append(record);break
            else:groups.append([record])
        return groups

    def _select(self,records,last,hypothesis,*,cad_model=None,return_options=False,selection_query=None):
        # Snapshot the CURRENT decision, not mutable records from an earlier
        # preparation pass. Early returns must not expose a previous audit.
        self._last_selection_competition=[]
        empty=[] if return_options else None
        if not records:return empty,[],['no_geometrically_valid_current_candidate']
        # NMS by measured mask overlap ONLY within the same camera.
        unique=[]
        score=lambda r:r.get('rank',r['mask'].score)
        for record in sorted(records,key=score,reverse=True):
            duplicate=False
            for other in unique:
                if record['frame'].camera!=other['frame'].camera:continue
                a,b=record['mask'].mask,other['mask'].mask
                # Different SAM masks must reach CAD before choosing a winner.
                threshold=.999 if cad_model is not None else .5
                if np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum())>threshold:
                    duplicate=True;break
            if not duplicate:unique.append(record)
        groups=self._groups(unique)
        groups.sort(key=lambda g:(len(g),np.mean([score(r) for r in g])),reverse=True)
        diagnostic_groups=groups
        if cad_model is not None:
            verifier=getattr(self.backend,'verify_cad',None)
            for record in unique:
                if self.config.require_cross_view_cad:
                    record['cross_view_records']=[other for other in unique
                        if other['frame'].camera!=record['frame'].camera]
                    record['require_cross_view_cad']=True
                    record['cross_view_frames']=[f for f in getattr(self,'_current_frames',[])
                        if f.camera!=record['frame'].camera]
                try:
                    if verifier is None:raise RuntimeError('CAD_backend_unavailable')
                    cache=getattr(self,'_cad_checks',{})
                    key=(record['frame'].camera,record['frame'].capture_id,repr(cad_model),
                         hashlib.sha256(record['mask'].mask.tobytes()).digest())
                    if key in cache:
                        result,render=cache[key]
                        if render is not None:record['cad_render_depth_m']=render
                    else:
                        result=verifier(record,cad_model)
                        cache[key]=(result,record.get('cad_render_depth_m'))
                    if self.config.complete_visible_surface and 'cad_render_depth_m' in record and result.get('accepted'):
                        # Rebuild visible geometry from measured pixels only;
                        # CAD origin remains a separate, model-inferred output.
                        from .cad_geometry import complete_measured_mask
                        if (result.get('visible_completion') or {}).get('accepted'):
                            refined,_=complete_measured_mask(record['frame'].depth_m,record['mask'].mask,
                                record['cad_render_depth_m'],self.config.cad,self.config.min_points,
                                frame=record['frame'] if self.config.body_guided_completion else None,
                                max_gap_m=self.config.completion_max_gap_m)
                            proposal=MaskCandidate(refined,record['mask'].box,record['mask'].score,
                                record['mask'].prompt,record['mask'].source,record['mask'].identity_evidence)
                            grid,valid=world_grid(record['frame'])
                            if selection_query is None:raise ValueError('Completion requires explicit query')
                            rebuilt=self._record(record['frame'],proposal,query=selection_query,
                                grid=grid,valid=valid)
                            if self.config.body_guided_completion:
                                # Already depth-connected and CAD checked. Generic
                                # component pruning/erosion can delete thin parts
                                # a second time; use all verified measured pixels.
                                camera=np.linalg.inv(record['frame'].world_to_camera)[:3,3]
                                shape,usable,points=geometry(grid[refined&valid],camera,self.config,
                                    selection_query.attributes)
                                rebuilt.update(mask=proposal,points=points,geometry=shape,usable=usable)
                                rebuilt['quality']=dict(rebuilt['quality'],
                                    verified_body_completion_preserved=True,
                                    selected_pixels=int(refined.sum()),geometry_points=len(points),
                                    mask_eroded_for_geometry=False)
                            record.update(rebuilt)
                    if not isinstance(result,dict):raise ValueError('Invalid CAD verifier response')
                    if result.get('accepted'):
                        rigid(result['object_to_world'])
                        if not np.isfinite(result['cost']) or result['cost']<0:
                            raise ValueError('Invalid CAD verification cost')
                    record['cad_verification']=result
                except (RuntimeError,ValueError,OSError,KeyError,ImportError) as error:
                    record['cad_verification']=dict(accepted=False,cost=None,reasons=[str(error)],
                        object_to_world=None)
            valid=[r for r in unique if r['cad_verification'].get('accepted') and
                   (not self.config.require_semantic_identity or identity_supported(r))]
            for r in unique:
                e=r['mask'].identity_evidence
                if self.config.evidence_ranked_cad and e.get('box_xyxy') is not None:
                    r['mask'].identity_evidence.update(semantic_evidence(r['mask'].mask,
                        [(e['detector_score'],e['box_xyxy'],e.get('prompt',''))]))
                r['identity_ranking']=identity_ranking(r)
                r['identity_gate']=dict(required=self.config.require_semantic_identity,
                    supported=identity_supported(r),geometry_alone_is_identity=False)
            # Collapse overlapping variants only AFTER their measured CAD fits.
            accepted=[]
            for record in sorted(valid,key=lambda r:selection_cost(r,self.config)):
                if any(record['frame'].camera==other['frame'].camera and
                       same_instance([record],[other],self.config)
                       for other in accepted):continue
                accepted.append(record)
            groups=self._groups(accepted,cad=True)
            groups.sort(key=lambda g:(-len(g),max(selection_cost(r,self.config) for r in g)))
        summaries=[dict(views=[r['frame'].camera for r in g],
            mean_semantic_score=float(np.mean([r['mask'].score for r in g])),
            mean_evidence_ranking_score=float(np.mean([score(r) for r in g])),
            candidates=[self._summary(r) for r in g]) for g in diagnostic_groups]
        if not groups:
            identity_only=(cad_model is not None and self.config.require_semantic_identity and
                           any((r.get('cad_verification') or {}).get('accepted') for r in unique))
            return empty,summaries,['no_semantically_supported_CAD_candidate' if identity_only else
                                    'no_CAD_consistent_current_candidate']
        active=[r for g in groups for r in g]
        if (not (cad_model is not None and self.config.require_cross_view_cad) and
            len({r['frame'].camera for r in active})>1 and all(len(g)==1 for g in groups)):
            return empty,summaries,['multi_view_disagreement_no_consistent_3D_target']
        if last is not None:
            predicted=np.asarray(last['observation'].geometry['position_world'])
            if cad_model is not None:
                previous=last['observation'].geometry.get('full_object_pose_world')
                if previous is not None:predicted=np.asarray(previous)[:3,3]
            if hypothesis is not None:predicted=hypothesis[:3,:3]@predicted+hypothesis[:3,3]
            # Check current-view agreement BEFORE prior association: a prior
            # must not silently erase a contradictory camera measurement.
            position=lambda r:(np.asarray(r['cad_verification']['object_to_world'])[:3,3]
                              if cad_model is not None else r['geometry']['position_world'])
            groups=[g for g in groups if all(np.linalg.norm(position(r)-predicted)
                    <=self.config.association_distance_m for r in g)]
            if not groups:return empty,summaries,['current_candidates_inconsistent_with_track_search_prior']
        if self.config.audit_candidate_selection and cad_model is not None:
            self._last_selection_competition=[dict(
                candidate_ids=[r.get('candidate_id') for r in g],
                geometric_cost=max(float(r['cad_verification']['cost']) for r in g),
                selection_cost=max(selection_cost(r,self.config) for r in g)) for g in groups]
        if return_options:return groups,summaries,[]
        if len(groups)>1:
            a,b=groups[:2]
            if cad_model is not None:
                gap=max(selection_cost(r,self.config) for r in b)-max(selection_cost(r,self.config) for r in a)
                if len(a)==len(b) and gap<self.config.cad.min_cost_gap:
                    return None,summaries,['ambiguous_CAD_consistent_candidates']
                return a,summaries,[]
            gap=float(np.mean([score(r) for r in a])-np.mean([score(r) for r in b]))
            if len(a)==len(b) and gap<self.config.ambiguity_score_gap:
                return None,summaries,['ambiguous_current_target_candidates']
        # A detector score is not identity proof; even a unique accepted group
        # retains that limitation in its evidence.
        return groups[0],summaries,[]

    def _prepare_query(self,query,frames,targets,last,hypothesis,grids):
        attempts=[];by_view={};parent=targets.get(query.parent)
        contexts={frame.camera:context(frame,query,self.config) for frame in frames}
        def collect(frame,refine):
            parent_record=None if parent is None else parent.views.get(frame.camera)
            if query.parent is not None and parent_record is None:
                attempts.append(dict(camera=frame.camera,reason='parent_not_resolved_in_current_view'));return []
            roi=None if parent_record is None else self._roi(parent_record)
            limit=(self.config.max_refined_candidates if refine else self.config.max_candidates)
            try:proposals=list(self.backend.propose(frame,query,roi=roi,refine=refine))[:limit]
            except RuntimeError as error:
                attempts.append(dict(camera=frame.camera,refine=refine,reason=str(error)));return []
            attempts.append(dict(camera=frame.camera,refine=refine,operation='semantic_mask_proposals',
                count=len(proposals),scores=[float(p.score) for p in proposals]))
            grid,valid=grids[frame.camera]
            if query.search_protrusions:
                return self._structure_records(query,frame,parent_record,proposals,grid,valid,attempts)
            result=[]
            for proposal in proposals:
                try:result.append(self._record(frame,proposal,query,grid,valid,contexts[frame.camera]))
                except RuntimeError as error:
                    attempts.append(dict(camera=frame.camera,refine=refine,reason=str(error)))
            return result
        for frame in frames:by_view[frame.camera]=collect(frame,False)
        def choose(*,cad=False,options=False):
            records=[r for values in by_view.values() for r in values]
            complete=[r for r in records if all(r['usable'].get(key,False)
                      for key in query.required if key not in ('structures','motion'))]
            return self._select(complete or records,last,hypothesis,
                cad_model=query.cad if cad else None,return_options=options,selection_query=query)
        selected,_,reasons=choose()
        initial_cad_selected=None
        initial_cad_reasons=[]
        if query.cad is not None and self.config.cad_refine_on_demand:
            initial_cad_selected,_,initial_cad_reasons=choose(cad=True)
            attempts.append(dict(
                operation='initial_candidates_CAD_gate',
                candidate_count=sum(len(values) for values in by_view.values()),
                resolved=initial_cad_selected is not None,
                reasons=initial_cad_reasons,
            ))
        if query.cad is not None and self.config.cad_refine_on_demand:
            # CAD remains mandatory.  Only the extra DINO/SAM recovery pass is
            # skipped when the initial candidates already produce a unique,
            # fully CAD-verified pose.  Rejected or ambiguous initial CAD
            # candidates retain the original exhaustive refinement path.
            needs_refine=initial_cad_selected is None
        else:
            needs_refine=(selected is None or query.hints is not None or query.cad is not None or any(
                r.get('clipped') or r.get('quality',{}).get('disconnected_major_components',0)>1 or
                not all(r['usable'].get(k,False) for k in query.required if k not in ('structures','motion'))
                for r in (selected or [])))
        if needs_refine:
            attempts.append(dict(operation='bounded_resegmentation_after_candidate_validation',
                first_selection_reasons=reasons))
            for frame in frames:
                refined=collect(frame,True)
                if refined:
                    # A worse retry must not erase the only correct CAD input.
                    if query.cad is not None:by_view[frame.camera].extend(refined)
                    else:by_view[frame.camera]=refined
        if query.cad is not None:
            for camera,values in by_view.items():
                seen=set();unique=[]
                for record in sorted(values,key=lambda r:r.get('rank',r['mask'].score),reverse=True):
                    digest=hashlib.sha256(record['mask'].mask.tobytes()).digest()
                    if digest in seen:continue
                    seen.add(digest);unique.append(record)
                by_view[camera]=unique
        records=[r for values in by_view.values() for r in values]
        for index,record in enumerate(records):record['candidate_id']=f'{query.key}_{index:02d}'
        options,summaries,reasons=choose(cad=query.cad is not None,options=True)
        return dict(query=query,last=last,attempts=attempts,records=records,options=options,
            summaries=summaries,reasons=reasons,choose=choose,tracked=last is not None)

    def observe(self,frames,queries,*,episode_id,search_motion=None):
        started=time.perf_counter();frames=list(frames);queries=ordered_queries(queries)
        if not episode_id or not frames or len(frames)>self.config.max_views:
            raise ValueError('Episode and bounded nonempty current frame batch required')
        if len({f.camera for f in frames})!=len(frames) or len({f.capture_id for f in frames})!=1:
            raise ValueError('Distinct cameras from the same capture group are required')
        timestamps=[f.timestamp_s for f in frames if f.timestamp_s is not None]
        if timestamps and len(timestamps)!=len(frames):raise ValueError('Mixed known/unknown sensor timestamps')
        if timestamps and max(timestamps)-min(timestamps)>self.config.max_sync_skew_s:
            raise ValueError('Asynchronous views cannot be fused as one current observation')
        timestamp=max(timestamps) if timestamps else None
        if self.episode_id!=episode_id:self.reset(episode_id)
        if frames[0].capture_id in self.seen_captures:
            raise ValueError('A repeated capture cannot become a fresh tracking observation')
        if timestamp is not None and self.last_timestamp is not None and timestamp<self.last_timestamp:
            raise ValueError('Out-of-order captures cannot update the current track')
        self.seen_captures.add(frames[0].capture_id);self.last_timestamp=timestamp
        self.capture_index+=1
        hypotheses={key:rigid(value) for key,value in (search_motion or {}).items()}
        grids={frame.camera:world_grid(frame) for frame in frames}
        targets={};self._cad_checks={};assignment_audits=[];self._current_frames=frames
        # Prepare all peers before assigning; only then resolve their children.
        pending=list(queries);prepared_queue=[]
        while pending or prepared_queue:
            if not prepared_queue:
                ready=[q for q in pending if q.parent is None or q.parent in targets]
                pending=[q for q in pending if q not in ready]
                for query in ready:
                    last=self._last(query.key,timestamp)
                    if last is not None and last['signature']!=repr(asdict(query)):
                        raise ValueError('Query changed within episode; use a new key or reset the episode')
                    prepared_queue.append(self._prepare_query(query,frames,targets,last,
                        hypotheses.get(query.key),grids))
                if len(prepared_queue)>1:
                    decisions,audit=assign_instances(prepared_queue,self.config)
                    assignment_audits.append(audit)
                    for item in prepared_queue:
                        decision=decisions[item['query'].key];item['selected']=decision.pop('selected')
                        item['assignment']=decision
                        if decision['reason']:item['reasons'].append(decision['reason'])
                else:
                    item=prepared_queue[0]
                    item['selected'],item['summaries'],item['reasons']=item['choose'](cad=item['query'].cad is not None)
                    if self.config.audit_candidate_selection:
                        item['assignment']=dict(final_competition=self._last_selection_competition,
                            selected_candidate_ids=[r.get('candidate_id') for r in (item['selected'] or [])],
                            reason=item['reasons'])
            item=prepared_queue.pop(0);query=item['query'];last=item['last']
            selected=item['selected'];summaries=item['summaries'];reasons=item['reasons']
            attempts=item['attempts'];candidate_records=item['records'];signature=repr(asdict(query))
            # Do not let a high detector score for a geometrically unsuitable
            # object displace a candidate satisfying the requested attributes.
            # If none satisfy them, preserve partial observations explicitly.
            if selected is None:
                status='ambiguous' if any(r in reasons for r in
                    ('ambiguous_current_target_candidates','ambiguous_CAD_consistent_candidates',
                     'ambiguous_joint_instance_assignment')) else 'not_observed'
                observation=TargetObservation(query.key,status,candidates=summaries,reasons=reasons+attempts,
                    candidate_records=candidate_records,assignment=item.get('assignment',{}))
                if last is not None:
                    observation.last_known=dict(capture_id=last['capture'],timestamp_s=last['timestamp'],
                        position_world=last['observation'].geometry['position_world'],
                        current_measurement=False,usable_for_current_control=False)
                targets[query.key]=observation;continue
            points=voxelize(np.concatenate([r['points'] for r in selected]),self.config.voxel_m)
            if self.config.require_cross_view_cad and query.cad is not None:
                # Report a measured single-view centre, not a view-dependent
                # union centre compared against a different camera's mask.
                # Other views certify the SAME CAD pose; all points remain
                # retained for motion tracking and independent diagnostics.
                selected=sorted(selected,key=lambda r:r['cad_verification']['cost'])
            camera=np.linalg.inv(selected[0]['frame'].world_to_camera)[:3,3]
            shape,usable,points=geometry(points,camera,self.config,query.attributes)
            if self.config.require_cross_view_cad and query.cad is not None:
                reference=selected[0]['geometry']
                shape['position_world']=reference['position_world']
                shape['bounds_world']=reference['bounds_world']
                shape['position_reference_camera']=selected[0]['frame'].camera
                shape['cross_view_points_retained']=True
            try:check_extent(shape,query)
            except RuntimeError as error:
                targets[query.key]=TargetObservation(query.key,'rejected',fresh=True,
                    geometry=shape,usable={key:False for key in usable},
                    evidence=[self._summary(r) for r in selected],candidates=summaries,
                    reasons=attempts+[str(error)],points=points,candidate_records=candidate_records)
                continue
            if query.cad is not None:
                best=min(selected,key=lambda r:r['cad_verification']['cost'])
                shape['full_object_pose_world']=best['cad_verification']['object_to_world']
                shape['cad_verification']=dict(accepted=True,source='static_CAD_fitted_to_current_RGBD',
                    best_camera=best['frame'].camera,cost=best['cad_verification']['cost'],
                    model_scale_fixed=True,symmetry_globally_resolved=False,
                    hidden_support_anchor_observed=False)
                shape['visible_surface_position_world']=np.asarray(shape['position_world']).copy()
                pose=np.asarray(shape['full_object_pose_world'])
                shape['cad_origin_world']=pose[:3,3].copy()
                shape['position_is_model_inferred']=query.position_reference!='visible_surface'
                if query.position_reference!='visible_surface':
                    local=np.zeros(3) if query.position_reference=='cad_origin' else np.asarray(query.anchor_object_m)
                    shape['position_world']=pose[:3,:3]@local+pose[:3,3]
                    shape['position_reference']=query.position_reference+'_model_inferred_not_directly_observed'
                    shape['anchor_object_m']=local
            if 'structures' in query.attributes:
                structures=[];structure_audit=[]
                for record in selected:
                    frame=record['frame'];grid,valid=grids[frame.camera]
                    roi=self._roi(record);yy,xx=np.indices(valid.shape)
                    region=valid&(xx>=roi[0])&(xx<=roi[2])&(yy>=roi[1])&(yy<=roi[3])
                    values,audit=protrusions(record['points'],grid[region],
                        np.linalg.inv(frame.world_to_camera)[:3,3],self.config)
                    for value in values:
                        value={key:v for key,v in value.items() if key!='points'}
                        structures.append(dict(value,camera=frame.camera))
                    structure_audit.append(dict(camera=frame.camera,**audit))
                shape['structures']=structures;shape['structure_search']=structure_audit
                usable['structures']=bool(structures)
            if 'motion' in query.attributes:
                if last is None:
                    shape['motion_audit']=dict(accepted=False,reason='no_previous_measured_surface')
                else:
                    motion,audit=surface_motion(last['observation'].points,points,self.config)
                    shape['motion_audit']=audit
                    if motion is not None:shape['motion_world']=motion;usable['motion']=True
            missing=[key for key in query.attributes if not usable.get(key,False)]
            required=all(usable.get(key,False) for key in query.required)
            observation=TargetObservation(query.key,'ok' if not missing else 'partial',required,True,
                shape,usable,evidence=[self._summary(r) for r in selected],candidates=summaries,
                reasons=attempts+(['unavailable_attributes:'+','.join(missing)] if missing else []),points=points,
                views={r['frame'].camera:r for r in selected})
            observation.candidate_records=candidate_records
            observation.assignment=item.get('assignment',{})
            targets[query.key]=observation
            # Store the current measured surface even if a requested motion or
            # plane is unavailable. Never store an unobserved/predicted surface.
            self.history[query.key]=dict(observation=observation,index=self.capture_index,
                timestamp=timestamp,capture=frames[0].capture_id,signature=signature)
        return SceneObservation(episode_id,frames[0].capture_id,targets,dict(
            elapsed_s=time.perf_counter()-started,configuration=asdict(self.config),
            cameras=[f.camera for f in frames],timestamp_s=timestamp,
            synchronization_source='sensor_timestamps' if timestamps else 'caller_declared_capture_group',
            uses_object_truth=False,uses_simulator_identity_masks=False,uses_contacts=False,
            uses_prepared_object_models=any(q.cad is not None for q in queries),
            cad_backend=(self.backend.cad_summary() if hasattr(self.backend,'cad_summary') else None),
            joint_instance_assignment=assignment_audits,
            semantic_task_planning=False,grasp_or_release_authorized=False,
            full_hidden_geometry_reconstructed=False))
