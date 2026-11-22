"""Fixed-candidate control vs frustum-prior selection, no detection or refit."""
import argparse,json,hashlib,sys,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.unified_vision import RGBDFrame,TargetQuery,VisionConfig
from adapters.unified_vision.contracts import MaskCandidate,serializable
from adapters.unified_vision.assignment import assign_instances
from adapters.fixed_candidate_selection_v1 import projection_visibility,frustum_cost
spec=importlib.util.spec_from_file_location('frustum_scoring',ROOT/'scripts/replay_vision_evidence_pair.py')
scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
TASKS=('open_microwave','open_laptop','rotate_qrcode','hanging_mug')


def select(candidates,queries,frame,enabled):
    items=[];lookup={};audits={};config=VisionConfig(audit_candidate_selection=True)
    for query in queries:
        values=[]
        for c in candidates[query.key]:
            cost=frustum_cost(c['original_cost'],c['frustum']) if enabled else c['original_cost']
            record=dict(frame=frame,mask=MaskCandidate(c['mask'],np.array(c['summary']['box_xyxy']),
                c['summary']['semantic_score'],c['summary']['prompt']),points=c['points'],
                geometry=dict(position_world=c['summary']['position_world']),
                cad_verification=dict(accepted=True,cost=cost,object_to_world=c['pose']),
                cad_render_depth_m=c['render'],candidate_id=c['id'])
            values.append(record);lookup[c['id']]=c
        # Original mask-IoU dedup rule, identical in both arms. Do NOT bundle
        # new containment clustering or largest-mask selection into this test.
        options=[]
        for r in sorted(values,key=lambda r:r['cad_verification']['cost']):
            m=r['mask'].mask
            if any((m&g[0]['mask'].mask).sum()/max(1,(m|g[0]['mask'].mask).sum())>.5 for g in options):continue
            options.append([r])
        items.append(dict(query=query,options=options,tracked=False))
        audits[query.key]=dict(finalists=[dict(id=g[0]['candidate_id'],cost=g[0]['cad_verification']['cost']) for g in options])
    if len(items)>1:
        decisions,global_audit=assign_instances(items,config)
    else:
        item=items[0];options=item['options'];chosen=options[0] if options else None;reason=None
        if len(options)>1 and options[1][0]['cad_verification']['cost']-options[0][0]['cad_verification']['cost']<config.cad.min_cost_gap:
            chosen=None;reason='ambiguous_CAD_consistent_candidates'
        if not options:reason='no_CAD_consistent_current_candidate'
        decisions={item['query'].key:dict(selected=chosen,reason=reason)};global_audit={}
    targets={}
    for q in queries:
        d=decisions[q.key];selected=d.pop('selected');audits[q.key].update(d)
        if not selected:
            targets[q.key]=dict(required_available=False,fresh=True,status='ambiguous' if 'ambiguous' in str(d.get('reason')) else 'not_observed',
                usable=dict(position=False),geometry={},evidence=[])
            continue
        c=lookup[selected[0]['candidate_id']];s=c['summary']
        # Keep the saved candidate centre and mask, no point-cloud completion,
        # averaging, new detections, re-registration or label-based reranking.
        targets[q.key]=dict(required_available=True,fresh=True,status='ok',usable=dict(position=True),
            geometry=dict(position_world=s['position_world'],bounds_world=s['bounds_world'],
                position_reference='saved_candidate_visible_surface_bounds_centre',
                full_object_pose_world=None,pose_resolved=False),
            evidence=[dict(measured_mask_path=c['mask_path'],camera='head',candidate_id=c['id'])])
        audits[q.key]['selected_candidate_id']=c['id']
    return targets,dict(targets=audits,scene_assignment=global_audit)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT):p.error('Use NEW ULA output')
    out.mkdir(exist_ok=False,parents=True)
    source=ROOT/'results/unified_vision50_known_cad_randomized_50per_task_20261110_v2_first3'
    hashes={str(f):scoring.digest(f) for f in [Path(__file__),ROOT/'adapters/fixed_candidate_selection_v1.py',
        ROOT/'adapters/unified_vision/assignment.py',ROOT/'diagnostics/unified_vision_evaluation.py',
        ROOT/'scripts/replay_vision_evidence_pair.py']}
    report=dict(status='running',scope='200_fixed_retained_CAD_candidates_not_robot_task_success',
        weight=2.,min_cost_gap=.15,rows=[],planned=200,sources=hashes,
        single_change='cost=archived_geometry_cost+2*(1-projected_hull_in_frame_fraction)',
        caveats=['Convex hull approximates silhouette; clipping is a soft prior, not identity proof.',
            'Both arms use identical saved candidates, archived best CAD pose and masks.',
            'Control is reconstructed fixed-candidate selection, not guaranteed identical to archived full pipeline.',
            'No CAD pose hypotheses reranked; no new detection/segmentation or pose validity proof.'])
    input_hashes={};meshes={}
    def save():
        temp=out/'report.json.tmp';temp.write_text(json.dumps(serializable(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(out/'report.json')
    save()
    try:
        for task in TASKS:
            rows=[r for r in scoring.read(source/'tasks'/task/'report.json')['rows'] if r.get('evaluation')]
            for row in rows:
                if any(scoring.digest(f)!=h for f,h in hashes.items()):raise RuntimeError('Frozen selector changed')
                folder=Path(row['directory']);sensor=folder/'rgbd.npz'
                if scoring.digest(sensor)!=row['sensor_sha256']:raise RuntimeError('Sensor mismatch')
                input_hashes[str(sensor)]=row['sensor_sha256']
                for f,h in scoring.read(folder/'prediction_locks.json').items():
                    if scoring.digest(f)!=h:raise RuntimeError('Archived prediction changed')
                    input_hashes[f]=h
                pred=scoring.read(Path(row['prediction_directory'])/'prediction.json')
                frame=RGBDFrame.from_npz(sensor,camera='head',capture_id='initial',depth_unit='mm')
                queries=[TargetQuery(**q) for q in scoring.read(folder/'public_queries.json')];candidates={}
                candidate_audit={}
                for query in queries:
                    values=[];seen=set()
                    for group in pred['targets'][query.key]['candidates']:
                        for c in group['candidates']:
                            v=c.get('cad_verification') or {};path=c.get('measured_mask_path')
                            if not v.get('accepted') or path in seen:continue
                            seen.add(path)
                            with np.load(path) as f:
                                mask=f['mask'].astype(bool);points=f['points_world'].copy()
                                render=f['cad_render_depth_m'].copy() if 'cad_render_depth_m' in f else None
                            model=query.cad;mk=repr(model)
                            if mk not in meshes:
                                mp=Path(model.mesh_path);mp=mp if mp.is_absolute() else ROOT/mp
                                mesh=trimesh.load(str(mp),force='scene');mesh=trimesh.util.concatenate(tuple(mesh.dump()))
                                mesh.apply_scale(model.scale)
                                if model.mesh_to_object is not None:mesh.apply_transform(model.mesh_to_object)
                                meshes[mk]=np.array(mesh.vertices,copy=True);input_hashes[str(mp)]=scoring.digest(mp)
                            vis=projection_visibility(meshes[mk],v['object_to_world'],frame.intrinsic,frame.world_to_camera,frame.depth.shape)
                            values.append(dict(id=c['candidate_id'],mask=mask,points=points,render=render,
                                pose=v['object_to_world'],summary=c,original_cost=v['cost'],frustum=vis,mask_path=path))
                    candidates[query.key]=values
                    candidate_audit[query.key]=[dict(id=c['id'],geometric_cost=c['original_cost'],frustum=c['frustum'],
                        new_cost=frustum_cost(c['original_cost'],c['frustum'])) for c in values]
                saved={};metric={};caseout=out/f'{task}_seed{row["seed"]:03d}';caseout.mkdir()
                for variant,enabled in [('control',False),('frustum',True)]:
                    targets,audit=select(candidates,queries,frame,enabled)
                    file=caseout/f'{variant}.json';file.write_text(json.dumps(serializable(dict(targets=targets,audit=audit,candidates=candidate_audit)),indent=2)+'\n')
                    saved[variant]=dict(targets=targets,path=str(file),sha256=scoring.digest(file))
                # First independent label read only AFTER both selections lock.
                with np.load(folder/'evaluation_only_masks.npz') as f:truth={k:f[k].copy() for k in f.files}
                groups={}
                for item in scoring.read(folder/'known_model_provenance.json'):groups.setdefault(item['equivalence_group'],[]).append(item['key'])
                for variant,item in saved.items():
                    values=scoring.score_prediction(item['targets'],truth,frame,groups)
                    metric[variant]=dict(all_localized=all(v['localization_verified'] for v in values.values()),targets=values,prediction=item['path'])
                    assert scoring.digest(item['path'])==item['sha256']
                report['rows'].append(dict(task=task,seed=row['seed'],variants=metric,
                    archived_all_localized=row['evaluation']['all_targets_localized']))
                taskrows=[r for r in report['rows'] if r['task']==task]
                report.setdefault('tasks',{})[task]=dict(completed=len(taskrows),
                    archived=sum(r['archived_all_localized'] for r in taskrows),
                    control=sum(r['variants']['control']['all_localized'] for r in taskrows),
                    frustum=sum(r['variants']['frustum']['all_localized'] for r in taskrows),
                    recovered=sum(not r['variants']['control']['all_localized'] and r['variants']['frustum']['all_localized'] for r in taskrows),
                    regressed=sum(r['variants']['control']['all_localized'] and not r['variants']['frustum']['all_localized'] for r in taskrows))
                save()
            print(task,report['tasks'][task],flush=True)
        report.update(status='finished',sources_unchanged=all(scoring.digest(f)==h for f,h in hashes.items()),
            inputs_unchanged=all(scoring.digest(f)==h for f,h in input_hashes.items()))
        (out/'input_hashes.json').write_text(json.dumps(input_hashes,indent=2)+'\n')
    except BaseException as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}');raise
    finally:save()

if __name__=='__main__':main()
