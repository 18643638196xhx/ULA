"""Paired archived RGB-D evaluation, no simulation or robot actions."""
import argparse,json,hashlib,time,sys,shutil
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.unified_vision import RGBDFrame,TargetQuery,VisionConfig,CadLimits
from adapters.unified_vision.worker import UnifiedVisionWorker
from adapters.unified_vision.contracts import serializable
from diagnostics.unified_vision_evaluation import evaluate_target

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def score_prediction(targets,truth,frame,groups):
    masks={}
    for k,t in targets.items():
        masks[k]=None
        if t['evidence']:
            with np.load(t['evidence'][0]['measured_mask_path']) as f:masks[k]=f['mask'].copy()
    values={}
    for group in groups.values():
        scores={(a,b):evaluate_target(targets[a],masks[a],truth[b],frame) for a in group for b in group}
        quality=np.array([[10*bool(scores[a,b]['localization_verified'])+bool(scores[a,b]['identity_match'])+
            .01*scores[a,b]['mask_metrics']['target_precision'] for b in group] for a in group])
        left,right=linear_sum_assignment(-quality)
        for i,j in zip(left,right):values[group[i]]=dict(scores[group[i],group[j]],matched_reference=group[j])
    return values

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=['open_microwave','open_laptop','rotate_qrcode','pick_dual_bottles'])
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=50)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT) or not 1<=a.episodes<=50:p.error('Use ULA output and 1..50 scenes')
    out.mkdir(exist_ok=False,parents=True)
    source=ROOT/'results/unified_vision50_known_cad_randomized_50per_task_20261110_v2_first3'
    task=read(source/'tasks'/a.task/'report.json');rows=[r for r in task['rows'] if r.get('evaluation')][:a.episodes]
    base=VisionConfig(require_semantic_identity=True,complete_visible_surface=True,verify_cad_color=True,
        audit_candidate_selection=True,cad=CadLimits(max_models=256))
    configs={'baseline':base,'evidence':replace(base,evidence_ranked_cad=True,verify_scene_structure=True)}
    files=[*sorted((ROOT/'adapters/unified_vision').glob('*.py')),Path(__file__),ROOT/'scripts/run_unified_vision.py',
        ROOT/'diagnostics/unified_vision_evaluation.py']
    hashes={str(f):digest(f) for f in files}
    snap=out/'frozen_code'
    for f in files:
        dest=snap/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    report=dict(status='running',scope='paired_archived_initial_head_RGBD_not_task_success',task=a.task,
        planned=len(rows),source=str(source),configurations={k:asdict(v) for k,v in configs.items()},
        rows=[],sources=hashes,labels_to_inference=False,robot_actions=0)
    workers={k:UnifiedVisionWorker(ROOT,out/k,gpu=a.gpu,config=v,timeout_s=600) for k,v in configs.items()}
    def save():
        tmp=out/'report.json.tmp';tmp.write_text(json.dumps(serializable(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(out/'report.json')
    save()
    try:
        for row in rows:
            if shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('Less than 8 GiB free; stopping without deleting data')
            if any(digest(f)!=h for f,h in hashes.items()):raise RuntimeError('Frozen code changed')
            case=Path(row['directory']);sensor=case/'rgbd.npz';queriesfile=case/'public_queries.json'
            inputs={str(f):digest(f) for f in (sensor,queriesfile)}
            frame=RGBDFrame.from_npz(sensor,camera='head',capture_id='initial',depth_unit='mm')
            queries=[TargetQuery(**q) for q in read(queriesfile)];predictions={};locks={};metrics={}
            seed=int(hashlib.sha256(f'{a.task}:{row["seed"]}'.encode()).hexdigest()[:8],16)
            for variant,worker in workers.items():
                started=time.perf_counter();scene=worker.observe([frame],queries,episode_id=f'{a.task}:{row["seed"]}',inference_seed=seed)
                directory=worker.output/f'request_{scene.audit["worker_request"]:05d}'/'prediction'
                locks[variant]={str(f):digest(f) for f in directory.iterdir()}
                (directory.parent/'prediction_locks.json').write_text(json.dumps(locks[variant],indent=2)+'\n')
                predictions[variant]=scene.to_dict()['targets'];metrics[variant]=dict(prediction_directory=str(directory),elapsed_s=time.perf_counter()-started)
            # Neither worker receives labels or equivalence provenance. FIRST
            # label read occurs only after BOTH variants' predictions are locked.
            with np.load(case/'evaluation_only_masks.npz') as f:truth={k:f[k].copy() for k in f.files}
            groups={}
            for item in read(case/'known_model_provenance.json'):groups.setdefault(item['equivalence_group'],[]).append(item['key'])
            for variant in workers:
                values=score_prediction(predictions[variant],truth,frame,groups)
                metrics[variant].update(targets=values,all_localized=all(v['localization_verified'] for v in values.values()))
                assert all(digest(f)==h for f,h in locks[variant].items())
            assert all(digest(f)==h for f,h in inputs.items())
            report['rows'].append(dict(seed=row['seed'],inference_seed=seed,variants=metrics,
                archived_all_localized=row['evaluation']['all_targets_localized'],input_hashes=inputs,
                predictions_unchanged=True,sensor_unchanged=True))
            report['summary']={k:sum(r['variants'][k]['all_localized'] for r in report['rows']) for k in workers}
            report['summary'].update(recovered=sum(not r['variants']['baseline']['all_localized'] and r['variants']['evidence']['all_localized'] for r in report['rows']),
                regressed=sum(r['variants']['baseline']['all_localized'] and not r['variants']['evidence']['all_localized'] for r in report['rows']))
            save();print(a.task,row['seed'],report['summary'],flush=True)
        report.update(status='finished',sources_unchanged=all(digest(f)==h for f,h in hashes.items()))
    except BaseException as e:
        report.update(status='error',error=f'{type(e).__name__}: {e}');raise
    finally:
        for worker in workers.values():worker.close()
        report['workers']={k:w.summary() for k,w in workers.items()};save()

if __name__=='__main__':main()
