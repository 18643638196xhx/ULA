"""All 50 tasks, full perception replay on identical archived RGB-D scenes."""
import argparse,json,sys,time,shutil,subprocess,importlib.util
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.unified_vision import RGBDFrame,TargetQuery,VisionConfig,CadLimits
from adapters.unified_vision.contracts import serializable
from adapters.frustum_selection_v1 import FrustumSelectionWorker
from diagnostics.vision50_catalog import TASK_TARGETS
spec=importlib.util.spec_from_file_location('frustum50_scoring',ROOT/'scripts/replay_vision_evidence_pair.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def dump(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(serializable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def check_freeze(out):
    for f,h in helpers.read(out/'freeze.json')['source_sha256'].items():
        if helpers.digest(f)!=h:raise RuntimeError('Frozen source changed: '+f)


def worker(a):
    task=a.worker_task;out=a.output/'tasks'/task;out.mkdir(parents=True,exist_ok=False)
    old=helpers.read(a.source/'tasks'/task/'report.json')
    rows=[r for r in old['rows'] if r.get('evaluation')]
    if len(rows)!=50:raise ValueError('Expected all 50 archived valid scenes')
    config=VisionConfig(cad=CadLimits(max_models=256))
    vision=FrustumSelectionWorker(ROOT,out/'vision_worker',gpu=a.worker_gpu,config=config,timeout_s=900)
    report=dict(task=task,status='running',planned=50,rows=[],gpu=a.worker_gpu,
        source=str(a.source),scope='full_perception_replay_same_archived_initial_RGBD',
        initialization_rejected_seeds=old['summary']['initialization_rejected_seeds'],
        configurations=asdict(config),frustum_weight=2.,robot_actions=0)
    def save():
        rs=report['rows'];good=sum(r['evaluation']['all_targets_localized'] for r in rs)
        per={}
        for r in rs:
            for key,v in r['evaluation']['targets'].items():
                c=per.setdefault(key,dict(scenes=0,localized=0,no_output=0,wrong_region=0,position_failed=0))
                c['scenes']+=1;c['localized']+=bool(v['localization_verified']);c['no_output']+=not v['output']
                c['wrong_region']+=bool(v['output'] and not v['identity_match'])
                c['position_failed']+=bool(v['output'] and v['identity_match'] and not v['localization_verified'])
        report['summary']=dict(evaluated_scenes=len(rs),all_targets_localized=good,
            localization_rate=good/len(rs) if rs else None,per_target=per,
            archived_passed=sum(r['archived_all_localized'] for r in rs),
            recovered=sum(not r['archived_all_localized'] and r['evaluation']['all_targets_localized'] for r in rs),
            regressed=sum(r['archived_all_localized'] and not r['evaluation']['all_targets_localized'] for r in rs))
        dump(out/'report.json',report)
    save();inputs={}
    try:
        for row in rows:
            check_freeze(a.output)
            if shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('Less than 8GiB free, stopped without deleting data')
            case=Path(row['directory']);sensor=case/'rgbd.npz';queryfile=case/'public_queries.json'
            if helpers.digest(sensor)!=row['sensor_sha256']:raise RuntimeError('Archived sensor hash mismatch')
            inputs[str(sensor)]=row['sensor_sha256'];inputs[str(queryfile)]=helpers.digest(queryfile)
            queries=[TargetQuery(**q) for q in helpers.read(queryfile)]
            for q in queries:
                if q.cad:
                    p=Path(q.cad.mesh_path);p=p if p.is_absolute() else ROOT/p
                    inputs[str(p)]=helpers.digest(p)
            frame=RGBDFrame.from_npz(sensor,camera='head',capture_id='initial',depth_unit='mm')
            start=time.perf_counter()
            scene=vision.observe([frame],queries,episode_id=f'{task}:{row["seed"]}')
            pred=vision.output/f'request_{scene.audit["worker_request"]:05d}'/'prediction'
            locks={str(p):helpers.digest(p) for p in pred.iterdir()};dump(pred.parent/'prediction_locks.json',locks)
            # First label/provenance reads AFTER persisted and locked prediction.
            with np.load(case/'evaluation_only_masks.npz') as f:truth={k:f[k].copy() for k in f.files}
            groups={}
            for item in helpers.read(case/'known_model_provenance.json'):groups.setdefault(item['equivalence_group'],[]).append(item['key'])
            values=helpers.score_prediction(scene.to_dict()['targets'],truth,frame,groups)
            assert all(helpers.digest(p)==h for p,h in locks.items())
            assert helpers.digest(sensor)==inputs[str(sensor)]
            report['rows'].append(dict(seed=row['seed'],directory=str(case),sensor_sha256=inputs[str(sensor)],
                prediction_directory=str(pred),inference_wall_s=time.perf_counter()-start,
                predictions_unchanged=True,archived_all_localized=row['evaluation']['all_targets_localized'],
                evaluation=dict(targets=values,all_targets_localized=all(v['localization_verified'] for v in values.values()),feeds_back_to_inference=False)))
            save();print(task,row['seed'],report['summary']['all_targets_localized'],len(report['rows']),flush=True)
        report.update(status='finished',inputs_unchanged=all(helpers.digest(p)==h for p,h in inputs.items()))
        dump(out/'input_hashes.json',inputs)
    except BaseException as e:
        report.update(status='error',error=f'{type(e).__name__}: {e}');raise
    finally:
        vision.close();report['worker']=vision.summary();save()


def coordinate(a):
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'logs').mkdir()
    files=[Path(__file__),ROOT/'scripts/run_frustum_selection_v1.py',ROOT/'adapters/frustum_selection_v1.py',
        ROOT/'adapters/fixed_candidate_selection_v1.py',ROOT/'scripts/run_unified_vision.py',
        ROOT/'scripts/replay_vision_evidence_pair.py',ROOT/'adapters/perception.py',
        ROOT/'diagnostics/unified_vision_evaluation.py',ROOT/'adapters/cup_initial_audit.py',
        ROOT/'diagnostics/vision50_catalog.py',*sorted((ROOT/'adapters/unified_vision').glob('*.py'))]
    hashes={str(f):helpers.digest(f) for f in files}
    for f in files:
        dest=a.output/'frozen_code'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    config=VisionConfig(cad=CadLimits(max_models=256))
    dump(a.output/'freeze.json',dict(source_sha256=hashes,source=str(a.source),tasks=list(TASK_TARGETS),
        scenes_per_task=50,frustum_weight=2.,vision_configuration=asdict(config),
        scope='full_detection_segmentation_CAD_registration_selection_on_2500_same_archived_RGBD',
        not_fixed_candidate_reranking=True,not_new_scene_generation=True,
        labels_to_inference=False,robot_actions=0,
        caveat='Current exact-surface and engine updates also differ from old archive; historical comparison not single-factor ablation'))
    gpus=[int(x) for x in a.gpus.split(',')];pending=list(TASK_TARGETS);active={};done={}
    report=dict(status='running',planned_tasks=50,planned_scenes=2500,tasks={},gpus=gpus)
    def save():
        tasks={}
        for name in TASK_TARGETS:
            p=a.output/'tasks'/name/'report.json'
            tasks[name]=helpers.read(p) if p.exists() else dict(status='pending')
            # Keep coordinator JSON small; per-frame detail remains task-owned.
            tasks[name]={k:v for k,v in tasks[name].items() if k in ('status','summary','error','gpu')}
            if name in done and done[name]!=0:tasks[name]['process_exit_code']=done[name]
        report.update(tasks=tasks,completed_tasks=sum(t['status']=='finished' for t in tasks.values()),
            evaluated_scenes=sum(t.get('summary',{}).get('evaluated_scenes',0) for t in tasks.values()),
            all_targets_localized=sum(t.get('summary',{}).get('all_targets_localized',0) for t in tasks.values()),
            active={str(g):dict(task=v[1],pid=v[0].pid) for g,v in active.items()})
        dump(a.output/'report.json',report)
    save()
    try:
        while pending or active:
            check_freeze(a.output)
            for gpu in gpus:
                if gpu in active:
                    process,name,log=active[gpu]
                    if process.poll() is None:continue
                    log.close();done[name]=process.returncode;del active[gpu]
                if pending:
                    if shutil.disk_usage(a.output).free<8*1024**3:raise RuntimeError('Insufficient disk space')
                    name=pending.pop(0);log=(a.output/'logs'/f'{name}.log').open('x')
                    command=[sys.executable,str(Path(__file__).resolve()),'--worker-task',name,'--worker-gpu',str(gpu),
                        '--source',str(a.source),'--output',str(a.output)]
                    proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    active[gpu]=(proc,name,log)
            save()
            if active:time.sleep(2)
        report.update(status='finished' if all(code==0 for code in done.values()) else 'finished_with_errors',sources_unchanged=True);save()
    except BaseException as e:
        # Coordinator failure must not silently leave task processes behind.
        for process,name,log in active.values():
            if process.poll() is None:process.terminate()
        report.update(status='error',error=f'{type(e).__name__}: {e}');save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'results/unified_vision50_known_cad_randomized_50per_task_20261110_v2_first3')
    p.add_argument('--gpus',default='0,1,2');p.add_argument('--worker-task',choices=list(TASK_TARGETS));p.add_argument('--worker-gpu',type=int,default=0)
    a=p.parse_args();a.output=a.output.resolve();a.source=a.source.resolve()
    if a.output==ROOT or not a.output.is_relative_to(ROOT) or not a.source.is_relative_to(ROOT):p.error('Use ULA paths')
    g=[int(v) for v in a.gpus.split(',')]
    if not g or len(g)!=len(set(g)):p.error('Distinct GPUs required')
    if a.worker_task:worker(a)
    else:coordinate(a)

if __name__=='__main__':main()
