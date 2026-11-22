"""Independent pilot using saved RGB-D; never overwrites an existing run."""
import argparse,json,hashlib,sys,time,importlib.util,shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.unified_vision import RGBDFrame,TargetQuery,VisionConfig,CadLimits
from adapters.unified_vision.contracts import serializable
from adapters.cad_prompted_sam_v1.worker import CadPromptedSamWorker
spec=importlib.util.spec_from_file_location('cad_prompt_pair_scoring',ROOT/'scripts/replay_vision_evidence_pair.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2);p.add_argument('--cases',nargs='+',default=['open_microwave:3','open_laptop:33'])
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT):p.error('Use new ULA output directory')
    out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'results/unified_vision50_known_cad_randomized_50per_task_20261110_v2_first3'
    protected=[*sorted((ROOT/'adapters/unified_vision').glob('*.py')),ROOT/'scripts/run_unified_vision.py',
        ROOT/'scripts/replay_vision_evidence_pair.py']
    additions=[*sorted((ROOT/'adapters/cad_prompted_sam_v1').glob('*.py')),Path(__file__),ROOT/'scripts/run_cad_prompted_sam_v1.py']
    hashes={str(f):helper.digest(f) for f in protected+additions}
    config=VisionConfig(require_semantic_identity=True,complete_visible_surface=True,verify_cad_color=True,
        audit_candidate_selection=True,cad=CadLimits(max_models=256))
    report=dict(status='running',scope='CAD_prompted_SAM_pilot_not_full_task_success',rows=[],planned=len(a.cases),
        configuration=asdict(config),base_refined_candidate_budget=12,extra_mask_budget=6,
        sources=hashes,protected_files=[str(f) for f in protected],robot_actions=0,labels_to_inference=False)
    workers={k:CadPromptedSamWorker(ROOT,out/k,gpu=a.gpu,config=config,timeout_s=900,use_cad_prompts=enabled)
        for k,enabled in [('baseline',False),('cad_prompted',True)]}
    def save():
        temp=out/'report.json.tmp';temp.write_text(json.dumps(serializable(report),ensure_ascii=False,indent=2)+'\n');temp.replace(out/'report.json')
    save()
    try:
        for item in a.cases:
            if shutil.disk_usage(out).free<8*1024**3:raise RuntimeError('Less than 8GiB free')
            if any(helper.digest(f)!=h for f,h in hashes.items()):raise RuntimeError('Frozen code changed')
            task,seed=item.rsplit(':',1);seed=int(seed)
            task_report=helper.read(source/'tasks'/task/'report.json')
            row=next(r for r in task_report['rows'] if r['seed']==seed and r.get('evaluation'))
            case=Path(row['directory']);sensor=case/'rgbd.npz';sensor_hash=helper.digest(sensor)
            frame=RGBDFrame.from_npz(sensor,camera='head',capture_id='initial',depth_unit='mm')
            queries=[TargetQuery(**q) for q in helper.read(case/'public_queries.json')]
            rngseed=int(hashlib.sha256(item.encode()).hexdigest()[:8],16);predictions={};locked={};metrics={}
            for variant,w in workers.items():
                started=time.perf_counter();scene=w.observe([frame],queries,episode_id=item,inference_seed=rngseed)
                folder=w.output/f'request_{scene.audit["worker_request"]:05d}'/'prediction'
                locked[variant]={str(f):helper.digest(f) for f in folder.iterdir()}
                (folder.parent/'prediction_locks.json').write_text(json.dumps(locked[variant],indent=2)+'\n')
                predictions[variant]=scene.to_dict()['targets'];metrics[variant]=dict(elapsed_s=time.perf_counter()-started,prediction_directory=str(folder))
            # Both predictions persisted before any label/provenance read.
            with np.load(case/'evaluation_only_masks.npz') as f:truth={k:f[k].copy() for k in f.files}
            groups={}
            for x in helper.read(case/'known_model_provenance.json'):groups.setdefault(x['equivalence_group'],[]).append(x['key'])
            for variant in workers:
                values=helper.score_prediction(predictions[variant],truth,frame,groups)
                metrics[variant].update(targets=values,all_localized=all(v['localization_verified'] for v in values.values()))
                assert all(helper.digest(f)==h for f,h in locked[variant].items())
            assert helper.digest(sensor)==sensor_hash
            report['rows'].append(dict(task=task,seed=seed,variants=metrics,sensor_sha256=sensor_hash,predictions_unchanged=True))
            save();print(item,{k:v['all_localized'] for k,v in metrics.items()},flush=True)
        report.update(status='finished',sources_unchanged=all(helper.digest(f)==h for f,h in hashes.items()))
    except BaseException as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}');raise
    finally:
        for w in workers.values():w.close()
        report['workers']={k:w.summary() for k,w in workers.items()};save()

if __name__=='__main__':main()
