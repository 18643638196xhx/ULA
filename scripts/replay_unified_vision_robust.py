"""Small frozen sensor-only regression; not a new full-task success benchmark."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.unified_vision import RGBDFrame,TargetQuery,VisionConfig,CadLimits
from adapters.unified_vision.worker import UnifiedVisionWorker
from adapters.unified_vision.contracts import serializable
from diagnostics.unified_vision_evaluation import evaluate_target


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--episodes-per-task',type=int,default=2)
    p.add_argument('--verify-cad-color',action='store_true')
    p.add_argument('--body-guided-completion',action='store_true')
    p.add_argument('--tasks',nargs='+',default=['shake_bottle','place_bread_skillet','move_stapler_pad','open_microwave'])
    a=p.parse_args();out=a.output.resolve();source=a.source.resolve()
    if not out.is_relative_to(ROOT) or not source.is_relative_to(ROOT) or a.episodes_per_task<1:
        p.error('Use ULA paths and positive episode count')
    out.mkdir(parents=True,exist_ok=False)
    config=VisionConfig(body_guided_completion=a.body_guided_completion,verify_cad_color=a.verify_cad_color,require_semantic_identity=True,complete_visible_surface=True,cad=CadLimits(max_models=256))
    files=[*sorted((ROOT/'adapters/unified_vision').glob('*.py')),Path(__file__),
           ROOT/'diagnostics/unified_vision_evaluation.py',ROOT/'scripts/run_unified_vision.py']
    hashes={str(f):digest(f) for f in files}
    report=dict(status='running',scope='small_initial_head_sensor_replay_not_full_task_success',
        configuration=asdict(config),source=str(source),rows=[],sources=hashes,
        scoring_feedback_to_inference=False,robot_actions=0)
    def save():
        path=out/'report.json';temp=out/'report.json.tmp'
        temp.write_text(json.dumps(serializable(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
    worker=UnifiedVisionWorker(ROOT,out/'worker',gpu=a.gpu,config=config)
    save()
    try:
        for task in a.tasks:
            old=json.loads((source/'tasks'/task/'report.json').read_text())
            rows=[r for r in old['rows'] if r.get('evaluation')][:a.episodes_per_task]
            for row in rows:
                if any(digest(f)!=h for f,h in hashes.items()):raise RuntimeError('Frozen replay code changed')
                directory=Path(row['directory']);sensor=directory/'rgbd.npz';sensor_hash=digest(sensor)
                frame=RGBDFrame.from_npz(sensor,camera='head',capture_id='initial',depth_unit='mm')
                queries=[TargetQuery(**q) for q in json.loads((directory/'public_queries.json').read_text())]
                start=time.perf_counter()
                scene=worker.observe([frame],queries,episode_id=f'{task}:{row["seed"]}')
                pred_dir=worker.output/f'request_{scene.audit["worker_request"]:05d}'/'prediction'
                locks={str(f):digest(f) for f in pred_dir.iterdir()}
                (pred_dir.parent/'prediction_locks.json').write_text(json.dumps(locks,indent=2)+'\n')
                # FIRST label read for this request occurs after saved prediction.
                with np.load(directory/'evaluation_only_masks.npz') as labels:
                    truth={k:labels[k].copy() for k in labels.files}
                values={}
                for key,target in scene.to_dict()['targets'].items():
                    mask=None
                    if target['evidence']:
                        with np.load(target['evidence'][0]['measured_mask_path']) as f:mask=f['mask'].copy()
                    values[key]=evaluate_target(target,mask,truth[key],frame)
                assert all(digest(f)==h for f,h in locks.items()) and digest(sensor)==sensor_hash
                report['rows'].append(dict(task=task,seed=row['seed'],elapsed_s=time.perf_counter()-start,
                    old_all_localized=row['evaluation']['all_targets_localized'],
                    all_localized=all(v['localization_verified'] for v in values.values()),
                    targets=values,prediction_directory=str(pred_dir),prediction_unchanged=True,sensor_unchanged=True))
                save();print(task,row['seed'],report['rows'][-1]['all_localized'],flush=True)
        report.update(status='finished',sources_unchanged=all(digest(f)==h for f,h in hashes.items()))
    except BaseException as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}');raise
    finally:
        worker.close();report['worker']=worker.summary();save()


if __name__=='__main__':main()
