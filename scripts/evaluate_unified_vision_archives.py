#!/usr/bin/env python3
"""Frozen initial-head RGB-D benchmark. ALL predictions precede label reads.

No simulator, robot action, policy inference or task scoring. Missing archived
frames and missing independent labels are disclosed, never fabricated.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.unified_vision import RGBDFrame,TargetQuery,VisionConfig
from adapters.unified_vision.worker import UnifiedVisionWorker
from adapters.unified_vision.contracts import serializable
from diagnostics.unified_vision_evaluation import evaluate_target,summarize_cohort


COHORTS={
    'cup':dict(root='results/cup_initial_randomized_50scenes_seed0_20261101_v2',
        sensor='rgbd.npz',config='cup_pad.json',labels=dict(object='cup',destination='pad'),
        baseline='results/cup_initial_blue_caption_replay50_seed0_20261101_v1',
        baseline_flags=dict(object='cup_output',destination='pad_output')),
    'bottle':dict(root='results/pillbottle_initial_cup_method_randomized_50valid_seed0_20261102_v1',
        sensor='rgbd.npz',config='bottle_pad.json',labels=dict(object='bottle',destination='pad'),
        baseline='results/pillbottle_initial_cup_method_randomized_50valid_seed0_20261102_v1',
        baseline_flags=dict(object='bottle_output',destination='pad_output')),
    'body_part':dict(root='results/microwave_no_extent_randomized_50valid_seed0_20261102_v1',
        sensor='perception/initial_0_head_camera/rgbd.npz',config='body_part.json',
        labels=dict(body=None,part=None),baseline=None),
}


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(serializable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def manifest(names):
    result=[]
    for name in names:
        specification=COHORTS[name]
        directories=sorted((ROOT/specification['root']).glob('episode_*_seed*'))
        if not directories:raise FileNotFoundError('No archive cases for '+name)
        for directory in directories:
            sensor=directory/specification['sensor']
            result.append(dict(cohort=name,seed=int(directory.name.rsplit('seed',1)[1]),
                archive_directory=str(directory),sensor_path=str(sensor),sensor_available=sensor.is_file(),
                sensor_sha256=digest(sensor) if sensor.is_file() else None))
    return result


def code_paths():
    return [Path(__file__),ROOT/'diagnostics/unified_vision_evaluation.py',
        ROOT/'scripts/run_unified_vision.py',ROOT/'scripts/run_unified_vision.sh',
        ROOT/'adapters/perception.py',ROOT/'adapters/rgbd_perception.py',
        ROOT/'adapters/cup_initial_audit.py',
        ROOT/'tra_pipe/RoboTwin/traditional_place_empty_cup/perception.py',
        *sorted((ROOT/'adapters/unified_vision').glob('*.py')),
        *sorted((ROOT/'configs/unified_vision').glob('*.json'))]


def load_mask(target,directory):
    evidence=[item for item in target.get('evidence',[]) if item['camera']=='head']
    if not evidence:return None
    if len(evidence)!=1:raise ValueError('Benchmark expects one initial head evidence per target')
    path=Path(evidence[0]['measured_mask_path']).resolve()
    if not path.is_relative_to(Path(directory).resolve()):raise ValueError('Evidence escaped frozen prediction directory')
    with np.load(path,allow_pickle=False) as raw:return raw['mask'].copy()


def baseline_targets(row,frame,truth,specification):
    if not specification['baseline']:return None,None
    directory=ROOT/specification['baseline']/Path(row['archive_directory']).name
    sensor=directory/'rgbd.npz'
    if not sensor.exists() or digest(sensor)!=row['sensor_sha256']:
        return None,'baseline_input_missing_or_not_byte_identical'
    if not (directory/'prediction.npz').exists():return None,'baseline_prediction_missing'
    original=json.loads((directory/'result.json').read_text())
    result={}
    with np.load(directory/'prediction.npz',allow_pickle=False) as raw:
        for key,flag in specification['baseline_flags'].items():
            available=bool(original.get(flag))
            mask_key='cup_mask' if key=='object' else 'pad_mask'
            mask=raw[mask_key].copy() if available else None
            position=(np.asarray(raw['cup_bounds_world']).mean(0) if key=='object'
                      else np.asarray(raw['coaster_world_center']))
            target=dict(required_available=available,fresh=available,status='legacy_output' if available else 'legacy_no_output',
                usable=dict(position=available),geometry=dict(position_world=position.tolist()))
            result[key]=evaluate_target(target,mask,truth.get(key),frame)
    return result,None


def tiles(output,rows):
    from PIL import Image,ImageDraw
    for cohort in sorted({row['cohort'] for row in rows}):
        selected=[row for row in rows if row['cohort']==cohort]
        for offset in range(0,len(selected),9):
            canvas=Image.new('RGB',(960,792),'#eeeeee')
            for index,row in enumerate(selected[offset:offset+9]):
                x,y=(index%3)*320,(index//3)*264
                path=row.get('audit_image')
                if path:
                    with Image.open(path) as tile:canvas.paste(tile.convert('RGB').resize((320,264)),(x,y))
                else:ImageDraw.Draw(canvas).text((x+6,y+8),f"seed {row['seed']}: no archived RGB-D",fill='black')
            canvas.save(output/f'{cohort}_audit_sheet_{offset//9+1:02d}.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--cohorts',nargs='+',choices=tuple(COHORTS),default=list(COHORTS))
    parser.add_argument('--known-cad',action='store_true',help='Optional bottle CAD ablation; requires --cohorts bottle')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if len(set(args.cohorts))!=len(args.cohorts):parser.error('Duplicate cohorts')
    if args.known_cad and args.cohorts!=['bottle']:parser.error('--known-cad currently supplies only the bottle asset; use --cohorts bottle')
    output=args.output.resolve()
    if not output.is_relative_to(ROOT):parser.error('Outputs must remain under ULA')
    output.mkdir(parents=True,exist_ok=False)
    sources=code_paths()
    if args.known_cad:
        sources += [ROOT/'tra_pipe/RoboTwin/third_party/FoundationPose'/name for name in
                    ('estimater.py','Utils.py')]
    before={str(path.relative_to(ROOT)):digest(path) for path in sources}
    for path in sources:
        destination=output/'frozen_code'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    assets={}
    def unchanged():
        return (before=={str(path.relative_to(ROOT)):digest(path) for path in sources} and
                all(digest(path)==value['sha256'] for path,value in assets.items()))
    rows=manifest(args.cohorts)
    configuration=VisionConfig()
    queries={name:[TargetQuery(**value) for value in json.loads(
        (ROOT/'configs/unified_vision'/('bottle_pad_cad.json' if args.known_cad else COHORTS[name]['config'])
        ).read_text())['queries']] for name in args.cohorts}
    for values in queries.values():
        for query in values:
            if query.cad is None:continue
            path=Path(query.cad.mesh_path)
            if not path.is_absolute():path=ROOT/path
            path=path.resolve()
            if not path.is_relative_to(ROOT):raise ValueError('Benchmark CAD asset must be under ULA for freezing')
            destination=output/'frozen_assets'/path.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
            assets[str(path)]=dict(sha256=digest(path),frozen_copy=str(destination),scale=query.cad.scale)
    freeze=dict(code_sha256=before,geometry_config=asdict(configuration),
        queries={name:[asdict(query) for query in values] for name,values in queries.items()},
        static_CAD_assets=assets,
        implementation_version='optional_CAD_selection_v1' if args.known_cad else 'shared_evidence_migration_v2',
        observe_signature_changed=False,output_schema_changed=False,
        optional_query_fields_added_since_v1=['hints','cad'],old_queries_remain_supported=True,
        thresholds_tuned_during_run=False,
        inference_scope='initial_head_RGBD_only_no_temporal_or_multiview_accuracy_claim',
        identity_rule='prediction_precision_at_least_0.8_and_at_least_20_intersecting_pixels',
        strict_mask_rule='IoU_at_least_0.5_reported_separately',
        position_rule='visible_surface_bounds_centre_error_at_most_0.02m_AND_identity_AND_required_output',
        missing_sensor_inputs_are_not_method_failures=True,
        absent_labels_do_not_imply_correct_predictions=True,
        dataset_selection='all_archived_episode_directories_no_filter_on_old_success',
        all_predictions_persisted_and_worker_closed_before_any_label_read=True)
    dump(output/'freeze.json',freeze);dump(output/'input_manifest.json',rows)
    report=dict(run_status='inference',freeze=str(output/'freeze.json'),rows=rows,
        robot_actions_executed=False,simulator_constructed=False,task_success_evaluated=False,
        sources_before=before,cohorts={},source_warning='cup archive has 42 valid frames; microwave archive has no independent labels')
    def save():dump(output/'report.json',report)
    worker=UnifiedVisionWorker(ROOT,output/'worker',gpu=args.gpu,config=configuration)
    save();prediction_locks={};total=sum(row['sensor_available'] for row in rows);done=0
    try:
        for row in rows:
            if not unchanged():raise RuntimeError('Frozen source/config changed; stop without retuning')
            if not row['sensor_available']:
                row['inference_status']='no_archived_sensor';continue
            if digest(row['sensor_path'])!=row['sensor_sha256']:raise RuntimeError('Archived sensor changed')
            frame=RGBDFrame.from_npz(row['sensor_path'],camera='head',capture_id='initial',depth_unit='mm')
            started=time.perf_counter()
            try:
                value=worker.observe([frame],queries[row['cohort']],
                    episode_id=row['cohort']+':'+Path(row['archive_directory']).name)
                request=value.audit['worker_request']
                directory=worker.output/f'request_{request:05d}'/'prediction'
                row.update(inference_status='ok',prediction_directory=str(directory),
                    inference_wall_s=time.perf_counter()-started,
                    inference_engine_wall_s=value.audit['elapsed_s'],worker_request=request,
                    first_worker_request=request==1,
                    outputs={key:dict(status=target.status,required_available=target.required_available,
                        usable=target.usable) for key,target in value.targets.items()})
                for path in directory.iterdir():prediction_locks[str(path)]=digest(path)
            except RuntimeError as error:
                row.update(inference_status='error',inference_error=str(error),
                           inference_wall_s=time.perf_counter()-started)
                # No hidden reload. An individual valid protocol error can be
                # counted and followed by the next frame; a dead worker aborts.
                if worker.process is None or worker.process.poll() is not None:raise
            done+=1;report.update(completed_predictions=done,total_sensor_frames=total)
            save();print(f'[unified-vision] predicted={done}/{total} cohort={row["cohort"]} seed={row["seed"]}',flush=True)
        worker.close()
        report.update(run_status='predictions_frozen',worker=worker.summary())
        if not unchanged():raise RuntimeError('Sources changed before evaluation')
        dump(output/'prediction_locks.json',prediction_locks);save()
        # STRICT PHASE BOUNDARY: labels and legacy prediction results have not
        # been opened above. The vision worker is now closed and cannot retry.
        report['run_status']='evaluating';save()
        from adapters.cup_initial_audit import overlay
        for row in rows:
            if not row['sensor_available']:continue
            spec=COHORTS[row['cohort']]
            if row.get('prediction_directory'):
                directory=Path(row['prediction_directory'])
                for path in directory.iterdir():
                    if digest(path)!=prediction_locks[str(path)]:raise RuntimeError('Frozen prediction changed')
                prediction=json.loads((directory/'prediction.json').read_text())
            else:
                directory=output/'absent_prediction'
                prediction=dict(targets={key:{} for key in spec['labels']})
            frame=RGBDFrame.from_npz(row['sensor_path'],camera='head',capture_id='initial',depth_unit='mm')
            label_file=Path(row['archive_directory'])/'evaluation_only_masks.npz'
            truth={key:None for key in spec['labels']}
            if label_file.exists():
                with np.load(label_file,allow_pickle=False) as raw:
                    truth={key:raw[value].copy() if value is not None and value in raw else None
                           for key,value in spec['labels'].items()}
                row['independent_label_sha256']=digest(label_file)
            masks={key:load_mask(target,directory) for key,target in prediction['targets'].items()}
            row['unified']={key:evaluate_target(target,masks[key],truth.get(key),frame)
                            for key,target in prediction['targets'].items()}
            baseline,error=baseline_targets(row,frame,truth,spec)
            if baseline is not None:row['baseline']=baseline
            if error:row['baseline_error']=error
            keys=list(spec['labels']);image_dir=output/'audit'/row['cohort'];image_dir.mkdir(parents=True,exist_ok=True)
            image=image_dir/f'seed{row["seed"]:05d}.png'
            def text(key):
                item=row['unified'][key]
                if not item['output']:return 'no_output'
                if not item['independent_label_available']:return 'UNVERIFIED'
                return 'match' if item['identity_match'] else 'wrong_region'
            overlay(frame.rgb,*[masks[key] if masks[key] is not None else np.zeros(frame.depth.shape,bool)
                for key in keys],f'{row["cohort"]} seed {row["seed"]} | '+
                ' '.join(key+':'+text(key) for key in keys),image,
                truth_masks=[value for value in truth.values() if value is not None])
            row['audit_image']=str(image)
        report['cohorts']={name:summarize_cohort([row for row in rows if row['cohort']==name],list(COHORTS[name]['labels']))
                           for name in args.cohorts}
        tiles(output,rows)
        after={str(path.relative_to(ROOT)):digest(path) for path in sources}
        report.update(run_status='finished',sources_after=after,sources_unchanged=before==after,
            static_CAD_assets_unchanged=all(digest(path)==value['sha256'] for path,value in assets.items()),
            predictions_unchanged=all(digest(path)==value for path,value in prediction_locks.items()),
            sensor_inputs_unchanged=all(not row['sensor_available'] or digest(row['sensor_path'])==row['sensor_sha256'] for row in rows))
        save()
        if (not report['sources_unchanged'] or not report['predictions_unchanged'] or
            not report['static_CAD_assets_unchanged']):raise RuntimeError('Freeze validation failed')
    except BaseException as error:
        report.update(run_status='error',error=f'{type(error).__name__}: {error}');save();raise
    finally:
        worker.close();report['worker']=worker.summary();save()


if __name__=='__main__':main()
