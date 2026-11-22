#!/usr/bin/env python3
"""50-task known-instance-CAD initial RGB-D localization, no task actions.

Coordinator runs one task process per selected GPU; each task has ONE persistent
vision worker. A fresh output is mandatory. --detach launches a durable local
coordinator; report.json includes every task's live progress and exclusions.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ROBOTWIN=ROOT/'tra_pipe/RoboTwin'
sys.path[:0]=[str(ROOT),str(ROBOTWIN),str(ROBOTWIN/'envs/curobo/src')]
from adapters.unified_vision import RGBDFrame,TargetQuery,VisionConfig,CadLimits
from adapters.unified_vision.worker import UnifiedVisionWorker
from adapters.unified_vision.contracts import serializable
from diagnostics.vision50_catalog import TASK_TARGETS,captions,targets
from diagnostics.vision50_assets import record_creations,AssetPreparer


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(serializable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def source_paths():
    return [Path(__file__),*sorted((ROOT/'adapters/unified_vision').glob('*.py')),
        *sorted((ROOT/'diagnostics').glob('vision50_*.py')),
        ROOT/'diagnostics/unified_vision_evaluation.py',ROOT/'adapters/cup_initial_audit.py',
        ROOT/'scripts/run_unified_vision.py',ROOT/'adapters/perception.py',
        ROBOTWIN/'envs/_base_task.py',ROBOTWIN/'envs/robot/robot.py',
        ROBOTWIN/'envs/camera/camera.py',ROBOTWIN/'envs/utils/create_actor.py',
        ROBOTWIN/'envs/utils/rand_create_actor.py',
        ROBOTWIN/'env_cfg/task_config/demo_randomized.yml',
        ROBOTWIN/'scripts/run_place_empty_cup_vision.py',
        *[ROBOTWIN/'envs'/(name+'.py') for name in TASK_TARGETS]]


def check_freeze(directory):
    frozen=json.loads((Path(directory)/'freeze.json').read_text())
    if any(digest(ROOT/name)!=value for name,value in frozen['source_sha256'].items()):
        raise RuntimeError('Frozen evaluation source changed; no retuning during sweep')


def scene_configuration(task_name,multi_view=False):
    spec=importlib.util.spec_from_file_location('ula_vision50_config',ROBOTWIN/'scripts/run_place_empty_cup_vision.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=module.load_task_args(str(ROBOTWIN/'env_cfg/task_config/demo_randomized.yml'),0,arm='auto')
    config.update(task_name=task_name,eval_mode=True,eval_video_log=False,save_data=False,
        need_plan=False,enable_mplib=False,head_camera_only=True,fast_eval_mode=True)
    config['data_type'].update(mesh_segmentation=False,actor_segmentation=False,pointcloud=False)
    if multi_view:
        config=deepcopy(config);config['head_camera_only']=False
        # Fixed calibrated workspace view, identical for every task and seed.
        # No actor location, target box or evaluation label is consulted.
        config['left_embodiment_config']['static_camera_list']=[
            x for x in config['left_embodiment_config']['static_camera_list'] if x['name']=='head_camera']+[
            dict(name='workspace_camera',type='D435',position=[0.,0.,2.2],
                 forward=[0.,0.,-1.],left=[-1.,0.,0.])]
    return config


def rendering_runtime():
    import sapien
    from sapien.pysapien.render import SapienRenderer as NativeRenderer
    device=sapien.Device('cuda:0')
    if not device.can_render():raise RuntimeError('Selected GPU cannot render')
    class Renderer(sapien.SapienRenderer):
        def __init__(self,**kwargs):NativeRenderer.__init__(self,device)
    sapien.SapienRenderer=Renderer


def block_planner_import():
    """Scene-only subprocess: no CuRobo import/JIT, no callable arm planner.

    Robot's existing _linear_gripper_plan retains the original 200-point
    initialization interpolation. No arm path is planned or substituted.
    """
    module=types.ModuleType('envs.robot.planner')
    class ForbiddenPlanner:
        def __init__(self,*args,**kwargs):raise RuntimeError('Arm planning forbidden in initial-vision benchmark')
    module.CuroboPlanner=ForbiddenPlanner;module.MplibPlanner=ForbiddenPlanner
    if 'envs.robot.planner' in sys.modules:raise RuntimeError('Install scene-only planner guard before importing tasks')
    sys.modules['envs.robot.planner']=module


@contextmanager
def no_planners():
    """Disable only unused planner construction in this private scene process.

    Initial joints, gripper opening, settling and the stability check remain.
    Never used in a rollout; the task-action guard also rejects all actions.
    """
    from envs.robot.robot import Robot
    original=Robot.set_planner
    def disabled(robot,scene=None):robot.communication_flag=False
    Robot.set_planner=disabled
    try:yield
    finally:Robot.set_planner=original


def coarse_color(value):
    if value is None:return None
    rgb=np.asarray(value,float)[:3]
    palette={'red':[1,0,0],'green':[0,1,0],'blue':[0,0,1],
        'yellow':[1,1,0],'white':[1,1,1],'black':[0,0,0]}
    color=min(palette,key=lambda key:np.linalg.norm(rgb-palette[key]))
    # Only name a color if it was explicitly supplied very near that palette
    # value. Do not turn arbitrary random RGB into a false hard color prior.
    return color if np.linalg.norm(rgb-palette[color])<.15 else None


def prepare_queries(task,name,creations,preparer):
    public=[];private=[]
    for key,actor in targets(task,name):
        if id(actor) not in creations:raise RuntimeError('No audited CAD creation descriptor for '+key)
        description=creations[id(actor)]
        fraction=.2 if name=='open_laptop' and key=='laptop' else None
        model=preparer.prepare(description,initial_joint_fraction=fraction)
        prompts=captions(description.get('modelname','box'))
        color=coarse_color(description.get('color'))
        if description['kind'] in ('create_box','create_visual_box'):
            noun='square pad' if min(description['half_size'])<.006 else 'block'
            prompts=((color+' ' if color else '')+noun,)
        query=dict(key=key,prompts=list(prompts),attributes=['position','bounds'],
            required=['position','bounds'],cad=model['cad'])
        if color:query['hints']=dict(colors=[color],appearance_required=True)
        public.append(query)
        # Identical model AND appearance instructions are exchangeable, e.g.
        # two visually identical bowls. Evaluation uses one-to-one assignment,
        # so returning the same instance twice can never count as two matches.
        equivalent=json.dumps(dict(cad=model['cad'],prompts=prompts,hints=query.get('hints')),sort_keys=True)
        private.append(dict(key=key,actor=actor,model_audit=model['audit'],equivalence_group=equivalent))
    return public,private


def entity_ids(actor):
    entity=getattr(actor,'actor',actor)
    entities=entity.get_links() if hasattr(entity,'get_links') else [entity]
    ids=[]
    for item in entities:
        value=getattr(item,'per_scene_id',None)
        if value is None:value=getattr(getattr(item,'entity',None),'per_scene_id',None)
        if value is None:raise RuntimeError('Missing independent renderer identity')
        ids.append(int(value))
    return ids


def evaluate_scene(task,private,scene,frame,directory,frames=None):
    from scipy.optimize import linear_sum_assignment
    from diagnostics.unified_vision_evaluation import evaluate_target
    from PIL import Image,ImageDraw
    labels=task.cameras.static_camera_list[task.cameras.static_camera_name.index('head_camera')].get_picture('Segmentation')[...,1].copy()
    actual={item['key']:np.isin(labels,entity_ids(item['actor'])) for item in private}
    frames={frame.camera:frame} if frames is None else {f.camera:f for f in frames}
    actual_by_view={frame.camera:actual}
    for camera_name in frames:
        if camera_name==frame.camera:continue
        camera=task.cameras.static_camera_list[task.cameras.static_camera_name.index(camera_name)]
        lab=camera.get_picture('Segmentation')[...,1].copy()
        actual_by_view[camera_name]={item['key']:np.isin(lab,entity_ids(item['actor'])) for item in private}
        np.savez_compressed(directory/('evaluation_only_masks_'+camera_name+'.npz'),**actual_by_view[camera_name])
    np.savez_compressed(directory/'evaluation_only_masks.npz',**actual)
    masks={};frozen=scene.to_dict()['targets'];results={};groups={}
    for item in private:
        key=item['key'];evidence=frozen[key].get('evidence',[])
        masks[key]=None
        if evidence:
            with np.load(evidence[0]['measured_mask_path'],allow_pickle=False) as raw:masks[key]=raw['mask'].copy()
        groups.setdefault(item['equivalence_group'],[]).append(key)
    for group in groups.values():
        scores={}
        for a in group:
            evidence=frozen[a].get('evidence',[])
            view=evidence[0]['camera'] if evidence else frame.camera
            if view not in frames:raise ValueError('Prediction evidence references an unscored camera')
            for b in group:
                scores[a,b]=dict(evaluate_target(frozen[a],masks[a],actual_by_view[view][b],frames[view]),
                                 evaluation_camera=view)
        quality=np.array([[float(bool(scores[a,b]['localization_verified']))*10+
            float(bool(scores[a,b]['identity_match']))+scores[a,b]['mask_metrics']['target_precision']*.01
            for b in group] for a in group])
        left,right=linear_sum_assignment(-quality)
        for i,j in zip(left,right):results[group[i]]=dict(scores[group[i],group[j]],matched_reference=group[j])
    canvas=np.array(frame.rgb,copy=True)
    palette=([0,210,255],[255,70,170],[80,255,100],[255,180,0])
    for index,item in enumerate(private):
        key=item['key'];mask=masks[key]
        if mask is not None and results[key]['evaluation_camera']==frame.camera:
            canvas[mask]=(.6*canvas[mask]+.4*np.asarray(palette[index%len(palette)])).astype(np.uint8)
        truth=actual[results[key]['matched_reference']];interior=np.zeros_like(truth)
        interior[1:-1,1:-1]=truth[1:-1,1:-1]&truth[:-2,1:-1]&truth[2:,1:-1]&truth[1:-1,:-2]&truth[1:-1,2:]
        canvas[truth&~interior]=[255,230,0]
    tile=Image.new('RGB',(canvas.shape[1],canvas.shape[0]+40),'white');tile.paste(Image.fromarray(canvas),(0,40))
    text=' | '.join(key+':'+('OK' if value['localization_verified'] else 'NO_OUTPUT' if not value['output'] else 'FAIL') for key,value in results.items())
    ImageDraw.Draw(tile).text((3,3),text,fill='black');tile.save(directory/'audit.png')
    return dict(targets=results,all_targets_localized=all(v['localization_verified'] for v in results.values()),
        all_targets_identified=all(v['identity_match'] for v in results.values()),
        source='same_frame_renderer_identity_after_persisted_prediction_only',feeds_back_to_inference=False)


def task_summary(rows):
    valid=[r for r in rows if r.get('initialized')]
    evaluated=[r for r in valid if r.get('evaluation')]
    good=sum(r['evaluation']['all_targets_localized'] for r in evaluated)
    per_target={}
    for row in evaluated:
        for key,target in row['evaluation']['targets'].items():
            counts=per_target.setdefault(key,dict(scenes=0,localized=0,identified=0,no_output=0,
                wrong_region=0,not_visible=0,identity_correct_but_position_unverified=0))
            counts['scenes']+=1;counts['localized']+=bool(target['localization_verified'])
            counts['identified']+=bool(target['identity_match']);counts['no_output']+=not target['output']
            counts['wrong_region']+=bool(target['output'] and not target['identity_match'])
            counts['not_visible']+=not target['mask_metrics']['target_visible']
            counts['identity_correct_but_position_unverified']+=bool(target['output'] and target['identity_match'] and not target['localization_verified'])
    # An infrastructure interruption is not a measured visual failure rate.
    rate=good/len(valid) if valid and len(evaluated)==len(valid) else None
    return dict(attempts=len(rows),initialized_scenes=len(valid),evaluated_scenes=len(evaluated),
        all_targets_localized=good,localization_rate=rate,per_target=per_target,
        initialization_rejected_seeds=[r['seed'] for r in rows if r.get('initialization_rejected')],
        infrastructure_error_seeds=[r['seed'] for r in rows if r.get('infrastructure_error')],
        targets=sum(len(r['evaluation']['targets']) for r in evaluated),
        targets_localized=sum(bool(t['localization_verified']) for r in evaluated for t in r['evaluation']['targets'].values()),
        task_success_evaluated=False)


def run_task(args):
    from PIL import Image
    from adapters.cup_initial_audit import no_task_actions
    name=args.worker_task;directory=args.output/'tasks'/name;directory.mkdir(parents=True,exist_ok=False)
    report=dict(task=name,status='running',pid=os.getpid(),gpu=args.worker_gpu,requested_valid_scenes=args.episodes,
        configuration='demo_randomized',scope='initial_head_whole_object_localization_only',
        known_instance_CAD_supplied=True,live_pose_or_masks_supplied_to_vision=False,
        robot_task_actions_executed=0,rows=[])
    if getattr(args,'multi_view',False):report['scope']='initial_head_and_fixed_overhead_whole_object_localization_only'
    rendering_runtime();block_planner_import();os.chdir(ROBOTWIN)
    module=importlib.import_module('envs.'+name)
    from envs.utils.create_actor import UnStableError
    task=getattr(module,name)();config=scene_configuration(name,getattr(args,'multi_view',False))
    preparer=AssetPreparer(ROBOTWIN,args.output/'generated_cad'/name)
    # Separate process per task bounds total CAD cache lifetime. Allow enough
    # variants for 50 scenes; this does not alter geometric acceptance limits.
    geometry=VisionConfig(cad=CadLimits(max_models=256),require_cross_view_cad=getattr(args,'multi_view',False))
    worker=UnifiedVisionWorker(ROOT,directory/'vision_worker',gpu=args.worker_gpu,config=geometry)
    started=time.perf_counter();failure=None
    def save():
        report.update(summary=task_summary(report['rows']),elapsed_s=time.perf_counter()-started,worker=worker.summary())
        dump(directory/'report.json',report)
    save()
    try:
        with no_planners():
            while sum(bool(r.get('initialized')) for r in report['rows'])<args.episodes:
                check_freeze(args.output)
                index=len(report['rows']);seed=args.seed+index
                if index>=max(200,5*args.episodes):raise RuntimeError('Scene initialization retry budget exhausted')
                case=directory/f'case_{index:04d}_seed{seed}';case.mkdir()
                row=dict(seed=seed,initialized=False,stage='setup',directory=str(case));report['rows'].append(row)
                try:
                    random.seed(seed)
                    with record_creations() as creations:
                        task.setup_demo(now_ep_num=index,seed=seed,**deepcopy(config))
                    row.update(initialized=True,stage='known_model_configuration')
                    with no_task_actions(task):
                        queries,private=prepare_queries(task,name,creations,preparer)
                        dump(case/'public_queries.json',queries)
                        dump(case/'known_model_provenance.json',[dict(key=x['key'],audit=x['model_audit'],equivalence_group=x['equivalence_group']) for x in private])
                        row['stage']='capture';camera=task.get_camera_observation('head_camera')
                        sensor=case/'rgbd.npz'
                        np.savez_compressed(sensor,rgb=camera['rgb'],depth=camera['depth'],intrinsic=camera['intrinsic_cv'],world_to_camera=camera['extrinsic_cv'])
                        Image.fromarray(camera['rgb'][...,:3]).save(case/'initial_head.png')
                        frame=RGBDFrame.from_npz(sensor,camera='head',capture_id='initial',depth_unit='mm')
                        frames=[frame]
                        if getattr(args,'multi_view',False):
                            extra=task.get_camera_observation('workspace_camera')
                            ep=case/'rgbd_workspace.npz'
                            np.savez_compressed(ep,rgb=extra['rgb'],depth=extra['depth'],intrinsic=extra['intrinsic_cv'],world_to_camera=extra['extrinsic_cv'])
                            Image.fromarray(extra['rgb'][...,:3]).save(case/'initial_workspace.png')
                            frames.append(RGBDFrame.from_npz(ep,camera='workspace_camera',capture_id='initial',depth_unit='mm'))
                            row['additional_sensor_sha256']={str(ep):digest(ep)}
                        row['sensor_sha256']=digest(sensor);row['stage']='inference';begin=time.perf_counter()
                        prediction=worker.observe(frames,[TargetQuery(**q) for q in queries],episode_id=name+':'+str(seed))
                        row['inference_wall_s']=time.perf_counter()-begin
                        row['prediction_directory']=str(worker.output/f"request_{prediction.audit['worker_request']:05d}"/'prediction')
                        row['outputs']={k:dict(status=v.status,required_available=v.required_available) for k,v in prediction.targets.items()}
                        # Lock ALL prediction artifacts before this case's truth
                        # is read. Evaluation never sends a second request.
                        locks={str(p):digest(p) for p in Path(row['prediction_directory']).iterdir()}
                        dump(case/'prediction_locks.json',locks)
                        row['stage']='evaluation';row['evaluation']=evaluate_scene(task,private,prediction,frame,case,frames=frames)
                        if any(digest(p)!=value for p,value in locks.items()):raise RuntimeError('Prediction changed during evaluation')
                        row['predictions_unchanged']=True;row['stage']='done'
                except UnStableError as error:
                    row.update(initialization_rejected=True,error=str(error))
                except Exception as error:
                    row.update(infrastructure_error=True,error=f'{type(error).__name__}: {error}')
                    traceback.print_exc();failure=error
                finally:
                    try:task.close_env()
                    except Exception as error:row['close_error']=str(error)
                    dump(case/'result.json',row);save()
                print(f'[vision50] {name} seed={seed} valid={report["summary"]["initialized_scenes"]}/{args.episodes} localized={report["summary"]["all_targets_localized"]} stage={row["stage"]}',flush=True)
                if failure is not None:raise failure
        check_freeze(args.output);report['status']='finished'
    except BaseException as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}');raise
    finally:worker.close();save()


def environment(gpu):
    env=os.environ.copy();cache=ROOT/'.runtime_cache';native=ROBOTWIN/'.conda_envs/RoboTwin_yzy'
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONPATH=os.pathsep.join(map(str,(ROOT,ROBOTWIN,ROBOTWIN/'envs/curobo/src'))),
        LD_LIBRARY_PATH=f'/usr/local/cuda-12.4/lib64:{native}/lib:/usr/lib/x86_64-linux-gnu:/usr/lib',
        VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MAX_JOBS='4',
        HF_HOME=str(cache/'hf_home'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TMPDIR=str(cache/'tmp'),MPLCONFIGDIR=str(cache/'matplotlib'),XDG_CACHE_HOME=str(cache/'xdg'),
        WARP_CACHE_PATH=str(cache/'warp_vision'),CUDA_CACHE_PATH=str(cache/'cuda'),
        TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'))
    env.pop('LD_PRELOAD',None)
    return env


def coordinate(args):
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'logs').mkdir()
    before={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    for name in before:
        destination=args.output/'frozen_code'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    selected=args.tasks or list(TASK_TARGETS)
    dump(args.output/'freeze.json',dict(source_sha256=before,tasks=selected,target_catalog=TASK_TARGETS,
        episodes_per_task=args.episodes,seed_start=args.seed,scene='demo_randomized',
        vision_configuration=asdict(VisionConfig(cad=CadLimits(max_models=256),require_cross_view_cad=args.multi_view)),
        CAD_prior='exact_instance_geometry_and_appearance_from_creation_parameters_explicitly_supplied',
        unknown_instance_identification_evaluated=False,initial_head_camera_only=not args.multi_view,
        multi_view=args.multi_view,view_configuration='head_plus_fixed_overhead_2p2m' if args.multi_view else 'head_only',
        localization_metric='required_output_AND_target_precision>=0.8_AND_overlap>=20pixels_AND_visible_surface_center_error<=0.02m',
        interchangeable_targets='one_to_one_assignment_within_identical_CAD_and_appearance_groups',
        ground_truth_use='per_case_post_persisted_prediction_evaluation_only_never_vision_input',
        whole_object_only=True,functional_parts_or_full_6D_pose_accuracy_claimed=False,
        robot_task_actions_executed=0,planner_construction_disabled_process_locally=True,
        original_scene_stability_check_retained=True,acceptance_thresholds_retuned=False))
    gpus=[int(v) for v in args.gpus.split(',')];pending=list(selected);active={};completed={};started=time.perf_counter()
    report=dict(status='running',pid=os.getpid(),gpus=gpus,requested_tasks=len(selected),
        episodes_per_task=args.episodes,requested_valid_scenes=len(selected)*args.episodes,tasks={})
    def save():
        tasks={}
        for name in selected:
            file=args.output/'tasks'/name/'report.json'
            if file.is_file():
                value=json.loads(file.read_text());tasks[name]={k:value.get(k) for k in ('status','gpu','summary','error','elapsed_s')}
            elif name in report.get('process_failures',{}):tasks[name]=dict(status='error',error='scene_process_failed_before_report')
            else:tasks[name]=dict(status='pending')
        report.update(tasks=tasks,elapsed_s=time.perf_counter()-started,
            active_processes={str(gpu):dict(pid=value[0].pid,task=value[1]) for gpu,value in active.items()},
            finished_tasks=sum(v['status']=='finished' for v in tasks.values()),
            failed_tasks=[k for k,v in tasks.items() if v['status']=='error'],
            initialized_scenes=sum((v.get('summary') or {}).get('initialized_scenes',0) for v in tasks.values()),
            evaluated_scenes=sum((v.get('summary') or {}).get('evaluated_scenes',0) for v in tasks.values()),
            all_targets_localized=sum((v.get('summary') or {}).get('all_targets_localized',0) for v in tasks.values()))
        dump(args.output/'report.json',report)
    save()
    try:
        while pending or active:
            check_freeze(args.output)
            for gpu in gpus:
                if gpu in active:
                    process,name,log=active[gpu]
                    if process.poll() is None:continue
                    log.close();completed[name]=process.returncode;del active[gpu]
                    if process.returncode:
                        report.setdefault('process_failures',{})[name]=dict(exit_code=process.returncode,log=str(args.output/'logs'/(name+'.log')))
                if pending:
                    name=pending.pop(0);log=(args.output/'logs'/(name+'.log')).open('x')
                    command=[sys.executable,str(Path(__file__).resolve()),'--worker-task',name,
                        '--worker-gpu',str(gpu),'--episodes',str(args.episodes),'--seed',str(args.seed),'--output',str(args.output)]
                    if args.multi_view:command.append('--multi-view')
                    process=subprocess.Popen(command,cwd=ROOT,env=environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    active[gpu]=(process,name,log)
            save()
            if active:time.sleep(2)
        report['status']='finished' if all(code==0 for code in completed.values()) else 'finished_with_errors'
        report['source_hashes_unchanged']=True;save()
    except BaseException as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}');save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--gpus',default='0,1,2,3')
    parser.add_argument('--tasks',nargs='+',choices=tuple(TASK_TARGETS))
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--multi-view',action='store_true',help='Head plus fixed workspace RGB-D; require same-pose CAD cross-view verification')
    parser.add_argument('--worker-task',choices=tuple(TASK_TARGETS))
    parser.add_argument('--worker-gpu',type=int,default=0)
    args=parser.parse_args();args.output=args.output.resolve()
    if not args.output.is_relative_to(ROOT) or args.output==ROOT:parser.error('Choose an output below ULA')
    if args.episodes<1 or args.seed<0:parser.error('Invalid episodes/seed')
    try:gpus=[int(v) for v in args.gpus.split(',')]
    except ValueError:parser.error('Use comma-separated GPU indices')
    if not gpus or len(gpus)!=len(set(gpus)):parser.error('Distinct GPUs required')
    if args.tasks and len(args.tasks)!=len(set(args.tasks)):parser.error('Duplicate tasks')
    if args.worker_task:
        try:return run_task(args)
        except BaseException as error:
            path=args.output/'tasks'/args.worker_task/'report.json'
            previous=json.loads(path.read_text()) if path.exists() else dict(task=args.worker_task,gpu=args.worker_gpu,rows=[])
            previous.update(status='error',error=f'{type(error).__name__}: {error}')
            path.parent.mkdir(parents=True,exist_ok=True);dump(path,previous)
            raise
    if args.output.exists():parser.error('Choose a fresh, nonexistent output directory')
    if args.detach:
        log_path=args.output.with_suffix('.log')
        with log_path.open('x') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                *[v for v in sys.argv[1:] if v!='--detach']],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=process.pid,output=str(args.output),log=str(log_path),detached=True)))
        return
    coordinate(args)


if __name__=='__main__':main()
