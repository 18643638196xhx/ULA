#!/usr/bin/env python3
"""Offline sensor-only replay, or persistent JSONL requests with one model load.

No simulator construction, robot action, actor pose, evaluation mask or score.
Persistent protocol: {episode_id, capture_id, depth_unit, queries:[...],
frames:[{path,camera,timestamp_s?}], output}. One JSON reply per request.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.unified_vision import RGBDFrame,TargetQuery,VisionConfig,UnifiedVision
from adapters.unified_vision.backends import DinoSamBackend


def isolate_protocol_output():
    """Reserve a clean pipe, then route Python AND native stdout to stderr.

    redirect_stdout alone cannot intercept printf/os.write from CAD libraries.
    This is process-local and is used only by this standalone worker entrypoint.
    """
    sys.stdout.flush();sys.stderr.flush()
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1,encoding='utf-8')
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    return protocol


def save(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Replay outputs must stay under ULA')
    path.mkdir(parents=True,exist_ok=False)
    for key,target in value.targets.items():
        if not key.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe target key for artifact name')
        np.savez_compressed(path/(key+'_surface.npz'),points_world=target.points)
        # Preserve actual candidate masks, including rejected CAD candidates.
        # Never reconstruct an unselected mask later from its bounding box.
        for index,record in enumerate(target.candidate_records):
            file=path/f'{key}_candidate_{index:02d}.npz'
            frame=record['frame'];payload=dict(mask=record['mask'].mask,
                points_world=record['points'],intrinsic=frame.intrinsic,
                world_to_camera=frame.world_to_camera,camera=np.asarray(frame.camera),
                capture_id=np.asarray(frame.capture_id))
            if 'cad_render_depth_m' in record:payload['cad_render_depth_m']=record['cad_render_depth_m']
            np.savez_compressed(file,**payload)
            for group in target.candidates:
                for candidate in group['candidates']:
                    if candidate.get('candidate_id')==record.get('candidate_id'):
                        candidate['measured_mask_path']=str(file)
        for index,record in enumerate(target.views.values()):
            file=path/f'{key}_view_{index:02d}.npz'
            frame=record['frame']
            np.savez_compressed(file,mask=record['mask'].mask,points_world=record['points'],
                intrinsic=frame.intrinsic,world_to_camera=frame.world_to_camera,
                camera=np.asarray(frame.camera),capture_id=np.asarray(frame.capture_id))
            target.evidence[index]['measured_mask_path']=str(file)
    (path/'prediction.json').write_text(json.dumps(value.to_dict(),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def execute(engine,request):
    seed=request.get('inference_seed')
    if seed is not None:
        if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**32:
            raise ValueError('Expected a uint32 inference seed')
        import random,torch
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    frames=[RGBDFrame.from_npz(item['path'],camera=item['camera'],capture_id=request['capture_id'],
        depth_unit=request['depth_unit'],timestamp_s=item.get('timestamp_s')) for item in request['frames']]
    queries=[TargetQuery(**item) for item in request['queries']]
    value=engine.observe(frames,queries,episode_id=request['episode_id'],search_motion=request.get('search_motion'))
    save(request['output'],value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--frame',type=Path,action='append',help='Repeat for synchronous views')
    parser.add_argument('--camera',action='append',help='One camera name per --frame')
    parser.add_argument('--episode-id',default='offline_replay')
    parser.add_argument('--capture-id',default='capture_000')
    parser.add_argument('--depth-unit',choices=('m','mm'),default='mm')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--requests',type=Path,help='JSONL manifest, all requests reuse the model stack')
    parser.add_argument('--serve',action='store_true',help='Read JSONL requests on stdin, reply on stdout')
    parser.add_argument('--geometry-config',type=Path,help='Shared VisionConfig JSON, not task-specific code')
    args=parser.parse_args()
    if sum((bool(args.requests),args.serve,bool(args.frame)))!=1:
        parser.error('Choose one-shot --frame, --requests, or --serve')
    protocol=isolate_protocol_output()
    sources=[Path(__file__),ROOT/'scripts/run_unified_vision.sh',*sorted((ROOT/'adapters/unified_vision').glob('*.py'))]
    def hashes():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    before=hashes();started=time.perf_counter()
    # Third-party initialization prints diagnostics. Keep stdout valid JSONL.
    from contextlib import redirect_stdout
    config=VisionConfig(**json.loads(args.geometry_config.read_text())) if args.geometry_config else VisionConfig()
    with redirect_stdout(sys.stderr):
        cad_output=(args.geometry_config.parent/'cad_runtime' if args.geometry_config else
                    ROOT/'.runtime_cache/unified_cad'/str(os.getpid()))
        backend=DinoSamBackend(ROOT/'tra_pipe/RoboTwin',max_candidates=config.max_candidates,config=config,
                              project_root=ROOT,cad_output=cad_output)
    engine=UnifiedVision(backend,config)
    if args.serve:
        print(json.dumps(dict(status='ready',pid=os.getpid(),model_load_s=time.perf_counter()-started)),file=protocol,flush=True)
    if args.requests:requests=[json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    elif args.serve:requests=(json.loads(line) for line in sys.stdin if line.strip())
    else:
        if args.config is None or args.output is None:parser.error('--config and --output required')
        cameras=args.camera or [f'camera_{i}' for i in range(len(args.frame))]
        if len(cameras)!=len(args.frame):parser.error('Number of camera names and frames differs')
        config=json.loads(args.config.read_text())
        requests=[dict(episode_id=args.episode_id,capture_id=args.capture_id,depth_unit=args.depth_unit,
            frames=[dict(path=str(path),camera=camera) for path,camera in zip(args.frame,cameras)],
            queries=config['queries'],output=str(args.output))]
    rows=[]
    for index,request in enumerate(requests):
        if request.get('op') in ('close','shutdown'):break
        try:
            with redirect_stdout(sys.stderr):value=execute(engine,request)
            reply=dict(id=request.get('id',index),status='ok',output=request['output'],
                targets={key:dict(status=target.status,required_available=target.required_available)
                         for key,target in value.targets.items()})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply=dict(id=request.get('id',index),status='error',error=f'{type(error).__name__}: {error}')
        rows.append(reply);print(json.dumps(reply,ensure_ascii=False),file=protocol,flush=True)
    if args.output is not None and (args.requests or args.serve):
        output=args.output.resolve()
        if not output.is_relative_to(ROOT):raise ValueError('Report must stay under ULA')
        output.mkdir(parents=True,exist_ok=True)
        report=dict(status='finished',requests=rows,model_stack_loads=1,backend=backend.summary(),
            elapsed_s=time.perf_counter()-started,sources_before=before,sources_after=hashes(),
            sources_unchanged=before==hashes(),robot_actions_executed=False,uses_object_truth=False,
            task_success_evaluated=False,not_a_localization_accuracy_benchmark=True)
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    protocol.close()


if __name__=='__main__':main()
