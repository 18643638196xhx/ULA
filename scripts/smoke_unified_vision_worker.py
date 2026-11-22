#!/usr/bin/env python3
"""Policy-Python -> persistent vision-Python sensor replay; no task execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.unified_vision import RGBDFrame,TargetQuery
from adapters.unified_vision.worker import UnifiedVisionWorker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    if not args.output.is_relative_to(ROOT):parser.error('Outputs must stay under ULA')
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'scripts/run_unified_vision.py',ROOT/'scripts/run_unified_vision.sh',
        ROOT/'adapters/perception.py',ROOT/'adapters/rgbd_perception.py',
        *sorted((ROOT/'adapters/unified_vision').glob('*.py')),
        *sorted((ROOT/'configs/unified_vision').glob('*.json'))]
    def hashes():return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report=dict(status='running',sources_before=hashes(),cases=[],robot_actions_executed=False,
        uses_object_truth=False,task_success_evaluated=False,not_a_localization_accuracy_benchmark=True)
    def save():
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    cases=[('cup','cup_pad.json',ROOT/'results/cup_initial_randomized_50scenes_seed0_20261101_v2/episode_000_seed0/rgbd.npz'),
        ('bottle','bottle_pad.json',ROOT/'results/pillbottle_initial_cup_method_randomized_50valid_seed0_20261102_v1/episode_000_seed0/rgbd.npz'),
        ('body_part','body_part.json',ROOT/'results/microwave_no_extent_randomized_50valid_seed0_20261102_v1/episode_000_seed0/perception/initial_0_head_camera/rgbd.npz')]
    worker=UnifiedVisionWorker(ROOT,args.output/'worker',gpu=args.gpu)
    save()
    try:
        for name,configuration,path in cases:
            queries=[TargetQuery(**value) for value in json.loads((ROOT/'configs/unified_vision'/configuration).read_text())['queries']]
            frame=RGBDFrame.from_npz(path,camera='head',capture_id='initial',depth_unit='mm')
            started=time.perf_counter()
            value=worker.observe([frame],queries,episode_id=name)
            row=dict(case=name,input_sensor=str(path),elapsed_s=time.perf_counter()-started,
                worker_pid=value.audit['worker_pid'],targets={key:dict(status=target.status,
                    required_available=target.required_available,usable=target.usable)
                    for key,target in value.targets.items()})
            report['cases'].append(row);save();print(json.dumps(row,ensure_ascii=False),flush=True)
        report['status']='finished'
    except BaseException as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        worker.close()
        report.update(worker=worker.summary(),sources_after=hashes(),sources_unchanged=report['sources_before']==hashes())
        save()


if __name__=='__main__':main()
