#!/usr/bin/env python3
"""Offline only: never constructs a simulator or calls a robot action."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from diagnostics.handover_obstacle_audit import analyze_snapshot
from diagnostics.handover_obstacle_snapshot import dump


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-for-capture',action='store_true',help='Process snapshots as unchanged replay finishes')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),ROOT/'diagnostics/handover_obstacle_audit.py',
           ROOT/'diagnostics/handover_obstacle_snapshot.py',ROOT/'adapters/handover/rgbd_obstacles.py']
    def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report=dict(status='running',pid=os.getpid(),sources_before=hashes(),snapshots=[],
                candidate_actions_executed=False,simulation_constructed=False)
    dump(args.output/'report.json',report)
    seen=set();deadline=time.monotonic()+1800
    while True:
        for path in sorted(args.capture_root.glob('rgbd_obstacles/scene_*/failure_snapshot')):
            if str(path) in seen or not (path/'metadata.json').exists():continue
            try:
                started=time.perf_counter()
                result=analyze_snapshot(path,args.output/path.parent.name)
                report['snapshots'].append(dict(snapshot=str(path),status='finished',
                    elapsed_s=time.perf_counter()-started,audit=str(args.output/path.parent.name/'audit.json')))
                seen.add(str(path))
            except Exception as error:
                report['snapshots'].append(dict(snapshot=str(path),status='error',error=f'{type(error).__name__}: {error}'))
                dump(args.output/'report.json',report)
                raise
            dump(args.output/'report.json',report)
        if not args.wait_for_capture:break
        import json
        capture=json.loads((args.capture_root/'report.json').read_text())
        expected=sum(bool(row.get('placement_control',{}).get('rgbd_obstacles',{}).get('diagnostic_snapshot'))
                     for row in capture['episodes'])
        if capture['run_status']=='finished' and len(seen)==expected:break
        if time.monotonic()>deadline:raise TimeoutError('Capture did not finish within 30 minutes')
        time.sleep(5)
    report.update(status='finished',sources_after=hashes(),sources_unchanged=report['sources_before']==hashes())
    dump(args.output/'report.json',report)


if __name__=='__main__':main()
