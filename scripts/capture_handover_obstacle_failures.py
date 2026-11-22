#!/usr/bin/env python3
"""Replay unchanged baseline, save failures AFTER control. No alternative actions."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
BASE_PATH=ROOT/'scripts/eval_pi05_handover_rgbd_obstacles.py'
spec=importlib.util.spec_from_file_location('obstacle_capture_base',BASE_PATH)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from diagnostics.handover_obstacle_snapshot import capture_failure,cost_flags,dump


class CapturingSession(base.ObstaclePlanningSession):
    def plan(self,*args,**kwargs):
        flags=cost_flags(self.planner.motion_gen)
        result=super().plan(*args,**kwargs)
        if result['status']!='Success':self._failure=(args,kwargs,result,flags)
        return result  # Never substituted, retried or used to select a diagnostic candidate.

    def __exit__(self,*exc):
        try:
            if getattr(self,'_failure',None) is not None:
                self.result['diagnostic_snapshot']=capture_failure(self,*self._failure)
        except Exception as error:
            self.result['diagnostic_capture_error']=f'{type(error).__name__}: {error}'
            dump(self.scene.directory/'capture_error.json',dict(error=self.result['diagnostic_capture_error']))
        finally:
            super().__exit__(*exc)


class CapturingObserver(base.MeasuredObstacleObserver):
    def observe(self,prior):
        pose,quality=super().observe(prior)
        self.planning.task._obstacle_audit_visual=(pose.copy(),quality)
        return pose,quality


def main():
    original_sources=base.full.shared.initial.sources
    def sources():
        files=[BASE_PATH,Path(__file__),ROOT/'scripts/capture_handover_obstacle_failures.sh',
               ROOT/'diagnostics/handover_obstacle_snapshot.py']
        return dict(original_sources(),**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    base.full.shared.initial.sources=sources
    original_dump=base.full.shared.dump
    def write(path,value):
        if Path(path).name=='report.json':value.update(diagnostic_capture_only=True,
            alternative_actions_executed=False,physical_truth_usage='separate_after_control_audit_only')
        original_dump(path,value)
    base.full.shared.dump=write
    base.ObstaclePlanningSession=CapturingSession
    base.MeasuredObstacleObserver=CapturingObserver
    # Preserve wrapper on --detach, while sources() also hashes the actual base.
    base.__file__=str(Path(__file__).resolve())
    base.main()


if __name__=='__main__':main()
