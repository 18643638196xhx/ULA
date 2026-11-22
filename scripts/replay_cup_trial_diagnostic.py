#!/usr/bin/env python3
"""Frozen old trial RGB-D replay. No simulator, VLA, actions or task scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tra_pipe/RoboTwin')]
from adapters.cup_trial_diagnostic import TrialCupWorker
from adapters.pointcloud_registration import RegistrationConfig, estimate_pointcloud_motion
from adapters.visual_grasp import visual_comotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(mode='frozen_old_trial_rgbd_replay', simulation=False, vla=False,
        task_success_not_evaluated=True, frames=[], completed_cases=0)
    source_hashes = {}
    worker = TrialCupWorker(ROOT, args.output/'vision_worker', backend='foundationpose')
    try:
        for seed in [9,21,32,48]:
            source = args.input/f'episode_{seed:03d}_seed{seed}'
            initial = source/'perception/rgbd.npz'
            files = [initial] + sorted((source/'trial_perception').glob('trial_before_*/rgbd.npz'))
            files += sorted((source/'trial_perception').glob('trial_after_*_0/rgbd.npz'))
            references = {}
            old = json.loads((source/'result.json').read_text())
            ee_delta = next((o['comotion']['ee_displacement_m'] for o in
                old['traditional_grasp_verification']['observations'] if 'comotion' in o), None)
            for frame in files:
                label = frame.parent.name
                directory = args.output/f'seed_{seed:02d}'/label
                directory.mkdir(parents=True, exist_ok=True)
                request = directory/'rgbd.npz'
                source_hashes[str(frame)] = hashlib.sha256(frame.read_bytes()).hexdigest()
                shutil.copy2(frame, request)
                item = dict(seed=seed, stage=label, source=str(frame),
                    source_sha256=source_hashes[str(frame)], output_directory=str(directory))
                report['frames'].append(item)
                try:
                    value = worker.estimate_files(request, directory/'estimate.npz')
                    item.update(returned=True, observation=value.observation_audit)
                    if label.startswith('trial_before_'):
                        references[label.removeprefix('trial_before_')] = value
                    elif label.startswith('trial_after_') and ee_delta is not None:
                        camera = label.removeprefix('trial_after_').removesuffix('_0')
                        if camera in references:
                            ref = references[camera]
                            motion, registration = estimate_pointcloud_motion(ref.cup_points_world,
                                value.cup_points_world, RegistrationConfig(max_rmse_m=.012, max_tilt_deg=35.))
                            item['registration'] = registration
                            if motion is not None:
                                item['comotion'] = visual_comotion(ref.cup_points_world, motion, ee_delta)
                except (RuntimeError, ValueError) as error:
                    item.update(returned=False, error=str(error))
                print(json.dumps({k:v for k,v in item.items() if k not in ('observation','registration')}, ensure_ascii=False), flush=True)
            report['completed_cases'] += 1
            (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    finally:
        worker.close()
        report.update(worker=worker.summary(), source_hashes=source_hashes,
            original_frames_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                                          for p,h in source_hashes.items()))
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
