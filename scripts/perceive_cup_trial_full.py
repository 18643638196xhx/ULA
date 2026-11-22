#!/usr/bin/env python3
"""Full-task worker: original known-CAD approach/place + semantic trial views."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tra_pipe/RoboTwin')]


def estimate_file(vision, source, destination):
    path = Path(source)
    initial = path.parent.name == 'perception' and path.name == 'rgbd.npz'
    placement = path.parent.name == 'placement_perception'
    if initial:
        vision.reset_episode()
    with np.load(source, allow_pickle=False) as frame:
        payload = dict(frame)
    allowed = {'rgb', 'depth', 'intrinsic', 'world_to_camera', 'cup_only', 'refine_cup'}
    if set(payload)-allowed:
        raise ValueError('Unexpected non-RGBD full-task input')
    value = vision.estimate(**payload, require_cad=initial or placement)
    output = dict(cup_bounds_world=value.cup_bounds_world, cup_points_world=value.cup_points_world,
        coaster_world_center=value.coaster_world_center,
        timings_json=np.asarray(json.dumps(value.timings_s)),
        observation_audit_json=np.asarray(json.dumps(value.observation_audit)))
    if value.cup_world_pose is not None:
        output['cup_world_pose'] = value.cup_world_pose
    if hasattr(value, 'cup_mask'):
        output['cup_mask'] = value.cup_mask
    np.savez(destination, **output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--debug-dir', type=Path, required=True)
    parser.add_argument('--fp-iterations', type=int, default=5)
    parser.add_argument('--backend', default='foundationpose')
    args = parser.parse_args()
    import torch
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    from adapters.cup_trial_diagnostic import TrialCupVision
    spec = importlib.util.spec_from_file_location('cup_trial_full_protocol', ROOT/'scripts/perceive_rgbd.py')
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    protocol.estimate_file = estimate_file
    return protocol.serve(lambda: TrialCupVision(ROOT/'tra_pipe/RoboTwin', args.fp_iterations, args.debug_dir))


if __name__ == '__main__':
    raise SystemExit(main())
