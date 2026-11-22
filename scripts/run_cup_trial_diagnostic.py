#!/usr/bin/env python3
"""Four explicitly selected failure seeds; stop after grasp trial, never place."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tra_pipe/RoboTwin')]
spec = importlib.util.spec_from_file_location('cup_trial_known_base', ROOT/'scripts/run_place_empty_cup_known_randomized.py')
known = importlib.util.module_from_spec(spec)
spec.loader.exec_module(known)
randomized = known.randomized
base = randomized.base
from adapters.cup_trial_diagnostic import TrialCupWorker, verify_segmented_trial


def terminal_audit(task, perception, arm, result, **kwargs):
    """Only after the controller returns/raises; never feed scoring back to it."""
    try:
        return verify_segmented_trial(task, perception, arm, result, **kwargs)
    finally:
        audit = dict(control_has_terminated=True, used_for_control=False,
            uses_simulator_truth=True, placement_will_not_execute=True)
        try:
            pose = task.cup.get_pose()
            bottom = task.cup.get_functional_point(0, 'pose').p
            audit.update(cup_world_pose=np.asarray(pose.to_transformation_matrix()).tolist(),
                annotated_bottom_reference_world=np.asarray(bottom).tolist(),
                actual_ee_pose=list(task.get_arm_pose(arm)),
                limitation='Annotated bottom reference is not the lowest physical point for a tilted cup.')
            tables = [o['perception']['table_height_m'] for o in result['observations']
                      if o.get('stage') == 'baseline' and o.get('perception')]
            if tables:
                table = float(np.median(tables))
                audit.update(visual_table_z_m=table, annotated_bottom_above_visual_table_m=float(bottom[2]-table))
        except Exception as error:
            audit['error'] = str(error)
        base.dump_json(perception.output/'evaluation_after_control.json', audit)


def main():
    args = base.parse_args()
    if args.mode != 'grasp-only' or args.approach_mode != 'pointcloud' or args.grasp_handoff != 'visual-trial-lift':
        raise ValueError('Diagnostic requires grasp-only + pointcloud + visual-trial-lift')
    if args.seed_cache or args.grasp_retries:
        raise ValueError('No seed filtering or automatic regrasp in this diagnostic')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh output directory')
    seeds = [int(s) for s in os.environ.get('ULA_TRIAL_SEEDS', '9,21,32,48').split(',')]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Provide distinct explicit diagnostic seeds')
    args.output.mkdir(parents=True, exist_ok=True)
    resource = randomized.configure_runtime()
    protected = randomized.fingerprints()
    original = (base.task_config, base.approach, base.verify_visual_trial_lift, base.WorldLA)
    config = randomized.randomized_config(base.task_config(args))
    base.task_config = lambda unused: deepcopy(config)
    base.approach = known.approach_known_cup
    base.verify_visual_trial_lift = terminal_audit
    policy = original[3](ROOT, args.model_dir, args.asset_checkpoint,
        blend=args.blend, replan_mode=args.replan_mode, inference_seed=args.inference_seed)
    base.WorldLA = lambda *a, **kw: policy
    report = dict(task_config='demo_randomized', purpose='targeted_grasp_diagnostic_not_task_success_rate',
        seeds=seeds, episodes=[], placement_executed=False, grasp_actions_unchanged=True,
        approach='unchanged_known_CAD_visual_pose', trial_perception='local_semantics_no_hard_color',
        new_observation_after_each_lift_segment=True, strict_slip_check=False,
        resource_policy=resource, protected_files_before=protected, run_status='running')
    worker = TrialCupWorker(ROOT, args.output/'vision_worker', args.fp_iterations, backend='foundationpose')
    previous = Path.cwd()
    output = args.output
    try:
        for seed in seeds:
            current = deepcopy(args)
            current.seed, current.episodes, current.output = seed, 1, output/f'seed_{seed:02d}'
            base.run_batch(current, worker)
            single = json.loads((current.output/'report.json').read_text())
            report['episodes'].extend(single['episodes'])
            base.dump_json(output/'report.json', report)
        report['run_status'] = 'finished'
    finally:
        worker.close()
        os.chdir(previous)
        base.task_config, base.approach, base.verify_visual_trial_lift, base.WorldLA = original
        after = randomized.fingerprints()
        report.update(perception_worker=worker.summary(), protected_files_after=after,
                      protected_files_unchanged=protected == after)
        report['diagnostic_visual_passes'] = sum(bool(e.get('grasp_handoff')) for e in report['episodes'])
        report['completed_episodes'] = len(report['episodes'])
        if report['run_status'] != 'finished':
            report['run_status'] = 'interrupted_or_error'
        base.dump_json(output/'report.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
