#!/usr/bin/env python3
"""Closed-command handoff directly to unchanged known-CAD placement.

--episodes counts INITIALIZED scenes. Rejected environment setups are retained
separately and consecutive seeds continue until the requested count is reached.
"""
from collections import defaultdict
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
spec = importlib.util.spec_from_file_location('cup_no_trial_known', ROOT/'scripts/run_place_empty_cup_known_randomized.py')
known = importlib.util.module_from_spec(spec)
spec.loader.exec_module(known)
randomized, base = known.randomized, known.randomized.base
from adapters.statistics import summarize_episodes
from adapters.timing import summarize_timings


class NoTrialVisionWorker(known.KnownCupWorker):
    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        selected = os.environ.get('ULA_NO_TRIAL_VISION_GPU')
        if selected is not None:
            if not selected.isdigit():
                raise ValueError('Choose one physical vision GPU')
            environment['CUDA_VISIBLE_DEVICES'] = selected
        return command, cwd, environment


def bypass_trial(task, perception, arm, result, **unused):
    """Permission only: NEVER move, observe, or claim that a grasp is verified."""
    result.update(complete=False, verification_performed=False, trial_lift_executed=False,
        slip_check_enabled=False, uses_simulator_truth_for_verification=False,
        decision='provisional_closed_command_direct_to_placement', segments=[], observations=[])
    return dict(ready=True, detector='closed_command_candidate_without_trial',
        lifted_grasp_verified=False, verification_skipped=True,
        placement_permission_only=True, uses_simulator_truth=False)


def valid_records(records):
    return [r for r in records if r.get('failure_stage') != 'setup']


def progress(records):
    valid = valid_records(records)
    successful = [r for r in valid if r['success']]
    rejected = [r for r in records if r.get('failure_stage') == 'setup']
    groups = defaultdict(list)
    for record in valid:
        if not record['success']:
            groups[record.get('failure_stage') or 'final_task_score'].append(record['seed'])
    return dict(raw_seed_attempts=len(records), completed_valid_episodes=len(valid),
        successes=len(successful), success_rate=len(successful)/len(valid) if valid else None,
        initialization_rejections=len(rejected), initialization_rejection_seeds=[r['seed'] for r in rejected],
        failed_valid_seeds=[r['seed'] for r in valid if not r['success']],
        valid_failure_groups=dict(groups), statistics=summarize_episodes(valid),
        timing_statistics=summarize_timings(valid))


def main():
    args = base.parse_args()
    if args.mode != 'hybrid' or args.approach_mode != 'pointcloud' or args.grasp_handoff != 'visual-trial-lift':
        raise ValueError('Use hybrid + pointcloud + visual-trial-lift legacy options; actual trial is bypassed')
    if args.seed_cache or args.episodes < 1:
        raise ValueError('Use positive initialized-scene count and raw consecutive seeds')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh output directory')
    output, target = args.output, args.episodes
    output.mkdir(parents=True, exist_ok=True)
    resource = randomized.configure_runtime()
    resource['vision_gpu'] = os.environ.get('ULA_NO_TRIAL_VISION_GPU', os.environ.get('CUDA_VISIBLE_DEVICES'))
    resource['jax_allocator'] = os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', 'default')
    before = randomized.fingerprints()
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = randomized.randomized_config(base.task_config(args))
    originals = (base.task_config, base.approach, base.finish_pointcloud_placement,
        base.verify_visual_trial_lift, base.WorldLA, base.ExternalPerception.select_reference, base.dump_json)
    policy = base.WorldLA(ROOT, args.model_dir, args.asset_checkpoint,
        blend=args.blend, replan_mode=args.replan_mode, inference_seed=args.inference_seed)
    worker = NoTrialVisionWorker(ROOT, output/'vision_worker', args.fp_iterations, backend='foundationpose')
    base.task_config = lambda unused: deepcopy(config)
    base.approach, base.finish_pointcloud_placement = known.approach_known_cup, known.place_known_cup
    base.verify_visual_trial_lift = bypass_trial
    base.WorldLA = lambda *a, **kw: policy
    def reference(self, estimate):
        originals[5](self, estimate)
        if estimate.cup_world_pose is not None:
            with np.load(self.reference_path, allow_pickle=False) as data:
                values = dict(data)
            values['cup_world_pose'] = estimate.cup_world_pose
            np.savez(self.reference_path, **values)
    base.ExternalPerception.select_reference = reference
    history = []
    report = dict(task_name='place_empty_cup', task_config='demo_randomized',
        target_valid_episodes=target, raw_seed_start=args.seed, run_status='running',
        control_sequence='traditional_approach -> VLA_closed_command_candidate -> traditional_place',
        traditional_trial_lift_enabled=False, extra_vla_actions_after_gate=0,
        grasp_verification_enabled=False, closed_command_is_not_verified_grasp=True,
        success_definition='official_complete_task_score_on_initialized_scenes',
        initialization_rejections_count_as_method_failures=False,
        approach_and_placement='unchanged_known_CAD_at_fresh_visual_pose',
        uses_prepared_model=True, control_uses_object_truth=False,
        evaluation_uses_object_truth=True, grasp_stage_truth_audit_enabled=False,
        model_dir=str(args.model_dir.resolve()), asset_checkpoint=str(args.asset_checkpoint.resolve()),
        instruction=args.instruction, arm=args.arm, inference_seed=args.inference_seed,
        blend=args.blend, replan_mode=args.replan_mode,
        effective_domain_randomization=config['domain_randomization'],
        resource_policy=resource, protected_files_before=before,
        implementation_hash=implementation_hash, episodes=[])
    def write_progress(records):
        report.update(progress(records), episodes=records, perception_worker=worker.summary())
        originals[6](output/'report.json', report)
    def dump(path, value):
        if Path(path).name == 'report.json' and Path(path).parent != output:
            value.update(task_config='demo_randomized', traditional_trial_lift_enabled=False,
                grasp_verification_enabled=False,
                control_sequence=report['control_sequence'],
                grasp_handoff_mode='closed_command_direct_placement_unverified',
                grasp_detector='provisional_closed_command_only',
                approach_mode='known_CAD_contact_at_visual_pose',
                placement_mode='unchanged_fresh_foundationpose_known_functional_point')
            value['early_handoff_configuration'].update(trial_lift_distance_m=0.,
                trial_slip_check_enabled=False, verification_uses_simulator_truth=False)
            originals[6](path, value)
            write_progress(history+value['episodes'])
        else:
            originals[6](path, value)
    base.dump_json = dump
    previous = Path.cwd()
    try:
        next_seed, round_index = args.seed, 0
        while len(valid_records(history)) < target:
            if len(history) >= max(200, target*5):
                raise RuntimeError('Too many setup rejections; initialized-scene target not reached')
            current = deepcopy(args)
            current.seed = next_seed
            current.episodes = target-len(valid_records(history))
            current.output = output/f'round_{round_index:02d}'
            base.run_batch(current, worker)
            records = json.loads((current.output/'report.json').read_text())['episodes']
            history.extend(records)
            next_seed += len(records)
            round_index += 1
            write_progress(history)
            resource_errors = [r for r in records if any(term in (r.get('error') or '')
                for term in ('RESOURCE_EXHAUSTED', 'Out of memory', 'CUDA out of memory'))]
            if resource_errors:
                raise RuntimeError('Resource error in evaluation; not a valid method comparison')
        report['run_status'] = 'finished'
    finally:
        worker.close()
        os.chdir(previous)
        (base.task_config, base.approach, base.finish_pointcloud_placement,
         base.verify_visual_trial_lift, base.WorldLA, base.ExternalPerception.select_reference, base.dump_json) = originals
        after = randomized.fingerprints()
        report.update(protected_files_after=after, protected_files_unchanged=before == after,
            implementation_unchanged=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==implementation_hash,
            perception_worker=worker.summary())
        if report['run_status'] != 'finished':
            report['run_status'] = 'interrupted_or_error'
        # Preserve the latest completed records even if a round aborts.
        originals[6](output/'report.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
