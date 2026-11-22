#!/usr/bin/env python3
"""Paired randomized seed9/21 grasp diagnostic. Never runs traditional placement."""
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
spec = importlib.util.spec_from_file_location('handoff_timing_base', ROOT/'scripts/run_cup_trial_diagnostic.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
base, randomized, known = diagnostic.base, diagnostic.randomized, diagnostic.known
from adapters.cup_trial_diagnostic import TrialCupWorker, verify_segmented_trial
from adapters.cup_handoff_timing import DelayedCommandMonitor, PairedObservationPolicy, prefix_comparison


class TimingVisionWorker(TrialCupWorker):
    """Optional physical GPU separation, without changing vision computation."""
    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        selected = os.environ.get('ULA_TIMING_VISION_GPU')
        if selected is not None:
            if not selected.isdigit():
                raise ValueError('Select one physical vision GPU index')
            environment['CUDA_VISIBLE_DEVICES'] = selected
        return command, cwd, environment


def scoring_sample(task, arm, phase, action=None):
    """Passive diagnostic trace. Its values never enter any control decision."""
    return dict(phase=phase, step=int(task.take_action_cnt),
        action=None if action is None else np.asarray(action).tolist(),
        actual_ee_pose=list(task.get_arm_pose(arm)),
        gripper_command=float(getattr(task.robot, f'get_{arm}_gripper_val')()),
        cup_world_pose=np.asarray(task.cup.get_pose().to_transformation_matrix()).tolist())


def original_policy_indices(report):
    index = -1
    found = {}
    for episode in report['episodes']:
        for attempt in episode.get('grasp_attempts', []):
            if attempt.get('vla_steps', 0) > 0:
                index += 1
        if episode['seed'] in (9,21):
            found[episode['seed']] = index
    return found


def main():
    args = base.parse_args()
    if args.mode != 'grasp-only' or args.grasp_handoff != 'visual-trial-lift' or args.approach_mode != 'pointcloud':
        raise ValueError('Use grasp-only + pointcloud + visual-trial-lift')
    if args.grasp_retries or args.seed_cache:
        raise ValueError('No automatic regrasp or seed cache in timing experiment')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT/'results/cup_known_semantic_trial_full_randomized_50eps_seed0_20261102_v1/report.json'
    indices = original_policy_indices(json.loads(source.read_text()))
    resource = randomized.configure_runtime()
    resource.update(vision_gpu=os.environ.get('ULA_TIMING_VISION_GPU', os.environ.get('CUDA_VISIBLE_DEVICES')),
                    split_vision_gpu='ULA_TIMING_VISION_GPU' in os.environ)
    protected = randomized.fingerprints()
    implementation_paths = ['scripts/run_cup_handoff_timing.py', 'adapters/cup_handoff_timing.py']
    implementation_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in implementation_paths}
    config = randomized.randomized_config(base.task_config(args))
    originals = (base.task_config, base.approach, base.make_grasp_monitor, base.rollout,
                 base.verify_visual_trial_lift, base.WorldLA)
    policy = PairedObservationPolicy(ROOT, args.model_dir, args.asset_checkpoint,
        blend=args.blend, replan_mode=args.replan_mode, inference_seed=args.inference_seed)
    context = {}
    policy.context = context
    base.task_config = lambda unused: deepcopy(config)
    base.approach = known.approach_known_cup
    base.WorldLA = lambda *a, **kw: policy
    base.make_grasp_monitor = lambda options, z: DelayedCommandMonitor(context,
        extra_actions=context['extra_actions'], stable_steps=options.grasp_stable_steps,
        gripper_threshold=options.contact_gripper_threshold)

    def traced_rollout(task, policy, z, record, monitor, arm, stall_check=None):
        original_action = task.take_action
        def action(command, *a, **kw):
            supplied = np.asarray(command, dtype=float).copy()
            step = int(task.take_action_cnt)
            reference = context.get('reference')
            # Freeze actual control commands before the counterfactual boundary.
            # With replayed observations and reset RNG these should already be
            # identical; any deviation is logged rather than hidden.
            if reference is not None and step < reference['first_gate_step']:
                expected = np.asarray(reference['actions'][step])
                context['max_prefix_policy_proposal_difference'] = max(
                    context['max_prefix_policy_proposal_difference'], float(np.max(np.abs(supplied-expected))))
                supplied = expected.copy()
            value = original_action(supplied, *a, **kw)
            context['actions'].append(supplied.tolist())
            context['trace'].append(scoring_sample(task, arm, 'vla', supplied))
            return value
        task.take_action = action
        try:
            return originals[3](task, policy, z, record, monitor, arm, stall_check)
        finally:
            task.take_action = original_action

    def trial(task, perception, arm, result, **kwargs):
        context['handoff_sample'] = scoring_sample(task, arm, 'before_traditional_trial')
        if context['condition'] == 'vla_only_60':
            result.update(complete=False, trial_not_executed=True,
                reason='VLA-only diagnostic; posthoc lift trace is not an online grasp certificate')
            return dict(ready=False, vla_only_diagnostic=True)
        if not context.get('last_closed_stable', True):
            raise RuntimeError('Delayed VLA state is not closed-stable; no traditional lift')
        original_dense = task.take_dense_action
        def dense(controls, *a, **kw):
            value = original_dense(controls, *a, **kw)
            context['traditional_trace'].append(scoring_sample(task, arm, 'traditional_lift_segment'))
            return value
        task.take_dense_action = dense
        try:
            return verify_segmented_trial(task, perception, arm, result, **kwargs)
        finally:
            task.take_dense_action = original_dense
            context['terminal_sample'] = scoring_sample(task, arm, 'after_control')

    # Record the closed-state qualification without substituting truth evidence.
    original_factory = base.make_grasp_monitor
    def factory(*a):
        monitor = original_factory(*a)
        update = monitor.update
        def checked(*args):
            decision = update(*args)
            context['last_closed_stable'] = decision['closed_stable_at_handoff']
            return decision
        monitor.update = checked
        return monitor
    base.make_grasp_monitor = factory
    base.rollout, base.verify_visual_trial_lift = traced_rollout, trial
    worker = TimingVisionWorker(ROOT, args.output/'vision_worker', args.fp_iterations, backend='foundationpose')
    output, previous = args.output, Path.cwd()
    previous_index = os.environ.get('WORLDLA_DIAG_EPISODE_INDEX')
    report = dict(purpose='paired_handoff_timing_diagnosis_not_task_success_rate',
        source=str(source), original_policy_episode_indices=indices, resource_policy=resource,
        scoring_truth_recorded_passively_not_used_for_control=True,
        traditional_placement_enabled=False, cases=[], run_status='running',
        protected_files_before=protected, implementation_hashes=implementation_hashes)
    try:
        for seed in [9,21]:
            reference = None
            for condition, extra in [('delay_0',0),('delay_10',10),('delay_20',20),('vla_only_60',60)]:
                context.clear()
                context.update(seed=seed, condition=condition, extra_actions=extra, reference=reference,
                    actions=[], trace=[], traditional_trace=[], prefix_observations={},
                    replayed_observation_starts=[], max_prefix_policy_proposal_difference=0.)
                os.environ['WORLDLA_DIAG_EPISODE_INDEX'] = str(indices[seed])
                current = deepcopy(args)
                current.seed, current.episodes = seed, 1
                current.output = output/f'seed_{seed:02d}_{condition}'
                base.run_batch(current, worker)
                episode = json.loads((current.output/'report.json').read_text())['episodes'][0]
                comparison = prefix_comparison(context)
                audit = {k:v for k,v in context.items() if k not in ('reference','prefix_observations')}
                audit.update(prefix_comparison=comparison,
                    visual_trial_pass=bool((episode.get('grasp_handoff') or {}).get('ready')),
                    episode_error=episode.get('error'), failure_stage=episode.get('failure_stage'),
                    uses_prepared_model_for_approach=True, trial_thresholds_unchanged=True)
                base.dump_json(current.output/'timing_audit.json', audit)
                report['cases'].append(dict(seed=seed, condition=condition,
                    first_gate_step=context.get('first_gate_step'), vla_steps=episode['vla_steps'],
                    visual_trial_pass=audit['visual_trial_pass'], prefix_comparison=comparison,
                    audit_path=str(current.output/'timing_audit.json'), error=episode.get('error')))
                if reference is None:
                    if 'first_gate_step' not in context:
                        raise RuntimeError('No baseline handoff gate; cannot create paired branches')
                    reference = deepcopy(context)
                base.dump_json(output/'report.json', report)
        report['run_status'] = 'finished'
    finally:
        worker.close()
        os.chdir(previous)
        (base.task_config, base.approach, base.make_grasp_monitor, base.rollout,
         base.verify_visual_trial_lift, base.WorldLA) = originals
        if previous_index is None:
            os.environ.pop('WORLDLA_DIAG_EPISODE_INDEX', None)
        else:
            os.environ['WORLDLA_DIAG_EPISODE_INDEX'] = previous_index
        after = randomized.fingerprints()
        report.update(perception_worker=worker.summary(), protected_files_after=after,
            protected_files_unchanged=protected == after, completed_cases=len(report['cases']),
            implementation_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
                                         for p,h in implementation_hashes.items()))
        if report['run_status'] != 'finished':
            report['run_status'] = 'interrupted_or_error'
        base.dump_json(output/'report.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
