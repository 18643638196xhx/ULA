"""Opt-in paired handoff experiment; no changes to production grasp control."""
from copy import deepcopy
import hashlib
import numpy as np

from adapters.visual_grasp import ClosedCommandMonitor
from adapters.worldla import WorldLA


def action_hash(actions):
    return hashlib.sha256(np.asarray(actions, dtype='<f8').tobytes()).hexdigest()


class DelayedCommandMonitor(ClosedCommandMonitor):
    """Fixed extra VLA actions after the original FIRST eligible command gate."""
    def __init__(self, context, *, extra_actions, stable_steps=3, gripper_threshold=.05):
        super().__init__(stable_steps=stable_steps, gripper_threshold=gripper_threshold)
        if not isinstance(extra_actions, int) or extra_actions < 0:
            raise ValueError('Invalid handoff delay')
        self.context, self.extra_actions = context, extra_actions
        self.first_step = None

    def update(self, gripper_command, step):
        original = super().update(gripper_command, step)
        if self.first_step is None and original['ready']:
            self.first_step = int(step)
            self.context['first_gate_step'] = int(step)
            self.context['first_gate_decision'] = deepcopy(original)
        elapsed = None if self.first_step is None else int(step)-self.first_step
        # Candidate is permission to inspect, not a grasp certificate. If the
        # delayed state reopened, the caller must refuse the traditional lift.
        return dict(original, ready=bool(elapsed is not None and elapsed >= self.extra_actions),
            original_gate_step=self.first_step, extra_vla_actions=self.extra_actions,
            actions_after_original_gate=elapsed, closed_stable_at_handoff=bool(original['ready']))


class PairedObservationPolicy(WorldLA):
    """Cache baseline observations; delayed branches replay that common prefix.

    Every branch also receives the same policy reset index. This prevents
    inference-seed drift across batch positions. After the common boundary,
    observation inputs are live again, preserving normal closed-loop inference.
    """
    def step(self, task, observation):
        context = self.context
        start = int(task.take_action_cnt)
        reference = context.get('reference')
        if reference is None:
            context['prefix_observations'][start] = deepcopy(observation)
        elif start < reference['first_gate_step']:
            observation = deepcopy(reference['prefix_observations'][start])
            context['replayed_observation_starts'].append(start)
        return super().step(task, observation)


def prefix_comparison(context):
    reference = context.get('reference')
    if reference is None:
        return dict(is_reference=True, first_gate_step=context.get('first_gate_step'))
    length = reference['first_gate_step']
    actions = context['actions'][:length]
    expected = reference['actions'][:length]
    enough = len(actions) == length
    command_equal = enough and np.array_equal(np.asarray(actions), np.asarray(expected))
    drift = None
    object_drift = None
    if enough:
        actual = np.asarray([r['actual_ee_pose'][:3] for r in context['trace'][:length]])
        before = np.asarray([r['actual_ee_pose'][:3] for r in reference['trace'][:length]])
        drift = float(np.linalg.norm(actual-before, axis=1).max())
        # Truth is used ONLY to audit experimental branch equivalence, never
        # as an action, segmentation prompt, grasp gate or planning target.
        actual_cup = np.asarray([r['cup_world_pose'] for r in context['trace'][:length]])[:, :3, 3]
        before_cup = np.asarray([r['cup_world_pose'] for r in reference['trace'][:length]])[:, :3, 3]
        object_drift = float(np.linalg.norm(actual_cup-before_cup, axis=1).max())
    same_gate = context.get('first_gate_step') == length
    return dict(is_reference=False, common_actions=length, enough_actions=enough,
        commands_identical=bool(command_equal), source_action_sha256=action_hash(expected),
        branch_action_sha256=action_hash(actions), same_first_gate=bool(same_gate),
        max_prefix_actual_ee_translation_difference_m=drift,
        max_prefix_cup_origin_difference_m=object_drift,
        physically_paired_within_1mm=bool(command_equal and same_gate
            and drift is not None and drift <= .001 and object_drift <= .001))
