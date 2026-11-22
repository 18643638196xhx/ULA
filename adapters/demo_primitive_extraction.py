"""Offline, task-independent demonstration events and primitive rule extraction.

The only mandatory input is recorded robot state.  Visual tracks are optional
and never synthesized from a closed gripper.  All semantic proposals retain
their evidence level; a task name cannot turn a kinematic cue into measured
contact.  Generated rules are input to a future runtime compiler, not legacy
``task_rules`` files, and are never installed automatically.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import median_filter

from adapters.primitive_geometry import (
    GEOMETRY_TEMPLATES, SCHEMA_VERSION as GEOMETRY_SCHEMA_VERSION,
    relation_record, relative_pose,
)


BASIC = ('move', 'grasp', 'release', 'push', 'pull', 'rotate', 'press', 'strike')
COMPOSITE = ('place', 'open', 'handover')


@dataclass(frozen=True)
class ExtractionConfig:
    quiet_fraction: float = .4
    noise_mad_scale: float = 6.
    smooth_frames: int = 3
    minimum_bout_extent_fraction: float = .01
    active_arm_fraction: float = .01
    gripper_open_value: float = 1.
    gripper_closed_value: float = 0.
    gripper_delta_fraction: float = .005
    minimum_gripper_ramp: float = .25
    ramp_gap_frames: int = 2
    local_approach_path_fraction: float = .2
    max_keyframes: int = 5

    def __post_init__(self):
        if (not 0 < self.quiet_fraction < 1 or self.noise_mad_scale <= 0
                or self.smooth_frames < 1 or self.smooth_frames % 2 != 1
                or self.gripper_open_value == self.gripper_closed_value
                or not 0 < self.minimum_bout_extent_fraction < 1
                or not 0 < self.active_arm_fraction < 1
                or not 0 < self.local_approach_path_fraction < 1
                or self.max_keyframes < 2):
            raise ValueError('Invalid shared extraction configuration')


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def pose_matrix(pose):
    pose = np.asarray(pose, float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-9:
        raise ValueError('Expected finite xyz+wxyz recorded EE pose')
    w, x, y, z = pose[3:] / np.linalg.norm(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = pose[:3]
    return matrix


def angles(quaternions, reference=None):
    q = np.asarray(quaternions, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    if reference is None:
        dot = np.sum(q[:-1] * q[1:], axis=-1)
    else:
        ref = np.asarray(reference, float)
        ref = ref / np.linalg.norm(ref)
        dot = np.sum(q * ref, axis=-1)
    return 2 * np.arccos(np.clip(np.abs(dot), 0., 1.))


def validate_library(value):
    if value.get('schema_version') != 1:
        raise ValueError('Expected primitive library schema_version: 1')
    definitions = value.get('primitives', {})
    if set(definitions) != set(BASIC + COMPOSITE):
        raise ValueError('Library must define the eight basic and three composite primitives')
    for name, definition in definitions.items():
        if not definition.get('recognition', {}).get('matcher') or not definition.get('description'):
            raise ValueError(f'Incomplete primitive definition: {name}')
        if definition.get('controller') not in ('traditional', 'vla', 'children'):
            raise ValueError(f'Invalid controller for {name}')
    return definitions


def _blocks(indices, gap=1):
    indices = np.asarray(indices, int)
    if not len(indices): return []
    return [block for block in np.split(indices, np.flatnonzero(np.diff(indices) > gap)+1)
            if len(block)]


def _threshold(values, config):
    quiet = values[values <= np.quantile(values, config.quiet_fraction)]
    centre = float(np.median(quiet))
    mad = float(np.median(np.abs(quiet-centre)))
    threshold = max(centre + config.noise_mad_scale*1.4826*mad, 1e-9)
    return threshold, dict(baseline=centre, mad=mad, threshold=threshold)


def detect_arm_events(poses, gripper, frequency, config=ExtractionConfig()):
    poses, gripper = np.asarray(poses, float), np.asarray(gripper, float).reshape(-1)
    if (poses.ndim != 2 or poses.shape[1] != 7 or len(poses) != len(gripper)
            or len(poses) < 2 or frequency <= 0 or not np.isfinite(poses).all()
            or not np.isfinite(gripper).all() or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-9)):
        raise ValueError('Invalid synchronized recorded arm state')
    aperture = (gripper-config.gripper_closed_value) / (
        config.gripper_open_value-config.gripper_closed_value)
    if np.any((aperture < -.1) | (aperture > 1.1)):
        raise ValueError('Gripper states disagree with the declared open/closed calibration')
    delta = np.diff(aperture)
    events = []
    for kind, indices in (
        ('gripper_close', np.flatnonzero(delta < -config.gripper_delta_fraction)+1),
        ('gripper_open', np.flatnonzero(delta > config.gripper_delta_fraction)+1),
    ):
        for block in _blocks(indices, config.ramp_gap_frames):
            start, end = int(block[0]-1), int(block[-1])
            change = float(aperture[end]-aperture[start])
            if abs(change) < config.minimum_gripper_ramp: continue
            events.append(dict(kind=kind, start_frame=start, end_frame=end,
                               start_value=float(aperture[start]), end_value=float(aperture[end]),
                               evidence='recorded_gripper_state', physical_grasp_proven=False))
    events.sort(key=lambda item: (item['start_frame'], item['end_frame']))
    linear = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)*frequency
    angular = angles(poses[:, 3:])*frequency
    linear_threshold, linear_audit = _threshold(linear, config)
    angular_threshold, angular_audit = _threshold(angular, config)
    moving = (median_filter(linear, size=config.smooth_frames, mode='nearest') > linear_threshold)
    moving |= median_filter(angular, size=config.smooth_frames, mode='nearest') > angular_threshold
    extent = float(np.linalg.norm(np.ptp(poses[:, :3], axis=0)))
    angular_extent = float(np.max(angles(poses[:, 3:], poses[0, 3:])))
    bouts = []
    for block in _blocks(np.flatnonzero(moving)+1):
        start, end = int(block[0]-1), int(block[-1])
        local_extent = float(np.linalg.norm(np.ptp(poses[start:end+1, :3], axis=0)))
        local_angle = float(np.max(angles(poses[start:end+1, 3:], poses[start, 3:])))
        significant = local_extent > max(config.minimum_bout_extent_fraction*extent, 1e-8)
        significant |= local_angle > max(config.minimum_bout_extent_fraction*angular_extent, 1e-6)
        if significant: bouts.append(dict(start_frame=start, end_frame=end))
    return dict(events=events, bouts=bouts, aperture=aperture,
                linear_noise=linear_audit, angular_noise=angular_audit,
                translation_extent_m=extent, rotation_extent_rad=angular_extent,
                path_length_m=float(linear.sum()/frequency))


class VisualTracks:
    """Explicit frame-indexed poses and optional current measured surfaces.

    A missing dynamic sample stays missing.  Only explicitly declared static
    entities can reuse a pose at a different frame.  Initial RGB-D anchors have
    separate semantics and are never mistaken for a post-grasp object pose.
    """

    def __init__(self, poses=None, points=None, static_entities=(), initial_anchors=None, metadata=None):
        self.poses = poses or {}
        self.points = points or {}
        self.static_entities = set(static_entities)
        self.initial_anchors = initial_anchors or {}
        self.metadata = metadata or {}
        for name, samples in self.poses.items():
            for frame, matrix in samples.items():
                matrix = np.asarray(matrix, float)
                if (int(frame) < 0 or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                        or not np.allclose(matrix[3], [0, 0, 0, 1])
                        or not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-4)
                        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-4)):
                    raise ValueError(f'Invalid visual pose: {name}:{frame}')

    @property
    def entities(self):
        return sorted(set(self.poses) | set(self.initial_anchors))

    def pose(self, entity, frame):
        samples = self.poses.get(entity, {})
        if frame in samples: return np.asarray(samples[frame], float)
        if entity in self.static_entities and samples:
            return np.asarray(samples[min(samples)], float)
        return None

    def initial(self, entity):
        value = self.pose(entity, 0)
        if value is None: value = self.initial_anchors.get(entity)
        return None if value is None else np.asarray(value, float)

    def identity_continuity(self, entity, frames):
        """Audit whether one visual track preserves identity at given frames.

        Identity evidence is supplied by the perception producer, not inferred
        from an entity name.  Every requested frame must also have an accepted
        measured pose, so metadata cannot fill a missing handover observation.
        """
        record = self.metadata.get('identity_continuity', {}).get(entity)
        samples = [int(frame) for frame in frames]
        available = all(self.pose(entity, frame) is not None for frame in samples)
        verified = bool(isinstance(record, dict) and record.get('method') and available)
        return dict(verified=verified, entity=entity, frames=samples,
                    all_required_visual_poses_available=available,
                    evidence=record if isinstance(record, dict) else None)

    def surface(self, entity, frame):
        samples = self.points.get(entity, {})
        value = samples.get(frame)
        if value is None and entity in self.static_entities and samples:
            value = samples[min(samples)]
        if value is None: return None
        value = np.asarray(value, float)
        return value[np.isfinite(value).all(axis=1)]

    def contact_proximity(self, entity, arm_poses, start, end):
        calibration = self.metadata.get('T_recorded_ee_contact')
        if calibration is None:
            return dict(available=False, reason='recorded_EE_to_contact_region_calibration_missing')
        calibration = np.asarray(calibration, float)
        if calibration.shape != (4, 4) or not np.isfinite(calibration).all():
            raise ValueError('Invalid T_recorded_ee_contact calibration')
        values = []
        for frame in range(start, end+1):
            surface = self.surface(entity, frame)
            if surface is None or len(surface) < 3: continue
            location = (pose_matrix(arm_poses[frame])@calibration)[:3, 3]
            diameter = float(np.linalg.norm(np.ptp(surface, axis=0)))
            distance = float(np.linalg.norm(surface-location, axis=1).min())
            uncertainty = float(self.metadata.get('observation_uncertainty_m', 0.))
            tolerance = max(uncertainty, .05*diameter, 1e-9)
            values.append(dict(frame=frame, nearest_surface_distance_m=distance,
                               demo_scaled_tolerance_m=tolerance, near=bool(distance <= tolerance)))
        if not values: return dict(available=False, reason='synchronous_measured_object_surface_missing')
        return dict(available=True, supported=any(value['near'] for value in values),
                    samples=values, physical_contact_proven=False,
                    reference='calibrated_contact_region_to_measured_RGBD_surface')

    def target_approach(self, entity, target, start, end):
        frames = [frame for frame in range(start, end+1)
                  if self.pose(entity, frame) is not None and self.pose(target, frame) is not None]
        if len(frames) < 2:
            return dict(available=False, reason='synchronous_object_and_target_poses_missing')
        distances = [float(np.linalg.norm(self.pose(entity, frame)[:3, 3]-self.pose(target, frame)[:3, 3]))
                     for frame in frames]
        return dict(available=True, supported=bool(distances[-1] < distances[0]),
                    start_distance_m=distances[0], end_distance_m=distances[-1],
                    sampled_frames=frames,
                    reference='declared_visual_pose_origins_not_hidden_support_points')

    def object_rotation(self, entity, start, end):
        frames = [frame for frame in range(start, end+1) if self.pose(entity, frame) is not None]
        if len(frames) < 2: return dict(available=False, reason='visual_object_rotation_samples_missing')
        initial = self.pose(entity, frames[0])[:3, :3]
        rotations = np.array([self.pose(entity, frame)[:3, :3] for frame in frames])
        excursion = np.arccos(np.clip((np.trace(initial.T@rotations, axis1=1, axis2=2)-1)/2, -1., 1.))
        return dict(available=True, maximum_excursion_deg=float(np.degrees(excursion.max())),
                    final_rotation_deg=float(np.degrees(excursion[-1])), sampled_frames=frames)

    def follow_evidence(self, entity, arm_poses, start, end):
        samples = [frame for frame in range(start, end+1) if self.pose(entity, frame) is not None]
        if len(samples) < 3:
            return dict(available=False, reason='insufficient_synchronous_visual_object_samples')
        relatives = [np.linalg.inv(pose_matrix(arm_poses[frame]))@self.pose(entity, frame)
                     for frame in samples]
        world = np.array([self.pose(entity, frame)[:3, 3] for frame in samples])
        hands = arm_poses[samples, :3]
        movement = np.linalg.norm(world[-1]-world[0])
        hand_movement = np.linalg.norm(hands[-1]-hands[0])
        translations = np.array([value[:3, 3] for value in relatives])
        translation_residual = float(np.max(np.linalg.norm(translations-np.median(translations, axis=0), axis=1)))
        rotations = np.array([value[:3, :3] for value in relatives])
        rotation_residual = float(np.max(np.arccos(np.clip(
            (np.trace(rotations[0].T@rotations, axis1=1, axis2=2)-1)/2, -1, 1))))
        # Ratios are shared detector settings; never a task-specific distance.
        scale = max(float(movement), float(hand_movement), 1e-6)
        supported = (min(movement, hand_movement) > 1e-5
                     and translation_residual <= .2*scale and rotation_residual <= np.deg2rad(10))
        return dict(available=True, supported=bool(supported), sampled_frames=samples,
                    translation_residual_m=translation_residual,
                    rotation_residual_deg=float(np.degrees(rotation_residual)),
                    object_displacement_m=float(movement), hand_displacement_m=float(hand_movement))


def infer_intents(task_name, description, definitions):
    text = (task_name.replace('_', ' ')+' '+description).lower()
    result = []
    for primitive in ('handover', 'open', 'strike', 'press', 'pull', 'push', 'rotate', 'place'):
        words = definitions[primitive].get('recognition', {}).get('intent_words', [])
        patterns = definitions[primitive].get('recognition', {}).get('intent_patterns', [])
        if (any(re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)', text) for word in words)
                or any(re.search(pattern, text) for pattern in patterns)):
            result.append(primitive)
    return result


def _frame_summary(poses, start, end, frequency):
    delta = poses[end, :3]-poses[start, :3]
    path = np.diff(poses[start:end+1, :3], axis=0)
    orientation = angles(poses[start:end+1, 3:], poses[start, 3:])
    return dict(delta_world_xyz_m=delta, duration_s=(end-start)/frequency,
                path_length_m=float(np.linalg.norm(path, axis=1).sum()),
                orientation_change_deg=float(np.degrees(orientation[-1])),
                angular_path_deg=float(np.degrees(angles(poses[start:end+1, 3:]).sum())),
                maximum_orientation_excursion_deg=float(np.degrees(orientation.max())))


def _select_keyframes(poses, start, end, maximum):
    selected = {start, end}
    if end-start > 2 and maximum > 2:
        points = poses[start:end+1, :3]
        delta = points[-1]-points[0]
        denominator = float(delta@delta)
        fraction = np.zeros(len(points)) if denominator < 1e-14 else np.clip(
            (points-points[0])@delta/denominator, 0, 1)
        residual = np.linalg.norm(points-(points[0]+fraction[:, None]*delta), axis=1)
        if residual.max() > 1e-7: selected.add(start+int(np.argmax(residual)))
        excursion = angles(poses[start:end+1, 3:], poses[start, 3:])
        selected.add(start+int(np.argmax(excursion)))
        if len(selected) < maximum: selected.add((start+end)//2)
    return sorted(selected)[:maximum-1]+([end] if end not in sorted(selected)[:maximum-1] else [])


def _approach_frame(bouts, previous_end, close_start, poses, config):
    before = [bout for bout in bouts if bout['end_frame'] <= close_start
              and bout['start_frame'] >= previous_end]
    if len(before) >= 2:
        return int(np.floor((before[-2]['end_frame']+before[-1]['start_frame'])/2+.5)), 'pause_before_local_approach'
    if not before: return close_start, 'no_preclose_motion_observed'
    start = before[-1]['start_frame']
    path = np.linalg.norm(np.diff(poses[start:close_start+1, :3], axis=0), axis=1)
    if path.sum() < 1e-9: return start, 'angular_approach_start'
    cumulative = np.r_[0., np.cumsum(path)]
    frame = start+int(np.searchsorted(cumulative, (1-config.local_approach_path_fraction)*cumulative[-1]))
    return frame, 'last_fraction_of_measured_approach_path_candidate'


def _make_stage(primitive, arm, start, end, definitions, recognition):
    definition = definitions[primitive]
    return dict(primitive=primitive, arm=arm, demo_frames=[int(start), int(end)],
                description=definition['description'], controller=definition['controller'],
                executor=definition.get('executor'), completion=definition['completion'],
                required_parameters=definition['parameters'], recognition=recognition,
                evidence_level='recorded_robot_state' if primitive in ('move', 'release')
                else 'candidate_from_robot_events', physical_contact_confirmed=False)


def _arm_stages(arm, poses, detected, frequency, definitions, config):
    events, bouts = detected['events'], detected['bouts']
    interactions, episodes = [], []
    previous_end = 0
    for event in events:
        if event.get('role') == 'gripper_preparation':
            previous_end = event['end_frame']
            continue
        if event['kind'] == 'gripper_open':
            interactions.append(_make_stage('release', arm, event['start_frame'], event['end_frame'],
                                           definitions, dict(matcher='gripper_opening', event=event)))
            previous_end = event['end_frame']
            continue
        start, source = _approach_frame(bouts, previous_end, event['start_frame'], poses, config)
        next_open = next((value for value in events if value['kind'] == 'gripper_open'
                          and value['start_frame'] >= event['end_frame']), None)
        limit = next_open['start_frame'] if next_open else len(poses)-1
        later = [bout for bout in bouts if event['end_frame'] <= bout['start_frame'] < limit]
        end = later[0]['start_frame'] if later else event['end_frame']
        stage = _make_stage('grasp', arm, start, end, definitions,
                            dict(matcher='gripper_closing', event=event, approach_reference_source=source,
                                 object_follows_hand=dict(available=False)))
        interactions.append(stage)
        episodes.append(dict(arm=arm, grasp=stage, close_event=event, release_event=next_open,
                             held_interval_is_candidate=True))
        previous_end = end
    stages, cursor = [], 0

    def add_motion(first, last):
        eligible = []
        for bout in bouts:
            start, end = max(first, bout['start_frame']), min(last, bout['end_frame'])
            if end > start: eligible.append((start, end))
        for index, (start, end) in enumerate(eligible):
            lower = first if index == 0 else int(np.floor((eligible[index-1][1]+start)/2+.5))
            upper = last if index == len(eligible)-1 else int(np.floor((end+eligible[index+1][0])/2+.5))
            if upper <= lower: continue
            stage = _make_stage('move', arm, lower, upper, definitions,
                                dict(matcher='measured_motion', active_bout=[start, end]))
            stage['gripper_state'] = 'closed_candidate' if np.median(detected['aperture'][lower:upper+1]) < .5 else 'open'
            stage['motion'] = _frame_summary(poses, lower, upper, frequency)
            stage['motion_role'] = ('lift' if stage['motion']['delta_world_xyz_m'][2] > 0 else 'lower') \
                if np.argmax(np.abs(stage['motion']['delta_world_xyz_m'])) == 2 else 'transport'
            stages.append(stage)

    for stage in sorted(interactions, key=lambda item: item['demo_frames'][0]):
        start, end = stage['demo_frames']
        if start < cursor:
            start = max(cursor, stage['recognition']['event']['start_frame'])
            stage['demo_frames'][0] = start
        add_motion(cursor, start)
        if end >= start: stages.append(stage)
        cursor = max(cursor, end)
    add_motion(cursor, len(poses)-1)
    for index, stage in enumerate(stages):
        stage['id'] = f'{arm}_{index:03d}_{stage["primitive"]}'
    return stages, episodes


def _episode_motions(episode, stages, end_frame=None):
    first = episode['grasp']['demo_frames'][1]
    release = episode['release_event']
    last = (release['start_frame'] if release else 10**12) if end_frame is None else end_frame
    return [stage for stage in stages if stage['arm'] == episode['arm'] and stage['primitive'] == 'move'
            and stage['demo_frames'][0] >= first and stage['demo_frames'][1] <= last]


def _relabel(stage, primitive, definitions, reason):
    definition = definitions[primitive]
    stage.update(primitive=primitive, description=definition['description'], controller=definition['controller'],
                 executor=definition.get('executor'), completion=definition['completion'],
                 required_parameters=definition['parameters'], evidence_level='task_conditioned_candidate')
    stage['recognition']['semantic_basis'] = reason
    stage['recognition']['physical_action_verified'] = False


def _make_composite(primitive, children, definitions, reason, **fields):
    first = min(stage['demo_frames'][0] for stage in children)
    last = max(stage['demo_frames'][1] for stage in children)
    controllers = set(stage['controller'] for stage in children)
    controller = next(iter(controllers)) if len(controllers) == 1 else 'mixed'
    return dict(id=f'{primitive}_{first}_{last}', primitive=primitive, description=definitions[primitive]['description'],
                controller=controller, demo_frames=[first, last], children=children,
                completion=definitions[primitive]['completion'], required_parameters=definitions[primitive]['parameters'],
                recognition=dict(matcher=definitions[primitive]['recognition']['matcher'], semantic_basis=reason),
                evidence_level='task_conditioned_candidate', **fields)


def _keyframe_geometry(stage, tracks, visual, object_name, target, initial_reference, maximum):
    arm = stage['arm']
    poses = tracks[arm]
    start, end = stage['demo_frames']
    matrices = {frame: pose_matrix(poses[frame]) for frame in _select_keyframes(poses, start, end, maximum)}
    start_pose = pose_matrix(poses[start])
    keyframes, missing = [], []
    for frame, matrix in matrices.items():
        value = dict(frame=frame, T_world_recorded_ee=matrix,
                     T_stage_start_ee_recorded_ee=np.linalg.inv(start_pose)@matrix,
                     relative_to_current_entities={}, relative_to_initial_visual_anchors={})
        for entity in visual.entities:
            current = visual.pose(entity, frame)
            initial = visual.initial(entity)
            if current is not None:
                value['relative_to_current_entities'][entity] = np.linalg.inv(current)@matrix
            if initial is not None:
                value['relative_to_initial_visual_anchors'][entity] = np.linalg.inv(initial)@matrix
        keyframes.append(value)
    geometry = dict(controlled_frame='recorded_ee', keyframes=keyframes,
                    requires_runtime_tool_frame_reconciliation=True,
                    reusable_object_goal=None, object_to_hand=None,
                    relation_template=GEOMETRY_TEMPLATES[stage['primitive']],
                    demonstrated_relations=[])
    reference = initial_reference or (object_name if stage['primitive'] == 'grasp' else target)
    stage['object'], stage['reference'] = object_name, reference
    current_reference = visual.pose(reference, end)
    reference_pose = current_reference
    reference_scope = 'current_visual_pose'
    if reference_pose is None and initial_reference:
        reference_pose = visual.initial(reference)
        reference_scope = 'initial_visual_anchor_fallback'
    if reference_pose is not None:
        geometry['T_reference_recorded_ee_hint'] = np.linalg.inv(reference_pose)@pose_matrix(poses[end])
        geometry['reference_scope'] = reference_scope
    else:
        missing.append('object_or_target_relative_goal')
    if stage['primitive'] == 'grasp' and object_name:
        precontact = visual.pose(object_name, start)
        if precontact is None and initial_reference == object_name:
            precontact = visual.initial(object_name)
        if precontact is not None:
            geometry['demonstrated_relations'].append(relation_record(
                'precontact', object_name, 'recorded_ee', start,
                precontact, pose_matrix(poses[start]), 'synchronous_visual_pose'))
        obj = visual.pose(object_name, end)
        if obj is not None:
            geometry['object_to_hand'] = relative_pose(pose_matrix(poses[end]), obj)
            geometry['demonstrated_relations'].append(relation_record(
                'attachment', 'recorded_ee', object_name, end,
                pose_matrix(poses[end]), obj, 'synchronous_postgrasp_visual_pose'))
        else: missing.append('postgrasp_object_to_hand')
    elif stage['primitive'] == 'move' and object_name:
        if (stage.get('gripper_state') == 'open' and initial_reference
                and reference_pose is not None):
            geometry['demonstrated_relations'].append(relation_record(
                'approach', object_name, 'recorded_ee', end,
                reference_pose, pose_matrix(poses[end]),
                'initial_visual_anchor' if reference_scope.startswith('initial_')
                else 'synchronous_visual_pose'))
        elif stage.get('gripper_state') == 'closed_candidate' and target:
            obj, support = visual.pose(object_name, end), visual.pose(target, end)
            if obj is not None and support is not None:
                relation = relation_record('held_transport', target, object_name, end,
                                           support, obj, 'synchronous_visual_pose')
                geometry['demonstrated_relations'].append(relation)
                geometry['reusable_object_goal'] = relation['T_reference_controlled']
    elif stage['primitive'] in ('push', 'pull', 'rotate', 'press', 'strike') and object_name:
        for frame in matrices:
            obj = visual.pose(object_name, frame)
            if obj is not None:
                geometry['demonstrated_relations'].append(relation_record(
                    'contact_path', object_name, 'recorded_ee', frame,
                    obj, pose_matrix(poses[frame]), 'synchronous_visual_pose'))
    if stage['primitive'] in ('push', 'pull', 'press', 'strike'):
        missing.append('verified_contact_reference_and_tool_contact_frame')
        path = poses[start:end+1, :3]
        delta = path[-1]-path[0]
        geometry['direction_world_hint'] = None if np.linalg.norm(delta) < 1e-9 else delta/np.linalg.norm(delta)
        geometry['speed_profile_m_s'] = np.linalg.norm(np.diff(path, axis=0), axis=1)*stage['frequency_hz']
        geometry['speed_profile_role'] = 'demonstrated_timing_evidence_not_runtime_force_control'
    if stage['primitive'] == 'rotate':
        geometry['relative_rotation'] = np.linalg.inv(start_pose)@pose_matrix(poses[end])
        geometry['relative_rotation_role'] = 'recorded_hand_motion_not_semantic_object_axis'
        missing.append('object_or_articulation_rotation_axis_and_goal')
    if stage['primitive'] == 'release':
        geometry['gripper_open_value'] = 1.
        missing = []
    stage['geometry'], stage['missing_geometry'] = geometry, sorted(set(missing))


def extract_rule(task, description, arm_tracks, grippers, frequency, library, *,
                 visual=None, object_name=None, target=None, config=ExtractionConfig()):
    definitions = validate_library(library)
    visual = visual or VisualTracks()
    if set(arm_tracks) != set(grippers) or not arm_tracks:
        raise ValueError('Pose and gripper arms must match')
    counts = {len(value) for value in arm_tracks.values()}
    if len(counts) != 1: raise ValueError('Arm tracks must be synchronized')
    frame_count = counts.pop()
    detected = {arm: detect_arm_events(track, grippers[arm], frequency, config)
                for arm, track in arm_tracks.items()}
    max_extent = max(value['translation_extent_m'] for value in detected.values())
    max_angle = max(value['rotation_extent_rad'] for value in detected.values())
    active = [arm for arm, value in detected.items() if value['events']
              or value['translation_extent_m'] > max(config.active_arm_fraction*max_extent, 1e-8)
              or value['rotation_extent_rad'] > max(config.active_arm_fraction*max_angle, 1e-6)]
    intents = infer_intents(task, description, definitions)
    preparations = []
    direct_press = ('press' in intents and 'place' not in intents
                    and (target is None or target == object_name)
                    and not re.search(r'\b(grasp|grab|pick|hold|lift)\b', description.lower()))
    for arm, value in detected.items():
        for event in value['events']:
            if event['kind'] != 'gripper_close': continue
            prior_motion = any(bout['end_frame'] <= event['start_frame'] for bout in value['bouts'])
            next_open = next((other['start_frame'] for other in value['events']
                              if other['kind'] == 'gripper_open' and other['start_frame'] > event['end_frame']), frame_count)
            repeated_close = any(other['kind'] == 'gripper_close'
                                 and event['end_frame'] < other['start_frame'] < next_open
                                 for other in value['events'])
            observed_follow = False
            if object_name:
                following = next((bout for bout in value['bouts'] if bout['start_frame'] >= event['end_frame']), None)
                if following:
                    observed_follow = visual.follow_evidence(object_name, arm_tracks[arm],
                                                             event['end_frame'], following['end_frame']).get('supported', False)
            preparation_candidate = (not prior_motion and repeated_close) or direct_press
            if preparation_candidate and not observed_follow:
                event['role'] = 'gripper_preparation'
                preparations.append(dict(arm=arm, action='configure_gripper', controller='traditional',
                                         demo_frames=[event['start_frame'], event['end_frame']],
                                         normalized_target_opening=event['end_value'],
                                         evidence_level='preparatory_close_candidate',
                                         description='接触操作前的夹爪配置，不将其当成已抓住物体'))
    leaves, episodes = [], []
    for arm in active:
        stages, local_episodes = _arm_stages(arm, arm_tracks[arm], detected[arm], frequency, definitions, config)
        leaves.extend(stages)
        episodes.extend(local_episodes)
    for episode in episodes:
        motions = _episode_motions(episode, leaves)
        episode['object'] = object_name
        episode['object_binding_source'] = 'declared_or_description_candidate' if object_name else 'unresolved'
        if object_name is None and motions:
            matches = []
            first, last = episode['grasp']['demo_frames'][1], motions[0]['demo_frames'][1]
            for entity in visual.entities:
                if entity == target: continue
                evidence = visual.follow_evidence(entity, arm_tracks[episode['arm']], first, last)
                obj = visual.pose(entity, first)
                if not evidence.get('supported') or obj is None: continue
                distance = float(np.linalg.norm(obj[:3, 3]-arm_tracks[episode['arm']][first, :3]))
                matches.append((distance, entity))
            matches.sort()
            if matches and (len(matches) == 1 or matches[1][0]-matches[0][0] > .05*max(matches[1][0], 1e-6)):
                episode['object'] = matches[0][1]
                episode['object_binding_source'] = 'visual_comotion_and_proximity_candidate'
        if episode['object'] and motions:
            evidence = visual.follow_evidence(episode['object'], arm_tracks[episode['arm']],
                                              episode['grasp']['demo_frames'][1], motions[0]['demo_frames'][1])
            episode['grasp']['recognition']['object_follows_hand'] = evidence
            if evidence.get('supported'): episode['grasp']['evidence_level'] = 'visual_motion_supported'

    for stage in leaves:
        local = [episode for episode in episodes if episode['arm'] == stage['arm']
                 and episode['grasp']['demo_frames'][0] <= stage['demo_frames'][0]
                 and (episode['release_event'] is None
                      or stage['demo_frames'][0] <= episode['release_event']['end_frame'])]
        if not local:
            local = [episode for episode in episodes if episode['arm'] == stage['arm']
                     and stage['demo_frames'][1] <= episode['grasp']['demo_frames'][0]]
        selected_episode = local[-1] if local and stage['demo_frames'][0] >= local[-1]['grasp']['demo_frames'][0] else (local[0] if local else None)
        stage['object'] = selected_episode['object'] if selected_episode else object_name
        stage['object_binding_source'] = selected_episode['object_binding_source'] if selected_episode else 'declared_or_unresolved'
        stage['object_candidates'] = [entity for entity in visual.entities if entity != target] if stage['object'] is None else [stage['object']]

    visual_events = []
    for stage in leaves:
        start, end = stage['demo_frames']
        local_object = stage['object']
        stage['visual_evidence'] = dict(
            contact_proximity=visual.contact_proximity(local_object, arm_tracks[stage['arm']], start, end),
            object_rotation=visual.object_rotation(local_object, start, end))
        if stage['primitive'] == 'move':
            follow = visual.follow_evidence(local_object, arm_tracks[stage['arm']], start, end)
            approach = visual.target_approach(local_object, target, start, end)
            stage['visual_evidence'].update(object_follows_hand=follow, object_approaches_target=approach)
            if follow.get('supported'):
                stage['holding_evidence'] = 'visual_rigid_comotion_not_physical_grasp_certificate'
                visual_events.append(dict(kind='object_follows_hand', arm=stage['arm'], object=local_object,
                                          start_frame=start, end_frame=end, evidence=follow))
            if approach.get('supported'):
                stage['variant'] = 'move_to'
                visual_events.append(dict(kind='object_approaches_target', arm=stage['arm'], object=local_object,
                                          target=target, start_frame=start, end_frame=end, evidence=approach))

    # Semantic labels are proposals. Without target tracking the evidence level
    # explicitly stays task-conditioned instead of claiming observed contact.
    contact_intent = next((kind for kind in ('strike', 'press', 'pull', 'push') if kind in intents), None)
    if contact_intent:
        for arm in active:
            candidates = [stage for stage in leaves if stage['arm'] == arm and stage['primitive'] == 'move']
            if not candidates: continue
            closed = [stage for stage in candidates if stage['gripper_state'] == 'closed_candidate']
            selected = (closed or candidates)[-1]
            if contact_intent == 'press':
                downward = [stage for stage in (closed or candidates) if stage['motion']['delta_world_xyz_m'][2] < 0]
                if downward: selected = max(downward, key=lambda stage: -stage['motion']['delta_world_xyz_m'][2])
            _relabel(selected, contact_intent, definitions, 'task_name_and_description_plus_measured_motion')
            selected['alternative_interpretations'] = ['move', 'unverified_contact_action']
            local_preparations = [value for value in preparations if value['arm'] == arm
                                  and value['demo_frames'][1] <= selected['demo_frames'][1]]
            if local_preparations:
                selected['gripper_configuration'] = local_preparations
                first = min(value['demo_frames'][0] for value in local_preparations)
                previous = [stage for stage in leaves if stage['arm'] == arm and stage is not selected
                            and stage['demo_frames'][0] <= first < stage['demo_frames'][1]
                            and stage['demo_frames'][1] <= selected['demo_frames'][0]]
                if previous:
                    previous[-1]['demo_frames'][1] = first
                    selected['demo_frames'][0] = first
                for preparation in local_preparations: preparation['handled_by_stage'] = selected['id']
    elif 'rotate' in intents and 'open' not in intents:
        for arm in active:
            candidates = [stage for stage in leaves if stage['arm'] == arm and stage['primitive'] == 'move']
            closed = [stage for stage in candidates if stage['gripper_state'] == 'closed_candidate']
            if closed or candidates:
                selected = max(closed or candidates, key=lambda stage: stage['motion']['angular_path_deg'])
                if selected['motion']['maximum_orientation_excursion_deg'] > 1e-5:
                    _relabel(selected, 'rotate', definitions, 'orientation_task_intent_and_recorded_angular_motion')

    roots, consumed, handovers = [], set(), []
    if 'handover' in intents:
        for receiver in episodes:
            receive_close = receiver['close_event']
            for donor in episodes:
                release = donor['release_event']
                if (donor['arm'] == receiver['arm'] or release is None
                        or donor['close_event']['end_frame'] >= receive_close['start_frame']
                        or receive_close['end_frame'] > release['start_frame']): continue
                if donor['object'] and receiver['object'] and donor['object'] != receiver['object']: continue
                if any(value['kind'] == 'gripper_open'
                       and donor['close_event']['end_frame'] < value['start_frame'] < receive_close['start_frame']
                       for value in detected[donor['arm']]['events']): continue
                donor_release = next((stage for stage in leaves if stage['arm'] == donor['arm']
                                      and stage['primitive'] == 'release'
                                      and stage['demo_frames'] == [release['start_frame'], release['end_frame']]), None)
                if donor_release is None or donor_release['id'] in consumed: continue
                children = [receiver['grasp'], donor_release]
                if any(stage['id'] in consumed for stage in children): continue
                # Waiting with the receiver closed belongs to the overlap,
                # not an unfinished grasp extending beyond donor release.
                receiver['grasp']['demo_frames'][1] = receive_close['end_frame']
                # Motion segmentation may include the receiver's waiting or
                # shared holding before donor release in its later carry bout.
                # Split at the semantic transfer boundary and keep that early
                # motion inside handover rather than prematurely inside place.
                receiver_after = []
                for movement in list(leaves):
                    first, last = movement['demo_frames']
                    boundary = release['end_frame']
                    if (movement['arm'] != receiver['arm'] or movement['primitive'] != 'move'
                            or first < receive_close['end_frame']): continue
                    if first < boundary < last:
                        overlap = deepcopy(movement)
                        overlap['id'] += '_handover_overlap'
                        overlap['demo_frames'][1] = boundary
                        movement['demo_frames'][0] = boundary
                        leaves.append(overlap)
                        children.append(overlap)
                        receiver_after.append(movement)
                    elif receive_close['end_frame'] <= first and last <= boundary:
                        children.append(movement)
                    elif first >= boundary:
                        receiver_after.append(movement)
                group = _make_composite('handover', children, definitions,
                                        'handover_intent_plus_receiver_close_before_donor_release',
                                        donor_arm=donor['arm'], receiver_arm=receiver['arm'],
                                        same_object_visually_verified=bool(
                                            donor['object'] and donor['object'] == receiver['object']
                                            and donor['object_binding_source'] == receiver['object_binding_source']
                                            == 'visual_comotion_and_proximity_candidate'))
                transfer_object = donor['object'] or receiver['object']
                relations, transfer_missing = [], []
                receiver_start = receiver['grasp']['demo_frames'][0]
                receiver_end = receive_close['end_frame']
                donor_release_start = release['start_frame']
                transfer_frame = donor_release_start
                identity = visual.identity_continuity(
                    transfer_object, (receiver_start, receiver_end, donor_release_start))
                group['same_object_visually_verified'] = bool(
                    group['same_object_visually_verified'] or identity['verified'])
                group['same_object_visual_evidence'] = identity
                for name, arm, frame, reference_first in (
                    ('receiver_precontact', receiver['arm'], receiver_start, True),
                    # Sample both attachments at the same instant: receiver
                    # closed, donor still closed. This is the physical
                    # transfer configuration used by the runtime.
                    ('receiver_attachment', receiver['arm'], transfer_frame, False),
                    ('donor_attachment', donor['arm'], transfer_frame, False),
                ):
                    obj = visual.pose(transfer_object, frame)
                    hand = pose_matrix(arm_tracks[arm][frame])
                    if obj is None:
                        transfer_missing.append(name)
                        continue
                    relations.append(relation_record(
                        name,
                        transfer_object if reference_first else f'{arm}_recorded_ee',
                        f'{arm}_recorded_ee' if reference_first else transfer_object,
                        frame, obj if reference_first else hand,
                        hand if reference_first else obj,
                        'synchronous_visual_pose'))
                group['geometry'] = dict(
                    schema_version=GEOMETRY_SCHEMA_VERSION,
                    relation_template=GEOMETRY_TEMPLATES['handover'],
                    object=transfer_object,
                    demonstrated_relations=relations,
                    runtime_rule='receiver_close_must_finish_before_donor_release')
                if target:
                    transfer_obj_pose = visual.pose(transfer_object, transfer_frame)
                    transfer_target_pose = visual.pose(target, transfer_frame)
                    if transfer_obj_pose is not None and transfer_target_pose is not None:
                        group['geometry']['reusable_object_goal'] = relative_pose(
                            transfer_target_pose, transfer_obj_pose
                        ).tolist()
                        group['geometry']['object_goal_frame'] = transfer_frame
                group['missing_geometry'] = transfer_missing
                if not group['same_object_visually_verified']:
                    group['missing_geometry'].append('same_object_tracking_across_handover')
                group['receiver_after_transfer'] = ([min(receiver_after, key=lambda stage: stage['demo_frames'][0])['id']]
                                                     if receiver_after else [])
                roots.append(group)
                handovers.append(group)
                consumed.update(stage['id'] for stage in children)
                break

    if 'open' in intents:
        for arm in active:
            interactions = [stage for stage in leaves if stage['arm'] == arm and stage['primitive'] == 'grasp']
            movements = [stage for stage in leaves if stage['arm'] == arm and stage['primitive'] == 'move']
            if not interactions and not movements: continue
            if 'place' in intents:
                engagement = [episode for episode in episodes if episode['arm'] == arm
                              and episode['release_event'] is None]
                if not engagement: continue
                interactions = [episode['grasp'] for episode in engagement]
            start = (interactions[0]['demo_frames'][0] if interactions else
                     movements[min(1, len(movements)-1)]['demo_frames'][0])
            children = [stage for stage in leaves if stage['arm'] == arm
                        and stage['demo_frames'][0] >= start and stage['id'] not in consumed]
            if not children: continue
            group = _make_composite('open', children, definitions, 'open_intent_with_observed_interaction_sequence', arm=arm)
            group['missing_geometry'] = ['articulation_axis_or_path', 'visual_open_state_goal']
            group['note'] = 'Interior motions remain move candidates until hinge/slide evidence identifies the constraint.'
            roots.append(group)
            consumed.update(stage['id'] for stage in children)

    if 'place' in intents or 'handover' in intents:
        for episode in episodes:
            release = episode['release_event']
            if release is None: continue
            candidates = [stage for stage in leaves if stage['arm'] == episode['arm']
                          and stage['id'] not in consumed
                          and stage['demo_frames'][0] >= episode['grasp']['demo_frames'][1]
                          and stage['demo_frames'][1] <= release['end_frame']]
            if (not any(stage['primitive'] in ('move', 'rotate') for stage in candidates)
                    or not any(stage['primitive'] == 'release' for stage in candidates)): continue
            # Do not absorb an already extracted non-placement interaction.
            if any(stage['primitive'] not in ('move', 'rotate', 'release') for stage in candidates): continue
            group = _make_composite('place', candidates, definitions,
                                    'placement_goal_and_postclose_motion_then_release', arm=episode['arm'])
            local_object = episode['object']
            obj = visual.pose(local_object, release['start_frame'])
            support = visual.pose(target, release['start_frame'])
            relative = None if obj is None or support is None else np.linalg.inv(support)@obj
            held_pose = visual.pose(local_object, episode['grasp']['demo_frames'][1])
            object_to_hand = None if held_pose is None else np.linalg.inv(
                pose_matrix(arm_tracks[episode['arm']][episode['grasp']['demo_frames'][1]]))@held_pose
            group['object_goal'] = dict(object=local_object, reference=target,
                                        T_reference_object=relative,
                                        orientation_policy='requires_task_success_constraint',
                                        source='synchronous_prerelease_visual_poses' if relative is not None else 'unresolved')
            group['demo_object_to_hand'] = object_to_hand
            group['runtime_object_to_hand'] = 'remeasure_after_this_run_grasp'
            group['geometry_template'] = GEOMETRY_TEMPLATES['place']
            group['geometry_schema_version'] = GEOMETRY_SCHEMA_VERSION
            group['missing_geometry'] = ([] if relative is not None else ['object_to_target_goal'])
            if object_to_hand is None: group['missing_geometry'].append('postgrasp_object_to_hand')
            roots.append(group)
            consumed.update(stage['id'] for stage in candidates)

    # Geometry is attached after relabelling, so parameters always agree with
    # the executable primitive's meaning instead of with a stale motion label.
    first_grasp = {arm: min((stage['demo_frames'][0] for stage in leaves
                            if stage['arm'] == arm and stage['primitive'] == 'grasp'), default=frame_count)
                   for arm in active}
    keyframes = {}
    for stage in leaves:
        stage['frequency_hz'] = frequency
        stage['motion'] = _frame_summary(arm_tracks[stage['arm']], *stage['demo_frames'], frequency)
        local_object = stage['object']
        initial_reference = local_object if stage['primitive'] == 'move' and stage['demo_frames'][1] <= first_grasp[stage['arm']] else None
        _keyframe_geometry(stage, arm_tracks, visual, local_object, target, initial_reference, config.max_keyframes)
        stage['keyframe_ids'] = []
        for value in stage['geometry'].pop('keyframes'):
            value['gripper_opening_normalized'] = float(detected[stage['arm']]['aperture'][value['frame']])
            identifier = f'{stage["id"]}_frame{value["frame"]}'
            keyframes[identifier] = value
            stage['keyframe_ids'].append(identifier)
    roots.extend(stage for stage in leaves if stage['id'] not in consumed)
    roots.sort(key=lambda stage: (stage['demo_frames'][0], stage['demo_frames'][1], stage['id']))

    # Leaves form per-arm chains.  Extra cross-arm edges enforce receive before
    # donor release; independent arms retain concurrency instead of global sort.
    edges = set()
    for arm in active:
        chain = sorted((stage for stage in leaves if stage['arm'] == arm), key=lambda stage: stage['demo_frames'][0])
        edges.update((left['id'], right['id']) for left, right in zip(chain, chain[1:]))
    for group in handovers:
        edges.add((group['children'][0]['id'], group['children'][1]['id']))
        for stage_id in group.get('receiver_after_transfer', []):
            edges.add((group['children'][1]['id'], stage_id))
    for stage in leaves:
        stage['depends_on'] = sorted(left for left, right in edges if right == stage['id'])
    observed = []
    for arm, value in detected.items():
        observed.extend(dict(arm=arm, **event) for event in value['events'])
    observed.extend(visual_events)
    observed.sort(key=lambda event: (event['start_frame'], event['arm']))
    blockers = sorted({item for stage in leaves+roots for item in stage.get('missing_geometry', [])})
    return plain(dict(
        schema='ula_demo_primitives_v1', task=task, description=description,
        geometry_schema_version=GEOMETRY_SCHEMA_VERSION,
        frame_count=frame_count, frequency_hz=frequency,
        frame_indexing='zero_based_inclusive_reference_frames', active_arms=active,
        task_intents=intents, bindings=dict(object=object_name, target=target),
        controller_assignment_source='primitive_library_ULA_policy_not_teacher_labels',
        config=asdict(config), stages=roots,
        gripper_configuration_events=preparations,
        execution_graph=dict(nodes=[stage['id'] for stage in leaves],
                             edges=[list(edge) for edge in sorted(edges)],
                             independent_arms_can_overlap=True),
        keyframes=keyframes, events=observed,
        arm_audit={arm: {key: value for key, value in audit.items() if key != 'aperture'}
                   for arm, audit in detected.items()},
        visual_input_provenance=visual.metadata,
        runtime_ready=False,
        runtime_blockers=blockers+['new_schema_requires_runtime_compiler', 'recorded_EE_to_tool_frame_mapping'],
        robot_actions=0, live_simulator_reads=False, uses_CAD=False,
        limitations=[
            'Gripper closure and measured hand motion are not physical grasp certificates.',
            'Task-conditioned contact/action labels remain candidates without visual evidence.',
            'Initial visual anchors do not substitute for post-grasp object poses.',
            'Demonstrated frame numbers and timings are evidence, not runtime stop commands.',
        ],
    ))
