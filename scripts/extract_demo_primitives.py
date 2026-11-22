#!/usr/bin/env python3
"""Extract shared ULA primitive rules from one or many RoboTwin demonstrations.

Single task (CPU only)::

  python scripts/extract_demo_primitives.py --task move_pillbottle_pad \
    --object pillbottle --target pad --output /data2/xuhuixin/ula_demo_rules/run1

All available tasks, one demo each::

  python scripts/extract_demo_primitives.py --all-tasks \
    --output /data2/xuhuixin/ula_demo_rules/all_tasks_run1

Optional visual NPZ format, with no pickle:
  frame_indices: [N] strictly increasing integer HDF5 frame indices
  pose__ENTITY: [N,4,4] world poses; NaN entries are missing samples
  points__ENTITY: optional [N,P,3] measured world surfaces, NaN padding
  metadata_json: scalar JSON with demo_sha256, world_frame='world', units='m',
                 source and optional static_entities.
An entity's pose coordinate convention must stay the same across all frames.
The hash must match this demo; static extrapolation requires an explicit
static_entities declaration. No dynamic pose interpolation is performed.

Optional task-spec YAML can contain object, target, static_entities, intent,
and description, or a 'tasks' mapping of task names to those fields. This is
declarative role binding, not executable per-task code. Existing runtime task
rules and robot runners are not changed by this offline tool.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.demo_primitive_extraction import (
    ExtractionConfig, VisualTracks, extract_rule, plain,
)
from diagnostics.vision50_catalog import TASK_TARGETS


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_visual_tracks(path, demo_hash, frame_count, static_entities):
    if path is None: return VisualTracks(static_entities=static_entities)
    with np.load(path, allow_pickle=False) as values:
        metadata = json.loads(str(values['metadata_json'].item()))
        if metadata.get('demo_sha256') != demo_hash:
            raise ValueError('Visual track demo_sha256 does not match this demonstration')
        if metadata.get('world_frame') != 'world' or metadata.get('units') != 'm':
            raise ValueError('Visual tracks require explicit world_frame=world and units=m')
        indices = np.asarray(values['frame_indices'])
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or not len(indices) or np.any(np.diff(indices) <= 0)
                or indices.min() < 0 or indices.max() >= frame_count):
            raise ValueError('Visual track frame_indices must be unique valid HDF5 indices')
        poses, points = {}, {}
        for key in values.files:
            if not key.startswith('pose__'): continue
            entity = key[len('pose__'):]
            matrices = np.asarray(values[key], float)
            if not entity or matrices.shape != (len(indices), 4, 4):
                raise ValueError('Invalid pose__ENTITY array')
            poses[entity] = {int(frame): matrix for frame, matrix in zip(indices, matrices)
                             if np.isfinite(matrix).all()}
        for key in values.files:
            if not key.startswith('points__'): continue
            entity = key[len('points__'):]
            surfaces = np.asarray(values[key], float)
            if surfaces.ndim != 3 or surfaces.shape[0] != len(indices) or surfaces.shape[2] != 3:
                raise ValueError('Invalid points__ENTITY array')
            points[entity] = {int(frame): surface for frame, surface in zip(indices, surfaces)}
    return VisualTracks(poses=poses, points=points,
                        static_entities=set(static_entities) | set(metadata.get('static_entities', [])),
                        metadata=dict(path=str(path), **metadata))


def initial_rgbd_anchors(directory, handle):
    """Optional verified initial anchors, never dynamic post-grasp poses."""
    import cv2
    with np.load(directory/'rgbd.npz', allow_pickle=False) as values:
        rgb = values['rgb'][..., :3]
        intrinsic = values['intrinsic'] if 'intrinsic' in values else values['intrinsic_cv']
        external = values['world_to_camera'] if 'world_to_camera' in values else values['extrinsic_cv']
    if external.shape == (3, 4): external = np.vstack((external, [0, 0, 0, 1]))
    demo_external = np.diag([1., -1., -1., 1.])@np.linalg.inv(handle['vision/cam_head/extrinsics_matrix'][0])
    if not (np.allclose(demo_external, external, atol=1e-5)
            and np.allclose(intrinsic, handle['vision/cam_head/intrinsic_matrix'][0])):
        raise ValueError('Initial RGB-D camera calibration does not match the demonstration')
    raw = np.frombuffer(handle['vision/cam_head/colors'][0], np.uint8)
    initial = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if initial.shape != rgb.shape: raise ValueError('Initial RGB-D image shape mismatch')
    anchors, audits = {}, {}
    with np.load(directory/'offline_template_masks.npz', allow_pickle=False) as masks, \
         np.load(directory/'template.npz', allow_pickle=False) as clouds:
        for entity in masks.files:
            if entity+'_points_world' not in clouds: continue
            y, x = np.nonzero(masks[entity])
            if not len(x): continue
            x0, x1 = max(0, int(x.min())-5), min(rgb.shape[1], int(x.max())+6)
            y0, y1 = max(0, int(y.min())-5), min(rgb.shape[0], int(y.max())+6)
            scores = cv2.matchTemplate(initial, rgb[y0:y1, x0:x1], cv2.TM_CCOEFF_NORMED)
            _, score, _, position = cv2.minMaxLoc(scores)
            error = float(np.linalg.norm(np.asarray(position)-[x0, y0]))
            audits[entity] = dict(match_score=score, pixel_error=error, accepted=score >= .9 and error <= 2)
            if not audits[entity]['accepted']: continue
            points = np.asarray(clouds[entity+'_points_world'], float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
                raise ValueError('Invalid clean-template points')
            matrix = np.eye(4)
            matrix[:3, 3] = np.quantile(points, [.01, .99], axis=0).mean(axis=0)
            anchors[entity] = matrix
    return anchors, dict(directory=str(directory), targets=audits,
                         scope='initial_snapshot_with_world_aligned_axes_not_semantic_CAD_frame',
                         segmentation_source='offline_renderer_identity_annotation',
                         dynamic_pose_measurements=False)


def success_source(task):
    path = ROOT/'tra_pipe/RoboTwin/envs'/f'{task}.py'
    if not path.is_file(): return dict(available=False)
    source = path.read_text()
    tree = ast.parse(source)
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'check_success']
    return dict(available=bool(methods), path=str(path), sha256=digest(path),
                methods=[ast.get_source_segment(source, node) for node in methods],
                use='static_task_goal_reference_only', method_executed=False)


def infer_bindings(description, entities, object_name, target):
    text = description.lower().replace('_', ' ')
    mentions = []
    for entity in entities:
        if entity.endswith('.*'): continue
        token = entity.replace('_', ' ')
        position = text.find(token)
        if position >= 0: mentions.append((position, entity))
    mentions.sort()
    if object_name is None and mentions: object_name = mentions[0][1]
    if target is None and len(mentions) > 1: target = mentions[-1][1]
    # Fall back to semantic role names, not task names.  This covers common
    # catalog aliases such as box/target_box when language says block/pad.
    target_tokens = {
        'target', 'pad', 'coaster', 'basket', 'plate', 'stand', 'scale',
        'cabinet', 'dustbin', 'rack', 'tray', 'skillet', 'plasticbox',
    }
    if target is None:
        candidates = [entity for entity in entities if not entity.endswith('.*')
                      and target_tokens.intersection(entity.lower().split('_'))]
        if len(candidates) == 1: target = candidates[0]
    if object_name is None and target is not None:
        candidates = [entity for entity in entities
                      if entity != target and not entity.endswith('.*')]
        if len(candidates) == 1: object_name = candidates[0]
    if object_name is None and len(entities) == 1 and not entities[0].endswith('.*'):
        object_name = entities[0]
    return object_name, target


def visit_stages(stages):
    for stage in stages:
        yield stage
        yield from visit_stages(stage.get('children', []))


def resolve_demo(args, task):
    if args.demo: return args.demo.resolve()
    matches = sorted((args.demos/task).glob(f'*/data/episode_{args.episode:07d}.hdf5'))
    if len(matches) != 1:
        raise ValueError(f'{task}: found {len(matches)} demos; specify --demo for a single task')
    return matches[0].resolve()


def extract_task(args, task, library, specifications):
    demo = resolve_demo(args, task)
    source_hash = digest(demo)
    specification = specifications.get('tasks', {}).get(task, {}) if 'tasks' in specifications else specifications
    language_path = args.descriptions/f'{task}.json'
    language = json.loads(language_path.read_text()) if language_path.is_file() else {}
    description = specification.get('description', language.get('full_description', task.replace('_', ' ')))
    entities = specification.get('entities', TASK_TARGETS.get(task, []))
    object_name = args.object or specification.get('object')
    target = args.target or specification.get('target')
    explicitly_bound = object_name is not None or target is not None
    object_name, target = infer_bindings(description, entities, object_name, target)
    static_entities = set(args.static_entity or []) | set(specification.get('static_entities', []))
    with h5py.File(demo, 'r') as handle:
        arm_tracks, grippers = {}, {}
        for arm in ('left', 'right'):
            if f'state/{arm}_ee_poses' not in handle: continue
            arm_tracks[arm] = np.asarray(handle[f'state/{arm}_ee_poses'], float)
            grippers[arm] = np.asarray(handle[f'state/{arm}_ee_joint_states'], float).reshape(-1)
        frequency = float(handle['additional_info/frequency'][()])
        count = len(next(iter(arm_tracks.values())))
        track_path = args.visual_tracks
        if track_path and track_path.is_dir(): track_path = track_path/'tasks'/task/'tracks.npz'
        visual = load_visual_tracks(track_path, source_hash, count, static_entities)
        anchor_audit = None
        if args.clean_sidecar:
            sidecar = args.clean_sidecar
            if (sidecar/'tasks'/task).is_dir(): sidecar = sidecar/'tasks'/task
            anchors, anchor_audit = initial_rgbd_anchors(sidecar, handle)
            visual.initial_anchors.update(anchors)
            for entity in visual.static_entities:
                if entity in anchors and entity not in visual.poses:
                    visual.poses[entity] = {0: anchors[entity]}
            visual.metadata['initial_RGBD'] = anchor_audit
        recognition_description = description+' '+str(specification.get('intent', ''))
        rule = extract_rule(task, recognition_description, arm_tracks, grippers, frequency, library,
                            visual=visual, object_name=object_name, target=target,
                            config=ExtractionConfig(gripper_open_value=args.gripper_open_value,
                                                    gripper_closed_value=args.gripper_closed_value))
    rule['description'] = description
    rule['declared_entities'] = entities
    rule['bindings']['source'] = ('explicit_specification' if explicitly_bound
                                  else 'description_or_semantic_role_name_candidate')
    rule['official_success'] = success_source(task)
    rule['source_demo'] = dict(path=str(demo), sha256=source_hash,
                               recorded_pose_convention='xyz_wxyz_world_recorded_EE',
                               controller_labels_present=False)
    rule['primitive_library_sha256'] = digest(args.primitive_library)
    keyframes = rule.pop('keyframes')
    events = dict(events=rule.pop('events'), arm_audit=rule.pop('arm_audit'))
    rule['keyframes_file'], rule['events_file'] = 'keyframes.json', 'events.json'
    destination = args.output/'tasks'/task
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'rule.yaml').write_text(yaml.safe_dump(rule, allow_unicode=True, sort_keys=False), encoding='utf-8')
    dump(destination/'keyframes.json', keyframes)
    dump(destination/'events.json', events)
    source_unchanged = digest(demo) == source_hash
    if not source_unchanged: raise RuntimeError('Demonstration changed during extraction')
    counts = Counter(stage['primitive'] for stage in visit_stages(rule['stages']))
    row = dict(task=task, status='extracted', source_demo=str(demo), source_unchanged=source_unchanged,
               frame_count=count, active_arms=rule['active_arms'], task_intents=rule['task_intents'],
               top_level_primitives=[stage['primitive'] for stage in rule['stages']],
               primitive_counts=dict(counts), keyframes=len(keyframes), events=len(events['events']),
               runtime_ready=rule['runtime_ready'], runtime_blockers=rule['runtime_blockers'],
               rule=str(destination/'rule.yaml'))
    dump(destination/'report.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument('--task')
    selected.add_argument('--tasks', nargs='+')
    selected.add_argument('--all-tasks', action='store_true')
    parser.add_argument('--demo', type=Path, help='explicit HDF5, single task only')
    parser.add_argument('--demos', type=Path, default=Path('/data2/xuhuixin/RoboTwin/data_official50/demo_clean'))
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--descriptions', type=Path, default=ROOT/'tra_pipe/RoboTwin/description/task_instruction')
    parser.add_argument('--primitive-library', type=Path, default=ROOT/'rules/primitive_library.yaml')
    parser.add_argument('--task-spec', type=Path)
    parser.add_argument('--object')
    parser.add_argument('--target')
    parser.add_argument('--visual-tracks', type=Path)
    parser.add_argument('--clean-sidecar', type=Path)
    parser.add_argument('--static-entity', action='append')
    parser.add_argument('--gripper-open-value', type=float, default=1.)
    parser.add_argument('--gripper-closed-value', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tasks = sorted(TASK_TARGETS) if args.all_tasks else args.tasks or [args.task]
    if len(tasks) != len(set(tasks)): parser.error('Duplicate tasks')
    if len(tasks) > 1 and (args.demo or args.object or args.target):
        parser.error('--demo/--object/--target are single-task options; use --task-spec for batch bindings')
    if any(not task.replace('_', '').isalnum() for task in tasks): parser.error('Invalid task name')
    args.output = args.output.resolve()
    if args.output.exists(): parser.error(f'Refusing to overwrite existing output: {args.output}')
    library = yaml.safe_load(args.primitive_library.read_text())
    specifications = yaml.safe_load(args.task_spec.read_text()) if args.task_spec else {}
    args.output.mkdir(parents=True)
    (args.output/'primitive_library.yaml').write_text(args.primitive_library.read_text(), encoding='utf-8')
    report = dict(schema='ula_demo_primitives_batch_v1', status='running', requested_tasks=tasks,
                  robot_actions=0, live_simulator_reads=False, rows=[])
    dump(args.output/'report.json', report)
    for task in tasks:
        try:
            row = extract_task(args, task, library, specifications)
            print(task, ' '.join(row['top_level_primitives']), f"keyframes={row['keyframes']}", flush=True)
        except Exception as error:
            row = dict(task=task, status='error', error=f'{type(error).__name__}: {error}')
            print(task, row['error'], file=sys.stderr, flush=True)
        report['rows'].append(row)
        report['summary'] = dict(requested=len(tasks), processed=len(report['rows']),
                                extracted=sum(item['status'] == 'extracted' for item in report['rows']),
                                errors=sum(item['status'] == 'error' for item in report['rows']))
        dump(args.output/'report.json', report)
    report['status'] = 'finished' if not report['summary']['errors'] else 'finished_with_errors'
    dump(args.output/'report.json', report)
    if report['summary']['errors']: raise SystemExit(1)


if __name__ == '__main__':
    main()
