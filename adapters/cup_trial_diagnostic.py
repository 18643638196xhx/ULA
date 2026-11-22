"""Opt-in grasp diagnostic. Existing cup approach/placement remain untouched."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from adapters.cup_known_model import KnownCupVision, KnownCupWorker
from adapters.cup_initial_perception import blue_evidence
from adapters.early_grasp import slower_trajectory
from adapters.handoff import handoff_ready
from adapters.pointcloud_geometry import estimate_visible_bounds
from adapters.pointcloud_registration import RegistrationConfig, estimate_pointcloud_motion
from adapters.reobservation import observation_quality
from adapters.visual_grasp import visual_comotion


def local_identity_geometry(bounds, points, anchor):
    """Partial visible surface is allowed; upright/full-cup shape is not required."""
    bounds, points, anchor = map(np.asarray, (bounds, points, anchor))
    spans = bounds[1] - bounds[0]
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 40
            or not np.isfinite(points).all() or not np.isfinite(bounds).all()
            or np.min(spans[:2]) < .008 or np.max(spans) > .18 or spans[2] < .015):
        raise RuntimeError('Untrusted partial cup extent')
    distance = float(np.linalg.norm(bounds.mean(0) - anchor))
    if distance > .20:
        raise RuntimeError('Candidate outside episode-local cup identity envelope')
    return distance


class TrialCupVision(KnownCupVision):
    """Keep the original known-CAD approach; replace only cup-only observations."""
    def estimate(self, rgb, depth, intrinsic, world_to_camera, cup_only=False,
                 refine_cup=False, require_cad=False):
        if not cup_only:
            value = super().estimate(rgb, depth, intrinsic, world_to_camera,
                cup_only=False, refine_cup=refine_cup, require_cad=require_cad)
            if require_cad:
                self.trial_anchor = value.cup_bounds_world.mean(0).copy()
                self.trial_initial_bounds = value.cup_bounds_world.copy()
            return value
        if not hasattr(self, 'trial_anchor'):
            raise RuntimeError('Missing visually established episode cup reference')
        started = time.perf_counter()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        anchor = self.trial_anchor
        empty = np.zeros(rgb.shape[:2], dtype=bool)
        audit = dict(source='episode_local_semantic_partial_surface',
            color_is_hard_gate=False, uses_prepared_object_model=False,
            uses_foundationpose=False, uses_simulator_truth_for_target=False,
            robot_region_filtering=False, initial_visual_anchor_world=anchor.tolist(),
            identity_envelope_m=.20, candidates=[], fallback_used=False)
        candidates = []

        def geometry(mask):
            bounds, item, points = estimate_visible_bounds(mask, empty, depth,
                intrinsic, world_to_camera, return_points=True)
            distance = local_identity_geometry(bounds, points, anchor)
            return bounds, item, points, distance

        def credible(mask):
            try:
                geometry(mask)
                return True
            except (RuntimeError, ValueError):
                return False

        # Blue is a language cue and small ranking bonus, never a pixel gate.
        # Generic fallback is explicit, bounded, and still spatially constrained.
        for caption in ('blue cup .', 'cup . mug .'):
            found, logits, _ = self._run_dino(rgb, caption)
            scores = logits.detach().cpu().numpy().reshape(-1)
            audit['fallback_used'] = caption != 'blue cup .'
            for index in np.argsort(scores)[::-1][:4]:
                box = self._xyxy(found[int(index)].detach().cpu().numpy(), rgb.shape[1], rgb.shape[0])
                item = dict(caption=caption, box=np.asarray(box).tolist(), accepted=False)
                audit['candidates'].append(item)
                try:
                    mask = self._refine_cup_mask(rgb, box, mask_validator=credible)
                    bounds, geom, points, distance = geometry(mask)
                    rank = float(scores[int(index)]) + .05*blue_evidence(rgb, mask) - 2*distance
                    item.update(accepted=True, distance_m=distance, rank=rank)
                    candidate = dict(mask=mask, bounds=bounds, points=points,
                                     geometry=geom, score=rank, item=item)
                    if not any(np.linalg.norm(bounds.mean(0)-c['bounds'].mean(0)) < .035
                               for c in candidates):
                        candidates.append(candidate)
                except (RuntimeError, ValueError) as error:
                    item['error'] = str(error)
            if candidates:
                break
        candidates.sort(key=lambda c: c['score'], reverse=True)
        if not candidates:
            raise RuntimeError(f'No local semantic cup surface: {audit}')
        if len(candidates) > 1 and candidates[0]['score']-candidates[1]['score'] < .10:
            raise RuntimeError(f'Local cup identity remains ambiguous: {audit}')
        value = candidates[0]
        rows, cols = np.nonzero(value['mask'])
        audit.update(value['geometry'])
        audit.update(selected=value['item'], cup_mask_touches_border=bool(
            np.any(rows < 2) or np.any(cols < 2)
            or np.any(rows >= rgb.shape[0]-2) or np.any(cols >= rgb.shape[1]-2)),
            cup_mask_bbox_pixels=[int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())])
        return SimpleNamespace(cup_bounds_world=value['bounds'], cup_points_world=value['points'],
            cup_world_pose=None, cup_mask=value['mask'], observation_audit=audit,
            coaster_world_center=np.full(3, np.nan),
            timings_s=dict(perception_total=time.perf_counter()-started))


class TrialCupWorker(KnownCupWorker):
    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(self.root/'scripts/perceive_cup_trial_diagnostic.py')
        return command, cwd, environment


def save_frames(task, perception, arm, label):
    """Save BOTH raw views even if inference subsequently rejects a frame."""
    from PIL import Image
    frames = {}
    for name in ('head_camera', f'{arm}_camera'):
        frame = deepcopy(task.get_camera_observation(name))
        frames[name] = frame
        directory = perception.output/'raw_frames'/label/name
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory/'rgbd.npz', rgb=frame['rgb'], depth=frame['depth'],
            intrinsic=frame['intrinsic_cv'], world_to_camera=frame['extrinsic_cv'],
            actual_ee_pose=np.asarray(task.get_arm_pose(arm)))
        Image.fromarray(np.asarray(frame['rgb'], dtype=np.uint8)[..., :3]).save(directory/'rgb.png')
    return frames


def verify_segmented_trial(task, perception, arm, result, *, lift_m=.05,
                           required_lift_m=.03, registration_config=None):
    """Intermediate observations are diagnostics, not early grasp certificates."""
    if not np.isfinite([lift_m, required_lift_m]).all() or not .03 <= lift_m <= .10 or not 0 < required_lift_m <= lift_m:
        raise ValueError('Invalid trial distances')
    config = registration_config or RegistrationConfig()
    result.update(complete=False, detector='segmented_rgbd_registered_surface_comotion',
        uses_simulator_truth_for_verification=False, slip_check_enabled=False,
        robot_region_filtering=False, observations=[], segments=[],
        failure_policy='stop_without_release_or_placement', lift_distance_m=lift_m,
        required_visual_lift_m=required_lift_m, intermediate_observations_diagnostic_only=True)
    names = ['head_camera', f'{arm}_camera']
    baseline = save_frames(task, perception, arm, 'before')
    references = {}
    start = np.asarray(task.get_arm_pose(arm), dtype=float)
    for name in names:
        item = dict(stage='baseline', camera=name)
        result['observations'].append(item)
        try:
            ref = perception.supplementary(baseline[name], f'trial_before_{name}', refine_cup=True)
            item['quality'] = observation_quality(ref)
            item['perception'] = ref.observation_audit
            if item['quality']['geometry_ok']:
                references[name] = ref
        except (RuntimeError, ValueError) as error:
            item['error'] = str(error)
    if not references:
        result['failure_category'] = 'observation_unavailable_before_lift'
        raise RuntimeError('No credible local cup baseline; no lift executed')
    for index in range(3):
        target = start.copy()
        target[2] += lift_m*(index+1)/3
        plan = getattr(task.robot, f'{arm}_plan_path')(target)
        segment = dict(index=index+1, target_ee_pose=target.tolist(), status=plan['status'])
        result['segments'].append(segment)
        if plan['status'] != 'Success':
            raise RuntimeError('Diagnostic trial lift planning failed')
        controls = dict(left_arm=None, right_arm=None, left_gripper=None, right_gripper=None)
        controls[f'{arm}_arm'] = slower_trajectory(plan, 2)
        moved = bool(task.take_dense_action(controls))
        actual = np.asarray(task.get_arm_pose(arm), dtype=float)
        segment['actual_ee_pose'] = actual.tolist()
        segment['arrival'] = handoff_ready(actual, target, moved and task.plan_success).to_dict()
        frames = save_frames(task, perception, arm, f'segment_{index+1}')
        if not segment['arrival']['ready']:
            raise RuntimeError('Diagnostic trial failed actual robot arrival check')
        decisions = []
        for name in names:
            if name not in references:
                continue
            ref = references[name]
            for refined in (False, True):
                item = dict(stage='after_segment', segment=index+1, camera=name, refined=refined)
                result['observations'].append(item)
                try:
                    after = perception.supplementary(frames[name],
                        f'trial_segment_{index+1}_{name}_{int(refined)}', refine_cup=refined)
                    item['perception'] = after.observation_audit
                    item['quality'] = observation_quality(after, ref.cup_bounds_world)
                    if not item['quality']['geometry_ok']:
                        continue
                    motion, item['registration'] = estimate_pointcloud_motion(
                        ref.cup_points_world, after.cup_points_world, config)
                    if motion is None:
                        continue
                    # Keep original FINAL 3 cm, ratio and direction thresholds.
                    decision = visual_comotion(ref.cup_points_world, motion,
                        actual[:3]-start[:3], minimum_lift=required_lift_m)
                    item['comotion'] = decision
                    decisions.append((name, decision))
                    break
                except (RuntimeError, ValueError) as error:
                    item['error'] = str(error)
        segment['registered_views'] = len(decisions)
        if index != 2:
            continue
        passing = [(name, d) for name, d in decisions if d['ready']]
        # A well-registered opposing view is not silently overridden by another.
        conflicting = bool(passing and any(not d['ready'] for _, d in decisions))
        if passing and not conflicting:
            name, decision = passing[0]
            result.update(complete=True, lifted_grasp_verified=True, selected_camera=name)
            return dict(decision, detector=result['detector'], matching_camera=name,
                lifted_grasp_verified=True, ee_pose=actual.tolist())
        result['failure_category'] = ('view_disagreement' if conflicting else
            'registered_surface_not_following' if decisions else 'observation_or_registration_unavailable')
    raise RuntimeError(f"Visual diagnostic rejected: {result['failure_category']}")
