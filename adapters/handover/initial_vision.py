"""Color/depth/semantic proposals + SAM, evaluated independently per target."""
import time
from types import SimpleNamespace

import numpy as np

from adapters.cup_target_focus import world_points_image, visual_table_height
from adapters.pillbottle_focus.frontend import components, mask_box, semantic_evidence, choose_unambiguous
from adapters.rgbd_perception import GroundedPointCloudPerception
from adapters.handover.initial_geometry import estimate_supported_upright_block, estimate_platform
from adapters.handover.initial_depth import clean_seed_component


def color_masks(rgb):
    red, green, blue = np.moveaxis(np.asarray(rgb)[..., :3].astype(float)/255., -1, 0)
    return ((red > .25) & (red-green > .15) & (red-blue > .15),
            (blue > .25) & (blue-red > .15) & (blue-green > .15))


class HandoverInitialVision(GroundedPointCloudPerception):
    def estimate(self, rgb, depth, intrinsic, world_to_camera):
        started = time.perf_counter()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        world, valid = world_points_image(depth, intrinsic, world_to_camera)
        table_z = visual_table_height(world, valid)
        red, blue = color_masks(rgb)
        from traditional_place_empty_cup.geometry import homogeneous_transform
        camera_center=np.linalg.inv(homogeneous_transform(world_to_camera))[:3,3]
        def block_geometry(points,height):
            return estimate_supported_upright_block(points,height,camera_center)
        result = SimpleNamespace(cup_world_pose=None, cup_bounds_world=np.full((2, 3), np.nan),
            cup_points_world=np.empty((0, 3)), coaster_world_center=np.full(3, np.nan),
            block_mask=np.zeros(valid.shape, bool), platform_mask=np.zeros(valid.shape, bool),
            observation_audit=dict(table_height_m=table_z, block_output=False, platform_output=False,
                uses_object_truth=False, uses_robot_identity_mask=False, uses_foundationpose=False,
                uses_prepared_cad_file=False, uses_declared_shape_dimensions=True,
                initial_upright_assumption=True, targets_independently_attempted=True), timings_s={})

        for name, color, prompt, eligible, estimator in (
            ("block", red, "red rectangular block . red cuboid",
             valid & (world[..., 2] > table_z+.005) & (world[..., 2] < table_z+.35), block_geometry),
            ("platform", blue, "blue square platform . blue square block . blue pad",
             valid & (np.abs(world[..., 2]-table_z) <= .025), estimate_platform)):
            attempt_started = time.perf_counter()
            attempts, accepted = [], []
            try:
                found, scores, _ = self._run_dino(rgb, prompt+" .")
                detections = [(self._xyxy(box.detach().cpu().numpy(), rgb.shape[1], rgb.shape[0]), float(score))
                              for box, score in zip(found, scores.detach().cpu().numpy().reshape(-1))]
                proposals = []
                masks=(clean_seed_component(color&eligible,world) if name=='block'
                       else components(color&eligible,40))
                for mask in masks:
                    evidence = semantic_evidence(mask, detections)
                    if evidence < .10:
                        attempts.append(dict(stage="proposal", reason="insufficient_matching_semantics",
                            box=mask_box(mask).tolist(), semantic_evidence=evidence))
                        continue
                    if name=='block':
                        points=world[mask]
                        low,high=np.quantile(points,[.01,.99],axis=0)
                        spans=high-low
                        if not .08<=spans[2]<=.24 or max(spans[:2])>.15:
                            attempts.append(dict(stage='proposal',reason='not_a_compact_tall_depth_component',
                                box=mask_box(mask).tolist(),spans_m=spans.tolist()))
                            continue
                    # Full block geometry is checked AFTER SAM. A cropped top
                    # or a stray boundary pixel must not prevent segmentation.
                    if name!='block':
                        try:
                            estimator(world[mask], table_z)
                        except (RuntimeError, ValueError) as error:
                            attempts.append(dict(stage="proposal", reason=str(error), box=mask_box(mask).tolist()))
                            continue
                    proposals.append((evidence, mask))
                for evidence, seed_mask in sorted(proposals, key=lambda item: item[0], reverse=True)[:3]:
                    box = mask_box(seed_mask)+[-3, -3, 3, 3]
                    y, x = np.nonzero(seed_mask)
                    xy = np.c_[x, y]
                    point = xy[np.argmin(np.linalg.norm(xy-np.median(xy, axis=0), axis=1))]
                    def measured(mask):
                        value=mask & color & eligible
                        return clean_seed_component(value,world,seed_mask) if name=='block' else value
                    def credible(mask):
                        if np.sum(mask & seed_mask)/seed_mask.sum() < .6:
                            return False
                        try:
                            estimator(world[measured(mask)], table_z)
                            return True
                        except (RuntimeError, ValueError):
                            return False
                    try:
                        mask = measured(self._refine_cup_mask(rgb, box, point, credible))
                        geometry, audit = estimator(world[mask], table_z)
                        center = geometry[:3, 3] if name == "block" else geometry
                        accepted.append(dict(score=float(2*evidence+min(mask.sum()/500., 1.)),
                            center=center, geometry=geometry, mask=mask, audit=audit,
                            semantic_evidence=evidence))
                    except (RuntimeError, ValueError) as error:
                        attempts.append(dict(stage="segmentation", box=box.tolist(), reason=str(error)))
                selected = choose_unambiguous(sorted(accepted, key=lambda item: item["score"], reverse=True))
                result.observation_audit[name+"_output"] = True
                result.observation_audit[name+"_geometry"] = selected["audit"]
                result.observation_audit[name+"_semantic_evidence"] = selected["semantic_evidence"]
                if name == "block":
                    result.block_mask = selected["mask"]
                    result.cup_world_pose = selected["geometry"]
                    result.cup_points_world = world[result.block_mask]
                    result.cup_bounds_world = np.quantile(result.cup_points_world, [.01, .99], axis=0)
                else:
                    result.platform_mask = selected["mask"]
                    result.coaster_world_center = selected["geometry"]
            except (RuntimeError, ValueError) as error:
                result.observation_audit[name+"_error"] = str(error)
            result.observation_audit[name+"_attempts"] = attempts
            result.timings_s[name+"_perception"] = time.perf_counter()-attempt_started
        result.timings_s["perception_total"] = time.perf_counter()-started
        return result
