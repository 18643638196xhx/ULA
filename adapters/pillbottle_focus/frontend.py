"""RGB-D candidate selection for the explicitly configured orange/white bottle.

Appearance is a declared task prior, not inferred from simulator identities.
Color proposes/ranks targets; SAM supplies the full measured bottle surface.
CAD fitting, motion planning and grasp handoff remain in the existing backend.
"""
import time
from types import SimpleNamespace

import numpy as np

from adapters.cup_target_focus import world_points_image, visual_table_height


def appearance_masks(rgb):
    colors = np.asarray(rgb)[..., :3].astype(float)/255.
    red, green, blue = np.moveaxis(colors, -1, 0)
    orange = (red > .28) & (red-green > .12) & (green-blue > .035)
    white = (colors.max(2) > .40) & (colors.max(2)-colors.min(2) < .30)
    pad = (blue > .25) & (blue-red > .15) & (blue-green > .15)
    return orange, white, pad


def components(mask, minimum):
    from scipy.ndimage import label, binary_closing
    labels, count = label(binary_closing(mask, iterations=1))
    result = []
    for index in range(1, count+1):
        selected = (labels == index) & mask
        if selected.sum() >= minimum:
            result.append(selected)
    return result


def mask_box(mask):
    y, x = np.nonzero(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=float)


def box_mask(box, shape):
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Invalid focused detection box")
    y, x = np.indices(shape)
    return (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])


def semantic_evidence(mask, detections):
    return max((float(score)*float(np.sum(mask & box_mask(box, mask.shape))/max(mask.sum(), 1))
                for box, score in detections), default=0.)


def bottle_proposals(rgb, world, valid, table_z, detections, hint=None):
    orange, white, _ = appearance_masks(rgb)
    foreground = valid & (world[..., 2] > table_z+.003) & (world[..., 2] < table_z+.35)
    proposals = [dict(box=np.asarray(box), semantic=float(score), source="semantic")
                 for box, score in detections[:5]]
    # Orange and white parts must be connected above the measured table. A
    # white tabletop or a separate orange distractor does not become a bottle.
    for mask in components((orange | white) & foreground, 40):
        if np.sum(mask & orange) < 6 or np.sum(mask & white) < 12:
            continue
        proposals.append(dict(box=mask_box(mask)+[-2, -2, 2, 2],
            semantic=semantic_evidence(mask, detections), source="color_depth"))
    if hint is not None:
        proposals.insert(0, dict(box=np.asarray(hint), semantic=0., source="projected_reference"))
    unique = []
    for proposal in proposals:
        try:
            support = box_mask(proposal["box"], valid.shape) & foreground & (orange | white)
        except ValueError:
            continue
        if support.sum() < 20:
            continue
        y, x = np.nonzero(support)
        # Use an actually measured bright foreground point, not the box center
        # which can land on a background hole or a neighboring gripper.
        xy = np.c_[x, y]
        proposal["point"] = xy[np.argmin(np.linalg.norm(xy-np.median(xy, axis=0), axis=1))]
        proposal["priority"] = proposal["semantic"]+min(np.sum(support & orange)/30., 1.)
        duplicate = next((item for item in unique
            if np.max(np.abs(item["box"]-proposal["box"])) < 4), None)
        if duplicate is None:
            unique.append(proposal)
        else:
            duplicate["semantic"] = max(duplicate["semantic"], proposal["semantic"])
    return sorted(unique, key=lambda item: item["priority"], reverse=True)[:6]


def bottle_metrics(mask, rgb, world, valid, table_z, *, held=False, semantic=0., hinted=False):
    orange, white, _ = appearance_masks(rgb)
    measured = mask & valid & (world[..., 2] > table_z+.003) & (world[..., 2] < table_z+.35)
    points = world[measured]
    reasons = []
    if len(points) < 40:
        return dict(accepted=False, reasons=["insufficient_measured_foreground"]), measured
    lower, upper = np.quantile(points, [.01, .99], axis=0)
    spans = upper-lower
    aspect = float(spans[2]/max(max(spans[:2]), .001))
    orange_count = int(np.sum(measured & orange))
    white_fraction = float(np.sum(measured & white)/len(points))
    if min(spans[:2]) < .008 or max(spans) > .25 or spans[2] < .015:
        reasons.append("not_compact_bottle_extent")
    if not held and not .65 <= aspect <= 4.:
        reasons.append("not_initial_upright_bottle")
    appearance_complete = orange_count >= 6 and white_fraction >= .12
    occluded_color_supported = held and white_fraction >= .25 and (semantic >= .30 or hinted)
    if not (appearance_complete or occluded_color_supported):
        reasons.append("orange_white_appearance_not_supported")
    if not hinted and semantic < .10:
        reasons.append("no_matching_semantic_evidence")
    return dict(accepted=not reasons, reasons=reasons, spans_m=spans.tolist(),
        center_world=((lower+upper)/2).tolist(), orange_pixels=orange_count,
        white_fraction=white_fraction, upright_aspect=aspect,
        cap_occlusion_allowed=bool(not appearance_complete and occluded_color_supported),
        score=float(2*semantic+min(orange_count/30., 1.)+white_fraction)), measured


def pad_candidates(rgb, world, valid, table_z, detections):
    _, _, blue = appearance_masks(rgb)
    plane = valid & (np.abs(world[..., 2]-table_z) <= .01)
    candidates = []
    for mask in components(blue & plane, 20):
        points = world[mask]
        lo, hi = np.quantile(points, [.01, .99], axis=0)
        spans = hi-lo
        aspect = min(spans[:2])/max(max(spans[:2]), .001)
        evidence = semantic_evidence(mask, detections)
        if min(spans[:2]) < .025 or max(spans[:2]) > .20 or aspect < .55 or evidence < .10:
            continue
        center = (lo+hi)/2
        center[2] = np.median(points[:, 2])
        candidates.append(dict(mask=mask, center=center, box=mask_box(mask),
            score=float(2*evidence+aspect), semantic=evidence, spans_m=spans.tolist()))
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def choose_unambiguous(candidates, *, center_key="center", margin=.12):
    if not candidates:
        raise RuntimeError("No color-depth-semantic candidate passed validation")
    best = candidates[0]
    for other in candidates[1:]:
        if np.linalg.norm(np.asarray(best[center_key])-other[center_key]) > .06:
            if best["score"]-other["score"] < margin:
                raise RuntimeError("Ambiguous color-depth-semantic targets; refusing identity jump")
    return best


class FocusedBottleFrontend:
    """Frontend interface consumed by the existing known-bottle CAD backend."""
    @staticmethod
    def estimate(self, rgb, depth, intrinsic, world_to_camera, cup_only=False,
                 refine_cup=False, object_roi_xyxy=None, object_point_xy=None, target_only=False):
        from adapters.pointcloud_geometry import estimate_visible_bounds
        from adapters.pillbottle import observation_quality

        start = time.perf_counter()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        world, valid = world_points_image(depth, intrinsic, world_to_camera)
        table_z = getattr(self, "table_height_world", None) if cup_only or target_only else None
        if table_z is None:
            table_z = visual_table_height(world, valid)

        def detect(prompt):
            found, scores, _ = self._run_dino(rgb, prompt+" .")
            scores = scores.detach().cpu().numpy().reshape(-1)
            order = np.argsort(scores)[::-1][:5]
            return [(self._xyxy(found[int(i)].detach().cpu().numpy(), rgb.shape[1], rgb.shape[0]),
                     float(scores[i])) for i in order]

        pad, pad_mask, pad_audit = np.full(3, np.nan), np.zeros(valid.shape, bool), {}
        if not cup_only or target_only:
            pads = pad_candidates(rgb, world, valid, table_z, detect(self.prompts["coaster"]))
            selected = choose_unambiguous(pads)
            pad, pad_mask = selected["center"], selected["mask"]
            pad_audit = {k: v.tolist() if isinstance(v, np.ndarray) else v
                         for k, v in selected.items() if k != "mask"}
        common = dict(table_height_m=float(table_z), uses_object_truth=False,
            uses_robot_identity_mask=False, uses_simulator_truth_for_target=False,
            perception_frontend="pillbottle_color_depth_semantic",
            appearance_prior="orange_cap_white_body_and_blue_square_pad",
            appearance_prior_source="explicit_task_configuration_not_simulator_identity",
            pad_focus=pad_audit)
        if target_only:
            return SimpleNamespace(cup_bounds_world=np.full((2, 3), np.nan),
                cup_points_world=np.empty((0, 3)), coaster_world_center=pad,
                observation_audit=dict(common, target_only=True),
                timings_s=dict(perception_total=time.perf_counter()-start))
        detections = detect("orange cap white pill bottle")
        # Generic class evidence is still useful under shadows and hand occlusion.
        detections += detect(self.prompts["cup"])
        detections.sort(key=lambda item: item[1], reverse=True)
        proposals = bottle_proposals(rgb, world, valid, table_z, detections, object_roi_xyxy)
        attempts, accepted = [], []
        for proposal in proposals:
            try:
                def validator(mask):
                    metrics, _ = bottle_metrics(mask, rgb, world, valid, table_z,
                        held=bool(cup_only), semantic=proposal["semantic"],
                        hinted=proposal["source"] == "projected_reference")
                    return metrics["accepted"]
                mask = self._refine_cup_mask(rgb, proposal["box"], proposal["point"], validator)
                metrics, measured = bottle_metrics(mask, rgb, world, valid, table_z,
                    held=bool(cup_only), semantic=proposal["semantic"],
                    hinted=proposal["source"] == "projected_reference")
                bounds, audit, points = estimate_visible_bounds(measured, pad_mask, depth,
                    intrinsic, world_to_camera, return_points=True)
                y, x = np.nonzero(measured)
                audit.update(common, cup_mask_touches_border=bool(np.any(x < 2) or np.any(y < 2)
                    or np.any(x >= rgb.shape[1]-2) or np.any(y >= rgb.shape[0]-2)),
                    image_size_hw=list(rgb.shape[:2]), coaster_observed=not cup_only,
                    cup_mask_bbox_pixels=mask_box(measured).astype(int).tolist(),
                    cup_segmentation_variant="color_depth_semantic_roi_positive_point",
                    object_focus=metrics, projected_reference_prompt=object_roi_xyxy is not None)
                value = SimpleNamespace(cup_bounds_world=bounds, cup_points_world=points,
                    coaster_world_center=pad, observation_audit=audit)
                quality = observation_quality(value)
                if not quality["geometry_ok"]:
                    raise RuntimeError(str(quality))
                audit["geometry_quality"] = quality
                accepted.append(dict(value=value, center=metrics["center_world"], score=metrics["score"]))
            except (RuntimeError, ValueError) as error:
                attempts.append(dict(box=proposal["box"].tolist(), source=proposal["source"], error=str(error)))
        try:
            value = choose_unambiguous(sorted(accepted, key=lambda item: item["score"], reverse=True))["value"]
        except RuntimeError as error:
            alternatives = [{k: v for k, v in item.items() if k != "value"} for item in accepted]
            raise RuntimeError(f"Focused bottle observation rejected: {error}; "
                               f"alternatives={alternatives}; attempts={attempts}") from error
        value.observation_audit["focused_candidate_attempts"] = attempts
        value.observation_audit["focused_candidates_accepted"] = len(accepted)
        value.timings_s = dict(perception_total=time.perf_counter()-start)
        return value
