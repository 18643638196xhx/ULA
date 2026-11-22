"""Opt-in blue-cup / stationary round-coaster tracking for cluttered RGB-D.

Uses explicit color, initial upright-cup and horizontal-table class priors,
not CAD, actor poses, simulation identity masks or hard-coded object locations.
"""
import os
import time
from types import SimpleNamespace

import numpy as np

from adapters.rgbd_perception import GroundedPointCloudPerception


def world_points_image(depth, intrinsic, extrinsic):
    from traditional_place_empty_cup.geometry import depth_to_metres, homogeneous_transform
    metres = depth_to_metres(depth)
    y, x = np.indices(metres.shape)
    rays = np.linalg.solve(np.asarray(intrinsic), np.stack((x.ravel(), y.ravel(), np.ones(x.size))))
    camera = (rays*metres.ravel()).T
    inverse = np.linalg.inv(homogeneous_transform(extrinsic))
    world = (camera@inverse[:3, :3].T+inverse[:3, 3]).reshape(*metres.shape, 3)
    valid = np.isfinite(world).all(2) & np.isfinite(metres) & (metres > 0)
    return world, valid


def visual_table_height(world, valid):
    values = world[..., 2][valid]
    if len(values) < 80:
        raise RuntimeError("Insufficient depth for focused table estimate")
    labels, counts = np.unique(np.floor(values/.005).astype(np.int64), return_counts=True)
    mode = (labels[np.argmax(counts)]+.5)*.005
    selected = values[np.abs(values-mode) <= .0075]
    if len(selected) < 80:
        raise RuntimeError("No supported horizontal table mode")
    return float(np.median(selected))


_HUE_RANGES = {
    "red": ((0.00, 0.045), (0.955, 1.00)),
    "orange": ((0.045, 0.115),),
    "yellow": ((0.115, 0.185),),
    "green": ((0.185, 0.45),),
    "cyan": ((0.45, 0.52),),
    "blue": ((0.52, 0.72),),
    "purple": ((0.72, 0.83),),
    "pink": ((0.83, 0.955),),
}


def _rgb_to_hsv(colors):
    r, g, b = colors[..., 0], colors[..., 1], colors[..., 2]
    hi, lo = np.max(colors, axis=-1), np.min(colors, axis=-1)
    delta = hi - lo
    hue = np.zeros_like(hi)
    moving = delta > 1e-6
    rmax = moving & (hi == r)
    gmax = moving & (hi == g) & ~rmax
    bmax = moving & ~rmax & ~gmax
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2.0
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4.0
    saturation = np.where(hi > 0, delta / np.maximum(hi, 1e-6), 0.0)
    return (hue / 6.0) % 1.0, saturation, hi


def hue_shape(colors):
    return np.asarray(colors)[..., 0].shape


def _color_mask(colors, names):
    """Boolean mask for the requested colour names; empty means 'any colour'."""
    if not names:
        # No colour prior: return everything and let CAD-derived size/shape decide.
        # The pillbottle bodies are white (saturation ~0), so any colour test would
        # drop them entirely.
        return np.ones(hue_shape(colors), bool)
    hue, saturation, value = _rgb_to_hsv(colors)
    mask = np.zeros(hue.shape, bool)
    for name in names:
        if str(name).lower() in ("white", "grey", "gray"):
            # Achromatic objects carry no hue; match on low saturation / high value
            # instead of a hue band (080_pillbottle bodies are white).
            mask |= (saturation < .18) & (value > .70)
            continue
        ranges = _HUE_RANGES.get(str(name).lower())
        if ranges is None:
            continue
        inside = np.zeros(hue.shape, bool)
        for lo, hi in ranges:
            inside |= (hue >= lo) & (hue <= hi)
        # A tan/brown background wall sits in the orange hue band with low
        # saturation; requiring a higher saturation keeps the coloured object and
        # drops the wall, which otherwise merges into one 86 cm component.
        mask |= inside & (saturation > .45) & (value > .15)
    return mask


def object_candidates(rgb, world, valid, table_z, *, colors=(), initial=True,
                      previous_center=None, aspect_bounds=None):
    from scipy.ndimage import label, binary_closing
    pixels = np.asarray(rgb)[..., :3].astype(float)/255.
    color = _color_mask(pixels, colors)
    foreground = valid & (world[..., 2] > table_z+.005) & (world[..., 2] < table_z+.35)
    # Remove the measured table BEFORE grouping blue fragments. A blue table
    # texture must not connect the cup to the entire image foreground.
    components, count = label(binary_closing(color & foreground, iterations=1) & foreground)
    candidates, rejected = [], []
    for component in range(1, count+1):
        mask = (components == component) & color & foreground
        y, x = np.nonzero(mask)
        if len(y) < 40:
            continue
        points = world[mask]
        lower, upper = np.quantile(points, [.01, .99], axis=0)
        spans = upper-lower
        ratio = float(spans[2]/max(np.max(spans[:2]), .001))
        reason = None
        if np.min(spans[:2]) < .008 or np.max(spans) > .25:
            reason = "not_compact_cup_extent"
        elif initial and aspect_bounds is not None and not aspect_bounds[0] <= ratio <= aspect_bounds[1]:
            reason = "outside_object_aspect_from_cad"
        elif initial and aspect_bounds is None and not .65 <= ratio <= 1.8:
            reason = "not_initial_upright_cup_shape"
        center = (lower+upper)/2
        distance = None if previous_center is None else float(np.linalg.norm(center-previous_center))
        if distance is not None and distance > .35:
            reason = "too_far_from_previous_visual_target"
        item = dict(pixels=len(y), bbox=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
            spans_m=spans.tolist(), upright_aspect=ratio, center_world=center.tolist(),
            distance_from_previous_m=distance)
        if reason:
            rejected.append(dict(item, reason=reason))
            continue
        median = np.median(np.c_[x, y], axis=0)
        seed = np.argmin(np.sum((np.c_[x, y]-median)**2, axis=1))
        candidates.append(dict(item, mask=mask, point_xy=np.array([x[seed], y[seed]], dtype=float)))
    return candidates, rejected


class FocusedCupPerception(GroundedPointCloudPerception):
    def reset_episode(self):
        self.reference_table_z = None
        self.previous_cup_center = None
        self.reference_pad = None

    def _object_aspect_bounds(self):
        """Expected height/planar ratio from the object's own CAD.

        The old window (0.65 .. 1.8) describes a *standing cup*.  Flat objects are
        rejected by it: the mouse measures 0.22 and the stapler 0.27, so neither
        could ever be localized.
        """
        import os as _os
        import json as _json
        from pathlib import Path as _Path
        model_dir = _os.environ.get("ULA_KNOWN_OBJECT_ASSET")
        if not model_dir:
            return None
        root = _Path("/data1/xuhuixin/ula/tra_pipe/RoboTwin/assets/objects") / model_dir
        for name in ("model_data0.json",):
            path = root / name
            if not path.is_file():
                continue
            try:
                meta = _json.loads(path.read_text())
                extents = [abs(float(v)) for v in meta["extents"]]
                scale = meta.get("scale", 1.0)
                if isinstance(scale, (int, float)):
                    extents = [v * float(scale) for v in extents]
                else:
                    extents = [v * float(s) for v, s in zip(extents, scale)]
                # The CAD's local axes are not aligned with world up: the mouse's
                # local z is its longest axis although it rests flat.  Rotate the box
                # by the extracted placement rotation (the pose the object actually
                # takes on its support), then read the world extents.
                rotation = None
                try:
                    from adapters.placement_relations import load_placement_relation
                    task = _os.environ.get("ULA_TASK_NAME")
                    payload = load_placement_relation(task) if task else None
                    if payload is not None:
                        rotation = payload["_matrix"][:3, :3]
                except Exception:
                    rotation = None
                corners = np.array([[sx * extents[0] / 2, sy * extents[1] / 2, sz * extents[2] / 2]
                                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                if rotation is not None:
                    corners = corners @ rotation.T
                world_spans = corners.max(axis=0) - corners.min(axis=0)
                planar = max(world_spans[0], world_spans[1])
                if planar <= 1e-6:
                    continue
                expected = world_spans[2] / planar
                return (expected * 0.40, expected * 2.6)
            except Exception:
                continue
        return None

    def _support_span_bounds(self):
        """Span bounds derived from the support's own CAD, not global constants.

        The original bounds (min 2.5 cm, max 25 cm, aspect >= 0.55) fit a round
        coaster.  Real pads are elongated: the shoe's is 26x10 cm (fails both the
        max and the aspect test), the mouse's 13x7 cm and the stapler's 11x6 cm
        (both miss the 0.55 aspect by ~0.01).
        """
        spec = {}
        try:
            from adapters.unified_perception import PERCEPTION_SPEC
            spec = PERCEPTION_SPEC.get("support_cad") or {}
        except Exception:
            spec = {}
        model_dir = spec.get("model_dir")
        # Workers run out-of-process, so fall back to the environment.
        if not model_dir:
            import os as _os
            model_dir = _os.environ.get("ULA_SUPPORT_MODEL_DIR")
            scale_env = _os.environ.get("ULA_SUPPORT_SCALE")
            if model_dir and scale_env and "scale" not in spec:
                try:
                    spec = dict(spec, scale=float(scale_env))
                except ValueError:
                    pass
        if not model_dir:
            return None
        import json as _json
        from pathlib import Path as _Path
        scale = spec.get("scale", 1.0)
        path = _Path("/data1/xuhuixin/ula/tra_pipe/RoboTwin/assets/objects") / model_dir / "model_data0.json"
        try:
            meta = _json.loads(path.read_text())
            extents = [abs(float(v)) for v in meta["extents"]]
            if isinstance(scale, (int, float)):
                extents = [v * float(scale) for v in extents]
            else:
                extents = [v * float(s) for v, s in zip(extents, scale)]
            planar = sorted(extents)[-2:]
            if len(planar) != 2 or planar[0] <= 0:
                return None
            small, large = planar[0], planar[1]
            return dict(min_span=small * 0.55, max_span=large * 1.6,
                        min_ratio=(small / large) * 0.5, source=model_dir,
                        expected=[round(small, 4), round(large, 4)])
        except Exception:
            return None

    def _coaster_by_color(self, rgb, world, valid, table_z):
        """Find the support by colour + planarity + CAD size, without any text prompt.

        DINO boxes a darkened table patch and misses a bright yellow pad that fills
        a sixth of the frame: "rectangular pad" has no appearance prior, while the
        pad itself is the most saturated flat region on the table.  Support pads
        come in eight colours (six saturated, plus black and grey), so segment on
        saturation and let geometry confirm.
        """
        bounds = self._support_span_bounds()
        if bounds is None:
            return None
        from scipy.ndimage import label
        colors = np.asarray(rgb)[..., :3].astype(float) / 255.0
        hue, saturation, value = _rgb_to_hsv(colors)
        # A known colour name is exact; without one, fall back to "any saturated".
        known = None
        try:
            from adapters.unified_perception import PERCEPTION_SPEC
            known = PERCEPTION_SPEC.get("support_color")
        except Exception:
            known = None
        if known:
            candidate = _color_mask(colors, (str(known),))
        else:
            candidate = (saturation >= 0.45) & (value >= 0.15)
        points_all = np.asarray(world)
        near_table = np.abs(points_all[..., 2] - table_z) <= 0.006
        mask = candidate & near_table & valid
        if mask.sum() < 200:
            return None
        components, count = label(mask)
        best = None
        for index in range(1, count + 1):
            component = components == index
            pixels = int(component.sum())
            if pixels < 200:
                continue
            points = points_all[component]
            if len(points) < 200:
                continue
            lo, hi = np.quantile(points, [.01, .99], axis=0)
            spans = hi - lo
            if (np.min(spans[:2]) < bounds["min_span"]
                    or np.max(spans[:2]) > bounds["max_span"]
                    or np.min(spans[:2]) / max(np.max(spans[:2]), 1e-6) < bounds["min_ratio"]):
                continue
            # Prefer the largest qualifying region.
            if best is None or pixels > best[0]:
                center = (lo + hi) / 2
                center[2] = float(np.median(points[:, 2]))
                best = (pixels, center, component)
        if best is None:
            return None
        pixels, center, component = best
        audit = dict(source="color_saturation_planar", pixels=pixels,
                     support_bounds=bounds, fresh_measurement=True,
                     uses_text_prompt=False)
        return center, component, audit

    def _coaster(self, rgb, world, valid, table_z):
        if self.reference_pad is not None:
            return self.reference_pad.copy(), np.zeros(valid.shape, bool), dict(
                source="initial_visual_static_coaster", fresh_measurement=False)
        # Colour segmentation first: it needs no prompt and the pad is the most
        # saturated flat region on the table.  DINO remains the fallback.
        by_color = None
        # Default OFF: enabling it measured worse than the DINO path
        # (0/5 vs 2/5), so it stays available for experiments only.
        if os.environ.get("ULA_SUPPORT_COLOR_FIRST") == "1":
            try:
                by_color = self._coaster_by_color(rgb, world, valid, table_z)
            except Exception:
                by_color = None
        if by_color is not None:
            return by_color
        prompt = self.prompts.get("coaster", "round coaster")
        # A support with a CAD asset can be found from several phrasings; pooling
        # them raises recall without trusting any single description.  The CAD span
        # filter then decides which candidate is really the support.  Prompts are
        # separated by "|" in the task config.
        variants = [v.strip() for v in str(prompt).split("|") if v.strip()] or [prompt]
        boxes, confidences = [], []
        for variant in variants:
            f, sc, _ = self._run_dino(rgb, f"{variant} .")
            f = f.detach().cpu().numpy()
            sc = sc.detach().cpu().numpy().reshape(-1)
            for box, score in zip(f, sc):
                boxes.append(box)
                confidences.append(float(score))
        if not boxes:
            raise RuntimeError(f"No semantic candidates for {prompt}")
        scores = self.torch.tensor(np.asarray(confidences))
        found = self.torch.tensor(np.asarray(boxes))
        order = np.argsort(np.asarray(confidences))[::-1][:6]
        attempts = []
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            self.sam.set_image(rgb)
        for index in order:
            box = self._xyxy(found[int(index)].detach().cpu().numpy(), rgb.shape[1], rgb.shape[0])
            with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                masks, _, _ = self.sam.predict(box=box, multimask_output=False)
            mask = np.asarray(masks).reshape(-1, *valid.shape)[0].astype(bool)
            points = world[mask & valid]
            if len(points) < 40:
                continue
            planar = points[np.abs(points[:, 2]-table_z) <= .01]
            attempt = dict(box=box.tolist(), plane_fraction=len(planar)/len(points), accepted=False)
            attempts.append(attempt)
            if len(planar) < 40 or len(planar)/len(points) < .7:
                continue
            lo, hi = np.quantile(planar, [.01, .99], axis=0)
            spans = hi-lo
            bounds = self._support_span_bounds()
            if bounds is None:
                min_span, max_span, min_ratio = .025, .25, .55
            else:
                min_span, max_span, min_ratio = (
                    bounds["min_span"], bounds["max_span"], bounds["min_ratio"])
            attempt["support_bounds"] = bounds
            if (np.min(spans[:2]) < min_span or np.max(spans[:2]) > max_span
                    or np.min(spans[:2])/np.max(spans[:2]) < min_ratio):
                continue
            center = (lo+hi)/2
            center[2] = np.median(planar[:, 2])
            attempt["accepted"] = True
            return center, mask, dict(source="fresh_rgbd_planar_round_coaster", attempts=attempts,
                fresh_measurement=True)
        raise RuntimeError(f"No credible planar support ({self.prompts.get('coaster', 'round coaster')}) among semantic candidates: {attempts}")

    def estimate(self, rgb, depth, intrinsic, world_to_camera, cup_only=False, refine_cup=False):
        from adapters.pointcloud_geometry import estimate_visible_bounds
        from adapters.reobservation import observation_quality
        start = time.perf_counter()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        world, valid = world_points_image(depth, intrinsic, world_to_camera)
        table_z = self.reference_table_z
        if table_z is None:
            table_z = visual_table_height(world, valid)
        initial = self.previous_cup_center is None
        from adapters.unified_perception import PERCEPTION_SPEC
        colors = tuple((PERCEPTION_SPEC.get("object_hints") or {}).get("colors") or ())
        candidates, rejected = object_candidates(
            rgb, world, valid, table_z, colors=colors, initial=initial,
            previous_center=self.previous_cup_center,
            aspect_bounds=self._object_aspect_bounds())
        if not candidates:
            raise RuntimeError(f"No credible {self.prompts.get('cup', 'blue cup')} candidate: {rejected}")
        found, logits, _ = self._run_dino(rgb, f"{self.prompts.get('cup', 'blue cup')} .")
        semantic = [(self._xyxy(b.detach().cpu().numpy(), rgb.shape[1], rgb.shape[0]), float(s))
            for b, s in zip(found, logits.detach().cpu().numpy().reshape(-1))]
        for candidate in candidates:
            y, x = np.nonzero(candidate["mask"])
            evidence = max((score*float(np.mean((x >= box[0]) & (x <= box[2])
                & (y >= box[1]) & (y <= box[3]))) for box, score in semantic), default=0.)
            shape = 1/(1+abs(candidate["upright_aspect"]-1.))
            distance = candidate["distance_from_previous_m"] or 0.
            candidate["score"] = float(np.log1p(candidate["pixels"])/10+shape+evidence-2*distance)
        candidates.sort(key=lambda c: c["score"], reverse=True)
        # A close score margin must not abort the episode: the loop below already
        # tries the top three candidates and validates each one geometrically
        # (mask agreement, visible bounds, observation quality), and the visual
        # trial lift remains the final verdict.  Keep the margin for diagnosis.
        self.ambiguity = dict(
            count=len(candidates),
            score_margin=(float(candidates[0]["score"]-candidates[1]["score"])
                          if len(candidates) > 1 else None))
        detection_s = time.perf_counter()-start
        pad, pad_mask, pad_audit = ((np.full(3, np.nan), np.zeros(valid.shape, bool), {}) if cup_only
            else self._coaster(rgb, world, valid, table_z))
        attempts = []
        for candidate in candidates[:3]:
            box = np.asarray(candidate["bbox"], dtype=float)+[-3, -3, 3, 3]
            box[:2] = np.maximum(box[:2], 0)
            box[2:] = np.minimum(box[2:], [rgb.shape[1]-1, rgb.shape[0]-1])
            def credible(mask):
                if np.sum(mask & candidate["mask"])/candidate["pixels"] < .6:
                    return False
                try:
                    bounds, audit, points = estimate_visible_bounds(mask, pad_mask, depth,
                        intrinsic, world_to_camera, return_points=True)
                    value = SimpleNamespace(cup_bounds_world=bounds, cup_points_world=points,
                        observation_audit=audit)
                    return observation_quality(value)["geometry_ok"]
                except (ValueError, RuntimeError):
                    return False
            try:
                mask = self._refine_cup_mask(rgb, box, candidate["point_xy"], credible)
                bounds, audit, points = estimate_visible_bounds(mask, pad_mask, depth,
                    intrinsic, world_to_camera, return_points=True)
                center = np.median(points, axis=0)
                if np.linalg.norm(center-np.asarray(candidate["center_world"])) > .08:
                    raise RuntimeError("SAM cup diverged from selected measured foreground")
                y, x = np.nonzero(mask)
                audit.update(cup_mask_touches_border=bool(np.any(x < 2) or np.any(y < 2)
                    or np.any(x >= rgb.shape[1]-2) or np.any(y >= rgb.shape[0]-2)),
                    cup_mask_bbox_pixels=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                    image_size_hw=list(rgb.shape[:2]), coaster_observed=not cup_only,
                    cup_segmentation_variant="focused_color_depth_semantic_sam",
                    target_focus=dict(selected={k:v for k,v in candidate.items() if k not in ("mask", "point_xy")},
                        rejected=rejected, attempts=attempts, pad=pad_audit,
                        assumptions=["blue_cup", "initial_upright_compact_cup", "static_round_coaster"],
                        uses_simulator_identity=False, uses_object_truth=False))
                self.previous_cup_center = center
                self.reference_table_z = table_z
                if not cup_only and self.reference_pad is None:
                    self.reference_pad = pad.copy()
                return SimpleNamespace(cup_bounds_world=bounds, cup_points_world=points,
                    coaster_world_center=pad, observation_audit=audit,
                    timings_s=dict(detection=detection_s, perception_total=time.perf_counter()-start))
            except (ValueError, RuntimeError) as error:
                attempts.append(dict(box=box.tolist(), error=str(error)))
        raise RuntimeError(f"Focused cup segmentation exhausted: {attempts}")
