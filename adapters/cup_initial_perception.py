"""Isolated pre-grasp cup/coaster perception: semantics + depth, soft color.

Only RGB-D and calibration enter inference. No CAD or simulator target IDs.
Cup and pad are reported separately, including when one of them is rejected.
"""
import time
from types import SimpleNamespace

import numpy as np

from adapters.cup_target_focus import FocusedCupPerception, world_points_image, visual_table_height
from adapters.perception import PerceptionWorker
from adapters.pointcloud_geometry import estimate_visible_bounds


def cup_geometry(bounds, points, table_z):
    bounds, points = np.asarray(bounds), np.asarray(points)
    spans = bounds[1]-bounds[0]
    ratio = float(spans[2]/max(np.max(spans[:2]), .001))
    if (len(points) < 40 or np.min(spans[:2]) < .008 or np.max(spans) > .25
            or not .55 <= ratio <= 2.2 or spans[2] < .025
            or bounds[1, 2] < table_z+.03):
        raise RuntimeError("Semantic cup mask failed compact upright foreground geometry")
    return dict(spans_m=spans.tolist(), upright_aspect=ratio,
                shape_score=1/(1+abs(ratio-1.)))


def blue_evidence(rgb, mask):
    colors = np.asarray(rgb)[np.asarray(mask, bool), :3].astype(float)/255.
    if not len(colors):
        return 0.
    red, green, blue = colors.T
    return float(np.mean((blue-red > .04) & (blue-green > -.015)
                         & (colors.max(1)-colors.min(1) > .06)))


class InitialCupPerception(FocusedCupPerception):
    def estimate(self, rgb, depth, intrinsic, world_to_camera):
        self.reset_episode()
        start = time.perf_counter()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        world, valid = world_points_image(depth, intrinsic, world_to_camera)
        table_z = visual_table_height(world, valid)
        audit = dict(source="semantic_candidates_depth_geometry_soft_color",
                     uses_simulator_truth_for_target=False, uses_prepared_object_model=False,
                     uses_foundationpose=False, color_is_hard_gate=False,
                     camera_scope="initial_head_rgbd_only", table_height_m=table_z,
                     cup_accepted=False, pad_accepted=False, cup_attempts=[])
        cup_mask = np.zeros(valid.shape, bool)
        pad_mask = np.zeros(valid.shape, bool)
        pad = np.full(3, np.nan)
        # Pad failure must not erase a perfectly usable cup result, or vice versa.
        try:
            pad, pad_mask, pad_audit = self._coaster(rgb, world, valid, table_z)
            audit.update(pad_accepted=True, pad=pad_audit)
        except (RuntimeError, ValueError) as error:
            audit["pad_error"] = str(error)
        bounds, points = np.full((2, 3), np.nan), np.empty((0, 3))
        candidates = []
        try:
            found, logits, _ = self._run_dino(rgb, "cup . mug .")
            scores = logits.detach().cpu().numpy().reshape(-1)
            order = np.argsort(scores)[::-1][:5]
            for index in order:
                box = self._xyxy(found[int(index)].detach().cpu().numpy(), rgb.shape[1], rgb.shape[0])
                item = dict(box=np.asarray(box).tolist(), semantic_score=float(scores[int(index)]),
                            accepted=False)
                audit["cup_attempts"].append(item)
                def geometry(mask):
                    b, g, p = estimate_visible_bounds(mask, pad_mask, depth, intrinsic,
                                                       world_to_camera, return_points=True)
                    return b, g, p, cup_geometry(b, p, table_z)
                def credible(mask):
                    try:
                        geometry(mask)
                        return True
                    except (RuntimeError, ValueError):
                        return False
                try:
                    mask = self._refine_cup_mask(rgb, box, mask_validator=credible)
                    b, g, p, shape = geometry(mask)
                    color = blue_evidence(rgb, mask)
                    # Color can add at most .10, never remove a candidate.
                    rank = float(scores[int(index)])+.15*shape["shape_score"]+.10*color
                    item.update(accepted=True, blue_pixel_fraction=color, ranking_score=rank,
                                geometry=shape, center_world=b.mean(0).tolist())
                    candidate = dict(mask=mask, bounds=b, points=p, geometry=g,
                                     score=rank, center=b.mean(0), audit=item)
                    if not any(np.linalg.norm(candidate["center"]-c["center"]) < .035
                               for c in candidates):
                        candidates.append(candidate)
                except (RuntimeError, ValueError) as error:
                    item["error"] = str(error)
            candidates.sort(key=lambda value: value["score"], reverse=True)
            if not candidates:
                raise RuntimeError("No semantic cup candidate passed measured depth geometry")
            if len(candidates) > 1 and candidates[0]["score"]-candidates[1]["score"] < .10:
                raise RuntimeError("Competing distinct cup candidates; refuse identity ambiguity")
            selected = candidates[0]
            cup_mask, bounds, points = selected["mask"], selected["bounds"], selected["points"]
            audit.update(cup_accepted=True, selected_cup=selected["audit"],
                         cup_geometry=selected["geometry"])
        except (RuntimeError, ValueError) as error:
            audit["cup_error"] = str(error)
        return SimpleNamespace(cup_bounds_world=bounds, cup_points_world=points,
            coaster_world_center=pad, cup_mask=cup_mask, pad_mask=pad_mask,
            observation_audit=audit, timings_s=dict(perception_total=time.perf_counter()-start))


class InitialCupWorker(PerceptionWorker):
    def __init__(self, root, output):
        super().__init__(root, output, backend="pointcloud")

    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(self.root/"scripts/perceive_cup_initial.py")
        return command, cwd, environment
