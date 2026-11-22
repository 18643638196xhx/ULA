"""Detection/segmentation only: no CAD, object metadata, or FoundationPose loading."""

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from traditional_place_empty_cup.perception import GroundedFoundationPerception
from traditional_place_empty_cup.geometry import (
    backproject_mask_to_world, depth_to_metres, robust_mask_center_world,
    homogeneous_transform,
)
from adapters.pointcloud_geometry import estimate_visible_bounds


def depth_foreground_seed(mask, depth, intrinsic, world_to_camera, table_z, box):
    """A measured foreground SAM prompt, not point removal or robot masking.

    In this table-top pilot, the lower foreground portion is a useful cue when
    a cup mask has grown into structures above it. It remains a heuristic.
    """
    rows, cols = np.nonzero(mask)
    d = depth_to_metres(depth)[rows, cols]
    inside = ((cols >= box[0]) & (cols <= box[2]) & (rows >= box[1]) & (rows <= box[3])
              & np.isfinite(d) & (d > 0))
    rows, cols, d = rows[inside], cols[inside], d[inside]
    rays = np.linalg.solve(np.asarray(intrinsic), np.stack([cols, rows, np.ones(len(rows))]))
    camera_points = (rays*d).T
    inverse = np.linalg.inv(homogeneous_transform(world_to_camera))
    points = camera_points@inverse[:3, :3].T+inverse[:3, 3]
    eligible = np.flatnonzero(points[:, 2] > table_z+.003)
    if len(eligible) < 20:
        raise RuntimeError("Insufficient measured foreground for depth-guided SAM prompt")
    lo, hi = np.quantile(points[eligible, 2], [.05, .35])
    low = eligible[(points[eligible, 2] >= lo) & (points[eligible, 2] <= hi)]
    if not len(low):
        raise RuntimeError("No supported depth layer for SAM foreground prompt")
    centre = np.median(points[low], axis=0)
    selected = low[np.argmin(np.linalg.norm(points[low]-centre, axis=1))]
    return np.array([cols[selected], rows[selected]], dtype=float)


class GroundedPointCloudPerception(GroundedFoundationPerception):
    # Reuse only the original DINO/SAM inference methods, NOT its model-based init.
    def __init__(self, root):
        start = time.perf_counter()
        root = Path(root)
        for folder in ("GroundingDINO", "sam2"):
            sys.path.insert(0, str(root / "third_party" / folder))
        import torch
        from groundingdino.datasets import transforms
        from groundingdino.models import build_model
        from groundingdino.util.inference import predict
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = torch.device("cuda")
        self.predict_dino = predict
        self.box_threshold = 0.30
        self.text_threshold = 0.25
        self.prompts = dict(cup="blue cup", coaster="round coaster")
        self.load_timings_s = {}
        stage = time.perf_counter()
        dino_root = root / "third_party/GroundingDINO"
        config = SLConfig.fromfile(str(dino_root / "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
        config.device = str(self.device)
        config.text_encoder_type = str(dino_root / "weights/bert-base-uncased")
        self.dino = build_model(config)
        weights = torch.load(dino_root / "weights/groundingdino_swint_ogc.pth",
                             map_location="cpu", weights_only=False)
        self.dino.load_state_dict(clean_state_dict(weights["model"]), strict=False)
        self.dino.to(self.device).eval()
        self.dino_transform = transforms.Compose([
            transforms.RandomResize([800], max_size=1333), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
        torch.cuda.synchronize()
        self.load_timings_s["groundingdino_load"] = time.perf_counter()-stage
        stage = time.perf_counter()
        self.sam = SAM2ImagePredictor(build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(root / "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"),
            device=str(self.device), apply_postprocessing=False))
        torch.cuda.synchronize()
        self.load_timings_s["sam2_load"] = time.perf_counter()-stage
        self.load_timings_s["model_stack_load_total"] = time.perf_counter()-start

    def _refine_cup_mask(self, rgb, box, point_xy=None, mask_validator=None):
        """A different SAM prompt on a cup ROI, not a robot/color point filter."""
        h, w = rgb.shape[:2]
        box = np.asarray(box, dtype=float)
        padding = max(4, int(max(box[2]-box[0], box[3]-box[1])*.1))
        x0, y0 = np.maximum(np.floor(box[:2]).astype(int)-padding, 0)
        x1, y1 = np.minimum(np.ceil(box[2:]).astype(int)+padding+1, [w, h])
        if x1-x0 < 3 or y1-y0 < 3:
            raise RuntimeError("Cup ROI too small for prompted resegmentation")
        local_box = box-np.array([x0, y0, x0, y0])
        centre = (((local_box[:2]+local_box[2:])/2) if point_xy is None
                  else np.asarray(point_xy)-[x0, y0])[None, :]
        with self.torch.inference_mode(), self.torch.autocast("cuda",dtype=self.torch.bfloat16):
            self.sam.set_image(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
            masks, scores, _ = self.sam.predict(box=local_box, point_coords=centre,
                point_labels=np.ones(1, dtype=np.int32), multimask_output=True)
        candidates = np.asarray(masks).reshape(-1, y1-y0, x1-x0).astype(bool)
        scores = np.asarray(scores).reshape(-1)
        valid = []
        for i, mask in enumerate(candidates):
            if not mask.any():
                continue
            full = np.zeros((h, w), dtype=bool)
            full[y0:y1, x0:x1] = mask
            if mask_validator is None or mask_validator(full):
                valid.append(i)
        if not valid:
            raise RuntimeError("Prompted cup resegmentation returned no credible foreground candidate")
        chosen = max(valid, key=lambda i: float(scores[i]))
        result = np.zeros((h, w), dtype=bool)
        result[y0:y1, x0:x1] = candidates[chosen]
        return result

    def estimate(self, rgb, depth, intrinsic, world_to_camera, cup_only=False, refine_cup=False):
        total = time.perf_counter()
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        stage = time.perf_counter()
        if cup_only:
            found, logits, _ = self._run_dino(rgb, f"{self.prompts['cup']} .")
            if len(found) == 0:
                raise RuntimeError("GroundingDINO did not detect cup in supplementary view")
            index = int(np.argmax(logits.detach().cpu().numpy().reshape(-1)))
            boxes = dict(cup=self._xyxy(found[index].detach().cpu().numpy(),rgb.shape[1],rgb.shape[0]))
        else:
            boxes = self._detect_both(rgb)
        self.torch.cuda.synchronize()
        timings = dict(detection=time.perf_counter()-stage)
        stage = time.perf_counter()
        if cup_only:
            with self.torch.inference_mode(), self.torch.autocast("cuda",dtype=self.torch.bfloat16):
                self.sam.set_image(rgb)
                values, _, _ = self.sam.predict(box=np.asarray([boxes["cup"]]),multimask_output=False)
            cup_mask = np.asarray(values).reshape(-1,*rgb.shape[:2])[0].astype(bool)
            masks = dict(cup=cup_mask,coaster=np.zeros_like(cup_mask))
        else:
            masks = self._segment_both(rgb, boxes)
        prompt_source = "box_centre"
        if refine_cup:
            seed = None
            try:
                _, initial_audit = estimate_visible_bounds(masks["cup"], masks["coaster"],
                    depth, intrinsic, world_to_camera)
                seed = depth_foreground_seed(masks["cup"], depth, intrinsic, world_to_camera,
                                             initial_audit["table_height_m"], boxes["cup"])
                prompt_source = "lower_measured_foreground_depth_layer"
            except (RuntimeError, ValueError):
                pass  # Fall back to the changed ROI/box prompt, never object truth.
            def credible_mask(mask):
                from adapters.reobservation import observation_quality

                try:
                    candidate_bounds, candidate_audit, candidate_points = estimate_visible_bounds(
                        mask, masks["coaster"], depth, intrinsic, world_to_camera, return_points=True)
                    candidate = SimpleNamespace(cup_bounds_world=candidate_bounds,
                        cup_points_world=candidate_points, observation_audit=candidate_audit)
                    return observation_quality(candidate)["geometry_ok"]
                except (RuntimeError, ValueError):
                    return False
            masks["cup"] = self._refine_cup_mask(rgb, boxes["cup"], seed, credible_mask)
        self.torch.cuda.synchronize()
        timings["segmentation"] = time.perf_counter()-stage
        stage = time.perf_counter()
        bounds, audit, points = estimate_visible_bounds(
            masks["cup"], masks["coaster"], depth, intrinsic, world_to_camera, return_points=True)
        coaster_points = (np.empty((0, 3)) if cup_only else backproject_mask_to_world(
            masks["coaster"], depth_to_metres(depth), intrinsic, world_to_camera))
        coaster_points = coaster_points[np.isfinite(coaster_points).all(axis=1)]
        coaster = (np.full(3,np.nan) if cup_only else robust_mask_center_world(
            masks["coaster"], depth_to_metres(depth), intrinsic, world_to_camera))
        rows, cols = np.nonzero(masks["cup"])
        margin = 2
        audit.update(cup_mask_touches_border=bool(
            np.any(rows < margin) or np.any(rows >= rgb.shape[0]-margin)
            or np.any(cols < margin) or np.any(cols >= rgb.shape[1]-margin)),
            cup_mask_bbox_pixels=[int(cols.min()),int(rows.min()),int(cols.max()),int(rows.max())],
            image_size_hw=list(rgb.shape[:2]),coaster_observed=not cup_only,
            cup_segmentation_variant="roi_box_and_positive_point" if refine_cup else "box_only",
            cup_foreground_prompt_source=prompt_source if refine_cup else None)
        timings["pointcloud_geometry"] = time.perf_counter()-stage
        timings["perception_total"] = time.perf_counter()-total
        return SimpleNamespace(cup_bounds_world=bounds, cup_points_world=points, observation_audit=audit,
                               coaster_world_center=coaster, coaster_points_world=coaster_points,
                               timings_s=timings)
