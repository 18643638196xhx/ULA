"""Unified-vision-backed perception for the generic support-placement entry.

It presents the same small interface the adaptive placement already consumes
(``estimate`` -> object/support points and centres, ``reference_path`` ->
pre-grasp reference npz) but resolves the object and support through the
shared GroundingDINO + SAM + RGB-D pipeline using the task's own prompts.

``PERCEPTION_SPEC`` is set once by the unified entry before the base runner
constructs its perception objects.
"""

from dataclasses import dataclass
from pathlib import Path
import itertools
import json
import os

import numpy as np

from adapters.unified_vision import (
    CadLimits,
    RGBDFrame,
    TargetQuery,
    VisionConfig,
    VisualHints,
)
from adapters.frustum_selection_v1 import FrustumSelectionWorker


_CAPTURE_COUNTER = itertools.count(1)



_TEMPLATE_MOD_CACHE = {}
_TEMPLATE_REC_CACHE = {}


def _load_demo_template_module():
    if "mod" not in _TEMPLATE_MOD_CACHE:
        import importlib.util
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "ula_demo_template_loc", root / "scripts/evaluate_demo_template_localization.py"
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _TEMPLATE_MOD_CACHE["mod"] = mod
    return _TEMPLATE_MOD_CACHE["mod"]


def _resolve_with_demo_template(worker_output: Path, request_num: int, target_key: str, template_spec: str):
    """Disambiguate multi-candidate unified vision output using a single clean-demo pointcloud template (zero CAD)."""
    path_text, sep, array_name = str(template_spec).partition("#")
    array_name = array_name if sep else "points_world"
    tpl_path = Path(path_text).resolve()
    if not tpl_path.is_file():
        return None, {"template_error": f"missing_template:{tpl_path}"}
    mod = _load_demo_template_module()
    cache_key = (str(tpl_path), array_name)
    if cache_key not in _TEMPLATE_REC_CACHE:
        _TEMPLATE_REC_CACHE[cache_key] = mod._template_record(target_key, tpl_path, array_name)
    tpl = _TEMPLATE_REC_CACHE[cache_key]

    pred_dir = Path(worker_output) / f"request_{int(request_num):05d}" / "prediction"
    cand_files = sorted(pred_dir.glob(f"{target_key}_candidate_*.npz"))
    if not cand_files:
        return None, {"template_error": f"no_candidates_in:{pred_dir}"}

    best = None
    scores = []
    for cf in cand_files:
        with np.load(cf, allow_pickle=False) as raw:
            if "points_world" not in raw:
                continue
            pts = np.asarray(raw["points_world"], dtype=float)
        match = mod._match(tpl, pts)
        rec = {"candidate": cf.name, "accepted": bool(match.get("accepted")), "cost": match.get("cost"), "reason": match.get("reason")}
        scores.append(rec)
        if match.get("accepted"):
            cost = float(match["cost"])
            if best is None or cost < best[0]:
                best = (cost, cf.name, pts, match)

    if best is None:
        return None, {"template_error": "no_accepted_candidate", "candidate_scores": scores}
    cost, name, pts, match = best
    return {
        "points": pts,
        "bounds_world": np.asarray(match["bounds_world"], dtype=float),
        "position_world": np.asarray(match["position_world"], dtype=float),
    }, {
        "template_disambiguated": True,
        "selected_candidate": name,
        "selected_cost": float(cost),
        "candidate_scores": scores,
        "uses_CAD": False,
    }



def estimate_template_pose(points_world: np.ndarray, template_spec: str) -> tuple[np.ndarray | None, dict]:
    """把示范点云模板配准到观测点云，返回完整 6-DoF 变换（模板→当前世界）。

    与 estimate_template_z_yaw_rotation 共用同一套配准与质量门控，
    区别是这里保留完整的旋转与平移，用于把整条示范轨迹映射到当前场景。
    """
    path_text, sep, array_name = str(template_spec).partition("#")
    array_name = array_name if sep else "points_world"
    tpl_path = Path(path_text).resolve()
    if not tpl_path.is_file():
        return None, {"pose_error": f"missing_template:{tpl_path}"}
    pts = np.asarray(points_world, dtype=float)
    if len(pts) < 20:
        return None, {"pose_error": "too_few_points"}
    mod = _load_demo_template_module()
    cache_key = (str(tpl_path), array_name)
    if cache_key not in _TEMPLATE_REC_CACHE:
        _TEMPLATE_REC_CACHE[cache_key] = mod._template_record("object", tpl_path, array_name)
    match = mod._match(_TEMPLATE_REC_CACHE[cache_key], pts)
    if not match.get("accepted") or "demo_template_to_current_world" not in match:
        return None, {"pose_accepted": False, "pose_reason": match.get("reason", "match_failed")}
    M = np.asarray(match["demo_template_to_current_world"], dtype=float)
    return M, {
        "pose_accepted": True,
        "match_cost": float(match.get("cost", 0.0)),
        "yaw_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2),
        "translation": M[:3, 3].tolist(),
        "uses_CAD": False,
    }


def estimate_template_z_yaw_rotation(points_world: np.ndarray, template_spec: str) -> tuple[np.ndarray | None, dict]:
    """Estimate pure Z-axis rotation R_z(yaw) aligning a single clean-demo pointcloud template to observed points (zero CAD)."""
    path_text, sep, array_name = str(template_spec).partition("#")
    array_name = array_name if sep else "points_world"
    tpl_path = Path(path_text).resolve()
    if not tpl_path.is_file():
        return None, {"yaw_error": f"missing_template:{tpl_path}"}
    pts = np.asarray(points_world, dtype=float)
    if len(pts) < 20:
        return None, {"yaw_error": "too_few_points"}
    mod = _load_demo_template_module()
    cache_key = (str(tpl_path), array_name)
    if cache_key not in _TEMPLATE_REC_CACHE:
        _TEMPLATE_REC_CACHE[cache_key] = mod._template_record("object", tpl_path, array_name)
    tpl = _TEMPLATE_REC_CACHE[cache_key]
    match = mod._match(tpl, pts)
    if not match.get("accepted") or "demo_template_to_current_world" not in match:
        return None, {"yaw_accepted": False, "yaw_reason": match.get("reason", "match_failed")}
    M = np.asarray(match["demo_template_to_current_world"], dtype=float)
    yaw_rad = float(np.arctan2(M[1, 0], M[0, 0]))
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rz, {
        "yaw_accepted": True,
        "yaw_deg": round(float(np.degrees(yaw_rad)), 2),
        "match_cost": float(match.get("cost", 0.0)),
        "uses_CAD": False,
    }


PERCEPTION_SPEC = {
    "object_prompt": "cup",
    "support_prompt": "coaster",
    "object_cad": None,
    "support_cad": None,
}


@dataclass
class SupportPlacementEstimate:
    cup_points_world: np.ndarray
    cup_bounds_world: np.ndarray
    coaster_world_center: np.ndarray
    coaster_points_world: np.ndarray
    observation_audit: dict
    timings_s: dict
    object_pose_world: np.ndarray | None = None


class UnifiedPerceptionWorker:
    """Runtime wrapper for the original 50-scene frustum vision evaluator.

    Keep this configuration identical to ``replay_frustum_vision50.py``.  The
    runtime and replay then differ only in where the RGB-D frame comes from, not
    in candidate generation, CAD verification, or candidate selection.
    """

    def __init__(self, ula_root, output_dir, fp_iterations=5, *, backend=None, timeout_s=300, **kwargs):
        self.fp_iterations = fp_iterations
        # run_ula_task.sh has already selected the physical card through
        # CUDA_VISIBLE_DEVICES.  Passing gpu=0 here would overwrite GPU=1/2 in
        # parallel runs and send every worker back to physical card 0.
        requested_gpu = os.environ.get("ULA_GPU")
        self._worker = FrustumSelectionWorker(
            ula_root, output_dir,
            gpu=(int(requested_gpu) if requested_gpu not in (None, "") else None),
            config=VisionConfig(cad=CadLimits(max_models=256)), timeout_s=timeout_s,
        )

    def close(self):
        self._worker.close()

    def summary(self):
        return self._worker.summary()


class UnifiedVisionPerception:
    """``ExternalPerception``-compatible object/support perception."""

    def __init__(self, ula_root, output_dir, fp_iterations=5, *, worker):
        self.output = Path(output_dir).resolve()
        self.worker = worker
        self.reference_path = self.output / "estimate.npz"
        spec = PERCEPTION_SPEC

        def _hints(value):
            return VisualHints(**value) if value else None

        # Prompts may list alternatives separated by "|" — the archived 50-scene
        # evaluator used exactly these alternatives (e.g. "kitchen scale" and
        # "digital scale" for the electronic scale) and scored 72-82% on them.
        def _prompts(value):
            parts = [p.strip() for p in str(value).split("|") if p.strip()]
            return tuple(parts) or (str(value),)

        self.queries = [
            TargetQuery(key="object", prompts=_prompts(spec["object_prompt"]),
                        attributes=("position", "bounds"), required=("position", "bounds"),
                        hints=_hints(spec.get("object_hints")), cad=spec.get("object_cad")),
        ]
        if spec.get("support_prompt"):
            self.queries.append(
                TargetQuery(key="support", prompts=_prompts(spec["support_prompt"]),
                            attributes=("position", "bounds"), required=("position", "bounds"),
                            hints=_hints(spec.get("support_hints")), cad=spec.get("support_cad")))
        self._episode = 0

    def _observe(self, rgb, depth, intrinsic, world_to_camera, queries, label):
        """Run a bounded query set on one real RGB-D frame."""
        index = next(_CAPTURE_COUNTER)
        frame = RGBDFrame(
            camera=str(label), capture_id=f"perception_{index}",
            rgb=np.asarray(rgb)[..., :3].astype(np.uint8),
            depth=np.asarray(depth), intrinsic=np.asarray(intrinsic, float),
            world_to_camera=np.asarray(world_to_camera, float), depth_unit="mm",
        )
        return self.worker._worker.observe(
            [frame], queries, episode_id=f"perception:{index}:{label}")

    @staticmethod
    def _trim_surface(points):
        points = np.asarray(points, float)
        if len(points):
            z = points[:, 2]
            lo, hi = np.quantile(z, 0.02), np.quantile(z, 0.98)
            points = points[(z >= lo) & (z <= hi)]
        return points

    def estimate(self, rgb, depth, intrinsic, world_to_camera):
        self.output.mkdir(parents=True, exist_ok=True)
        scene = self._observe(
            rgb, depth, intrinsic, world_to_camera, self.queries, "head_camera")
        obj = scene.targets.get("object")
        sup = scene.targets.get("support")
        if obj is None:
            raise RuntimeError("Unified vision returned no object target")
        req_num = scene.audit.get("worker_request")
        tpl_audit = {}
        obj_tpl_res = None
        sup_tpl_res = None
        if not obj.required_available and PERCEPTION_SPEC.get("object_template") and req_num is not None:
            obj_tpl_res, obj_tpl_info = _resolve_with_demo_template(
                self.worker._worker.output, req_num, "object", PERCEPTION_SPEC["object_template"]
            )
            tpl_audit["object"] = obj_tpl_info
        if obj_tpl_res is None and not obj.required_available:
            raise RuntimeError(
                f"Unified vision could not localize the object: {obj.status}")

        if sup is not None and not sup.required_available and PERCEPTION_SPEC.get("support_template") and req_num is not None:
            sup_tpl_res, sup_tpl_info = _resolve_with_demo_template(
                self.worker._worker.output, req_num, "support", PERCEPTION_SPEC["support_template"]
            )
            tpl_audit["support"] = sup_tpl_info
        if sup is not None and sup_tpl_res is None and not sup.required_available:
            raise RuntimeError(
                f"Unified vision could not localize the support: {sup.status}")

        if obj_tpl_res is not None:
            obj_points = np.asarray(obj_tpl_res["points"], float)
            obj_bounds = np.asarray(obj_tpl_res["bounds_world"], float)
            obj_pose = None
        else:
            obj_points = np.asarray(obj.points, float)
            obj_bounds = np.asarray(obj.geometry.get("bounds_world"), float)
            obj_pose_value = obj.geometry.get("full_object_pose_world")
            obj_pose = (None if obj_pose_value is None else np.asarray(obj_pose_value, float))

        if sup_tpl_res is not None:
            sup_points = np.asarray(sup_tpl_res["points"], float)
            sup_center = np.asarray(sup_tpl_res["position_world"], float)
        else:
            sup_points = (np.asarray(sup.points, float) if sup is not None else np.empty((0, 3)))
            sup_center = (np.asarray(sup.geometry.get("position_world"), float)
                          if sup is not None else np.zeros(3))
        # Geometry contract: the support point set is its measured top surface,
        # and the anchor height is the table (the support's lower edge), not its
        # mid-surface, so the object seats on the same plane the old perception used.
        if len(sup_points):
            sup_points = self._trim_surface(sup_points)
            table_height = float(np.quantile(sup_points[:, 2], 0.02))
            # Thick supports (scale ~4 cm, display stand more) must be addressed at
            # their TOP surface; the bounds centre sits inside them, which made every
            # placement plan infeasible or landed the object 18-76 cm away.
            top_z = float(np.quantile(sup_points[:, 2], 0.98))
            top = sup_points[sup_points[:, 2] >= top_z - 0.005]
            if len(top) >= 20:
                sup_center = np.array([float(np.median(top[:, 0])),
                                       float(np.median(top[:, 1])), top_z])
        else:
            table_height = float(np.min(obj_points[:, 2]))
        estimate = SupportPlacementEstimate(
            cup_points_world=obj_points,
            cup_bounds_world=obj_bounds,
            coaster_world_center=sup_center,
            coaster_points_world=sup_points,
            observation_audit=dict(
                table_height_m=table_height, source="unified_vision",
                object_status=obj.status,
                support_status=(sup.status if sup is not None else "not_requested"),
                template_disambiguation=tpl_audit,
                object_pose_source=("CAD_verified_RGBD" if obj_pose is not None
                                    else "visible_bounds_only")),
            timings_s=dict(),
            object_pose_world=obj_pose,
        )
        np.savez_compressed(
            self.output / "rgbd.npz", rgb=np.asarray(rgb), depth=np.asarray(depth),
            intrinsic=np.asarray(intrinsic), world_to_camera=np.asarray(world_to_camera))
        saved = dict(
            cup_points_world=estimate.cup_points_world,
            cup_bounds_world=estimate.cup_bounds_world,
            coaster_world_center=estimate.coaster_world_center,
            coaster_points_world=estimate.coaster_points_world,
            observation_audit_json=np.asarray(json.dumps(estimate.observation_audit)),
            timings_json=np.asarray(json.dumps(estimate.timings_s)))
        if obj_pose is not None:
            saved["object_pose_world"] = obj_pose
        np.savez(self.reference_path, **saved)
        return estimate

    def supplementary(self, camera, label, *, cup_only=True, refine_cup=False):
        """Wrist/close view used by the trial lift and reobservation.

        cup_only=True (wrist cameras, trial lift) must localize ONLY the held
        object: the wrist is close enough that the distant support is not in
        frame, and demanding it would abort the verification.  cup_only=False
        (head pre-approach reobservation) still asks for the full estimate.
        """
        if cup_only:
            return self.object_only(camera, label)
        return self.estimate(camera["rgb"], camera["depth"],
                             camera["intrinsic_cv"], camera["extrinsic_cv"])

    def object_only(self, camera, label="object_only"):
        """Localize only the manipulated object when its support is occluded."""
        scene = self._observe(
            camera["rgb"], camera["depth"], camera["intrinsic_cv"],
            camera["extrinsic_cv"], [self.queries[0]], label)
        obj = scene.targets.get("object")
        if obj is None or not obj.required_available:
            status = "not_returned" if obj is None else obj.status
            raise RuntimeError(f"Unified vision could not localize the object: {status}")
        points = np.asarray(obj.points, float)
        bounds = np.asarray(obj.geometry.get("bounds_world"), float)
        pose_value = obj.geometry.get("full_object_pose_world")
        pose = None if pose_value is None else np.asarray(pose_value, float)
        return SupportPlacementEstimate(
            cup_points_world=points,
            cup_bounds_world=bounds,
            coaster_world_center=np.full(3, np.nan),
            coaster_points_world=np.empty((0, 3)),
            observation_audit=dict(
                table_height_m=float(np.min(points[:, 2])), source="unified_vision_object_only",
                object_status=obj.status, support_status="not_requested", camera=str(label),
                object_pose_source=("CAD_verified_RGBD" if pose is not None
                                    else "visible_bounds_only")),
            timings_s=dict(),
            object_pose_world=pose,
        )

    def support_only(self, camera, label="support_only"):
        """Localize only the support when the held object is self-occluded."""
        if len(self.queries) < 2:
            raise RuntimeError("This task has no support query")
        scene = self._observe(
            camera["rgb"], camera["depth"], camera["intrinsic_cv"],
            camera["extrinsic_cv"], [self.queries[1]], label)
        sup = scene.targets.get("support")
        if sup is None or not sup.required_available:
            status = "not_returned" if sup is None else sup.status
            raise RuntimeError(f"Unified vision could not localize the support: {status}")
        points = self._trim_surface(np.asarray(sup.points, float))
        return SupportPlacementEstimate(
            cup_points_world=np.empty((0, 3)),
            cup_bounds_world=np.full((2, 3), np.nan),
            coaster_world_center=np.asarray(sup.geometry.get("position_world"), float),
            coaster_points_world=points,
            observation_audit=dict(
                table_height_m=float(np.quantile(points[:, 2], 0.02)),
                source="unified_vision_support_only", object_status="not_requested",
                support_status=sup.status, camera=str(label)),
            timings_s=dict(),
        )

    def placement_observation(self, task, arm):
        """Independently recover object and support from bounded real views.

        A failed joint object+support query no longer erases a valid target.  No
        camera motion is introduced; head, active wrist and peer wrist are the
        only finite candidates.
        """
        head = task.get_camera_observation("head_camera")
        attempts = []
        try:
            value = self.estimate(
                head["rgb"], head["depth"], head["intrinsic_cv"], head["extrinsic_cv"])
            value.observation_audit["placement_observation"] = dict(
                mode="joint_head_query", attempts=attempts)
            return value
        except (RuntimeError, ValueError) as error:
            attempts.append(dict(camera="head_camera", query="joint", error=str(error)))

        names = ("head_camera", f"{arm}_camera",
                 "right_camera" if str(arm) == "left" else "left_camera")
        obj = sup = None
        for name in names:
            camera = task.get_camera_observation(name)
            if obj is None:
                try:
                    obj = self.object_only(camera, f"placement_object_{name}")
                    attempts.append(dict(camera=name, query="object", accepted=True))
                except (RuntimeError, ValueError) as error:
                    attempts.append(dict(camera=name, query="object", error=str(error)))
            if sup is None:
                try:
                    sup = self.support_only(camera, f"placement_support_{name}")
                    attempts.append(dict(camera=name, query="support", accepted=True))
                except (RuntimeError, ValueError) as error:
                    attempts.append(dict(camera=name, query="support", error=str(error)))
            if obj is not None and sup is not None:
                break
        if obj is None:
            raise RuntimeError(f"No camera localized the manipulated object: {attempts}")
        if sup is None:
            raise RuntimeError(f"No camera localized the support target: {attempts}")
        audit = dict(obj.observation_audit)
        audit.update(
            table_height_m=sup.observation_audit["table_height_m"],
            support_status=sup.observation_audit["support_status"],
            placement_observation=dict(mode="independent_multicamera_queries", attempts=attempts),
        )
        return SupportPlacementEstimate(
            cup_points_world=obj.cup_points_world,
            cup_bounds_world=obj.cup_bounds_world,
            coaster_world_center=sup.coaster_world_center,
            coaster_points_world=sup.coaster_points_world,
            observation_audit=audit,
            timings_s=dict(),
            object_pose_world=obj.object_pose_world,
        )

    def capture_reference_view(self, camera, camera_name):
        directory = self.output / "reference_views"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{camera_name}.npz"
        np.savez_compressed(path, rgb=camera["rgb"], depth=camera["depth"],
                            intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"])
        return path

    def select_reference(self, estimate):
        self.output.mkdir(parents=True, exist_ok=True)
        self.reference_path = self.output / "selected_reference.npz"
        saved = dict(
            cup_points_world=estimate.cup_points_world,
            cup_bounds_world=estimate.cup_bounds_world,
            coaster_world_center=estimate.coaster_world_center,
            coaster_points_world=getattr(estimate, "coaster_points_world", np.empty((0, 3))),
            observation_audit_json=np.asarray(json.dumps(estimate.observation_audit)),
            timings_json=np.asarray(json.dumps(estimate.timings_s)))
        if getattr(estimate, "object_pose_world", None) is not None:
            saved["object_pose_world"] = estimate.object_pose_world
        np.savez(self.reference_path, **saved)


def _pca_major_axis_angle_xy(points_world: np.ndarray) -> tuple[float, float]:
    """Return (major_axis_angle_rad, aspect_ratio) of a 3D pointcloud projected onto XY."""
    xy = np.asarray(points_world, dtype=float)[:, :2]
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = (xy.T @ xy) / max(len(xy), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order: eigvals[1] >= eigvals[0]
    lam_max = max(float(eigvals[1]), 1e-12)
    lam_min = max(float(eigvals[0]), 1e-12)
    aspect = float(np.sqrt(lam_max / lam_min))
    v = eigvecs[:, 1]
    angle = float(np.arctan2(v[1], v[0]))
    return angle, aspect


def estimate_object_z_yaw_adaptive(
    points_world: np.ndarray,
    template_spec: str | None = None,
    *,
    symmetric_180: bool = False,
    aspect_ratio_thresh: float = 1.15,
) -> tuple[np.ndarray | None, dict]:
    """Symmetry-aware pure Z-yaw estimation (zero CAD, zero truth).

    - Radially symmetric objects (XY aspect_ratio < aspect_ratio_thresh, e.g. cups/cans):
      return identity I_3 to avoid random 360-deg wrist twisting.
    - 180-deg symmetric elongated / two-handled objects (symmetric_180=True, e.g. pots/rollers):
      align 2D PCA major axis directly to the template's 2D PCA major axis (wrapped to [-90, +90] deg).
    - Asymmetric elongated objects: fall back to full 360-deg template matching.
    """
    pts = np.asarray(points_world, dtype=float)
    if len(pts) < 20:
        return None, {"yaw_error": "too_few_points", "uses_CAD": False}
    cur_angle, aspect = _pca_major_axis_angle_xy(pts)
    if aspect < float(aspect_ratio_thresh):
        return np.eye(3, dtype=float), {
            "method": "radially_symmetric_skip",
            "yaw_accepted": True,
            "yaw_deg": 0.0,
            "aspect_ratio": round(aspect, 3),
            "uses_CAD": False,
        }
    if symmetric_180:
        tpl_angle = 0.0
        if template_spec:
            path_text, sep, array_name = str(template_spec).partition("#")
            array_name = array_name if sep else "points_world"
            tpl_path = Path(path_text).resolve()
            if tpl_path.is_file():
                with np.load(tpl_path, allow_pickle=False) as z:
                    if array_name in z.files:
                        tpl_angle, _ = _pca_major_axis_angle_xy(np.asarray(z[array_name], dtype=float))
        # Wrap delta yaw into [-pi/2, +pi/2] (shortest 180-deg-symmetric alignment)
        dyaw = float((cur_angle - tpl_angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
        c, s = float(np.cos(dyaw)), float(np.sin(dyaw))
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return Rz, {
            "method": "pca_axis_180",
            "yaw_accepted": True,
            "yaw_deg": round(float(np.degrees(dyaw)), 2),
            "aspect_ratio": round(aspect, 3),
            "uses_CAD": False,
        }
    if not template_spec:
        return np.eye(3, dtype=float), {
            "method": "no_template_identity",
            "yaw_accepted": True,
            "yaw_deg": 0.0,
            "aspect_ratio": round(aspect, 3),
            "uses_CAD": False,
        }
    return estimate_template_z_yaw_rotation(pts, template_spec)
