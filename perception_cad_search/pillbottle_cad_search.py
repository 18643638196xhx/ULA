"""CAD-ranked pill-bottle localization with bounded non-semantic recovery.

Unlike the 35.7% baseline, colour and an upright cup aspect ratio never decide
whether an object proposal may reach CAD verification.  DINO/SAM proposals are
tried first for speed.  If they do not produce a unique CAD-consistent result,
the shared backend adds a bounded set of measured RGB-D connected regions and
resegments them with SAM.  Only a candidate accepted by exact-model
FoundationPose, point-to-CAD distance and rendered-depth agreement is returned.
"""

from __future__ import annotations

from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

from adapters.perception import PerceptionWorker
from adapters.unified_vision import (
    CadLimits,
    CadModel,
    RGBDFrame,
    TargetQuery,
    UnifiedVision,
    VisionConfig,
    VisualHints,
)
from adapters.unified_vision.backends import DinoSamBackend


_CAPTURES = itertools.count(1)


def build_vision_config(fp_iterations: int = 5) -> VisionConfig:
    """Return an explicit bounded search budget, not task-name conditionals."""
    if not isinstance(fp_iterations, int) or isinstance(fp_iterations, bool) or fp_iterations < 1:
        raise ValueError("fp_iterations must be a positive integer")
    return VisionConfig(
        cad_refine_on_demand=True,
        evidence_ranked_cad=True,
        audit_candidate_selection=True,
        # A detector hit is useful ranking evidence, but exact CAD/depth is the
        # identity gate.  This permits white/brown bottles after a DINO miss.
        require_semantic_identity=False,
        verify_scene_structure=True,
        verify_cad_color=False,
        complete_visible_surface=False,
        max_candidates=4,
        recovery_max_boxes=8,
        max_refined_candidates=10,
        max_masks_per_box=2,
        max_recovery_tiles=2,
        max_depth_regions=4,
        cad=CadLimits(
            iterations=fp_iterations,
            max_hypotheses=8,
            max_models=5,
            depth_tolerance_m=.015,
            max_median_error_m=.008,
            min_observed_inliers=.65,
            min_mask_precision=.65,
            min_render_coverage=.55,
            min_visible_fraction=.35,
            max_free_space_fraction=.20,
            min_cost_gap=.15,
        ),
    )


def load_model_spec(root: Path, object_asset: str, model_id: int) -> dict:
    """Load only the explicitly supplied static CAD and metadata."""
    root = Path(root).resolve()
    if not object_asset or not object_asset.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Unsafe or empty object asset")
    model_id = int(model_id)
    directory = root / "tra_pipe/RoboTwin/assets/objects" / object_asset
    metadata_path = directory / f"model_data{model_id}.json"
    mesh_path = directory / "visual" / f"base{model_id}.glb"
    if not metadata_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError(
            f"Missing exact CAD for {object_asset}/base{model_id}: "
            f"metadata={metadata_path.is_file()} mesh={mesh_path.is_file()}"
        )
    metadata = json.loads(metadata_path.read_text())
    scale = np.asarray(metadata.get("scale"), dtype=float)
    extents = np.asarray(metadata.get("extents"), dtype=float)
    if scale.shape != (3,) or extents.shape != (3,) or np.any(scale <= 0) or np.any(extents <= 0):
        raise ValueError("Exact CAD metadata requires finite positive 3-D scale/extents")
    physical_extents = scale * extents
    transform = np.asarray(metadata.get("transform_matrix", np.eye(4)), dtype=float)
    mesh_to_object = None if np.allclose(transform, np.eye(4)) else tuple(map(tuple, transform))
    return dict(
        object_asset=object_asset,
        model_id=model_id,
        mesh_path=str(mesh_path.relative_to(root)),
        scale=tuple(float(value) for value in scale),
        mesh_to_object=mesh_to_object,
        physical_extents_m=physical_extents,
        diameter_m=float(np.linalg.norm(physical_extents)),
    )


def build_object_query(spec: dict, prompt: str = "pill bottle") -> TargetQuery:
    """Build a high-recall query whose final identity is exact CAD agreement."""
    # Preserve the specific task phrase, then add identity-neutral aliases.  A
    # colour word can help DINO but is never installed as an appearance gate.
    prompts = []
    for value in (prompt, "pill bottle", "medicine bottle", "bottle with cap"):
        value = str(value).strip()
        if value and value not in prompts:
            prompts.append(value)
    diameter = float(spec["diameter_m"])
    cad = CadModel(
        mesh_path=spec["mesh_path"],
        scale=spec["scale"],
        mesh_to_object=spec.get("mesh_to_object"),
    )
    return TargetQuery(
        key="object",
        prompts=tuple(prompts),
        attributes=("position", "bounds", "axis"),
        required=("position", "bounds"),
        # The initial object is supported above the measured table.  No colour
        # and no fixed cup aspect window are used.
        hints=VisualHints(up_world=(0., 0., 1.), support_relation="above"),
        cad=cad,
        position_reference="cad_origin",
        # A world-axis box may be smaller under occlusion, so only a generous
        # upper bound is safe. Exact shape is checked later in CAD coordinates.
        max_extent_m=(1.35 * diameter,) * 3,
    )


def _support_extents(model_id: int | None = None):
    """Support extents from its generated CAD, in metres.

    The sizes below used to be the pillbottle pad's 8x8x1 mm window, which rejects
    every other support: the shoe's is 26x10 cm.  Read the asset that
    gen_pad_assets.py produced instead.
    """
    model_dir = os.environ.get("ULA_SUPPORT_MODEL_DIR")
    if not model_dir:
        return None
    model_id = (os.environ.get("ULA_SUPPORT_MODEL_ID", "0")
                if model_id is None else int(model_id))
    base = Path("/data1/xuhuixin/ula/tra_pipe/RoboTwin/assets/objects") / model_dir
    path = base / f"model_data{int(model_id)}.json"
    if not path.is_file():
        _cands = sorted(base.glob("model_data*.json"))
        if not _cands:
            return None
        path = _cands[0]
    try:
        meta = json.loads(path.read_text())
        extents = [abs(float(v)) for v in meta["extents"]]
        scale = meta.get("scale", 1.0)
        if isinstance(scale, (int, float)):
            extents = [v * float(scale) for v in extents]
        else:
            extents = [v * float(s) for v, s in zip(extents, scale)]
        planar = sorted(extents)[-2:]
        if len(planar) != 2 or planar[0] <= 0:
            return None
        return planar[0], planar[1], min(extents)
    except Exception:
        return None


def build_support_query(prompt: str | None = None, colour: str | None = None,
                        model_id: int | None = None) -> TargetQuery:
    resolved = (prompt or os.environ.get("ULA_KNOWN_SUPPORT_PROMPT") or "pad").strip()
    extents = _support_extents(model_id)
    # A thin support (pad, coaster, plate) is a planar region: planarity, "on the
    # table" and the aspect gate all help.  A thick support (kitchen scale,
    # display stand, cabinet) is an object, and those constraints made the
    # archived-scene replay reject it with candidate_not_on_measured_support /
    # measured_surface_disagrees_with_CAD even when the CAD was correct.
    if extents is not None and extents[2] > 0.03:
        return TargetQuery(
            key="support",
            prompts=tuple(x.strip() for x in str(resolved).split("|") if x.strip()) or (resolved,),
            attributes=("position", "bounds"),
            required=("position", "bounds"),
            hints=VisualHints(appearance_required=False, min_planar_aspect=None),
            min_extent_m=(extents[0] * 0.5, extents[0] * 0.5, 0.0),
            max_extent_m=(extents[1] * 1.7, extents[1] * 1.7, extents[2] * 1.7),
        )
    if extents is not None:
        small, large, thickness = extents
        min_extent_m = (small * 0.5, small * 0.5, 0.0)
        max_extent_m = (large * 1.7, large * 1.7, max(0.03, thickness * 6.0))
        min_aspect = (small / large) * 0.5
    else:
        min_extent_m, max_extent_m, min_aspect = (.025, .025, 0.), (.20, .20, .025), .55
    # A colour word in the prompt is a real prior when present ("blue square pad"),
    # but several tasks pick the pad colour per episode and pass a colourless
    # prompt ("rectangular pad").  Require appearance only in the first case, and
    # let CAD size plus planarity decide otherwise.
    words = resolved.lower().split()
    # Must stay inside VisualHints' calibrated palette: cyan, magenta, brown and
    # pink appear on real pads but are rejected by the query validator, and passing
    # one turned every fan/stapler/mouse episode into a ValueError.
    known = {"red", "orange", "yellow", "green", "blue", "purple",
             "white", "black", "gray", "grey"}
    # The runner forwards the pad colour chosen in setup_demo; that is exact and
    # beats guessing it from the prompt.
    colour = str(colour).lower() if colour else next((w for w in words if w in known), None)
    if colour == "grey":
        colour = "gray"
    if colour is not None and colour not in known:
        colour = None
    if colour:
        prompts = (f"{colour} pad",)
        hints = VisualHints(colors=(colour,), appearance_required=True,
                            support_relation="on", up_world=(0., 0., 1.),
                            min_planar_aspect=min_aspect)
    else:
        prompts = (resolved,)
        hints = VisualHints(appearance_required=False,
                            support_relation="on", up_world=(0., 0., 1.),
                            min_planar_aspect=min_aspect)
    # A "|"-separated prompt lists alternatives, matching the archived evaluator
    # ("kitchen scale" and "digital scale" for the electronic scale).
    # Keep the colour-derived prompt when there is one; otherwise treat "|" as a
    # list of alternative phrasings.
    if colour:
        _alts = (f"{colour} pad",)
    else:
        _alts = tuple(x.strip() for x in str(resolved).split("|") if x.strip()) or (resolved,)
    return TargetQuery(
        key="support",
        prompts=_alts,
        attributes=("position", "bounds", "plane"),
        required=("position", "plane"),
        min_extent_m=min_extent_m,
        max_extent_m=max_extent_m,
        hints=hints,
    )


class PillbottleCadSearchVision:
    """Old worker-compatible estimate API backed by the shared CAD engine."""

    def __init__(self, root, output, fp_iterations=5, *, object_asset="080_pillbottle",
                 object_prompt="orange pill bottle", support_prompt="blue square pad"):
        self.root = Path(root).resolve()
        self.output = Path(output).resolve()
        self.object_asset = object_asset
        self.object_prompt = object_prompt
        self.support_prompt = support_prompt
        self.config = build_vision_config(fp_iterations)
        started = time.perf_counter()
        self.backend = DinoSamBackend(
            self.root / "tra_pipe/RoboTwin",
            max_candidates=self.config.max_candidates,
            config=self.config,
            project_root=self.root,
            cad_output=self.output / "cad_runtime",
        )
        self.engine = UnifiedVision(self.backend, self.config)
        self.load_timings_s = dict(
            self.backend.model.load_timings_s,
            model_stack_load_total=time.perf_counter() - started,
        )

    def estimate(self, rgb, depth, intrinsic, world_to_camera, *, model_id,
                 object_asset=None, cup_only=False, refine_cup=False,
                 support_only=False, support_color=None):
        del refine_cup  # The engine performs bounded resegmentation on demand.
        started = time.perf_counter()
        asset = str(object_asset or self.object_asset)
        spec = load_model_spec(self.root, asset, int(model_id))
        object_query = build_object_query(spec, self.object_prompt)
        support_query = build_support_query(self.support_prompt, support_color)
        if bool(support_only):
            # Placement only needs the destination support; the object is already
            # in the gripper and re-detecting it would only add occlusion error.
            queries = [support_query]
        elif bool(cup_only):
            queries = [object_query]
        else:
            queries = [object_query, support_query]
        capture = next(_CAPTURES)
        frame = RGBDFrame(
            camera="head_or_requested_camera",
            capture_id=f"pillbottle_cad_search_{capture}",
            rgb=np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8),
            depth=np.asarray(depth),
            intrinsic=np.asarray(intrinsic, dtype=float),
            world_to_camera=np.asarray(world_to_camera, dtype=float),
            depth_unit="mm",
        )
        scene = self.engine.observe(
            [frame], queries,
            episode_id=f"pillbottle:{capture}:{asset}:base{int(model_id)}",
        )
        obj = scene.targets.get("object")
        if obj is None or not obj.required_available:
            status = "not_returned" if obj is None else obj.status
            reasons = [] if obj is None else obj.reasons
            raise RuntimeError(f"No exact-CAD-consistent pill bottle: {status}; reasons={reasons}")
        pose = obj.geometry.get("full_object_pose_world")
        if pose is None:
            raise RuntimeError("CAD search accepted an object without a full 6-D CAD pose")
        support = scene.targets.get("support")
        if not cup_only and (support is None or not support.required_available):
            status = "not_returned" if support is None else support.status
            raise RuntimeError(f"No credible support pad: {status}")
        points = np.asarray(obj.points, dtype=float)
        bounds = np.asarray(obj.geometry["bounds_world"], dtype=float)
        support_points = (np.empty((0, 3)) if support is None
                          else np.asarray(support.points, dtype=float))
        support_center = (np.full(3, np.nan) if support is None
                          else np.asarray(support.geometry["position_world"], dtype=float))
        audit = dict(
            source="bounded_semantic_plus_RGBD_regions_exact_CAD_search",
            exact_model=f"{asset}/base{int(model_id)}",
            object_status=obj.status,
            support_status="not_requested" if support is None else support.status,
            color_is_hard_gate=False,
            fixed_cup_aspect_is_hard_gate=False,
            semantic_identity_required=False,
            final_identity_gate="FoundationPose_point_surface_and_rendered_depth_agreement",
            candidate_search_bounded=True,
            configuration=asdict(self.config),
            scene_audit=scene.audit,
            object_evidence=obj.evidence,
            object_candidates=obj.candidates,
            uses_object_truth=False,
            uses_simulator_segmentation=False,
        )
        return type("CadSearchEstimate", (), dict(
            cup_world_pose=np.asarray(pose, dtype=float),
            cup_bounds_world=bounds,
            cup_points_world=points,
            coaster_world_center=support_center,
            coaster_points_world=support_points,
            observation_audit=audit,
            timings_s=dict(perception_total=time.perf_counter() - started),
        ))()


class _SupportOnlyResult:
    """Minimal stand-in for the estimate object `finish_placement` reads."""

    def __init__(self, coaster_world_center, timings_s=None):
        self.coaster_world_center = coaster_world_center
        self.timings_s = timings_s or {}
        self.cup_world_pose = None
        self.object_pose_world = None


class PillbottleCadSearchWorker(PerceptionWorker):
    """Persistent old-protocol worker used by the existing ULA runner."""

    def __init__(self, root, output, fp_iterations=5, **kwargs):
        super().__init__(root, output, fp_iterations=fp_iterations,
                         timeout_s=kwargs.get("timeout_s", 300), backend="foundationpose")

    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(self.root / "perception_cad_search/perceive_pillbottle_cad_search.py")
        environment.update(
            ULA_CAD_SEARCH_OBJECT_ASSET=os.environ.get(
                "ULA_KNOWN_OBJECT_ASSET", "080_pillbottle"),
            ULA_CAD_SEARCH_OBJECT_PROMPT=os.environ.get(
                "ULA_KNOWN_OBJECT_PROMPT", "orange pill bottle"),
            ULA_CAD_SEARCH_SUPPORT_PROMPT=os.environ.get(
                "ULA_KNOWN_SUPPORT_PROMPT", "blue square pad"),
        )
        return command, cwd, environment
