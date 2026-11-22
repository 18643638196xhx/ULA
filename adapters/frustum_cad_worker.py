"""Bridge the original frustum vision worker to the known-CAD controller API.

The known-CAD controller writes one legacy RGB-D NPZ and expects one legacy
estimate NPZ. ``FrustumSelectionWorker`` instead accepts ``RGBDFrame`` and
``TargetQuery`` objects through ``observe``. This adapter performs only that
wire-format conversion; detection, segmentation, CAD registration and frustum
selection remain inside the original unified-vision worker.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import itertools
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from adapters.frustum_selection_v1 import FrustumSelectionWorker
from adapters.unified_vision import (
    CadLimits,
    CadModel,
    RGBDFrame,
    TargetQuery,
    VisionConfig,
    VisualHints,
)
from adapters.unified_vision.contracts import serializable
from perception_cad_search.pillbottle_cad_search import (
    build_support_query,
    load_model_spec,
)


_CAPTURES = itertools.count(1)


def _scalar(payload, key, default=None):
    if key not in payload:
        return default
    return np.asarray(payload.pop(key)).item()


def _support_cad(root: Path, model_id: int = 0):
    """Resolve the static support mesh supplied by the task registry, if any."""
    model_dir = os.environ.get("ULA_SUPPORT_MODEL_DIR")
    if not model_dir:
        return None
    directory = root / "tra_pipe/RoboTwin/assets/objects" / model_dir
    # The support also selects a per-episode CAD model (072_electronicscale ships
    # base0/base1/base2/base5/base6 and the 50 archived scenes cover 19/16/10/5).
    # Hard-coding base0 made 31 of 50 scenes use the wrong support CAD.
    model_id = int(model_id)
    mesh = directory / f"visual/base{int(model_id)}.glb"
    metadata = directory / f"model_data{int(model_id)}.json"
    if not mesh.is_file() or not metadata.is_file():
        candidates = sorted((directory / "visual").glob("base*.glb"))
        if not candidates:
            return None
        mesh = candidates[0]
        stem = mesh.stem.replace("base", "")
        metadata = directory / f"model_data{stem}.json"
        if not metadata.is_file():
            return None
    values = json.loads(metadata.read_text())
    configured_scale = os.environ.get("ULA_SUPPORT_SCALE")
    scale = configured_scale if configured_scale is not None else values.get("scale", 1.0)
    if isinstance(scale, str):
        scale = float(scale)
    if isinstance(scale, (int, float)):
        scale = (float(scale),) * 3
    return CadModel(
        mesh_path=str(mesh.relative_to(root)),
        scale=tuple(float(value) for value in scale),
    )


def _object_query(model, prompt, hints):
    """Build a task-generic exact-CAD query without pill-bottle aliases."""
    cad = CadModel(
        mesh_path=model["mesh_path"],
        scale=model["scale"],
        mesh_to_object=model.get("mesh_to_object"),
    )
    diameter = float(model["diameter_m"])
    _parts = [x.strip() for x in str(prompt).split("|") if x.strip()]
    return TargetQuery(
        key="object",
        prompts=tuple(_parts) or ("object",),
        attributes=("position", "bounds", "axis"),
        required=("position", "bounds"),
        hints=VisualHints(**hints) if hints else None,
        cad=cad,
        position_reference="cad_origin",
        max_extent_m=(1.35 * diameter,) * 3,
    )


class FrustumCadWorker(FrustumSelectionWorker):
    """Original frustum selection with the legacy known-CAD NPZ interface."""

    def __init__(self, root, output, fp_iterations=5, *, gpu=None, config=None,
                 timeout_s=300, **unused):
        self.fp_iterations = int(fp_iterations)
        self.object_asset = os.environ.get("ULA_KNOWN_OBJECT_ASSET", "")
        self.object_prompt = os.environ.get("ULA_KNOWN_OBJECT_PROMPT", "object")
        self.object_hints = json.loads(os.environ.get("ULA_KNOWN_OBJECT_HINTS", "{}"))
        self.support_prompt = os.environ.get("ULA_KNOWN_SUPPORT_PROMPT", "support")
        if config is None:
            config = VisionConfig(cad=CadLimits(
                iterations=self.fp_iterations, max_models=256))
        super().__init__(root, output, gpu=gpu, config=config, timeout_s=timeout_s)

    def estimate_files(self, request, result):
        started = time.perf_counter()
        with np.load(request, allow_pickle=False) as values:
            payload = {key: values[key].copy() for key in values.files}

        model_id = int(_scalar(payload, "model_id", 0))
        support_model_id = int(_scalar(
            payload, "support_model_id",
            os.environ.get("ULA_SUPPORT_MODEL_ID", "0")))
        object_asset = str(_scalar(payload, "object_asset", self.object_asset) or "")
        cup_only = bool(_scalar(payload, "cup_only", False))
        support_only = bool(_scalar(payload, "support_only", False))
        _scalar(payload, "refine_cup", False)
        support_color = _scalar(payload, "support_color", None)
        required = {"rgb", "depth", "intrinsic", "world_to_camera"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Legacy RGB-D request is missing {sorted(missing)}")

        queries = []
        exact_model = None
        if not support_only:
            if not object_asset:
                raise ValueError("Known-CAD frustum request has no object asset")
            model = load_model_spec(self.root, object_asset, model_id)
            queries.append(_object_query(model, self.object_prompt, self.object_hints))
            exact_model = f"{object_asset}/base{model_id}"
        if not cup_only:
            support = build_support_query(
                self.support_prompt, support_color, model_id=support_model_id)
            cad = _support_cad(self.root, support_model_id)
            if cad is not None:
                _cad_path = Path(cad.mesh_path)
                _cad_path = (_cad_path if _cad_path.is_absolute()
                             else self.root / _cad_path)
                if not _cad_path.is_file():
                    # Several assets ship without base0.glb (072_electronicscale,
                    # 080_pillbottle); fall back to the first model that exists.
                    from pathlib import Path as _P
                    _dir = _cad_path.parent
                    _cands = sorted(_dir.glob("base*.glb"))
                    if _cands:
                        cad = replace(cad, mesh_path=str(_cands[0]))
                support = replace(support, cad=cad)
                import os as _os, json as _json
                if _os.environ.get("ULA_DIAG_CAD_QUERY") == "1":
                    print(f"[cadq] support mesh={cad.mesh_path} scale={cad.scale} "
                          f"prompts={support.prompts} "
                          f"exists={__import__('pathlib').Path(cad.mesh_path).is_file()}", flush=True)
            queries.append(support)

        capture = next(_CAPTURES)
        frame = RGBDFrame(
            camera="head_or_requested_camera",
            capture_id=f"known_cad_frustum_{capture}",
            rgb=np.ascontiguousarray(payload["rgb"][..., :3], dtype=np.uint8),
            depth=np.asarray(payload["depth"]),
            intrinsic=np.asarray(payload["intrinsic"], dtype=float),
            world_to_camera=np.asarray(payload["world_to_camera"], dtype=float),
            depth_unit="mm",
        )
        scene = self.observe(
            [frame], queries,
            episode_id=f"known-cad-frustum:{capture}:{exact_model or 'support-only'}",
        )

        obj = scene.targets.get("object")
        support = scene.targets.get("support")
        if not support_only and (obj is None or not obj.required_available):
            status = "not_returned" if obj is None else obj.status
            reasons = [] if obj is None else obj.reasons
            raise RuntimeError(
                f"Unified vision could not localize exact CAD object: {status}; "
                f"reasons={reasons}")
        if not cup_only and (support is None or not support.required_available):
            status = "not_returned" if support is None else support.status
            reasons = [] if support is None else support.reasons
            raise RuntimeError(
                f"Unified vision could not localize support: {status}; reasons={reasons}")

        output = {}
        if obj is not None:
            pose = obj.geometry.get("full_object_pose_world")
            if pose is None:
                raise RuntimeError("Unified CAD result has no full 6-D object pose")
            output.update(
                cup_world_pose=np.asarray(pose, dtype=float),
                cup_bounds_world=np.asarray(obj.geometry["bounds_world"], dtype=float),
                cup_points_world=np.asarray(obj.points, dtype=float),
            )
        if support is None:
            output.update(
                coaster_world_center=np.full(3, np.nan),
                coaster_points_world=np.empty((0, 3)),
            )
        else:
            output.update(
                coaster_world_center=np.asarray(
                    support.geometry["position_world"], dtype=float),
                coaster_points_world=np.asarray(support.points, dtype=float),
            )
            support_pose = support.geometry.get("full_object_pose_world")
            if support_pose is not None:
                output["support_world_pose"] = np.asarray(support_pose, dtype=float)

        audit = dict(
            source="original_unified_frustum_at_known_cad_control_interface",
            exact_model=exact_model,
            exact_support_model=(None if cup_only else
                                 f"{os.environ.get('ULA_SUPPORT_MODEL_DIR', '')}/base{support_model_id}"),
            object_status="not_requested" if obj is None else obj.status,
            support_status="not_requested" if support is None else support.status,
            configuration=asdict(self.geometry_config),
            scene_audit=scene.audit,
            uses_object_truth=False,
            uses_simulator_segmentation=False,
        )
        output["observation_audit_json"] = np.asarray(
            json.dumps(serializable(audit)))
        output["timings_json"] = np.asarray(json.dumps(dict(
            perception_total=time.perf_counter() - started)))
        Path(result).parent.mkdir(parents=True, exist_ok=True)
        np.savez(result, **output)
        return self._load_legacy_result(result, started)

    def _load_legacy_result(self, result, started):
        """Return the namespace expected by ``ExternalPerception`` callers."""
        with np.load(result, allow_pickle=False) as values:
            timings = json.loads(str(values["timings_json"]))
            timings.update(
                perception_subprocess_wall=time.perf_counter() - started,
                vision_worker_pid=(None if self.process is None else self.process.pid),
                vision_request_id=self._stats["requests"],
            )
            pose = (values["cup_world_pose"].copy()
                    if "cup_world_pose" in values else None)
            return SimpleNamespace(
                cup_world_pose=pose,
                object_pose_world=pose,
                cup_bounds_world=(values["cup_bounds_world"].copy()
                                  if "cup_bounds_world" in values else None),
                cup_points_world=(values["cup_points_world"].copy()
                                  if "cup_points_world" in values else None),
                coaster_world_center=values["coaster_world_center"].copy(),
                coaster_points_world=values["coaster_points_world"].copy(),
                support_world_pose=(values["support_world_pose"].copy()
                                    if "support_world_pose" in values else None),
                observation_audit=json.loads(str(values["observation_audit_json"])),
                timings_s=timings,
            )
