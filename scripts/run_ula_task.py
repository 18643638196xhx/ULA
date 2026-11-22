#!/usr/bin/env python3
"""Unified ULA task entry: one command for any registered support-placement task.

    run_ula_task.py --task <name> [the base runner's own arguments...]

The entry keeps each task's own base runner (env + approach primitive) and
replaces only the perception, the post-VLA placement and the fixed trial lift
with the shared implementations, driven by the task's declarative rule.
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWN_SPEC = {}

from adapters.ula_task_registry import TASKS  # noqa: E402
from adapters.unified_perception import PERCEPTION_SPEC  # noqa: E402
from adapters.task_rules import (  # noqa: E402
    load_articulated_container_rule, load_container_rule, load_press_rule,
    load_shake_rule, load_support_placement_rule, load_task_rule,
)


def _support_prompt(spec):
    """Resolve the support prompt for this task.

    place_mouse_pad and move_stapler_pad declare
    `support_prompt_format="{color_name} rectangular pad"`, but the pad colour is
    chosen inside setup_demo, i.e. per episode and long after the worker's prompt
    is fixed.  Rather than guess the colour, drop the placeholder and let the
    generated pad asset's span filter reject the wrong rectangles: "rectangular
    pad" plus CAD extents is enough to disambiguate.
    """
    prompt = spec["support_prompt"]
    template = spec.get("support_prompt_format")
    if template and "{color_name}" in str(template):
        prompt = str(template).replace("{color_name} ", "").replace("{color_name}", "")
    return " ".join(str(prompt).split())

def _load_module(path):
    spec = importlib.util.spec_from_file_location("ula_unified_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_adaptive_place(rule):
    def adaptive_place(*args, **kwargs):
        task, perception, arm, reference_path = args[0], args[1], args[2], args[3]
        result = args[-1]
        return finish_adaptive_support_placement(
            task, perception, arm, reference_path, result,
            registration_config=kwargs.get("registration_config"),
            release_gap_m=kwargs.get("release_gap_m"),
            rule=rule,
        )
    return adaptive_place


def _provisional(*args, **kwargs):
    result = None
    for value in args:
        if isinstance(value, dict):
            result = value
            break
    if result is not None:
        result.update(
            complete=False,
            verification_performed=False,
            trial_lift_executed=False,
            decision="closed_command_direct_to_adaptive_support_placement",
            uses_simulator_truth=False,
        )
    return dict(
        ready=True,
        detector="stable_closed_gripper_command",
        placement_permission_only=True,
        lifted_grasp_verified=False,
        uses_simulator_truth=False,
    )


def _build_task_adapter(spec):
    """Subclass the task env and alias the base pipeline's actor names."""
    os.chdir(ROOT / "tra_pipe/RoboTwin")
    env_module = importlib.import_module(spec["env_module"])
    env_class = getattr(env_module, spec["env_attr"])

    class TaskAdapter(env_class):
        def load_actors(self, *args, **kwargs):
            super().load_actors(*args, **kwargs)
            KNOWN_SPEC["task"] = self
            self.cup = getattr(self, spec["object_actor"])
            self.coaster = getattr(self, spec["support_actor"])
            # Some supports choose their colour during setup_demo.  Build the
            # same concrete query the archived 50-scene evaluator received,
            # after that per-episode choice is known and before perception is
            # constructed.
            support_template = spec.get("support_prompt_format")
            support_colour = getattr(self, "color_name", None)
            if support_template and support_colour:
                colour = str(support_colour).strip().lower()
                # Keep the colour out of the text prompt: the query validator
                # rejects unknown colour words, and several pads use cyan, magenta,
                # brown or pink, which are outside its calibrated palette.  The
                # structured hint below carries the colour only when supported.
                PERCEPTION_SPEC["support_prompt"] = str(support_template).replace(
                    "{color_name} ", "").replace("{color_name}", "")
                colour = "gray" if colour == "grey" else colour
                # VisualHints has a deliberately small calibrated palette.  The
                # semantic prompt can still name cyan/magenta; do not turn an
                # unsupported colour into a hard appearance gate.
                if colour in {
                    "red", "orange", "yellow", "green", "blue", "purple",
                    "white", "black", "gray",
                }:
                    PERCEPTION_SPEC["support_hints"] = dict(
                        colors=(colour,), appearance_required=True)
                else:
                    PERCEPTION_SPEC["support_hints"] = {}
            for key, asset in (("object_cad", spec.get("object_asset")),
                               ("support_cad", spec.get("support_asset"))):
                if not asset:
                    continue
                attr = asset.get("model_id_attr")
                model_id = getattr(self, attr, 0) if attr else 0
                model_dir = asset.get("model_dir")
                if model_dir is None and asset.get("model_dir_attr"):
                    model_dir = str(getattr(self, asset["model_dir_attr"], "") or "")
                if not model_dir:
                    continue
                scale = asset.get("scale")
                if scale is None:
                    import json as _json
                    metadata = (ROOT / "tra_pipe/RoboTwin/assets/objects" / model_dir
                                / f"model_data{model_id}.json")
                    scale = float(_json.loads(metadata.read_text())["scale"][0])
                if isinstance(scale, (int, float)):
                    scale = (scale, scale, scale)
                try:
                    import json as _json
                    _meta = _json.loads((ROOT / "tra_pipe/RoboTwin/assets/objects" / model_dir
                                         / f"model_data{model_id}.json").read_text())
                    _ext = np.asarray(_meta["extents"], dtype=float) * np.asarray(scale, dtype=float)
                    _sorted = np.sort(np.abs(_ext))
                    if _sorted[-1] > 1e-6:
                        PERCEPTION_SPEC["object_upright_ratio"] = float(_sorted[-1] / max(_sorted[-2], 1e-6))
                except Exception:
                    pass

                if key == "object_cad":
                    KNOWN_SPEC["model_id"] = int(model_id)
                    KNOWN_SPEC["model_dir"] = model_dir
                PERCEPTION_SPEC[key] = dict(
                    mesh_path=(f"tra_pipe/RoboTwin/assets/objects/{model_dir}"
                               f"/visual/base{model_id}.glb"),
                    scale=tuple(scale),
                )

    env_module.__dict__[spec["env_attr"]] = TaskAdapter
    # The shared base runner imports the cup env module, so patch that symbol too.
    cup_module = importlib.import_module("envs.place_empty_cup")
    cup_module.place_empty_cup = TaskAdapter
    return TaskAdapter


def load_demo_instruction(task, keyword=None):
    """Read the official task instruction from the recorded demos."""
    import glob
    import json

    import h5py

    files = sorted(glob.glob(
        f"/data2/xuhuixin/RoboTwin/data_official50/demo_clean/{task}/*/data/*.hdf5"))
    if not files:
        return None
    values = []
    for path in files[:3]:
        with h5py.File(path, "r") as handle:
            raw = handle["instructions"][()]
        values.extend(json.loads(raw.decode() if isinstance(raw, bytes) else raw))
    if not values:
        return None
    if keyword:
        key = str(keyword).lower()
        for text in values:
            if key in text.lower():
                return text
    return values[0]


def main():
    # Adapters (placement, relations) need the registry task name; RoboTwin's
    # class decorator renames env classes, so type(task).__name__ is unreliable.
    import os as _os
    if "--task" in sys.argv:
        try:
            _os.environ["ULA_TASK_NAME"] = sys.argv[sys.argv.index("--task") + 1]
        except (ValueError, IndexError):
            pass
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True)
    parser.add_argument("--instruction-from-demo", action="store_true")
    known, rest = parser.parse_known_args()
    spec = TASKS.get(known.task)
    if spec is None:
        raise SystemExit(f"Unknown task '{known.task}'. Known: {sorted(TASKS)}")
    try:
        rule = load_support_placement_rule(ROOT / spec["rule"])
    except ValueError:
        # Container/shake/press rules have different stage sequences; dispatch by
        # the declared sequence instead of assuming a support-placement rule.
        rule = load_task_rule(ROOT / spec["rule"])
    _build_task_adapter(spec)
    PERCEPTION_SPEC.update(
        object_prompt=spec["object_prompt"],
        support_prompt=spec["support_prompt"],
        object_hints=spec.get("object_hints") or {},
        support_hints=spec.get("support_hints") or {},
        object_cad=None,
        support_cad=spec.get("support_cad"),
    )
    base_path = ROOT / spec["base"]
    base = _load_module(base_path)
    # Default composition: original unified/frustum perception feeding the
    # existing CAD-annotation approach and placement controllers. Both older
    # alternatives remain explicit for controlled ablations.
    pipeline = os.environ.get("ULA_PIPELINE", "known_cad_unified")
    if pipeline in ("known_cad", "known_cad_unified"):
        # Generic unified vision (DINO+SAM+CAD, 100% localization on the 9 tasks)
        # supplies the 6D pose; the official RoboTwin CAD annotations drive the
        # approach and placement; the base keeps its real visual trial lift.
        from adapters.known_cad_model import (
            KnownCadWorker, approach_known_object, place_known_object,
        )
        if pipeline == "known_cad_unified":
            from adapters.frustum_cad_worker import FrustumCadWorker
            worker_class = FrustumCadWorker
        else:
            worker_class = KnownCadWorker
            worker_spec = spec.get("known_cad_worker")
            if worker_spec:
                module_name, class_name = worker_spec.split(":", 1)
                worker_class = getattr(importlib.import_module(module_name), class_name)
        fixed_asset = spec["object_asset"].get("model_dir") or ""
        contact_ids = spec.get("grasp_contact_ids") or {}
        # FoundationPose-backed known-CAD vision: the reference implementation from
        # the 84% baseline (1s/frame instead of the unified CAD verification's 18s).
        os.environ["ULA_KNOWN_OBJECT_ASSET"] = fixed_asset
        os.environ["ULA_KNOWN_OBJECT_PROMPT"] = spec["object_prompt"]
        os.environ["ULA_KNOWN_OBJECT_HINTS"] = json.dumps(spec.get("object_hints") or {})
        os.environ["ULA_KNOWN_SUPPORT_PROMPT"] = _support_prompt(spec)
        # The perception worker runs in a separate process, so PERCEPTION_SPEC is
        # not visible there.  Pass the support's CAD identity via the environment.
        support_asset = spec.get("support_asset") or {}
        if support_asset.get("model_dir"):
            os.environ["ULA_SUPPORT_MODEL_DIR"] = str(support_asset["model_dir"])
        scale = support_asset.get("scale")
        if scale is not None:
            os.environ["ULA_SUPPORT_SCALE"] = str(scale)
        setattr(base, spec["worker_class"], worker_class)
        # Persist the pre-grasp 6D pose beside the reference so the placement can
        # restore the upright orientation.  The 82% baseline depended on this;
        # without it the placement falls back to the (possibly tilted) current pose.
        has_external = hasattr(base, "ExternalPerception")
        original_reference = (base.ExternalPerception.select_reference
                              if has_external else None)

        def reference(self, value):
            original_reference(self, value)
            pose = getattr(value, "cup_world_pose", None)
            if pose is None:
                pose = getattr(value, "object_pose_world", None)
            if pose is not None:
                with np.load(self.reference_path, allow_pickle=False) as data:
                    payload = dict(data)
                payload["cup_world_pose"] = np.asarray(pose, dtype=float)
                np.savez(self.reference_path, **payload)

        if has_external:
            base.ExternalPerception.select_reference = reference
        # The stock verify_visual_trial_lift is object-agnostic once the vision
        # exposes supplementary(); it is re-used unchanged for every task.
        # ULA_SKIP_TRIAL_LIFT=1 cancels the visual trial lift entirely (permission
        # is granted without verification) to measure how much it costs/gains.
        if os.environ.get("ULA_SKIP_TRIAL_LIFT") == "1":
            setattr(base, spec["trial_lift_function"], _provisional)

        def resolved_asset():
            return KNOWN_SPEC.get("model_dir") or fixed_asset

        def resolved_model_id():
            # Resolve lazily from the live task object: at _build_task_adapter time
            # the env has not yet chosen its per-episode asset id, so the cached
            # KNOWN_SPEC["model_id"] is a stale 0 (e.g. 080_pillbottle has no base0).
            task = KNOWN_SPEC.get("task")
            live = None
            # Each task names its per-episode asset id differently
            # (pillbottle_id, stapler_id, fan_id, mouse_id, shoe_id, ...).
            attr_name = (spec.get("object_asset") or {}).get("model_id_attr")
            candidates = []
            if attr_name:
                candidates.append(str(attr_name))
            candidates += ["model_id", "object_model_id", "selected_model_id"]
            for attr in candidates:
                if live is not None:
                    break
                live = getattr(task, attr, None) if task is not None else None
            if live is None:
                live = KNOWN_SPEC.get("model_id", 0)
            value = int(live or 0)
            print(f"[ula] model_id resolve -> {value} "
                  f"(task={'set' if task is not None else 'None'}, "
                  f"pillbottle_id={getattr(task, 'pillbottle_id', 'N/A')})", flush=True)
            return value

        def resolved_support_model_id():
            """Resolve the support CAD selected by this live episode."""
            task = KNOWN_SPEC.get("task")
            attr_name = (spec.get("support_asset") or {}).get("model_id_attr")
            live = (getattr(task, str(attr_name), None)
                    if task is not None and attr_name else None)
            return int(live or 0)

        # The per-episode model id must travel with every perception request:
        # base scripts write only the RGB-D payload, so the worker would fall
        # back to model 0 (e.g. 080_pillbottle has no base0.glb).
        def _attach_model_id(fn):
            def wrapper(self, *args, **kwargs):
                import numpy as _np
                original = _np.savez_compressed
                def savez(path, **payload):
                    payload.setdefault("model_id", _np.asarray(resolved_model_id()))
                    payload.setdefault("object_asset", _np.asarray(resolved_asset()))
                    payload.setdefault(
                        "support_model_id", _np.asarray(resolved_support_model_id()))
                    # The support's colour is chosen per episode inside setup_demo,
                    # so it can only be forwarded here.  Workers start before that
                    # and would otherwise see a prompt fixed at launch time.
                    _live = KNOWN_SPEC.get("task")
                    _colour = getattr(_live, "color_name", None) if _live is not None else None
                    if _colour:
                        payload.setdefault("support_color", _np.asarray(str(_colour)))
                    return original(path, **payload)
                _np.savez_compressed = savez
                try:
                    return fn(self, *args, **kwargs)
                finally:
                    _np.savez_compressed = original
            return wrapper
        # Patch the class itself, not the base module's alias: pillbottle's base
        # imports PillbottlePerception (a subclass) and never exposes
        # ExternalPerception, so a hasattr guard on the base silently skipped this.
        import adapters.perception as _perception_module
        _cls = _perception_module.ExternalPerception
        _cls.estimate = _attach_model_id(_cls.estimate)
        _cls.supplementary = _attach_model_id(_cls.supplementary)
        _cls.support_only = _attach_model_id(_cls.support_only)

        desired_rotation = None
        if rule is not None and getattr(rule, "desired_orientation", None) == "fixed_quaternion":
            from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
            desired_rotation = pose_wxyz_to_matrix(
                np.array([0., 0., 0., *rule.desired_quaternion_wxyz], dtype=float))[:3, :3]

        base.approach = (lambda task, perception, args: approach_known_object(
            task, perception, args, object_asset=resolved_asset(), root=ROOT,
            contact_ids=contact_ids, model_id=resolved_model_id(),
            support_asset=(spec.get("support_asset") or {}).get("model_dir"),
            support_model_id=resolved_support_model_id()))
        setattr(base, spec["placement_function"],
                lambda task, perception, arm, reference_path, result, **kw: place_known_object(
                    task, perception, arm, reference_path, result,
                    object_asset=resolved_asset(), root=ROOT,
                    model_id=resolved_model_id(), desired_rotation=desired_rotation,
                    support_asset=(spec.get("support_asset") or {}).get("model_dir"),
                    support_model_id=resolved_support_model_id(),
                    allow_reorientation=(getattr(rule, "desired_orientation", None)
                                         == "unconstrained"), **kw))
    elif pipeline == "unified_vision":
        # This is the same DINO + SAM + RGB-D + CAD + frustum-selection stack as
        # scripts/replay_frustum_vision50.py.  Patch both names imported by the
        # base runner; changing only the worker would leave the old NPZ protocol
        # object in place and silently evaluate a different interface.
        from adapters.unified_perception import (
            UnifiedPerceptionWorker,
            UnifiedVisionPerception,
        )
        setattr(base, spec["worker_class"], UnifiedPerceptionWorker)
        setattr(base, spec["perception_class"], UnifiedVisionPerception)
    else:
        raise SystemExit(
            f"Unknown ULA_PIPELINE={pipeline!r}; expected 'known_cad_unified', "
            "'unified_vision', or 'known_cad'"
        )

    original_task_config = getattr(base, "task_config", None)
    if original_task_config is not None:
        def task_config(args):
            config = original_task_config(args)
            config["task_name"] = spec["env_attr"]
            return config

        base.task_config = task_config
    if "--instruction" not in rest:
        instruction = None
        if known.instruction_from_demo:
            instruction = load_demo_instruction(known.task, spec.get("object_actor"))
        rest = rest + ["--instruction", instruction or spec["instruction"]]
    extra = list(spec.get("extra_args", ()))
    if spec.get("demo_approach_prior") and "--demo-approach-prior" not in rest:
        extra.extend(("--demo-approach-prior", str(ROOT / spec["demo_approach_prior"])))
    index = 0
    while index < len(extra):
        flag = extra[index]
        value = extra[index + 1] if index + 1 < len(extra) else None
        if flag not in rest:
            rest = rest + [flag] + ([value] if value is not None else [])
        index += 2 if value is not None else 1
    sys.argv = [str(base_path)] + rest
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
