#!/usr/bin/env python3
"""CLI entrypoint for compiling and executing ``ula_demo_primitives_v1`` rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ULA_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = ULA_ROOT / "tra_pipe" / "RoboTwin"
for p in (str(ULA_ROOT), str(ROBOTWIN_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapters.rule_compiler import load_and_compile_rule
from adapters.runtime_frame_calibration import FrameCalibration, load_calibration


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rule", required=True, help="Path to generated ula_demo_primitives_v1 rule.yaml")
    ap.add_argument("--right-rule", default=None, help="Optional right-arm rule.yaml for dual-side single-arm tasks")
    ap.add_argument("--calibration", default=None, help="Path to calibrated robot EE frame JSON")
    ap.add_argument("--output", required=True, help="New output directory (refuses to overwrite existing)")
    ap.add_argument("--dry-run", action="store_true", help="Compile, validate, and emit compiled_plan.json without running RoboTwin")
    ap.add_argument("--task-config", default="demo_clean", choices=("demo_clean", "demo_randomized"))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pickup-steps", type=int, default=200)
    ap.add_argument("--take-steps", type=int, default=200)
    ap.add_argument("--left-checkpoint", type=Path, default=None)
    ap.add_argument("--right-checkpoint", type=Path, default=None)
    ap.add_argument("--policy-backend", default="pi05", choices=("pi05", "worldla"))
    ap.add_argument("--worldla-model-dir", type=Path, default=None)
    ap.add_argument("--worldla-asset-checkpoint", type=Path, default=None)
    ap.add_argument("--receiver-standoff-m", type=float, default=0.095)
    args = ap.parse_args()

    # The live runner changes into RoboTwin before constructing episode paths.
    # Resolve once here so all artifacts stay under the requested directory.
    out_dir = Path(args.output).expanduser().resolve()
    if out_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {out_dir}")

    plan = load_and_compile_rule(args.rule)
    cal = load_calibration(args.calibration) if args.calibration else FrameCalibration.identity()

    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "compiled_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "calibration_used.json").write_text(
        json.dumps(cal.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = {
        "task": plan.task,
        "stages": [
            {
                "stage_id": s.stage_id,
                "primitive": s.primitive,
                "arm": s.arm,
                "controller": s.controller,
                "executor": s.executor,
                "role": s.role,
                "depends_on": list(s.depends_on),
                "relations": sorted(s.relations.keys()),
            }
            for s in plan.stages
        ],
        "warnings": list(plan.warnings),
        "calibration_robot": cal.robot_name,
        "dry_run": bool(args.dry_run),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if plan.task == "move_pillbottle_pad":
        from adapters.pillbottle_compiled_runner import (
            DEFAULT_WORLDLA_ASSET_CKPT as _PB_WL_ASSET,
            DEFAULT_WORLDLA_MODEL_DIR as _PB_WL_MODEL,
            run_pillbottle_compiled_episodes,
        )
        live_summary = run_pillbottle_compiled_episodes(
            plan,
            cal,
            output_dir=out_dir,
            task_config_name=args.task_config,
            episodes=args.episodes,
            seed=args.seed,
            pickup_steps=args.pickup_steps,
            worldla_model_dir=args.worldla_model_dir or _PB_WL_MODEL,
            worldla_asset_checkpoint=args.worldla_asset_checkpoint or _PB_WL_ASSET,
            right_plan=load_and_compile_rule(args.right_rule) if args.right_rule else None,
        )
        print(json.dumps(live_summary, indent=2))
        return
    if plan.task == "place_empty_cup":
        from adapters.place_empty_cup_compiled_runner import (
            DEFAULT_WORLDLA_ASSET_CKPT as _CUP_WL_ASSET,
            DEFAULT_WORLDLA_MODEL_DIR as _CUP_WL_MODEL,
            run_place_empty_cup_compiled_episodes,
        )
        live_summary = run_place_empty_cup_compiled_episodes(
            plan,
            cal,
            output_dir=out_dir,
            task_config_name=args.task_config,
            episodes=args.episodes,
            seed=args.seed,
            pickup_steps=args.pickup_steps,
            worldla_model_dir=args.worldla_model_dir or _CUP_WL_MODEL,
            worldla_asset_checkpoint=args.worldla_asset_checkpoint or _CUP_WL_ASSET,
            right_plan=load_and_compile_rule(args.right_rule) if args.right_rule else None,
        )
        print(json.dumps(live_summary, indent=2))
        return 0
    if plan.task == "place_mouse_pad":
        from adapters.place_mouse_pad_compiled_runner import (
            DEFAULT_WORLDLA_ASSET_CKPT as _MOUSE_WL_ASSET,
            DEFAULT_WORLDLA_MODEL_DIR as _MOUSE_WL_MODEL,
            run_place_mouse_pad_compiled_episodes,
        )
        live_summary = run_place_mouse_pad_compiled_episodes(
            plan,
            cal,
            output_dir=out_dir,
            task_config_name=args.task_config,
            episodes=args.episodes,
            seed=args.seed,
            pickup_steps=args.pickup_steps,
            worldla_model_dir=args.worldla_model_dir or _MOUSE_WL_MODEL,
            worldla_asset_checkpoint=args.worldla_asset_checkpoint or _MOUSE_WL_ASSET,
            right_plan=load_and_compile_rule(args.right_rule) if args.right_rule else None,
        )
        print(json.dumps(live_summary, indent=2))
        return 0
    if plan.task == "place_fan":
        from adapters.place_fan_compiled_runner import (
            DEFAULT_WORLDLA_ASSET_CKPT as _FAN_WL_ASSET,
            DEFAULT_WORLDLA_MODEL_DIR as _FAN_WL_MODEL,
            run_place_fan_compiled_episodes,
        )
        live_summary = run_place_fan_compiled_episodes(
            plan,
            cal,
            output_dir=out_dir,
            task_config_name=args.task_config,
            episodes=args.episodes,
            seed=args.seed,
            pickup_steps=args.pickup_steps,
            worldla_model_dir=args.worldla_model_dir or _FAN_WL_MODEL,
            worldla_asset_checkpoint=args.worldla_asset_checkpoint or _FAN_WL_ASSET,
            right_plan=load_and_compile_rule(args.right_rule) if args.right_rule else None,
        )
        print(json.dumps(live_summary, indent=2))
        return 0
    if plan.task != "handover_block":
        raise SystemExit(f"Live perception/VLA adapter currently wired for handover_block; got {plan.task!r}")

    from adapters.handover_compiled_runner import (
        DEFAULT_LEFT_CKPT,
        DEFAULT_RIGHT_CKPT,
        DEFAULT_WORLDLA_MODEL_DIR,
        DEFAULT_WORLDLA_ASSET_CKPT,
        run_handover_compiled_episodes,
    )
    live_summary = run_handover_compiled_episodes(
        plan,
        cal,
        output_dir=out_dir,
        task_config_name=args.task_config,
        episodes=args.episodes,
        seed=args.seed,
        left_checkpoint=args.left_checkpoint or DEFAULT_LEFT_CKPT,
        right_checkpoint=args.right_checkpoint or DEFAULT_RIGHT_CKPT,
        policy_backend=args.policy_backend,
        worldla_model_dir=args.worldla_model_dir or DEFAULT_WORLDLA_MODEL_DIR,
        worldla_asset_checkpoint=args.worldla_asset_checkpoint or DEFAULT_WORLDLA_ASSET_CKPT,
        receiver_preapproach_standoff_m=args.receiver_standoff_m,
        pickup_steps=args.pickup_steps,
        take_steps=args.take_steps,
    )
    print(json.dumps(live_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
