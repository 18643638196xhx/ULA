#!/usr/bin/env python3
"""Calibrate recorded_ee <-> plan_target / ee_link / TCP transforms in RoboTwin."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ULA_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = ULA_ROOT / "tra_pipe" / "RoboTwin"
for p in (str(ULA_ROOT), str(ROBOTWIN_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml

from adapters.runtime_frame_calibration import calibrate_robot_frames, save_calibration


def _load_task_config(task_name: str, task_config_name: str) -> dict:
    cfg_path = ROBOTWIN_ROOT / "task_config" / f"{task_config_name}.yml"
    if not cfg_path.exists():
        cfg_path = ROBOTWIN_ROOT / "env_cfg" / "task_config" / f"{task_config_name}.yml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["task_name"] = task_name
    emb_cfg_path = ROBOTWIN_ROOT / "task_config" / "_embodiment_config.yml"
    if not emb_cfg_path.exists():
        emb_cfg_path = ROBOTWIN_ROOT / "env_cfg" / "task_config" / "_embodiment_config.yml"
    if emb_cfg_path.exists() and "embodiment" in cfg:
        embodiments = yaml.safe_load(emb_cfg_path.read_text(encoding="utf-8"))
        emb = cfg["embodiment"][0] if isinstance(cfg["embodiment"], list) else cfg["embodiment"]
        if emb in embodiments:
            robot_file = ROBOTWIN_ROOT / embodiments[emb]["file_path"]
            cfg["left_robot_file"] = str(robot_file)
            cfg["right_robot_file"] = str(robot_file)
            cfg["dual_arm_embodied"] = True
            robot_yml = yaml.safe_load((robot_file / "config.yml").read_text(encoding="utf-8"))
            cfg["left_embodiment_config"] = robot_yml
            cfg["right_embodiment_config"] = robot_yml
    cam_cfg_path = ROBOTWIN_ROOT / "task_config" / "_camera_config.yml"
    if not cam_cfg_path.exists():
        cam_cfg_path = ROBOTWIN_ROOT / "env_cfg" / "task_config" / "_camera_config.yml"
    if cam_cfg_path.exists() and "camera" in cfg:
        cams = yaml.safe_load(cam_cfg_path.read_text(encoding="utf-8"))
        head_type = cfg["camera"].get("head_camera_type")
        if head_type in cams:
            cfg["head_camera_h"] = cams[head_type]["h"]
            cfg["head_camera_w"] = cams[head_type]["w"]
            cfg["head_camera_fovy"] = cams[head_type]["fovy"]
    cfg["need_plan"] = False
    cfg["render_freq"] = 0
    cfg["save_data"] = False
    cfg["eval_mode"] = True
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="handover_block")
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {out}")

    os.chdir(ROBOTWIN_ROOT)
    mod = importlib.import_module(f"envs.{args.task}")
    task = getattr(mod, args.task)()
    cfg = _load_task_config(args.task, args.task_config)
    try:
        task.setup_demo(now_ep_num=0, seed=args.seed, **cfg)
        cal = calibrate_robot_frames(task, num_samples=args.samples, seed=args.seed, source_task=args.task)
        saved = save_calibration(cal, out, overwrite=args.overwrite)
        print(json.dumps({"saved": str(saved), "calibration": cal.to_dict()}, indent=2, ensure_ascii=False))
    finally:
        if hasattr(task, "close_env"):
            task.close_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
