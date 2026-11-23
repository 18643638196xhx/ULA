#!/usr/bin/env python3
"""保存示范抓取帧的物体点云，作为运行时配准模板。

用法: build_grasp_cloud_template.py <demo.hdf5> <object> <prompt> <frame> <out.npz> <anchor_pose_json>
输出: npz，含 <object>_points_world 与 anchor（4x4，示范物体位姿）
"""
import json
import sys
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ULA = Path("/data1/xuhuixin/ula")
sys.path[:0] = [str(ULA), str(ULA / "tra_pipe/RoboTwin")]
from adapters.unified_perception import (  # noqa: E402
    PERCEPTION_SPEC, UnifiedPerceptionWorker, UnifiedVisionPerception,
)

demo, obj, prompt, frame = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
out, anchor_json = Path(sys.argv[5]), Path(sys.argv[6])
anchor = np.asarray(json.loads(anchor_json.read_text()), dtype=float)

PERCEPTION_SPEC.update({
    "object_prompt": prompt, "support_prompt": None,
    "object_cad": None, "support_cad": None, "object_hints": None, "support_hints": None,
})

with h5py.File(demo, "r") as f:
    cam = f["vision/cam_head"]
    rgb = np.asarray(Image.open(BytesIO(bytes(cam["colors"][frame]))).convert("RGB"))
    depth = np.asarray(cam["depths"][frame], dtype=float)
    if np.nanmax(depth) < 20.0:
        depth = depth * 1000.0
    K = np.asarray(cam["intrinsic_matrix"][frame], dtype=float)
    E = np.asarray(cam["extrinsics_matrix"][frame], dtype=float)

ext4 = np.eye(4)
ext4[:E.shape[0], :E.shape[1]] = E
ext_cv = np.linalg.inv(ext4 @ np.diag([1.0, -1.0, -1.0, 1.0]))[:3, :4]

worker = UnifiedPerceptionWorker(ULA, out.parent / (out.stem + "_worker"))
try:
    perc = UnifiedVisionPerception(ULA, out.parent / (out.stem + "_perc"), worker=worker)
    est = perc.estimate(rgb, depth, K, ext_cv)
    pts = np.asarray(est.cup_points_world, dtype=float)
finally:
    worker.close()

np.savez_compressed(out, **{f"{obj}_points_world": pts}, anchor=anchor)
print(f"保存 {out}  点数={len(pts)}  anchor={np.round(anchor[:3,3],4)}")
