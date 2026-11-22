"""Inspect an adjust_bottle demonstration without using simulator object truth.

Writes a compact RGB contact sheet and measured robot-state summary. The HDF5
demonstration does not contain depth or bottle pose, so this script deliberately
does not infer an object-relative TCP target or claim motion-planning success.
"""

import argparse
import io
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CAMERAS = ("cam_head", "cam_left_wrist", "cam_third_view")
FRAMES = (40, 45, 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    with h5py.File(args.demo) as demo:
        freq = int(demo["additional_info/frequency"][()])
        gripper = np.asarray(demo["state/left_ee_joint_states"]).reshape(-1)
        right_gripper = np.asarray(demo["state/right_ee_joint_states"]).reshape(-1)
        poses = np.asarray(demo["state/left_ee_poses"])
        closures = np.flatnonzero(np.diff(gripper) < -0.02) + 1
        if not len(closures):
            raise ValueError("No left gripper closure in demonstration")
        first, last = int(closures[0]), int(closures[-1])
        near_closure = poses[first - 1, :3]

        fig, axes = plt.subplots(len(CAMERAS), len(FRAMES), figsize=(13.5, 9))
        rows = []
        for col, frame in enumerate(FRAMES):
            pose = poses[frame]
            rows.append({
                "frame": frame,
                "time_s": frame / freq,
                "left_ee_pose_xyz_wxyz": pose.astype(float).tolist(),
                "left_gripper_state": float(gripper[frame]),
                "tcp_distance_to_preclosure_tcp_m": float(np.linalg.norm(pose[:3] - near_closure)),
                "tcp_vertical_difference_to_preclosure_m": float(pose[2] - near_closure[2]),
            })
            for row, camera in enumerate(CAMERAS):
                data = bytes(demo[f"vision/{camera}/colors"][frame])
                axes[row, col].imshow(Image.open(io.BytesIO(data)).convert("RGB"))
                axes[row, col].axis("off")
                axes[row, col].set_title(f"frame {frame} | {frame/freq:.2f}s | z={pose[2]:.3f}m")
                if col == 0:
                    axes[row, col].set_ylabel(camera, fontsize=12)
        fig.suptitle("adjust_bottle episode 0000000: before left-gripper closure", fontsize=16)
        fig.tight_layout()
        fig.savefig(args.output / "frames_40_45_50.png", dpi=150)
        plt.close(fig)

        summary = {
            "demo": str(args.demo),
            "frequency_hz": freq,
            "frames_total": len(gripper),
            "closure_start_frame": first,
            "closure_end_frame": last,
            "closure_start_s": first / freq,
            "closure_end_s": last / freq,
            "left_gripper_after_closure": float(gripper[last]),
            "right_gripper_initial_final": [float(right_gripper[0]), float(right_gripper[-1])],
            "reference_preclosure_frame": first - 1,
            "suggested_handoff_frame": 40,
            "handoff_selection_basis": "RGB view shows the gripper above the bottle; TCP is about 8 cm above the pre-closure TCP, leaving local VLA approach",
            "candidate_frames": rows,
            "object_relative_target_available": False,
            "reason": "HDF5 contains RGB and robot states, but no depth or bottle 3D pose",
        }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
