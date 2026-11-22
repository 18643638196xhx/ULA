#!/usr/bin/env python3
"""Build visual pose tracks from a synchronized RGB-D demonstration.

The first measured object cloud is the geometric template.  Later keyframes
are registered to that template, so no CAD mesh, actor pose, renderer label,
contact record or success state is consumed.  Colour names are declarative
segmentation cues for clean demonstrations; geometry comes from depth.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tra_pipe/RoboTwin")]

from adapters.cup_target_focus import _color_mask, world_points_image
from adapters.handover.initial_depth import split_depth_masks
from adapters.handover.initial_geometry import rectangle_xy
from adapters.pointcloud_registration import RegistrationConfig, estimate_pointcloud_motion
from capture_demo_keyframe_rgbd import required_frames

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def rgb(dataset, frame: int) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(bytes(dataset[frame]))).convert("RGB"))


def world_to_camera(cam2world_gl: np.ndarray) -> np.ndarray:
    # RoboTwin stores SAPIEN/OpenGL camera-to-world in HDF5.  Depth rays use
    # OpenCV axes, hence the Y/Z sign conversion before world projection.
    return np.diag([1., -1., -1., 1.]) @ np.linalg.inv(np.asarray(cam2world_gl, float))


def measured_cloud(handle, frame: int, colour: str) -> tuple[np.ndarray, dict]:
    clouds, views = [], {}
    for camera in CAMERAS:
        group = handle[f"vision/{camera}"]
        image = rgb(group["colors"], frame)
        depth = np.asarray(group["depths"][frame], float)
        intrinsic = np.asarray(group["intrinsic_matrix"][frame], float)
        external = world_to_camera(group["extrinsics_matrix"][frame])
        world, valid = world_points_image(depth, intrinsic, external)
        mask = _color_mask(image.astype(float) / 255., (colour,)) & valid
        # Anti-aliased silhouette pixels can carry the depth of the table or
        # wall behind the object.  Keep the largest coherent 3-D colour
        # component in each view before fusing cameras.
        from scipy.ndimage import binary_erosion
        interior = binary_erosion(mask, iterations=1)
        if interior.sum() >= 25:
            mask = interior
        components = split_depth_masks(mask, world, minimum=25)
        points = world[components[0]] if components else np.empty((0, 3))
        # Keep the robot workspace and reject colour pixels on the distant wall.
        points = points[(points[:, 2] > .65) & (points[:, 2] < 1.35)
                        & (np.abs(points[:, 0]) < .8) & (np.abs(points[:, 1]) < .8)]
        views[camera] = int(len(points))
        if len(points):
            clouds.append(points)
    if not clouds:
        raise RuntimeError(f"No {colour} RGB-D surface at frame {frame}")
    points = np.concatenate(clouds)
    _, indices = np.unique(np.floor(points / .0025).astype(np.int64), axis=0,
                           return_index=True)
    return points[np.sort(indices)], views


def horizontal_pose(points: np.ndarray) -> tuple[np.ndarray, list[float]]:
    points = np.asarray(points, float)
    lower, upper = np.quantile(points, [.01, .99], axis=0)
    robust = points[np.all((points >= lower) & (points <= upper), axis=1)]
    center_xy, sides, yaw = rectangle_xy(robust)
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    pose[:3, 3] = [*center_xy, (lower[2] + upper[2]) / 2.]
    return pose, np.r_[sides, upper[2] - lower[2]].tolist()


def pad(points: np.ndarray, limit: int = 2048) -> np.ndarray:
    points = np.asarray(points, float)
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit).astype(int)]
    result = np.full((limit, 3), np.nan)
    result[:len(points)] = points
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", required=True, type=Path)
    parser.add_argument("--rule", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--object-colour", required=True)
    parser.add_argument("--target-colour", required=True)
    args = parser.parse_args()
    args.demo, args.rule, args.output = args.demo.resolve(), args.rule.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Refusing to overwrite existing output")
    args.output.mkdir(parents=True)
    frames = required_frames(args.rule)
    audit = {"status": "running", "frames": frames, "uses_CAD": False,
             "uses_actor_pose": False, "uses_renderer_labels": False,
             "uses_contacts": False, "records": []}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(audit, indent=2) + "\n")

    poses, surfaces = [], []
    with h5py.File(args.demo, "r") as handle:
        required = [f"vision/{camera}/depths" for camera in CAMERAS]
        if any(path not in handle for path in required):
            raise ValueError("Demonstration does not contain all three depth streams")
        reference, initial_views = measured_cloud(handle, frames[0], args.object_colour)
        initial_pose, object_extent = horizontal_pose(reference)
        target_points, target_views = measured_cloud(handle, frames[0], args.target_colour)
        target_pose, target_extent = horizontal_pose(target_points)
        robust_lower, robust_upper = np.quantile(reference, [.01, .99], axis=0)
        diameter = float(np.linalg.norm(robust_upper - robust_lower))
        config = RegistrationConfig(
            voxel_m=.003,
            correspondence_m=float(np.clip(.25 * diameter, .025, .06)),
            inlier_m=float(np.clip(.10 * diameter, .009, .018)),
            max_rmse_m=float(np.clip(.075 * diameter, .007, .014)),
            min_observation_overlap=.25,
            min_reference_overlap=.15,
            max_tilt_deg=45.,
            iterations=45,
            min_points=40,
            yaw_starts_rad=tuple(np.linspace(-np.pi, np.pi, 16, endpoint=False)),
        )
        for frame in frames:
            points, views = measured_cloud(handle, frame, args.object_colour)
            if frame == frames[0]:
                motion, registration = np.eye(4), {"accepted": True, "reason": "template_frame"}
            else:
                motion, registration = estimate_pointcloud_motion(reference, points, config)
            accepted = motion is not None
            poses.append(motion @ initial_pose if accepted else np.full((4, 4), np.nan))
            surfaces.append(pad(points))
            audit["records"].append({
                "frame": frame, "accepted": accepted, "points": int(len(points)),
                "views": views, "registration": registration,
            })
            print(f"[demo-track] frame={frame} accepted={accepted} points={len(points)}", flush=True)

    target_poses = np.full((len(frames), 4, 4), np.nan)
    target_poses[0] = target_pose
    target_surfaces = np.full((len(frames), 2048, 3), np.nan)
    target_surfaces[0] = pad(target_points)
    accepted_frames = [row["frame"] for row in audit["records"] if row["accepted"]]
    metadata = {
        "demo_sha256": digest(args.demo), "world_frame": "world", "units": "m",
        "source": "clean_demo_synchronized_RGBD_template_registration",
        "static_entities": [args.target], "uses_CAD": False,
        "uses_actor_pose": False, "uses_renderer_labels": False,
        "uses_contact_truth": False, "object_template_frame": frames[0],
        "object_template_extent_m": object_extent, "target_extent_m": target_extent,
        "target_initial_view_points": target_views, "object_initial_view_points": initial_views,
        "observation_uncertainty_m": .01,
        "identity_continuity": {
            args.object: {
                "method": "same_initial_RGBD_template_registered_without_candidate_rebinding",
                "accepted_frames": accepted_frames,
                "semantic_identity_scope": "single_clean_demonstration_colour_component",
            }
        },
    }
    np.savez_compressed(
        args.output / "tracks.npz", frame_indices=np.asarray(frames, np.int64),
        **{f"pose__{args.object}": np.asarray(poses),
           f"points__{args.object}": np.asarray(surfaces),
           f"pose__{args.target}": target_poses,
           f"points__{args.target}": target_surfaces},
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    accepted = sum(row["accepted"] for row in audit["records"])
    audit.update(status="finished", accepted_frames=accepted,
                 rejected_frames=len(frames) - accepted,
                 tracks=str((args.output / "tracks.npz").resolve()), metadata=metadata)
    report_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
