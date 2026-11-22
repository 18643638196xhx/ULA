"""Independent RGB-D corner tracking for ambiguous bottle surface ICP.

No expected hand displacement is supplied to this estimator. Actual observed
feature motion is fitted first; the caller must separately test co-motion.
"""
import numpy as np


def feature_motion(before, after, reference_points):
    import cv2
    from traditional_place_empty_cup.geometry import depth_to_metres, homogeneous_transform
    from adapters.pointcloud_registration import _rigid_fit

    audit = dict(method="rgbd_lk_forward_backward_ransac", accepted=False,
        uses_object_truth=False, expected_motion_used=False)
    first, second = np.asarray(before["rgb"], dtype=np.uint8), np.asarray(after["rgb"], dtype=np.uint8)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("Expected same-camera RGB-D frames")
    h, w = first.shape[:2]
    reference = np.asarray(reference_points, dtype=float)
    transform = homogeneous_transform(before["extrinsic_cv"])
    xyz = reference@transform[:3, :3].T+transform[:3, 3]
    xyz = xyz[xyz[:, 2] > .01]
    pixels = xyz@np.asarray(before["intrinsic_cv"], dtype=float).T
    pixels = np.rint(pixels[:, :2]/pixels[:, 2:]).astype(int)
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    pixels = pixels[valid]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[pixels[:, 1], pixels[:, 0]] = 255
    mask = cv2.erode(cv2.dilate(mask, np.ones((3, 3), np.uint8)), np.ones((3, 3), np.uint8))
    gray0 = cv2.cvtColor(first[..., :3], cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(second[..., :3], cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray0, maxCorners=150, qualityLevel=.01,
        minDistance=2, mask=mask, blockSize=3)
    if corners is None or len(corners) < 6:
        audit["reason"] = "insufficient_object_image_features"
        return None, audit
    tracked, good, error = cv2.calcOpticalFlowPyrLK(gray0, gray1, corners, None,
        winSize=(11, 11), maxLevel=3)
    back, back_good, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, tracked, None,
        winSize=(11, 11), maxLevel=3)
    keep = (good[:, 0] > 0) & (back_good[:, 0] > 0) & (error[:, 0] < 25.)
    keep &= np.linalg.norm(back[:, 0]-corners[:, 0], axis=1) < 1.
    uv0, uv1 = corners[:, 0][keep], tracked[:, 0][keep]

    def world(frame, uv):
        depth = depth_to_metres(frame["depth"])
        xy = np.rint(uv).astype(int)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy = np.clip(xy, [0, 0], [w-1, h-1])
        z = depth[xy[:, 1], xy[:, 0]]
        inside &= np.isfinite(z) & (z > .01)
        camera = (np.linalg.solve(np.asarray(frame["intrinsic_cv"]), np.c_[uv, np.ones(len(uv))].T)*z).T
        inverse = np.linalg.inv(homogeneous_transform(frame["extrinsic_cv"]))
        return camera@inverse[:3, :3].T+inverse[:3, 3], inside
    source, a = world(before, uv0)
    target, b = world(after, uv1)
    source, target = source[a & b], target[a & b]
    audit.update(detected_features=len(corners), valid_rgbd_tracks=len(source))
    if len(source) < 6:
        audit["reason"] = "insufficient_consistent_rgbd_tracks"
        return None, audit
    rng = np.random.default_rng(0)
    best = np.zeros(len(source), dtype=bool)
    for _ in range(128):
        indices = rng.choice(len(source), 3, replace=False)
        try:
            rotation, translation = _rigid_fit(source[indices], target[indices])
        except RuntimeError:
            continue
        keep = np.linalg.norm(source@rotation.T+translation-target, axis=1) < .008
        if keep.sum() > best.sum():
            best = keep
    audit.update(inliers=int(best.sum()), inlier_fraction=float(best.mean()))
    if best.sum() < 6 or best.mean() < .6:
        audit["reason"] = "inconsistent_object_feature_motion"
        return None, audit
    try:
        rotation, translation = _rigid_fit(source[best], target[best])
    except RuntimeError:
        audit["reason"] = "degenerate_object_feature_distribution"
        return None, audit
    residual = np.linalg.norm(source[best]@rotation.T+translation-target[best], axis=1)
    rmse = float(np.sqrt(np.mean(residual**2)))
    if rmse > .005:
        audit["reason"] = "rgbd_feature_fit_residual_too_large"
        return None, audit
    motion = np.eye(4)
    motion[:3, :3], motion[:3, 3] = rotation, translation
    audit.update(accepted=True, rmse_m=rmse, measured_rigid_motion=motion.tolist(),
        observed_track_median_displacement_m=np.median(target[best]-source[best], axis=0).tolist())
    return motion, audit
