"""Optional soft background suppression, preserving RGB geometry and robot ROI."""
import numpy as np


def corners(bounds):
    bounds = np.asarray(bounds, dtype=float)
    return np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])


def projected_rectangle(points, frame, padding=10):
    from traditional_place_empty_cup.geometry import homogeneous_transform
    transform = homogeneous_transform(frame["extrinsic_cv"])
    xyz = np.asarray(points)@transform[:3, :3].T+transform[:3, 3]
    xyz = xyz[xyz[:, 2] > .01]
    if not len(xyz):
        return None
    pixels = xyz@np.asarray(frame["intrinsic_cv"]).T
    xy = pixels[:, :2]/pixels[:, 2:]
    h, w = frame["rgb"].shape[:2]
    lo, hi = np.floor(xy.min(0)).astype(int)-padding, np.ceil(xy.max(0)).astype(int)+padding
    lo, hi = np.maximum(lo, 0), np.minimum(hi, [w-1, h-1])
    if np.any(hi-lo < 3):
        return None
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def soften_background(rgb, rectangles):
    from PIL import Image, ImageFilter
    value = np.asarray(rgb)
    result = value.copy()
    mask = np.zeros(value.shape[:2], np.uint8)
    for x0, y0, x1, y1 in rectangles:
        mask[y0:y1+1, x0:x1+1] = 255
    smooth = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(3))).astype(float)/255
    weight = np.maximum(mask.astype(float)/255, smooth)[..., None]
    original = value[..., :3].astype(float)
    blurred = np.asarray(Image.fromarray(value[..., :3]).filter(ImageFilter.GaussianBlur(10))).astype(float)
    background = .2*original+.8*blurred
    result[..., :3] = np.rint(weight*original+(1-weight)*background).astype(value.dtype)
    return result


def focused_observation(task, observation, worker):
    bounds = getattr(worker, "focus_cup_bounds", None)
    target = getattr(worker, "focus_pad_center", None)
    audit = dict(applied=False, cameras={})
    if bounds is None or target is None:
        audit["reason"] = "no_accepted_visual_target"
        return observation, audit
    cup = corners(np.asarray(bounds)+[[-.025, -.025, -.025], [.025, .025, .025]])
    pad = corners(np.array([target+[-.07, -.07, -.01], target+[.07, .07, .01]]))
    hands = []
    for arm in ("left", "right"):
        # Robot FK only. Include both the TCP and end-effector/palm, not just
        # the object's pixels. Do not query robot or object identity masks.
        for center in (np.asarray(task.get_arm_pose(arm))[:3],
                np.asarray(getattr(task.robot, f"get_{arm}_tcp_pose")())[:3]):
            hands.append(corners(np.array([center-.05, center+.05])))
    result = dict(observation)
    result["observation"] = dict(observation["observation"])
    for name in ("head_camera", "left_camera", "right_camera"):
        frame = task.get_camera_observation(name)
        cup_rect = projected_rectangle(cup, frame)
        if cup_rect is None:
            audit["cameras"][name] = dict(applied=False, reason="cup_outside_camera_keep_original")
            continue
        rectangles = [cup_rect]
        for points in (pad, *hands):
            rectangle = projected_rectangle(points, frame)
            if rectangle is not None:
                rectangles.append(rectangle)
        original = observation["observation"][name]
        result["observation"][name] = dict(original,
            rgb=soften_background(original["rgb"], rectangles))
        audit["cameras"][name] = dict(applied=True, preserved_rectangles=rectangles)
        audit["applied"] = True
    return result, audit
