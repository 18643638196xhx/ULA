"""Post-inference diagnostics only; NEVER passed to target perception."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np


@contextmanager
def no_task_actions(task):
    names = ("move", "take_action", "take_dense_action", "grasp_actor", "place_actor",
             "open_gripper", "close_gripper", "move_by_displacement", "check_success")
    restored = []
    def forbidden(*args, **kwargs):
        raise RuntimeError("Initial-perception-only test forbids task actions and task success scoring")
    try:
        for name in names:
            if hasattr(task, name):
                restored.append((name, getattr(task, name), name in vars(task)))
                setattr(task, name, forbidden)
        yield
    finally:
        for name, value, owned in reversed(restored):
            if owned:
                setattr(task, name, value)
            else:
                delattr(task, name)


def mask_metrics(predicted, actual):
    predicted, actual = np.asarray(predicted, bool), np.asarray(actual, bool)
    intersection = int((predicted & actual).sum())
    union = int((predicted | actual).sum())
    available = bool(actual.any())
    return dict(target_visible=available, predicted_pixels=int(predicted.sum()),
                target_pixels=int(actual.sum()), intersection_pixels=intersection,
                iou=intersection/union if union else None,
                target_precision=intersection/int(predicted.sum()) if predicted.any() else 0.,
                target_recall=intersection/int(actual.sum()) if actual.any() else None,
                identity_match=bool(available and predicted.any() and intersection/union >= .5))


def evaluation_labels(task, camera_name):
    """Call ONLY after prediction files are written, on the same rendered frame."""
    cameras = task.cameras
    if camera_name == "left_camera":
        camera = cameras.left_camera
    elif camera_name == "right_camera":
        camera = cameras.right_camera
    else:
        camera = cameras.static_camera_list[cameras.static_camera_name.index(camera_name)]
    labels = camera.get_picture("Segmentation")[..., 1].copy()
    ids = {}
    for name, actor in (("cup", task.cup.actor), ("pad", task.coaster.actor)):
        value = getattr(actor, "per_scene_id", None)
        if value is None:
            raise RuntimeError("Cannot independently validate renderer entity IDs")
        ids[name] = int(value)
    return labels == ids["cup"], labels == ids["pad"], ids


def overlay(rgb, cup_mask, pad_mask, title, path, truth_masks=None):
    from PIL import Image, ImageDraw
    array = np.asarray(rgb)[..., :3].copy()
    for mask, color in ((cup_mask, [0, 210, 255]), (pad_mask, [255, 70, 170])):
        mask = np.asarray(mask, bool)
        array[mask] = (.60*array[mask]+.40*np.asarray(color)).astype(np.uint8)
    if truth_masks is not None:
        for target in truth_masks:
            target = np.asarray(target, bool)
            interior = np.zeros_like(target)
            interior[1:-1, 1:-1] = (target[1:-1, 1:-1] & target[:-2, 1:-1]
                                    & target[2:, 1:-1] & target[1:-1, :-2] & target[1:-1, 2:])
            array[target & ~interior] = [255, 230, 0]
    frame = Image.fromarray(array)
    canvas = Image.new("RGB", (frame.width, frame.height+24), "white")
    canvas.paste(frame, (0, 24))
    ImageDraw.Draw(canvas).text((5, 6), title, fill="black")
    canvas.save(path)


def contact_sheets(output, records):
    from PIL import Image, ImageDraw
    output = Path(output)
    for offset in range(0, len(records), 9):
        canvas = Image.new("RGB", (960, 792), "#ededed")
        for index, record in enumerate(records[offset:offset+9]):
            x, y = (index % 3)*320, (index//3)*264
            directory = output/f"episode_{offset+index:03d}_seed{record['seed']}"
            image = directory/"audit_head_camera.png"
            if not image.is_file():
                image = directory/"overlay_head_camera.png"
            if image.is_file():
                with Image.open(image) as tile:
                    canvas.paste(tile.convert("RGB"), (x, y))
            else:
                ImageDraw.Draw(canvas).text((x+8, y+12),
                    f"seed {record['seed']}: {record.get('failure_stage', 'unavailable')}", fill="black")
        canvas.save(output/f"contact_sheet_{offset//9+1:02d}.png")
