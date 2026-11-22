#!/usr/bin/env python3
"""Visualize frozen prediction-vs-target mask bounds; no inference or rescoring."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def box(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Cannot illustrate an empty mask")
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new diagnostic directory")
    args.output.mkdir(parents=True)
    report = json.loads((args.input/"report.json").read_text())
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    evidence = []
    for seed in args.seeds:
        index = next(i for i, e in enumerate(report["episodes"]) if e["seed"] == seed)
        directory = args.input/f"episode_{index:03d}_seed{seed}"
        with np.load(directory/"prediction.npz", allow_pickle=False) as data:
            predicted = data["cup_mask"].astype(bool)
        with np.load(directory/"evaluation_only_masks.npz", allow_pickle=False) as data:
            target = data["cup"].astype(bool)
        with Image.open(directory/"initial_head_camera.png") as raw:
            frame = raw.convert("RGB").resize((640, 480), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (640, 536), "white")
        canvas.paste(frame, (0, 56))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"Seed {seed}: frozen initial cup selection", fill="black", font=font)
        draw.text((8, 30), "RED = selected as cup", fill="#df2020", font=font)
        draw.text((320, 30), "GREEN = actual target cup", fill="#008000", font=font)
        for mask, color, label in ((predicted, "#ff2020", "SELECTED"),
                                   (target, "#00ff50", "TARGET")):
            x0, y0, x1, y1 = box(mask)
            coordinates = [2*x0, 2*y0+56, 2*x1, 2*y1+56]
            draw.rectangle(coordinates, outline=color, width=4)
            tx = min(coordinates[0]+5, 525)
            ty = min(coordinates[1]+5, 510)
            text_box = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle(text_box, fill="white")
            draw.text((tx, ty), label, fill="#df2020" if label == "SELECTED" else "#008000", font=font)
        canvas.save(args.output/f"seed_{seed:02d}_selection.png")
        evidence.append(dict(seed=seed, prediction_box=box(predicted), target_box=box(target),
                             intersection_pixels=int((predicted & target).sum())))
    (args.output/"evidence.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
