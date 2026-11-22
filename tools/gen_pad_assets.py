"""Generate CAD templates for support actors that are `create_box` primitives.

RoboTwin builds most supports as programmatic boxes (the pillbottle pad is
`half_size=[0.04, 0.04, 0.0005]`), so they have no asset directory and the vision
cannot verify them against CAD.  Their geometry is fully specified in the env
source, so a template can be generated once and registered.

Reading the env source is offline asset generation, not runtime truth: the mesh
and its extents are static task priors, exactly like a shipped CAD file.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xuhuixin/ula")
SIM = ROOT / "tra_pipe/RoboTwin"
sys.path[:0] = [str(ROOT)]


def find_box_spec(env_path: Path, actor: str):
    """Locate `self.<actor> = create_box(..., half_size=(a,b,c), ...)` in the source."""
    source = env_path.read_text()
    # Match the assignment and the following create_box call up to its close paren.
    pattern = re.compile(
        rf"self\.{re.escape(actor)}\s*=\s*create_box\((.*?)\n\s*\)", re.S)
    match = pattern.search(source)
    if not match:
        return None
    body = match.group(1)
    half = re.search(r"half_size\s*=\s*\(([^)]*)\)", body)
    if not half:
        half = re.search(r"half_size\s*=\s*\[([^\]]*)\]", body)
    if half:
        values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                              half.group(1))]
        if len(values) == 3:
            return dict(half_size=values, body=body)
    # Many envs assign `half_size = [...]` on its own line, then pass the name.
    named = re.search(r"half_size\s*=\s*([A-Za-z_][A-Za-z_0-9]*)", body)
    if named:
        var = re.escape(named.group(1))
        assign = re.search(rf"^\s*{var}\s*=\s*\[([^\]]*)\]", source, re.M)
        if assign:
            values = [float(v) for v in re.findall(
                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", assign.group(1))]
            if len(values) == 3:
                return dict(half_size=values, body=body)
    return None


def generate(task: str, actor: str, name: str, env_module: str):
    env_path = SIM / (env_module.replace(".", "/") + ".py")
    if not env_path.is_file():
        raise SystemExit(f"env not found: {env_path}")
    spec = find_box_spec(env_path, actor)
    if spec is None:
        raise SystemExit(f"no create_box found for self.{actor} in {env_path}")

    half = np.asarray(spec["half_size"], dtype=float)
    extents = (half * 2.0).tolist()
    print(f"  {task}: self.{actor} half_size={half.tolist()} -> extents={extents}")

    try:
        import trimesh
    except Exception as error:
        raise SystemExit(f"trimesh unavailable: {error}")

    out = SIM / f"assets/objects/{name}"
    (out / "visual").mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=extents)
    mesh.export(str(out / "visual/base0.glb"))

    meta = {
        "center": [0.0, 0.0, 0.0],
        "extents": extents,
        "scale": [1.0, 1.0, 1.0],
        "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "functional_matrix": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
        "functional_point_discription": ["centre of the pad top surface"],
        "contact_points_pose": [],
        "stable": True,
        "generated_from": f"{env_module}:self.{actor}",
    }
    (out / "model_data0.json").write_text(json.dumps(meta, indent=2))
    print(f"    written -> {out}")
    return name, extents


TARGETS = [
    ("place_shoe", "target_block", "shoe_pad", "envs.place_shoe"),
    ("place_fan", "pad", "fan_pad", "envs.place_fan"),
    ("place_mouse_pad", "target", "mouse_pad", "envs.place_mouse_pad"),
    ("move_stapler_pad", "pad", "stapler_pad", "envs.move_stapler_pad"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    results = []
    for task, actor, name, module in TARGETS:
        if args.only and args.only != task:
            continue
        try:
            results.append(generate(task, actor, name, module))
        except SystemExit as error:
            print(f"  {task}: SKIP ({error})")
    print("\n生成汇总:")
    for name, extents in results:
        print(f"  {name}: {[round(v,4) for v in extents]} m")


if __name__ == "__main__":
    main()
