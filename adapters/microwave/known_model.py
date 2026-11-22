"""Opt-in known door CAD experiment. No reads of a live object actor."""

import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from adapters.microwave.perception import MicrowaveVision, MicrowaveWorker


def load_door_model(directory):
    """Assemble the MOVING link in its hinge frame, in metres, offline.

    Registering a whole appliance as one rigid body after its door opens would
    be invalid. Only the URDF door link is assembled here. Root placement and
    model_data.transform_matrix are NOT applied: FP estimates the link pose.
    """
    import trimesh
    directory = Path(directory).resolve()
    metadata = json.loads((directory/"model_data.json").read_text())
    root = ET.parse(directory/"mobility.urdf").getroot()
    joint = next(j for j in root.findall("joint")
                 if j.get("type") == "revolute" and j.find("child").get("link") == "link_0")
    scale = float(metadata["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid known model scale")
    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
    axis /= np.linalg.norm(axis)
    parts, handles = [], []
    for visual in root.find("link[@name='link_0']").findall("visual"):
        mesh_node = visual.find("geometry/mesh")
        mesh = trimesh.load(directory/mesh_node.get("filename"), force="mesh", process=False)
        origin = visual.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        transform = trimesh.transformations.euler_matrix(*rpy)
        transform[:3, 3] = xyz
        mesh.apply_transform(transform)
        mesh.apply_scale(scale)
        # Convert each texture before concatenation to preserve its appearance
        # without relying on a single material shared by unrelated OBJ parts.
        if hasattr(mesh.visual, "to_color"):
            try:
                colours = mesh.visual.to_color()
            except IndexError:
                # Some exported UVs slightly exceed 1.0. Only compensate for
                # affected trimesh versions; preserve the working vision-env
                # conversion used in the actual single robot experiment.
                if getattr(mesh.visual, "uv", None) is None:
                    raise
                mesh.visual.uv = np.clip(np.asarray(mesh.visual.uv), 0., 1.)
                colours = mesh.visual.to_color()
            mesh.visual = colours
        parts.append(mesh)
        if visual.get("name", "").startswith("handle"):
            handles.append(np.asarray(mesh.vertices))
    if not handles:
        raise ValueError("Known model lacks a labelled handle mesh")
    vertices = np.vstack(handles)
    # Match the front visible handle surface used by the RGB-D baseline, not
    # the unobserved inside of the bar. Both supplied assets face local +Z.
    handle = (vertices.min(0)+vertices.max(0))/2
    handle[2] = vertices[:, 2].max()
    return dict(mesh=trimesh.util.concatenate(parts), handle_local=handle,
                axis_local=axis, front_local=np.array([0., 0., 1.]),
                directory=directory, scale=scale, hinge_local=np.zeros(3))


def model_hinge(panel, handle):
    """Use the VISUALLY ESTIMATED CAD origin, not the visible opposite edge."""
    normal = np.asarray(panel["normal"])
    hinge = np.asarray(panel["known_model_hinge_world"], dtype=float).copy()
    axis = np.asarray(panel["known_model_axis_world"], dtype=float)
    if (not np.isfinite(hinge).all() or not np.isfinite(axis).all()
            or abs(axis@np.array([0., 0., 1.])) < np.cos(np.radians(10))):
        raise RuntimeError("Known-model visual hinge is not credibly vertical")
    hinge[2] = handle[2]  # Same vertical line; planar motion is unchanged.
    radial = np.asarray(handle)-hinge
    radius = float(np.linalg.norm(radial[:2]))
    outward = float(np.cross([0., 0., 1.], radial)@normal)
    if not .07 <= radius <= .9 or abs(outward) < .04:
        raise RuntimeError("Known-model visual hinge/handle geometry is implausible")
    return dict(hinge_world=hinge, axis_world=np.array([0., 0., 1.]),
        model_axis_estimated_world=axis, opening_sign=1. if outward > 0 else -1.,
        initial_normal=normal.copy(), initial_panel=panel, radius_m=radius,
        initial_handle=np.asarray(handle).copy(), uses_object_truth=False,
        source="known_urdf_door_origin_transformed_by_foundationpose_visual_estimate",
        uses_prepared_object_model=True, hinge_at_panel_edge_assumed=False,
        vertical_axis_assumed=True, hidden_hinge_directly_observed=False)


class KnownModelWorker(MicrowaveWorker):
    def __init__(self, root, output, *, model_directory, fp_iterations=5, **kwargs):
        super().__init__(root, output, **kwargs)
        self.backend = "foundationpose"  # Include FP/mycpp paths in child env.
        self.model_directory = Path(model_directory).resolve()
        self.fp_iterations = fp_iterations

    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(self.root/"scripts/perceive_microwave_known_model.py")
        command += ["--model-directory", str(self.model_directory),
                    "--debug-dir", str(self.output/"foundationpose"),
                    "--fp-iterations", str(self.fp_iterations)]
        return command, cwd, environment

    def summary(self):
        return dict(super().summary(), uses_prepared_object_model=True,
                    uses_foundationpose=True, model_directory=str(self.model_directory),
                    model_scope="moving_door_link_only")


class KnownModelVision(MicrowaveVision):
    """Initial RGB-D + known door mesh registration. No GT initialization."""
    def __init__(self, root, door_prompt, handle_prompt, model_directory, debug_dir, fp_iterations):
        super().__init__(root, door_prompt, handle_prompt)
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
        start = time.perf_counter()
        self.asset = load_door_model(model_directory)
        mesh = self.asset["mesh"]
        self.fp = FoundationPose(model_pts=np.asarray(mesh.vertices),
            model_normals=np.asarray(mesh.vertex_normals), mesh=mesh,
            scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(), debug=0, debug_dir=str(debug_dir))
        self.fp_iterations = fp_iterations
        self.models.torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        self.load_timings_s["foundationpose_load"] = elapsed
        self.load_timings_s["model_stack_load_total"] += elapsed

    def estimate(self, rgb, depth, intrinsic, world_to_camera, **kwargs):
        from traditional_place_empty_cup.geometry import depth_to_metres, homogeneous_transform
        from adapters.microwave.geometry import fit_panel, isolate_door_surface
        result = super().estimate(rgb, depth, intrinsic, world_to_camera, **kwargs)
        cloud, measured_handle = result["door_points"], result["handle_world"].copy()
        panel = fit_panel(cloud, result["camera_world"]-np.median(cloud, 0))
        cloud, _ = isolate_door_surface(cloud, panel, measured_handle)
        panel = fit_panel(cloud, panel["normal"])
        extrinsic = homogeneous_transform(world_to_camera)
        camera_points = cloud@extrinsic[:3, :3].T+extrinsic[:3, 3]
        pixels = camera_points@np.asarray(intrinsic).T
        pixels = np.rint(pixels[:, :2]/pixels[:, 2:]).astype(int)
        mask = np.zeros(np.asarray(depth).shape, bool)
        valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < mask.shape[1])
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] < mask.shape[0]))
        mask[pixels[valid, 1], pixels[valid, 0]] = True
        start = time.perf_counter()
        with self.models.torch.inference_mode():
            pose_camera = self.fp.register(K=np.asarray(intrinsic), rgb=rgb,
                depth=depth_to_metres(depth), ob_mask=mask, iteration=self.fp_iterations)
        self.models.torch.cuda.synchronize()
        if pose_camera is None:
            raise RuntimeError("Known door FoundationPose registration failed")
        pose = np.linalg.inv(extrinsic)@np.asarray(pose_camera)
        model_handle = pose[:3, :3]@self.asset["handle_local"]+pose[:3, 3]
        front = pose[:3, :3]@self.asset["front_local"]
        axis = pose[:3, :3]@self.asset["axis_local"]
        error = float(np.linalg.norm(model_handle-measured_handle))
        if (not np.isfinite(pose).all() or error > .025
                or front@panel["normal"] < np.cos(np.radians(15))
                or abs(axis[2]) < np.cos(np.radians(10))):
            raise RuntimeError(f"Known door pose fails fresh-depth validation: handle_error_m={error:.5f}, "
                               f"normal_cos={front@panel['normal']:.4f}, vertical_cos={abs(axis[2]):.4f}")
        result["handle_world"] = model_handle
        result["audit"].update(uses_prepared_object_model=True, uses_foundationpose=True,
            known_model=dict(directory=str(self.asset["directory"]), scope="moving_door_link",
                pose_world=pose.tolist(), hinge_world=pose[:3, 3].tolist(), axis_world=axis.tolist(),
                measured_handle_world=measured_handle.tolist(), model_handle_world=model_handle.tolist(),
                handle_validation_error_m=error, uses_object_truth=False, uses_gt_pose_initialization=False))
        result["timings_s"]["foundationpose_registration_s"] = time.perf_counter()-start
        return result
