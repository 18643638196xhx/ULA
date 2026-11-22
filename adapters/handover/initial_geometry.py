"""Initial upright cuboid localization from measured RGB-D surfaces.

Declared task priors: red square-section block 6x6x20 cm; blue square platform
approximately 10x10 cm; initial block upright on a horizontal table. No object
pose, simulator identity labels, CAD file or FoundationPose is consumed here.
"""
import numpy as np

BLOCK_SIZE = np.array([.06, .06, .20])


def rectangle_xy(points):
    import cv2
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 20 or not np.isfinite(points).all():
        raise ValueError("Insufficient finite points for a visible horizontal rectangle")
    rect = cv2.minAreaRect(np.ascontiguousarray(points[:, :2], dtype=np.float32))
    corners = cv2.boxPoints(rect).astype(float)
    edge = corners[1]-corners[0]
    return np.asarray(rect[0], float), np.sort(rect[1]), float(np.arctan2(edge[1], edge[0]))


def cuboid_surface_distance(points, pose, size=BLOCK_SIZE):
    local = (np.asarray(points)-pose[:3, 3])@pose[:3, :3]
    delta = np.abs(local)-np.asarray(size)/2
    return np.linalg.norm(np.maximum(delta, 0), axis=1)+np.abs(np.minimum(delta.max(axis=1), 0))


def estimate_upright_block(points, table_z):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 60 or not np.isfinite(points).all():
        raise ValueError("Insufficient finite block depth")
    lo, hi = np.quantile(points, [.01, .99], axis=0)
    spans = hi-lo
    if spans[2] < .10 or spans[2] > .27 or max(spans[:2]) > .16 or min(spans[:2]) < .02:
        raise RuntimeError("Measured extent is not a compact upright long block")
    upper = np.quantile(points[:, 2], .98)
    top = points[np.abs(points[:, 2]-upper) <= .004]
    center_xy, sides, yaw = rectangle_xy(top)
    if min(sides) < .035 or max(sides) > .085:
        raise RuntimeError(f"Visible top does not support the configured square section: {sides.tolist()}")
    top_z = float(np.median(top[:, 2]))
    bottom_z = top_z-BLOCK_SIZE[2]
    if not -.015 <= bottom_z-table_z <= .025:
        raise RuntimeError("Configured block bottom is inconsistent with the visual table")
    cosine, sine = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    pose[:3, 3] = [*center_xy, top_z-BLOCK_SIZE[2]/2]
    distances = cuboid_surface_distance(points, pose)
    median, fraction = float(np.median(distances)), float(np.mean(distances <= .015))
    if median > .008 or fraction < .65:
        raise RuntimeError(f"Visible depth disagrees with configured cuboid: median={median}, fraction={fraction}")
    return pose, dict(top_surface_points=len(top), observed_top_sides_m=sides.tolist(),
        measured_top_height_m=top_z, inferred_bottom_height_m=bottom_z,
        cuboid_surface_median_m=median, fraction_within_15mm=fraction,
        configured_block_size_m=BLOCK_SIZE.tolist(), yaw_symmetry_degrees=90,
        source="visible_top_rectangle_and_declared_upright_cuboid_dimensions",
        hidden_center_inferred_not_directly_observed=True, uses_object_truth=False)


def estimate_platform(points, table_z):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 40 or not np.isfinite(points).all():
        raise ValueError("Insufficient finite platform depth")
    center_xy, sides, yaw = rectangle_xy(points)
    height = float(np.median(points[:, 2]))
    flatness = float(np.quantile(points[:, 2], .95)-np.quantile(points[:, 2], .05))
    if not .06 <= min(sides) or max(sides) > .16 or min(sides)/max(sides) < .65:
        raise RuntimeError(f"Blue surface is not a plausible square platform: {sides.tolist()}")
    if abs(height-table_z) > .025 or flatness > .015:
        raise RuntimeError("Blue platform is not a horizontal surface near the visual table")
    return np.r_[center_xy, height], dict(sides_m=sides.tolist(), top_height_m=height,
        height_spread_m=flatness, source="measured_blue_platform_top_rectangle", uses_object_truth=False)


def estimate_supported_upright_block(points, table_z, camera_center, *, plane_tolerance_m=.0015):
    """Use visible side extent when the top is cropped, not a measured top.

    Explicit assumptions: known square section/height, upright on the measured
    horizontal table, and at least one substantially visible full-width side.
    Camera position disambiguates which side of a visible plane contains the
    hidden center. Small/ambiguous patches are rejected, never filled by truth.
    """
    if not np.isfinite(plane_tolerance_m) or plane_tolerance_m<=0:
        raise ValueError('A positive sensor-dependent plane tolerance is required')
    try:
        pose,audit=estimate_upright_block(points, table_z)
        planar=float(np.mean(cuboid_surface_distance(points,pose)<plane_tolerance_m))
        if planar>.85:
            audit.update(planar_face_fraction=planar,plane_tolerance_m=plane_tolerance_m)
            return pose,audit
    except (ValueError, RuntimeError):
        pass
    from scipy.optimize import least_squares
    points=np.asarray(points,dtype=float)
    camera=np.asarray(camera_center,dtype=float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<60 or
        not np.isfinite(points).all() or camera.shape!=(3,) or not np.isfinite(camera).all()):
        raise ValueError('Insufficient finite side surfaces or camera calibration')
    lo,hi=np.quantile(points,[.01,.99],axis=0)
    spans=hi-lo
    if spans[2]<.10 or spans[2]>.22 or hi[2]>table_z+BLOCK_SIZE[2]+.012 or max(spans[:2])>.12:
        raise RuntimeError('Measured side surfaces do not support the declared upright block')
    xy,sides,yaw=rectangle_xy(points)
    if max(sides)<.045 or max(sides)>.095:
        raise RuntimeError('No substantially visible square-section side width')
    _,indices=np.unique(np.floor(points/.002).astype(np.int64),axis=0,return_index=True)
    cloud=points[np.sort(indices)]
    if len(cloud)>1200:cloud=cloud[np.linspace(0,len(cloud)-1,1200).astype(int)]
    from adapters.handover.held_vision import planes
    angles=[yaw]
    for normal,_,count in planes(cloud):
        if abs(normal[2])<.15 and count>=60:
            angle=float(np.arctan2(normal[1],normal[0]))
            if all(abs((angle-a+np.pi/4)%(np.pi/2)-np.pi/4)>.03 for a in angles):
                angles.append(angle)
    candidates=[];attempts=[]
    for angle in angles:
        c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        local=cloud@rotation
        lower,upper=np.quantile(local,[.005,.995],axis=0)
        center=(lower[:2]+upper[:2])/2
        camera_local=camera@rotation
        supported=0
        for axis in range(2):
            width=upper[axis]-lower[axis]
            if width>=.045:
                supported+=1
            else:
                # Looking at a narrow projected slab does not reveal the
                # opposite face. Infer the center away from the camera.
                direction=camera_local[axis]-center[axis]
                if abs(direction)<.04:break
                center[axis]=(upper[axis]-.03 if direction>0 else lower[axis]+.03)
        else:
            if not supported:continue
            start_xy=(rotation[:2,:2]@center)
            start=np.r_[start_xy,angle]
            def pose_of(parameters):
                cosine,sine=np.cos(parameters[2]),np.sin(parameters[2])
                pose=np.eye(4)
                pose[:3,:3]=[[cosine,-sine,0.],[sine,cosine,0.],[0.,0.,1.]]
                pose[:3,3]=[*parameters[:2],table_z+BLOCK_SIZE[2]/2]
                return pose
            def residual(parameters):
                # A weak silhouette prior retains the measured side midpoint
                # along directions not constrained by point-to-plane distance.
                return np.r_[cuboid_surface_distance(cloud,pose_of(parameters)),
                             .02*(parameters[:2]-start_xy),.0002*(parameters[2]-angle)]
            fit=least_squares(residual,start,loss='soft_l1',f_scale=.002,max_nfev=80,
                bounds=(start- [.015,.015,.15],start+[.015,.015,.15]))
            pose=pose_of(fit.x)
            errors=cuboid_surface_distance(cloud,pose)
            local=(cloud-pose[:3,3])@pose[:3,:3]
            # A fitted box cannot put the measured face behind the object.
            nearest=np.argmin(np.abs(np.abs(local)-BLOCK_SIZE/2),axis=1)
            signs=np.sign(local[np.arange(len(local)),nearest])
            camera_local=(camera-pose[:3,3])@pose[:3,:3]
            visible=float(np.mean(signs*camera_local[nearest]>BLOCK_SIZE[nearest]/2))
            median=float(np.median(errors));fraction=float(np.mean(errors<.006))
            planar=float(np.mean(errors<plane_tolerance_m))
            attempts.append(dict(median_m=median,fraction_6mm=fraction,visible_fraction=visible,
                                 planar_face_fraction=planar,center=pose[:3,3].tolist(),yaw_rad=float(fit.x[2])))
            if median<.003 and fraction>.90 and visible>.90 and planar>.85:
                candidates.append((float(np.sqrt(np.mean(errors**2))),pose,dict(
                    source='visible_side_width_camera_visibility_and_visual_table_support',
                    measured_top_required=False,initial_upright_on_table_assumed=True,
                    visible_side_span_m=spans.tolist(),projected_rectangle_sides_m=sides.tolist(),
                    cuboid_surface_median_m=median,fraction_within_6mm=fraction,
                    camera_visible_surface_fraction=visible,configured_block_size_m=BLOCK_SIZE.tolist(),
                    planar_face_fraction=planar,plane_tolerance_m=plane_tolerance_m,
                    inferred_bottom_height_m=float(table_z),hidden_center_inferred_not_directly_observed=True,
                    uses_object_truth=False)))
    if not candidates:raise RuntimeError('Visible sides cannot constrain a consistent camera-facing upright cuboid: '+str(attempts))
    candidates.sort(key=lambda item:item[0])
    _,pose,audit=candidates[0]
    for error,alternative,_ in candidates[1:]:
        if error<=candidates[0][0]+.001 and np.linalg.norm(alternative[:3,3]-pose[:3,3])>.01:
            raise RuntimeError('Ambiguous hidden center from initial visible sides')
    return pose,audit


def yaw_error_modulo_square_degrees(estimated, actual):
    first, second = np.asarray(estimated), np.asarray(actual)
    delta = np.arctan2(first[1, 0], first[0, 0])-np.arctan2(second[1, 0], second[0, 0])
    return float(np.degrees(abs((delta+np.pi/4) % (np.pi/2)-np.pi/4)))
