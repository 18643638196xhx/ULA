"""Associate measured held-block components before estimating their pose.

Only RGB-D points, the previous visual/robot-motion prediction and the declared
primitive dimensions are used. Missing surfaces are allowed; a prediction never
supplies evidence that an occluded end has been observed.
"""
import numpy as np


def connected_clouds(points, *, voxel_m=.003, connection_m=.012):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Target components require finite measured XYZ points")
    if len(points) == 0:
        return []
    _, first, inverse = np.unique(np.floor(points/voxel_m).astype(np.int64),
                                  axis=0, return_index=True, return_inverse=True)
    samples = points[first]
    pairs = cKDTree(samples).query_pairs(connection_m, output_type="ndarray")
    if len(pairs):
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(samples), len(samples))).tocsr()
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(len(samples))
    result = []
    for label in np.unique(labels):
        selected = labels == label
        if selected.sum() >= 30:
            result.append(dict(points=points[selected[inverse]],
                               distinct_points=int(selected.sum())))
    return sorted(result, key=lambda item: -item["distinct_points"])


def fit_target_cloud(points, prior, *, max_candidates=4,min_visible_span_m=.07,
                     resolve_axis_ambiguity_with_prior=False):
    from adapters.handover.held_vision import HALF, fit_cuboid, surface_distance
    from scipy.spatial.distance import pdist
    prior = np.asarray(prior, dtype=float)
    if not np.isfinite(min_visible_span_m) or not 0<min_visible_span_m<=2*HALF[2]:
        raise ValueError('Invalid minimum visible target span')
    if prior.shape != (4, 4) or not np.isfinite(prior).all():
        raise ValueError("Invalid target association prior")
    components = connected_clouds(points)
    audit = dict(method="measured_3D_components_shape_and_motion_association",
                 uses_object_truth=False, uses_robot_identity_mask=False,
                 voxel_m=.003, connection_m=.012, prediction_margin_m=.035,
                 input_points=len(points), components=len(components), candidates=[],
                 min_visible_span_m=float(min_visible_span_m))
    audit['resolve_axis_ambiguity_with_prior']=bool(resolve_axis_ambiguity_with_prior)
    hypotheses = []
    fragments = []
    for index, component in enumerate(components):
        cloud = component["points"]
        local = (cloud-prior[:3, 3])@prior[:3, :3]
        inside = float(np.mean(np.all(np.abs(local) <= HALF+.035, axis=1)))
        span = np.ptp(cloud, axis=0)
        sample = cloud[np.linspace(0, len(cloud)-1, min(512, len(cloud))).astype(int)]
        diameter = float(np.max(pdist(sample)))
        # An upper extent limit rejects broad surfaces. A lower extent only
        # asks for a useful visible patch, never all six faces or full height.
        record = dict(component=index, points=len(cloud),
                      distinct_points=component["distinct_points"],
                      prediction_volume_fraction=inside,
                      world_spans_m=span.tolist(), diameter_m=diameter)
        audit["candidates"].append(record)
        if inside < .5 or diameter > np.linalg.norm(2*HALF)+.025:
            record.update(eligible=False, reason="incompatible_extent_or_prediction_region")
            continue
        # Rank by agreement with predicted surfaces, not raw pixel count:
        # a large nearby distractor must not automatically become the target.
        distance = float(np.median(np.abs(surface_distance(cloud, prior))))
        record.update(eligible=True, predicted_surface_median_m=distance)
        fragments.append((distance, index, cloud))
        if max(span) >= min_visible_span_m:
            hypotheses.append((distance, (index,), cloud, record))
    # A gripper can hide the middle of the block and split its visible surface.
    # Test bounded unions as well as single components, rather than require
    # each fragment to reveal the full long-axis span by itself.
    from itertools import combinations
    nearby = sorted([f for f in fragments if f[0] < .01], key=lambda f:f[0])[:4]
    for first, second in combinations(nearby, 2):
        cloud = np.concatenate((first[2], second[2]))
        span = np.ptp(cloud, axis=0)
        sample = cloud[np.linspace(0, len(cloud)-1, min(512, len(cloud))).astype(int)]
        if max(span) < min_visible_span_m or np.max(pdist(sample)) > np.linalg.norm(2*HALF)+.025:
            continue
        distance = float(np.median(np.abs(surface_distance(cloud, prior))))
        ids = (first[1], second[1])
        record = dict(components=list(ids), points=len(cloud), eligible=True,
                      predicted_surface_median_m=distance)
        audit['candidates'].append(record)
        hypotheses.append((distance, ids, cloud, record))
    accepted = []
    for _, ids, cloud, record in sorted(hypotheses, key=lambda item: (item[0], item[1]))[:max_candidates]:
        try:
            pose, quality = fit_cuboid(cloud, prior,min_visible_span_m=min_visible_span_m,
                resolve_axis_ambiguity_with_prior=resolve_axis_ambiguity_with_prior)
            shift = float(np.linalg.norm(pose[:3, 3]-prior[:3, 3]))
            angle = float(np.degrees(np.arccos(np.clip(pose[:3, 2]@prior[:3, 2], -1, 1))))
            if shift > .12 or angle > 65.:
                raise RuntimeError("Component pose is inconsistent with tracked motion")
            record.update(accepted=True, rmse_m=quality["rmse_m"],
                          center_change_m=shift, axis_change_deg=angle)
            score = float(np.median(np.abs(surface_distance(cloud, prior)))) + quality["rmse_m"]
            accepted.append((score, ids, cloud, pose, quality))
        except RuntimeError as error:
            record.update(accepted=False, error=str(error))
    if not accepted:
        raise RuntimeError("No trustworthy target component: "+str(audit))
    accepted.sort(key=lambda item: item[0])
    score, ids, selected, pose, quality = accepted[0]
    for other_score, other_ids, _, other_pose, _ in accepted[1:]:
        if other_score > score+.005:
            continue
        if set(ids) & set(other_ids):
            continue
        separation = float(np.linalg.norm(other_pose[:3, 3]-pose[:3, 3]))
        if separation > .02:
            raise RuntimeError("Ambiguous measured target components: "+str(audit))
    # Occlusion may split two sides/end faces of the SAME block. Reunite only
    # entire components already consistent with the measured candidate surface;
    # do not trim residuals until a bad candidate passes the existing fit gate.
    included = list(ids)
    extra = []
    for component_index, component in enumerate(components):
        if component_index in ids:
            continue
        errors = np.abs(surface_distance(component["points"], pose))
        if np.mean(errors < .006) >= .9:
            extra.append(component["points"])
            included.append(component_index)
    if extra:
        combined = np.concatenate([selected, *extra])
        try:
            combined_pose, combined_quality = fit_cuboid(combined, prior,min_visible_span_m=min_visible_span_m,
                resolve_axis_ambiguity_with_prior=resolve_axis_ambiguity_with_prior)
            shift = np.linalg.norm(combined_pose[:3, 3]-prior[:3, 3])
            angle = np.degrees(np.arccos(np.clip(combined_pose[:3, 2]@prior[:3, 2], -1, 1)))
            if shift <= .12 and angle <= 65.:
                pose, quality, selected = combined_pose, combined_quality, combined
            else:
                included = list(ids)
        except RuntimeError:
            included = list(ids)
    audit.update(selected_components=included, selected_points=len(selected),
                 rejected_points=int(len(points)-len(selected)))
    quality["target_selection"] = audit
    return pose, quality, selected
