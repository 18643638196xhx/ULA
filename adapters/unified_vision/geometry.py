"""Class-independent visible surfaces, local patches and protrusion candidates.

Every centre/boundary describes OBSERVED points, not a hidden object centre,
CAD frame, bottom, complete support region or certified grasp affordance.
"""
import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def world_grid(frame):
    depth=frame.depth_m
    y,x=np.indices(depth.shape)
    rays=np.linalg.solve(frame.intrinsic,np.stack((x.ravel(),y.ravel(),np.ones(x.size))))
    camera=(rays*depth.reshape(1,-1)).T
    pose=np.linalg.inv(frame.world_to_camera)
    points=(camera@pose[:3,:3].T+pose[:3,3]).reshape(*depth.shape,3)
    valid=(depth>0)&np.isfinite(points).all(2)
    return points,valid


def voxelize(points,pitch):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all():
        raise ValueError('Expected finite measured Nx3 points')
    if not len(points):return points.copy()
    _,index=np.unique(np.floor(points/pitch).astype(np.int64),axis=0,return_index=True)
    return points[np.sort(index)]


def plane(points,camera,config,*,min_fraction=None):
    points=voxelize(points,config.voxel_m)
    if len(points)<config.min_points:return None
    sample=points[np.linspace(0,len(points)-1,min(1800,len(points))).astype(int)]
    rng=np.random.default_rng(0);best=np.zeros(len(sample),bool)
    for _ in range(100):
        a,b,c=sample[rng.choice(len(sample),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-10:continue
        keep=np.abs((sample-a)@(normal/length))<=config.plane_tolerance_m
        if keep.sum()>best.sum():best=keep
    fraction=config.plane_min_fraction if min_fraction is None else min_fraction
    if best.sum()<config.min_points or best.mean()<fraction:return None
    selected=sample[best]
    for _ in range(2):
        centre=np.median(selected,axis=0)
        _,singular,vt=np.linalg.svd(selected-centre,full_matrices=False)
        if singular[1]/np.sqrt(len(selected))<2*config.voxel_m:return None
        normal=vt[-1]
        if normal@(np.asarray(camera)-centre)<0:normal=-normal
        keep=np.abs((points-centre)@normal)<=config.plane_tolerance_m
        if keep.sum()<config.min_points or keep.mean()<fraction:return None
        selected=points[keep]
    centre=selected.mean(0)
    x=vt[0];y=np.cross(normal,x)
    coordinates=(selected-centre)@np.column_stack((x,y))
    hull=ConvexHull(coordinates)
    footprint=centre+coordinates[hull.vertices]@np.vstack((x,y))
    return dict(centre_world=centre,normal_world=normal,tangent_basis_world=np.column_stack((x,y)),
        tangent_bounds=np.quantile(coordinates,[.01,.99],axis=0),
        visible_projection_hull_world=footprint,inlier_fraction=float(keep.mean()),
        rmse_m=float(np.sqrt(np.mean(((selected-centre)@normal)**2))),
        normal_sign_source='toward_observing_camera',hidden_boundary_observed=False,
        placement_region_certified=False)


def geometry(points,camera,config,attributes):
    points=voxelize(points,config.voxel_m)
    if len(points)<config.min_points:raise RuntimeError('insufficient_distinct_valid_depth')
    bounds=np.quantile(points,[.01,.99],axis=0)
    output=dict(position_world=bounds.mean(0),bounds_world=bounds,point_count=len(points),
        position_reference='visible_surface_bounds_centre_not_hidden_object_centre',
        position_error_bound_m=None,full_object_pose_world=None,
        extent_is_complete=False,hidden_support_anchor_observed=False)
    usable=dict(position=True,bounds=True,plane=False,axis=False,structures=False,motion=False)
    if 'plane' in attributes or 'structures' in attributes:
        estimate=plane(points,camera,config)
        if estimate is not None:output['plane']=estimate;usable['plane']=True
    if 'axis' in attributes:
        values,vectors=np.linalg.eigh(np.cov(points.T))
        gap=float(values[-1]/max(values[-2],1e-12))
        if gap>=2.:
            output['axis']=dict(direction_world=vectors[:,-1],sign_observable=False,
                source='visible_points_principal_axis_not_semantic_object_axis',eigenvalue_ratio=gap)
            usable['axis']=True
    return output,usable,points


def surface_patches(points,camera,config):
    """A global patch when supported, otherwise local patches on curved surfaces.

    No vertical-plane prior, world-Z bar shape or object-class switch. Local
    tangent patches are approximations; edges/large residuals are rejected.
    """
    points=voxelize(points,config.voxel_m)
    main=plane(points,camera,config)
    if main is not None:return [dict(main,source='supported_global_surface_patch')]
    if len(points)<config.min_points:return []
    # A body may expose several disconnected/recessed surface layers. Extract
    # measured 3-D plane patches without the old vertical-door constraint;
    # also retain local tangent patches for genuinely curved bodies.
    result=[];remaining=points.copy()
    for _ in range(4):
        patch=plane(remaining,camera,config,min_fraction=.12)
        if patch is None:break
        support=np.abs((remaining-patch['centre_world'])@patch['normal_world'])<=config.plane_tolerance_m
        if support.sum()<max(config.min_points,int(.12*len(points))):break
        result.append(dict(patch,source='measured_multi_surface_layer_patch'))
        remaining=remaining[~support]
        if len(remaining)<config.min_points:break
    tree=cKDTree(points)
    seed_points=voxelize(points,config.local_surface_radius_m*.8)
    for seed in seed_points[np.linspace(0,len(seed_points)-1,min(64,len(seed_points))).astype(int)]:
        indices=tree.query_ball_point(seed,config.local_surface_radius_m)
        if len(indices)<config.min_points:continue
        selected=points[indices]
        centre=selected.mean(0)
        _,singular,vt=np.linalg.svd(selected-centre,full_matrices=False)
        if (singular[1]/np.sqrt(len(selected))<2*config.voxel_m or
            singular[-1]/np.sqrt(len(selected))>config.plane_tolerance_m):continue
        normal=vt[-1]
        if normal@(np.asarray(camera)-centre)<0:normal=-normal
        x=vt[0];y=np.cross(normal,x);basis=np.column_stack((x,y))
        result.append(dict(centre_world=centre,normal_world=normal,tangent_basis_world=basis,
            tangent_bounds=np.quantile((selected-centre)@basis,[.02,.98],axis=0),
            source='local_tangent_patch_of_measured_surface'))
    return result


def protrusions(body_points,search_points,camera,config):
    """Generic surface-relative projections; NOT a handle or grasp decision.

    Parent association is geometric proximity, not proof of physical connection.
    Camera-visible outward protrusions only; unseen/rear structures are unknown.
    """
    body=voxelize(body_points,config.voxel_m)
    search=voxelize(search_points,config.voxel_m)
    if len(body)<config.min_points or len(search)<config.min_points:
        return [],dict(reason='insufficient_parent_or_search_depth',patches=0)
    patches=surface_patches(body,camera,config)
    eligible=np.zeros(len(search),bool)
    height=np.zeros(len(search))
    for patch in patches:
        delta=search-patch['centre_world'];signed=delta@patch['normal_world']
        projected=delta@patch['tangent_basis_world']
        low,high=patch['tangent_bounds']
        inside=np.all((projected>=low-config.attachment_distance_m)&
                      (projected<=high+config.attachment_distance_m),axis=1)
        keep=inside&(signed>=config.protrusion_min_m)&(signed<=config.protrusion_max_m)
        eligible|=keep;height[keep]=np.maximum(height[keep],signed[keep])
    indices=np.flatnonzero(eligible)
    if len(indices)<config.min_points:
        return [],dict(reason='no_supported_outward_structure',patches=len(patches))
    # Bounded memory: voxelize already removed duplicate pixels; cap before
    # neighbour graph creation rather than constructing an unbounded dense graph.
    if len(indices)>12000:indices=indices[np.linspace(0,len(indices)-1,12000).astype(int)]
    points=search[indices]
    pairs=cKDTree(points).query_pairs(max(3*config.voxel_m,.012),output_type='ndarray')
    adjacency=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
    count,labels=connected_components(adjacency,directed=False)
    tree=cKDTree(body);candidates=[]
    for label in range(count):
        selected=points[labels==label]
        if len(selected)<config.min_points:continue
        attachment,_=tree.query(selected)
        if attachment.min()>config.attachment_distance_m:continue
        bounds=np.quantile(selected,[.01,.99],axis=0)
        candidates.append(dict(points=selected,position_world=bounds.mean(0),bounds_world=bounds,
            point_count=len(selected),protrusion_height_m=float(np.median(height[indices[labels==label]])),
            nearest_parent_surface_m=float(attachment.min()),
            geometric_parent_association_only=True,physical_attachment_proven=False,
            semantic_identity_confirmed=False,graspable=None,graspability_evaluated=False,
            source='surface_relative_measured_depth',visible_outward_structure_only=True))
    candidates.sort(key=lambda c:c['point_count'],reverse=True)
    return candidates[:config.max_candidates],dict(patches=len(patches),
        patch_sources=sorted({patch['source'] for patch in patches}),
        candidate_count=len(candidates),reference='measured_parent_surface_not_camera_depth_order')
