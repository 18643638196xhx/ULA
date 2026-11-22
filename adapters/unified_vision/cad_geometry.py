"""CPU-only checks of a registered CAD against observed RGB-D, not labels.

An arbitrary returned registration is not a match. Check observed surface,
projected coverage, visible rendered depth and impossible empty-space overlap.
"""
import numpy as np

from .contracts import rigid,CadLimits


class TriangleSurface:
    """Cached static CAD surface in metres, not a sampled point cloud.

    An unsigned distance to an actual triangle works for open meshes as well
    as closed solids. A point inside a solid is NOT automatically on its surface.
    Lazy imports keep model-free queries independent of CAD dependencies.
    """
    distance_method='exact_unsigned_point_to_triangle_surface'

    def __init__(self,vertices,faces):
        import trimesh
        import open3d as o3d
        vertices=np.asarray(vertices,float);faces=np.asarray(faces)
        if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or len(vertices)<3 or
            not np.isfinite(vertices).all() or faces.ndim!=2 or faces.shape[1:]!=(3,) or
            len(faces)==0 or not np.issubdtype(faces.dtype,np.integer) or
            np.any(faces<0) or np.any(faces>=len(vertices))):
            raise ValueError('Expected finite CAD vertices and valid integer triangle indices')
        triangles=vertices[faces]
        valid=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                     triangles[:,2]-triangles[:,0]),axis=1)>0
        if not valid.any():raise ValueError('CAD has no nondegenerate triangle surface')
        self.degenerate_faces_removed=int(np.sum(~valid))
        # Private copies: estimator centering or caller edits cannot mutate the
        # fixed metric surface. The BVH selects a triangle, then its closest
        # point is projected in the ORIGINAL float64 CAD coordinates.
        faces=faces[valid].copy();self._triangles=triangles[valid].copy()
        self.triangle_count=len(faces)
        low,high=vertices.min(0),vertices.max(0)
        self._origin=(low+high)/2.;self._scale=float(np.max(high-low))
        normalized=np.asarray((vertices-self._origin)/self._scale,np.float32)
        if not np.isfinite(normalized).all():raise ValueError('CAD coordinates cannot form a finite spatial index')
        self._tensor=o3d.core.Tensor
        self._scene=o3d.t.geometry.RaycastingScene(nthreads=1)
        self._scene.add_triangles(self._tensor(normalized),self._tensor(np.asarray(faces,np.uint32)))
        self._project=trimesh.triangles.closest_point
        self.distance_accelerator='CPU_triangle_BVH_float32_search_float64_projection'

    def distances(self,points):
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all():
            raise ValueError('Expected finite Nx3 points in the static CAD frame')
        result=np.empty(len(points),float)
        # Bound temporary queries; no measured-points x model-faces product.
        # Normalization keeps large asset-origin offsets out of the float32
        # BVH search; neither the metric coordinates nor thresholds are scaled.
        for start in range(0,len(points),4096):
            batch=points[start:start+4096]
            normalized=np.asarray((batch-self._origin)/self._scale,np.float32)
            if not np.isfinite(normalized).all():raise ValueError('Query cannot form finite BVH coordinates')
            closest=self._scene.compute_closest_points(self._tensor(normalized),nthreads=1)
            ids=closest['primitive_ids'].numpy().astype(np.int64)
            if np.any(ids<0) or np.any(ids>=self.triangle_count):
                raise RuntimeError('CAD spatial query returned no valid triangle; no sampled fallback')
            projected=self._project(self._triangles[ids],batch)
            result[start:start+len(batch)]=np.linalg.norm(batch-projected,axis=1)
        if not np.isfinite(result).all() or np.any(result<0):
            raise RuntimeError('Exact CAD surface query returned invalid distances')
        return result


def point_agreement(points,pose,model_surface,limits=CadLimits()):
    pose=rigid(pose);points=np.asarray(points,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<3 or
        not np.isfinite(points).all() or not isinstance(model_surface,TriangleSurface)):
        raise ValueError('Expected finite measured Nx3 points and a static triangle surface')
    # Inverse-transform current measured points, leaving the CAD and its index
    # cached in object coordinates. Rigid transforms preserve metric distance.
    local=(points-pose[:3,3])@pose[:3,:3]
    distance=model_surface.distances(local)
    median=float(np.median(distance));fraction=float(np.mean(distance<=limits.depth_tolerance_m))
    return dict(accepted=median<=limits.max_median_error_m and fraction>=limits.min_observed_inliers,
        median_point_to_model_m=median,observed_surface_inliers=fraction,
        distance_method=model_surface.distance_method,model_triangles=model_surface.triangle_count,
        distance_accelerator=model_surface.distance_accelerator,
        degenerate_faces_removed=model_surface.degenerate_faces_removed,sampled_model_points=False)


def render_agreement(observed_depth,observed_mask,rendered_depth,limits=CadLimits(),min_pixels=40):
    observed=np.asarray(observed_depth,float);mask=np.asarray(observed_mask,bool)
    rendered=np.asarray(rendered_depth,float)
    if observed.ndim!=2 or observed.shape!=mask.shape or rendered.shape!=observed.shape:
        raise ValueError('Expected matching measured/rendered depth and candidate mask')
    valid=np.isfinite(observed)&(observed>0)
    rendered_valid=np.isfinite(rendered)&(rendered>0)
    measured=mask&valid
    # Surfaces seen closer than the CAD may occlude it. A CAD cannot simply
    # disappear behind everything: require enough non-occluded visible support.
    occluded=rendered_valid&valid&(observed<rendered-limits.depth_tolerance_m)
    expected=rendered_valid&valid&~occluded
    overlap=expected&measured
    visible_fraction=float(expected.sum()/max(1,rendered_valid.sum()))
    precision=float(overlap.sum()/max(1,measured.sum()))
    coverage=float(overlap.sum()/max(1,expected.sum()))
    free=float(np.sum(expected&(observed>rendered+limits.depth_tolerance_m))/max(1,expected.sum()))
    residual=np.abs(observed[overlap]-rendered[overlap])
    median=float(np.median(residual)) if len(residual) else None
    inliers=float(np.mean(residual<=limits.depth_tolerance_m)) if len(residual) else 0.
    reasons=[]
    if overlap.sum()<min_pixels:reasons.append('insufficient_visible_CAD_overlap')
    if visible_fraction<limits.min_visible_fraction:reasons.append('CAD_mostly_occluded_or_unmeasured')
    if precision<limits.min_mask_precision:reasons.append('candidate_surface_not_explained_by_CAD')
    if coverage<limits.min_render_coverage:reasons.append('visible_CAD_not_covered_by_candidate')
    if free>limits.max_free_space_fraction:reasons.append('CAD_occupies_observed_empty_space')
    if median is None or median>limits.max_median_error_m or inliers<limits.min_observed_inliers:
        reasons.append('rendered_depth_disagrees_with_RGBD')
    cost=None if median is None else float(median/limits.max_median_error_m+
        (1-inliers)+(1-precision)+(1-coverage)+free)
    return dict(accepted=not reasons,reasons=reasons,overlap_pixels=int(overlap.sum()),
        rendered_pixels=int(rendered_valid.sum()),visible_render_fraction=visible_fraction,
        mask_precision=precision,render_coverage=coverage,free_space_fraction=free,
        median_rendered_depth_error_m=median,rendered_depth_inliers=inliers,cost=cost,
        hidden_CAD_surfaces_required=False,object_identity_proven=False)


def structure_depth_evidence(observed_depth,rendered_depth,limits=CadLimits(),min_pixels=40):
    """Local free-space veto: a good large surface cannot hide a wrong part.

    Four-by-four footprint tiles are spatial tests, not semantic CAD parts.
    Unknown depth and nearer occluders never count as positive support.
    """
    observed=np.asarray(observed_depth,float);rendered=np.asarray(rendered_depth,float)
    valid=np.isfinite(observed)&(observed>0)
    footprint=np.isfinite(rendered)&(rendered>0)
    visible=footprint&valid&(observed>=rendered-limits.depth_tolerance_m)
    free=visible&(observed>rendered+limits.depth_tolerance_m)
    ys,xs=np.nonzero(footprint);tiles=[]
    if len(xs):
        ye=np.linspace(ys.min(),ys.max()+1,5,dtype=int);xe=np.linspace(xs.min(),xs.max()+1,5,dtype=int)
        for i in range(4):
            for j in range(4):
                region=np.s_[ye[i]:ye[i+1],xe[j]:xe[j+1]]
                n=int(visible[region].sum());nf=int(free[region].sum())
                if n<min_pixels:continue
                tiles.append(dict(tile=[i,j],visible_pixels=n,free_pixels=nf,
                    free_fraction=nf/n,contradicted=nf/n>limits.max_free_space_fraction))
    bad=[t for t in tiles if t['contradicted']]
    return dict(accepted=not bad,state='contradicted' if bad else 'not_contradicted' if tiles else 'unobservable',
        tiles=tiles,occlusion_or_missing_depth_is_support=False,
        source='measured_scene_depth_at_4x4_rendered_footprint_tiles',semantic_parts_inferred=False)


def scene_depth_evidence(observed_depth,rendered_depth,limits=CadLimits(),min_pixels=40):
    """Independent depth: distinguish support, contradiction and no visibility.

    Occluder pixels cannot prove identity or depth disagreement behind them.
    Measured empty space in front of the CAD remains contradictory even when
    most of the model is hidden. No acceptance thresholds are relaxed.
    """
    observed=np.asarray(observed_depth,float);rendered=np.asarray(rendered_depth,float)
    if observed.shape!=rendered.shape or observed.ndim!=2:raise ValueError('Matching depth images required')
    model=np.isfinite(rendered)&(rendered>0)
    valid=np.isfinite(observed)&(observed>0)
    visible=model&valid&(observed>=rendered-limits.depth_tolerance_m)
    fit=render_agreement(observed,visible,rendered,limits,min_pixels)
    count=int(visible.sum())
    if count>=min_pixels and (fit['free_space_fraction']>limits.max_free_space_fraction or
        fit['median_rendered_depth_error_m'] is None or
        fit['median_rendered_depth_error_m']>limits.max_median_error_m or
        fit['rendered_depth_inliers']<limits.min_observed_inliers):
        state='contradicted'
    elif fit['accepted']:state='supported'
    else:state='unobservable'
    return dict(state=state,visible_pixels=count,render_agreement=fit,
                semantic_identity_proven=False)


def bridge_measured_components(frame,original,allowed,rendered,max_gap_m,limits):
    """Bounded paths over a continuous rendered surface; output measured only.

    A closer measured occluder blocks a path. CAD projection continuity is
    evidence, not proof of topological ownership or physical attachment.
    """
    import heapq
    from dataclasses import replace
    from .geometry import world_grid
    from .evidence import components_3d
    grid,_=world_grid(frame)
    labels,_=components_3d(allowed,grid,frame)
    seeds=np.unique(labels[original[allowed]])
    reached=np.zeros_like(original);reached[allowed]=np.isin(labels,seeds)
    initial=int(reached.sum())
    if max_gap_m<=0 or not reached.any():return reached,0
    rf=replace(frame,depth=rendered,depth_unit='m')
    rg,rv=world_grid(rf)
    # Do not bridge through a foreground object in front of the hypothesized
    # surface. Missing depth or a farther background is not synthesized.
    rv&=~((frame.depth_m>0)&(frame.depth_m<rendered-limits.max_median_error_m))
    h,w=allowed.shape;distance=np.full((h,w),np.inf)
    distance[reached&rv]=0
    heap=[(0.,int(y),int(x)) for y,x in zip(*np.nonzero(reached&rv))]
    heapq.heapify(heap)
    while heap:
        cost,y,x=heapq.heappop(heap)
        if cost>distance[y,x]:continue
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if not (0<=yy<h and 0<=xx<w and rv[yy,xx]):continue
            step=float(np.linalg.norm(rg[yy,xx]-rg[y,x]))
            continuity=max(.012,4*max(rendered[y,x],rendered[yy,xx])/min(frame.intrinsic[0,0],frame.intrinsic[1,1]))
            if step>continuity or cost+step>max_gap_m:continue
            next_cost=0. if allowed[yy,xx] else cost+step
            if next_cost<distance[yy,xx]:
                distance[yy,xx]=next_cost
                heapq.heappush(heap,(next_cost,yy,xx))
    reached|=allowed&np.isfinite(distance)
    return reached,int(reached.sum())-initial


def complete_measured_mask(observed_depth,mask,rendered_depth,limits=CadLimits(),min_pixels=40,*,frame=None,max_gap_m=0.):
    """Recover measured CAD-supported pixels; never synthesize hidden points."""
    depth=np.asarray(observed_depth,float);rendered=np.asarray(rendered_depth,float)
    original=np.asarray(mask,bool)
    if depth.shape!=original.shape or rendered.shape!=depth.shape:raise ValueError('Matching arrays required')
    added=(np.isfinite(depth)&(depth>0)&np.isfinite(rendered)&(rendered>0)&
           (np.abs(depth-rendered)<=limits.max_median_error_m))
    if frame is not None:
        # Grow only through measured 3-D-continuous neighbours connected to the
        # trusted body. No image dilation across missing depth or CAD-only fill.
        from .geometry import world_grid
        grid,valid=world_grid(frame)
        allowed=added&valid
        if allowed.any():
            added,bridged=bridge_measured_components(frame,original,allowed,rendered,max_gap_m,limits)
        else:bridged=0
    candidate=original|added
    fit=render_agreement(depth,candidate,rendered,limits,min_pixels)
    audit=dict(accepted=fit['accepted'],added_pixels=int((candidate&~original).sum()),
        source='measured_depth_consistent_with_verified_CAD_render',hidden_pixels_generated=False,
        render_agreement=fit)
    audit['body_connected_measured_growth']=frame is not None
    audit['bounded_gap_recovered_pixels']=bridged if frame is not None else 0
    audit['max_gap_m']=max_gap_m if frame is not None else 0.
    return (candidate if fit['accepted'] else original.copy()),audit
