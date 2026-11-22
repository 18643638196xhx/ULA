"""Experimental selector for frozen CAD-verified candidates; no detector calls.

Evidence-only functions below never receive evaluation labels or seed/task IDs.
Projection completeness is a prior, not proof of target identity.
"""
import numpy as np
from scipy.spatial import ConvexHull


def frustum_cost(geometric_cost,visibility,weight=2.):
    """Soft prior only; never turn a CAD-rejected candidate into an accepted one."""
    if not np.isfinite(geometric_cost) or geometric_cost<0:raise ValueError('Invalid geometric cost')
    fraction=float(visibility['fraction'])
    if not np.isfinite(fraction) or not 0<=fraction<=1 or not np.isfinite(weight) or weight<0:
        raise ValueError('Invalid frustum coverage or weight')
    return float(geometric_cost+weight*(1-fraction))


def clip_polygon(polygon,axis,bound,greater):
    if not len(polygon):return polygon
    output=[];last=polygon[-1];last_in=last[axis]>=bound if greater else last[axis]<=bound
    for point in polygon:
        inside=point[axis]>=bound if greater else point[axis]<=bound
        if inside!=last_in:
            t=(bound-last[axis])/(point[axis]-last[axis])
            output.append(last+t*(point-last))
        if inside:output.append(point)
        last,last_in=point,inside
    return np.asarray(output).reshape(-1,2)


def area(poly):
    if len(poly)<3:return 0.
    return abs(float(np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))))/2


def projection_visibility(vertices,pose,intrinsic,world_to_camera,shape):
    transform=np.asarray(world_to_camera)@np.asarray(pose)
    camera=np.asarray(vertices)@transform[:3,:3].T+transform[:3,3]
    # A model crossing the camera plane cannot form a finite reliable hull.
    if not np.isfinite(camera).all() or np.any(camera[:,2]<=1e-4):
        return dict(fraction=0.,state='camera_plane_crossing',is_identity_evidence=False)
    projected=camera@np.asarray(intrinsic).T;uv=projected[:,:2]/projected[:,2:]
    try:poly=uv[ConvexHull(uv).vertices]
    except Exception:
        return dict(fraction=0.,state='degenerate_projection',is_identity_evidence=False)
    total=area(poly);clipped=poly
    for axis,bound,greater in ((0,0,True),(1,0,True),(0,shape[1]-1,False),(1,shape[0]-1,False)):
        clipped=clip_polygon(clipped,axis,bound,greater)
    return dict(fraction=float(np.clip(area(clipped)/max(total,1e-9),0,1)),
        state='measured_camera_frustum_prior',projected_hull_area_px2=total,
        is_identity_evidence=False,convex_hull_not_exact_silhouette=True)


def same_observed_region(a,b):
    x,y=a['mask'],b['mask'];intersection=int((x&y).sum())
    if intersection/max(1,int((x|y).sum()))>.5:return True
    if intersection/max(1,min(int(x.sum()),int(y.sum())))<.8:return False
    # Containment alone can merge separate touching objects. Require the two
    # full projected CAD surfaces to agree too, not just nearby centres.
    ra,rb=a.get('render'),b.get('render')
    if ra is None or rb is None:return False
    ma=(ra>0)&np.isfinite(ra);mb=(rb>0)&np.isfinite(rb);overlap=ma&mb
    return bool(overlap.sum()/max(1,(ma|mb).sum())>.5 and overlap.any() and
                np.median(np.abs(ra[overlap]-rb[overlap]))<=.008)


def rank_candidates(candidates,depth):
    """Keep maximal consistent measured support per observed-region group.

    Fixed cost = archived geometry cost + 2*(1-frustum hull coverage).
    No class/seed branch, confidence threshold tuning or truth-guided choices.
    """
    groups=[]
    for candidate in sorted(candidates,key=lambda c:c['original_cost']):
        # Complete-link clustering avoids a broad bridging mask merging two
        # otherwise disjoint regions through transitive overlap.
        for group in groups:
            if all(same_observed_region(candidate,other) for other in group):
                group.append(candidate);break
        else:groups.append([candidate])
    ranked=[]
    for group in groups:
        def support(c):
            r=c.get('render')
            if r is None:return 0
            return int(np.sum(c['mask']&(depth>0)&(r>0)&(np.abs(depth-r)<=.015)))
        rep=max(group,key=lambda c:(support(c),-c['original_cost']))
        cost=frustum_cost(rep['original_cost'],rep['frustum'])
        ranked.append(dict(representative=rep,selection_cost=cost,
            candidate_ids=[c['id'] for c in group],measured_support_pixels=support(rep),
            pose_ambiguity_not_resolved=True))
    return sorted(ranked,key=lambda g:g['selection_cost'])
