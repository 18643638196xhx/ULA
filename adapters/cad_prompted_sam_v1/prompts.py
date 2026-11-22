"""Sensor-supported point prompts, not a CAD silhouette used as ground truth."""
import numpy as np
from scipy.ndimage import binary_dilation,distance_transform_edt


def spaced_points(mask,limit):
    mask=np.asarray(mask,bool);remaining=mask.copy();points=[]
    spacing=max(2.,np.sqrt(mask.sum()/max(1,limit))/2)
    y,x=np.indices(mask.shape)
    for _ in range(limit):
        if not remaining.any():break
        score=distance_transform_edt(mask)*remaining
        yy,xx=np.unravel_index(np.argmax(score),score.shape)
        points.append([float(xx),float(yy)])
        remaining&=(x-xx)**2+(y-yy)**2>spacing**2
    return np.asarray(points,float).reshape(-1,2)


def cad_prompts(depth,rendered,all_footprints=None,*,tolerance_m=.008,min_pixels=40,max_points=5):
    depth=np.asarray(depth,float);rendered=np.asarray(rendered,float)
    if depth.ndim!=2 or depth.shape!=rendered.shape:raise ValueError('Matching scalar depth images required')
    valid=np.isfinite(depth)&(depth>0);footprint=np.isfinite(rendered)&(rendered>0)
    supported=valid&footprint&(np.abs(depth-rendered)<=tolerance_m)
    audit=dict(positive_support_pixels=int(supported.sum()),hidden_pixels_generated=False,
        source='coarse_CAD_projection_AND_current_measured_depth',identity_proven=False)
    if supported.sum()<min_pixels:return None,dict(audit,accepted=False,reason='insufficient_depth_supported_CAD_pixels')
    union=footprint if all_footprints is None else np.asarray(all_footprints,bool)|footprint
    if union.shape!=depth.shape:raise ValueError('Matching alternative footprint required')
    # Negative prompts lie outside ALL retained coarse hypotheses, never on a
    # surface merely hidden by a closer object inside the model silhouette.
    outside=binary_dilation(union,iterations=4)&~binary_dilation(union,iterations=1)&valid
    positive=spaced_points(supported,max_points);negative=spaced_points(outside,max_points)
    y,x=np.nonzero(footprint);pad=4
    box=np.array([max(0,x.min()-pad),max(0,y.min()-pad),
        min(depth.shape[1]-1,x.max()+pad),min(depth.shape[0]-1,y.max()+pad)],float)
    coordinates=np.concatenate([positive,negative]);labels=np.r_[np.ones(len(positive),np.int32),np.zeros(len(negative),np.int32)]
    return dict(box=box,point_coords=coordinates,point_labels=labels,multimask_output=True),dict(audit,
        accepted=True,positive_points=positive.tolist(),negative_points=negative.tolist(),box_xyxy=box.tolist())
