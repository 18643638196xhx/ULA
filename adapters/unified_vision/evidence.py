"""Generic RGB-D evidence, foreground proposals and conservative cleanup.

No object/task names, CAD, actor state or renderer labels. Explicit query hints
are optional; an arbitrary object is not assumed upright, colorful or on a table.
"""
import numpy as np
from scipy.ndimage import binary_closing,label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geometry import world_grid,voxelize


def color_masks(rgb,colors):
    values=np.asarray(rgb)[...,:3].astype(float)/255.
    r,g,b=np.moveaxis(values,-1,0);hi=values.max(2);lo=values.min(2);delta=hi-lo
    palette={
        'blue':(b-r>.04)&(b-g>-.015)&(delta>.06),
        'orange':(r>.28)&(r-g>.12)&(g-b>.035),
        'white':(hi>.40)&(delta<.30),
        'red':(r>.25)&(r-g>.12)&(r-b>.12),
        'yellow':(r>.35)&(g>.28)&(b<np.minimum(r,g)-.12),
        'green':(g>.22)&(g-r>.06)&(g-b>.04),
        'purple':(r-g>.07)&(b-g>.07)&(np.minimum(r,b)>.20),
        'black':hi<.20,
        'gray':(delta<.12)&(hi>=.20)&(hi<=.70),
    }
    return {name:palette[name] for name in colors}


def support_plane(grid,valid,up,config):
    """Measured dominant support layer along an EXPLICIT calibrated up axis."""
    if up is None:return None
    points=voxelize(grid[valid],max(config.voxel_m,.006))
    if len(points)<80:return None
    up=np.asarray(up,float);heights=points@up
    bins,counts=np.unique(np.floor(heights/.005).astype(np.int64),return_counts=True)
    centre=(bins[np.argmax(counts)]+.5)*.005
    selected=points[np.abs(heights-centre)<=.0075]
    if len(selected)<80:return None
    tangent=basis(up);spans=np.ptp(selected@tangent,axis=0)
    if np.min(spans)<.10:return None
    # Use observed height mode, not a known simulator table height.
    return dict(normal_world=up,offset_m=float(np.median(selected@up)),
        measured_points=len(selected),source='RGBD_dominant_layer_along_explicit_up',
        physical_support_proven=False)


def basis(normal):
    normal=np.asarray(normal,float)
    axis=np.eye(3)[np.argmin(np.abs(normal))]
    x=np.cross(normal,axis);x/=np.linalg.norm(x)
    return np.column_stack((x,np.cross(normal,x)))


def context(frame,query,config):
    grid,valid=world_grid(frame);hints=query.hints
    support=support_plane(grid,valid,None if hints is None else hints.up_world,config)
    return dict(grid=grid,valid=valid,support=support,
        colors={} if hints is None else color_masks(frame.rgb,hints.colors))


def foreground(context,query):
    valid=context['valid'].copy();hints=query.hints;support=context['support']
    if hints is not None and hints.support_relation is not None and support is not None:
        distance=context['grid']@support['normal_world']-support['offset_m']
        if hints.support_relation=='above':valid&=distance>.003
        else:valid&=np.abs(distance)<=.01
    return valid


def appearance(mask,context,query):
    hints=query.hints
    if hints is None or not hints.colors:return {},0.,True
    count=max(1,int(mask.sum()));fractions={k:float(np.sum(mask&v)/count) for k,v in context['colors'].items()}
    scores=[min(fractions[k]/(.12 if k in ('white','gray','black') else .08),1.) for k in hints.colors]
    supported=[int(np.sum(mask&context['colors'][k]))>=6 and fractions[k]>=(.08 if k=='white' else .015)
               for k in hints.colors]
    matched=all(supported) if hints.all_colors else any(supported)
    return fractions,float(np.mean(scores) if hints.all_colors else max(scores)),matched


def components_3d(mask,grid,frame):
    """Pixel adjacency with measured 3-D continuity, not object identity."""
    indices=np.full(mask.shape,-1,int);indices[mask]=np.arange(mask.sum())
    edges=[]
    for a,b,pa,pb,da,db in (
        (indices[:-1],indices[1:],grid[:-1],grid[1:],frame.depth_m[:-1],frame.depth_m[1:]),
        (indices[:,:-1],indices[:,1:],grid[:,:-1],grid[:,1:],frame.depth_m[:,:-1],frame.depth_m[:,1:])):
        keep=(a>=0)&(b>=0)
        distances=np.linalg.norm(pa-pb,axis=2)
        # At lower image resolution neighbouring pixels may span several mm.
        scale=4*np.maximum(da,db)/min(frame.intrinsic[0,0],frame.intrinsic[1,1])
        keep&=distances<=np.maximum(.012,scale)
        edges.append(np.c_[a[keep],b[keep]])
    pairs=np.concatenate(edges);n=int(mask.sum())
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n))
    count,labels=connected_components(graph,directed=False)
    sizes=np.bincount(labels,minlength=count)
    return labels,sizes


def clean(mask,frame,query,config,ctx=None):
    ctx=context(frame,query,config) if ctx is None else ctx
    mask=np.asarray(mask,bool)&ctx['valid'];raw=int(mask.sum());audit={}
    hints=query.hints
    if hints is not None and hints.support_relation is not None:
        if ctx['support'] is None:raise RuntimeError('explicit_support_relation_not_observed')
        keep=foreground(ctx,query)
        fraction=float(np.sum(mask&keep)/max(raw,1))
        if hints.support_relation=='on' and fraction<.70:
            raise RuntimeError('candidate_not_on_measured_support')
        mask&=keep;audit['support_fraction']=fraction
    if mask.sum()<config.min_points:raise RuntimeError('insufficient_measured_foreground')
    labels,sizes=components_3d(mask,ctx['grid'],frame)
    # Remove only small disconnected islands. Several major visible components
    # are retained and flagged, not silently assumed to be separate objects.
    accepted=sizes>=max(config.min_points,int(.20*sizes.max()))
    selected=accepted[labels];cleaned=np.zeros_like(mask);cleaned[mask]=selected
    if cleaned.sum()>=config.min_points:mask=cleaned
    fractions,color_score,color_ok=appearance(mask,ctx,query)
    if hints is not None and hints.appearance_required and not color_ok:
        raise RuntimeError('explicit_appearance_not_supported')
    interior=np.zeros_like(mask)
    interior[1:-1,1:-1]=(mask[1:-1,1:-1]&mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:])
    erode=interior.sum()>=max(config.min_points,.6*mask.sum())
    points=ctx['grid'][interior if erode else mask]
    bounds=np.quantile(points,[.01,.99],axis=0);shape_score=0.
    spans=bounds[1]-bounds[0]
    if query.min_extent_m is not None and np.any(spans<query.min_extent_m):
        raise RuntimeError('visible_extent_below_explicit_prior')
    if query.max_extent_m is not None and np.any(spans>query.max_extent_m):
        raise RuntimeError('visible_extent_above_explicit_prior')
    if hints is not None and hints.vertical_aspect_range is not None:
        up=np.asarray(hints.up_world);tangent=basis(up)
        height=np.diff(np.quantile(points@up,[.01,.99]))[0]
        width=np.max(np.diff(np.quantile(points@tangent,[.01,.99],axis=0),axis=0))
        aspect=float(height/max(width,.001));low,high=hints.vertical_aspect_range
        if not low<=aspect<=high:raise RuntimeError('explicit_vertical_shape_not_supported')
        shape_score=1/(1+abs(aspect-np.sqrt(low*high)))
        audit['vertical_aspect']=aspect
    if hints is not None and hints.min_planar_aspect is not None:
        if ctx['support'] is None:raise RuntimeError('planar_compactness_requires_measured_support')
        projected=points@basis(ctx['support']['normal_world'])
        spans=np.diff(np.quantile(projected,[.01,.99],axis=0),axis=0)[0]
        aspect=float(spans.min()/max(spans.max(),.001))
        if aspect<hints.min_planar_aspect:raise RuntimeError('explicit_planar_shape_not_supported')
        shape_score=aspect;audit['planar_aspect']=aspect
    major=int(np.sum(sizes>=max(config.min_points,int(.20*sizes.max()))))
    audit.update(raw_valid_pixels=raw,selected_pixels=int(mask.sum()),geometry_points=len(points),
        mask_eroded_for_geometry=bool(erode),disconnected_major_components=major,
        color_fractions=fractions,appearance_supported=color_ok,appearance_score=color_score,
        shape_score=shape_score,support=ctx['support'],
        removed_pixels=raw-int(mask.sum()),full_object_identity_proven=False)
    return mask,points,audit


def region_proposals(ctx,query,semantic,limit):
    """Supplement semantic ROIs using declared appearance + measured support.

    Color alone cannot name an object: retain semantic overlap evidence and
    require it before returning a candidate.
    """
    if not ctx['colors']:return []
    mask=np.logical_or.reduce(list(ctx['colors'].values()))&foreground(ctx,query)
    groups,count=label(binary_closing(mask,iterations=1)&foreground(ctx,query))
    values=[]
    for i in range(1,count+1):
        selected=(groups==i)&mask
        if selected.sum()<40:continue
        _,color_score,color_ok=appearance(selected,ctx,query)
        if query.hints.appearance_required and not color_ok:continue
        y,x=np.nonzero(selected)
        evidence=max((score*float(np.mean((x>=box[0])&(x<=box[2])&(y>=box[1])&(y<=box[3])))
                      for score,box,_ in semantic),default=0.)
        if evidence<.10:continue
        values.append((float(evidence),np.array([x.min()-2,y.min()-2,x.max()+2,y.max()+2],float),query.prompts[0]))
    return sorted(values,key=lambda v:v[0],reverse=True)[:limit]


def seed_point(box,ctx,query):
    y,x=np.indices(ctx['valid'].shape)
    selected=foreground(ctx,query)&(x>=box[0])&(x<=box[2])&(y>=box[1])&(y<=box[3])
    if ctx['colors']:
        colored=selected&np.logical_or.reduce(list(ctx['colors'].values()))
        if colored.sum()>=20:selected=colored
    ys,xs=np.nonzero(selected)
    if not len(ys):return (np.asarray(box[:2])+box[2:])/2
    xy=np.c_[xs,ys];return xy[np.argmin(np.linalg.norm(xy-np.median(xy,axis=0),axis=1))]


def depth_region_proposals(ctx,query,frame,config,*,diameter_m=None,roi=None):
    """Unlabelled RGB-D regions, eligible ONLY for explicit CAD verification.

    These are search seeds, not identified objects. No simulator labels, table
    height, class-specific dimensions or mandatory upright assumption.
    """
    if query.cad is None:return []
    valid=foreground(ctx,query).copy();h,w=valid.shape
    if roi is not None:
        y,x=np.indices(valid.shape)
        valid&=(x>=roi[0])&(x<=roi[2])&(y>=roi[1])&(y<=roi[3])
    n=int(valid.sum())
    if n<config.min_points:return []
    indices=np.full(valid.shape,-1,int);indices[valid]=np.arange(n)
    rgb=np.asarray(frame.rgb,float)/255.;grid=ctx['grid'];depth=frame.depth_m
    edges=[]
    for a,b,pa,pb,ca,cb,da,db in (
        (indices[:-1],indices[1:],grid[:-1],grid[1:],rgb[:-1],rgb[1:],depth[:-1],depth[1:]),
        (indices[:,:-1],indices[:,1:],grid[:,:-1],grid[:,1:],rgb[:,:-1],rgb[:,1:],depth[:,:-1],depth[:,1:])):
        scale=4*np.maximum(da,db)/min(frame.intrinsic[0,0],frame.intrinsic[1,1])
        keep=(a>=0)&(b>=0)&(np.linalg.norm(pa-pb,axis=2)<=np.maximum(.012,scale))
        keep&=np.max(np.abs(ca-cb),axis=2)<=.18
        edges.append(np.c_[a[keep],b[keep]])
    pairs=np.concatenate(edges)
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n))
    count,labels=connected_components(graph,directed=False)
    sizes=np.bincount(labels,minlength=count);values=[]
    yy,xx=np.nonzero(valid)
    for i in np.argsort(-sizes):
        if sizes[i]<config.min_points:break
        # A huge background component is not an object proposal.
        if sizes[i]>.70*h*w:continue
        selected=labels==i;y,x=yy[selected],xx[selected]
        points=grid[y,x];span=np.ptp(points,axis=0)
        if diameter_m is not None and np.linalg.norm(span)>1.5*diameter_m+.015:continue
        # No semantic claim is attached to this low-score hypothesis.
        values.append((0.,np.array([x.min(),y.min(),x.max()+1,y.max()+1],float),
                       query.prompts[0] if query.prompts else 'unlabelled RGB-D region'))
        if len(values)>=config.max_depth_regions:break
    return values
