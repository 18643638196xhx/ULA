"""Measured depth components for initial RGB-D target masks, no identity IDs."""
import numpy as np


def split_depth_masks(mask,world,*,minimum=60,voxel_m=.003,connection_m=.012):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    mask=np.asarray(mask,dtype=bool)
    flat=np.flatnonzero(mask)
    if len(flat)<minimum:return []
    points=np.asarray(world).reshape(-1,3)[flat]
    if not np.isfinite(points).all():raise ValueError('Depth mask contains nonfinite XYZ')
    _,first,inverse=np.unique(np.floor(points/voxel_m).astype(np.int64),axis=0,
                             return_index=True,return_inverse=True)
    pairs=cKDTree(points[first]).query_pairs(connection_m,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),
                    shape=(len(first),len(first))).tocsr()
    _,labels=connected_components(graph,directed=False)
    result=[]
    for label in np.unique(labels):
        selected=labels[inverse]==label
        if selected.sum()<minimum:continue
        value=np.zeros(mask.shape,dtype=bool)
        value.flat[flat[selected]]=True
        result.append(value)
    return sorted(result,key=lambda value:-value.sum())


def clean_seed_component(mask,world,seed_mask=None):
    from scipy.ndimage import binary_erosion
    interior=binary_erosion(mask,iterations=1)
    if interior.sum()>=60:mask=interior
    candidates=split_depth_masks(mask,world)
    if not candidates:raise RuntimeError('No coherent measured target depth component')
    if seed_mask is None:return candidates
    ranked=sorted(candidates,key=lambda value:np.sum(value&seed_mask),reverse=True)
    selected=ranked[0]
    if np.sum(selected&seed_mask)/max(seed_mask.sum(),1)<.5:
        raise RuntimeError('Segmented depth is inconsistent with the proposed target')
    return selected
