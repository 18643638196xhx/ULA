"""Measured surface history. Missing/current data are never interchangeable."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import voxelize


def surface_motion(previous,current,config):
    """Bounded local ICP with geometric observability and symmetry disclosure.

    No world-Z yaw starts or upright-object assumptions. This local solver is
    deliberately conservative and may reject larger motion or occlusion.
    """
    reference=voxelize(previous,config.voxel_m)
    observed=voxelize(current,config.voxel_m)
    if min(len(reference),len(observed))<config.min_points:
        return None,dict(accepted=False,reason='insufficient_matching_surfaces')
    reference=reference[np.linspace(0,len(reference)-1,min(1800,len(reference))).astype(int)]
    observed=observed[np.linspace(0,len(observed)-1,min(1800,len(observed))).astype(int)]
    tree=cKDTree(observed)
    rotation=np.eye(3);translation=np.median(observed,axis=0)-np.median(reference,axis=0)
    keep=None
    for iteration in range(25):
        moved=reference@rotation.T+translation
        distance,nearest=tree.query(moved)
        keep=distance<=min(.04,float(np.quantile(distance,.8))+1e-12)
        if keep.sum()<config.min_points:
            return None,dict(accepted=False,reason='too_few_motion_correspondences')
        source,target=reference[keep],observed[nearest[keep]]
        a,b=source.mean(0),target.mean(0)
        u,singular,vt=np.linalg.svd((source-a).T@(target-b))
        if singular[1]<1e-10:return None,dict(accepted=False,reason='degenerate_motion_correspondences')
        new_rotation=vt.T@u.T
        if np.linalg.det(new_rotation)<0:
            vt[-1]*=-1;new_rotation=vt.T@u.T
        new_translation=b-new_rotation@a
        change=np.linalg.norm(new_translation-translation)+Rotation.from_matrix(new_rotation@rotation.T).magnitude()
        rotation,translation=new_rotation,new_translation
        if change<1e-6:break
    moved=reference@rotation.T+translation
    distance,nearest=tree.query(moved)
    reverse=cKDTree(moved).query(observed)[0]
    keep=distance<=.012
    rmse=float(np.sqrt(np.mean(distance[keep]**2))) if keep.any() else float('inf')
    # Point-to-plane information exposes flat-plane and symmetric-surface
    # ambiguity which a low point-to-point ICP residual alone would conceal.
    _,neighbours=tree.query(observed,k=min(15,len(observed)))
    normals=[]
    for indices in neighbours:
        patch=observed[indices];_,_,vt=np.linalg.svd(patch-patch.mean(0),full_matrices=False)
        normals.append(vt[-1])
    normals=np.asarray(normals)[nearest[keep]]
    centred=moved[keep]-moved[keep].mean(0) if keep.any() else np.empty((0,3))
    jac=np.column_stack((np.cross(centred,normals),normals))
    if len(jac):
        scale=np.linalg.norm(jac,axis=0)
        information=np.linalg.svd(jac/np.maximum(scale,1e-9),compute_uv=False)
        rank=int(np.sum(information>max(information[0]*.02,1e-8)))
    else:rank=0
    accepted=bool(keep.mean()>=.55 and np.mean(reverse<=.012)>=.55 and rmse<=.008 and rank==6)
    audit=dict(accepted=accepted,method='local_trimmed_3D_ICP_with_surface_information_rank',
        iterations=iteration+1,reference_overlap=float(keep.mean()),
        observation_overlap=float(np.mean(reverse<=.012)),inlier_rmse_m=rmse if np.isfinite(rmse) else None,
        observable_rank=rank,source='consecutive_measured_visible_surfaces',
        semantic_identity_proven=False,symmetry_globally_resolved=False,
        calibrated_error_bound=False,physical_rigid_attachment_proven=False)
    if not accepted:
        audit['reason']='overlap_residual_or_motion_observability_failed'
        return None,audit
    motion=np.eye(4);motion[:3,:3]=rotation;motion[:3,3]=translation
    return motion,audit
