"""Measured RGB-D surfaces of the tracked red primitive, including tilted poses.

Uses declared 6x6x20 cm dimensions, NOT a CAD mesh, actor pose, contact labels
or robot identity mask. A previous visual/kinematic estimate is a search/symmetry
prior, not an observation. Unobservable pose fits are rejected explicitly.
"""
from pathlib import Path
import numpy as np

HALF=np.array([.03,.03,.10])


def surface_distance(points,pose):
    delta=np.abs((np.asarray(points)-pose[:3,3])@pose[:3,:3])-HALF
    return np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)


def planes(points):
    remaining=np.asarray(points)
    rng=np.random.default_rng(20261107)
    found=[]
    for _ in range(4):
        if len(remaining)<35:
            break
        best=None
        for _ in range(140):
            trio=remaining[rng.choice(len(remaining),3,replace=False)]
            normal=np.cross(trio[1]-trio[0],trio[2]-trio[0])
            length=np.linalg.norm(normal)
            if length<1e-8:
                continue
            normal/=length
            mask=np.abs((remaining-trio[0])@normal)<.0025
            if best is None or mask.sum()>best.sum():
                best=mask
        if best is None or best.sum()<35:
            break
        patch=remaining[best]
        center=patch.mean(axis=0)
        _,_,vt=np.linalg.svd(patch-center,full_matrices=False)
        normal=vt[-1]
        found.append((normal,center,len(patch)))
        remaining=remaining[~best]
    return found


def canonical_rotation(rotation,previous):
    from scipy.spatial.transform import Rotation
    # The unmarked primitive also has visually identical ends. Preserve the
    # nearest continuous frame; a 180-degree equivalent must not be mistaken
    # for an uncertain long-axis direction or a command to flip the wrist.
    candidates=[]
    for flip in (0.,np.pi):
        for k in range(4):
            candidate=rotation@Rotation.from_euler("x",flip).as_matrix()@Rotation.from_euler("z",k*np.pi/2).as_matrix()
            candidates.append((Rotation.from_matrix(candidate@previous.T).magnitude(),candidate))
    return min(candidates,key=lambda item:item[0])[1]


def fit_cuboid(points,prior,*,min_visible_span_m=.07,resolve_axis_ambiguity_with_prior=False):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    points,prior=np.asarray(points,dtype=float),np.asarray(prior,dtype=float)
    if not np.isfinite(min_visible_span_m) or not 0<min_visible_span_m<=2*HALF[2]:
        raise ValueError('A positive visible-span requirement within object length is required')
    if points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<60 or not np.isfinite(points).all():
        raise RuntimeError("Insufficient finite held-object depth")
    if prior.shape!=(4,4) or not np.isfinite(prior).all():
        raise ValueError("Invalid RGB-D tracking prior")
    _,indices=np.unique(np.floor(points/.002).astype(np.int64),axis=0,return_index=True)
    cloud=points[np.sort(indices)]
    if len(cloud)<60:
        raise RuntimeError("Insufficient distinct held-object surface points")
    if len(cloud)>2500:
        cloud=cloud[np.linspace(0,len(cloud)-1,2500).astype(int)]
    normals=planes(cloud)
    seeds=[prior[:3,:3]]
    _,vectors=np.linalg.eigh(np.cov(cloud.T))
    bases=[vectors]
    for i,(first,_,_) in enumerate(normals):
        for second,_,_ in normals[i+1:]:
            if abs(first@second)>.25:
                continue
            second=second-first*(first@second)
            second/=np.linalg.norm(second)
            bases.append(np.column_stack((first,second,np.cross(first,second))))
    for basis in bases:
        for z_index in range(3):
            axis=basis[:,z_index].copy()
            if axis@prior[:3,2]<0:
                axis=-axis
            x=basis[:,(z_index+1)%3].copy()
            x-=axis*(x@axis)
            x/=np.linalg.norm(x)
            rotation=canonical_rotation(np.column_stack((x,np.cross(axis,x),axis)),prior[:3,:3])
            if not any(Rotation.from_matrix(rotation@value.T).magnitude()<.05 for value in seeds):
                seeds.append(rotation)
    observed=cloud[np.linspace(0,len(cloud)-1,min(700,len(cloud))).astype(int)]
    candidates=[]
    for rotation in seeds:
        local=cloud@rotation
        low,high=np.quantile(local,[.002,.998],axis=0)
        previous=prior[:3,3]@rotation
        center=np.clip(previous,high-HALF,low+HALF)
        start=np.r_[np.zeros(3),rotation@center]
        def residual(parameters):
            pose=np.eye(4)
            pose[:3,:3]=Rotation.from_rotvec(parameters[:3]).as_matrix()@rotation
            pose[:3,3]=parameters[3:]
            # Very weak priors only break symmetries; quality is evaluated on
            # measured surfaces and on the geometry-only Jacobian below.
            return np.r_[surface_distance(observed,pose),
                .0002*parameters[:3],.002*(parameters[3:]-prior[:3,3])]
        result=least_squares(residual,start,loss="soft_l1",f_scale=.002,
            max_nfev=55,diff_step=1e-5)
        pose=np.eye(4)
        pose[:3,:3]=canonical_rotation(Rotation.from_rotvec(result.x[:3]).as_matrix()@rotation,prior[:3,:3])
        pose[:3,3]=result.x[3:]
        errors=surface_distance(cloud,pose)
        rmse=float(np.sqrt(np.mean(errors**2)))
        overlap=float(np.mean(np.abs(errors)<.006))
        jac=result.jac[:len(observed)]
        _,singular,vt=np.linalg.svd(jac,full_matrices=False)
        covariance=vt.T@np.diag((max(rmse,.001)/np.maximum(singular,1e-8))**2)@vt
        position_std=float(np.sqrt(max(np.linalg.eigvalsh(covariance[3:,3:]))))
        axis=pose[:3,2]
        projector=np.eye(3)-np.outer(axis,axis)
        axis_std=float(np.degrees(np.sqrt(max(0.,max(np.linalg.eigvalsh(projector@covariance[:3,:3]@projector))))))
        span=np.ptp((cloud-pose[:3,3])@pose[:3,:3],axis=0)
        credible=bool(rmse<.004 and overlap>.85 and span[2]>min_visible_span_m and axis_std<5.)
        candidates.append(dict(pose=pose,rmse_m=rmse,inlier_fraction=overlap,
            position_std_m=position_std,axis_std_deg=axis_std,
            covariance=covariance,
            visible_spans_m=span.tolist(),credible=credible))
    accepted=sorted([c for c in candidates if c["credible"]],key=lambda c:c["rmse_m"])
    audit=dict(source="measured_RGBD_red_primitive_surface_fit",uses_object_truth=False,
        uses_robot_identity_mask=False,uses_prepared_CAD=False,known_dimensions_m=(2*HALF).tolist(),
        measured_points=len(cloud),planes=len(normals),initializations=len(seeds),
        min_visible_span_m=float(min_visible_span_m),
        resolve_axis_ambiguity_with_prior=bool(resolve_axis_ambiguity_with_prior),
        attempts=[{k:v for k,v in c.items() if k not in ("pose","covariance")} for c in candidates],accepted=False)
    if not accepted:
        raise RuntimeError("Held-object surfaces do not constrain a credible pose: "+str(audit))
    best=accepted[0]
    near=[c for c in accepted if c['rmse_m']<=best['rmse_m']*1.25+.00005]
    ambiguous=any(np.degrees(np.arccos(np.clip(c['pose'][:3,2]@best['pose'][:3,2],-1,1)))>8.
                  for c in near)
    if ambiguous and not resolve_axis_ambiguity_with_prior:
        raise RuntimeError("Held-object axis is ambiguous across visible-surface fits")
    if ambiguous:
        # Explicit ablation: prefer the robot-transfer orientation, not an
        # arbitrary equally good visual solution. Do not call it observed.
        best=min(near,key=lambda c:Rotation.from_matrix(c['pose'][:3,:3]@prior[:3,:3].T).magnitude())
    position_observable=best["position_std_m"]<.01 and not ambiguous
    maximum_shift=0.
    for alternative in accepted[1:]:
        if alternative["rmse_m"]>best["rmse_m"]*1.25+.00005:
            continue
        shift=np.linalg.norm(alternative["pose"][:3,3]-best["pose"][:3,3])
        angle=np.degrees(np.arccos(np.clip(alternative["pose"][:3,2]@best["pose"][:3,2],-1,1)))
        maximum_shift=max(maximum_shift,float(shift))
        if angle>8. and not resolve_axis_ambiguity_with_prior:
            raise RuntimeError("Held-object axis is ambiguous across visible-surface fits")
        if shift>.015:
            position_observable=False
    pose=best["pose"].copy()
    if not position_observable:
        # A visible side plane constrains tilt/axis but can leave the hidden
        # ends unobservable. Keep a declared longitudinal prior, not a fake
        # measurement of the concealed bottom. Release is forbidden in this
        # state; normal transport/descent may acquire a more informative view.
        axis=pose[:3,2]
        low,high=np.quantile(cloud@axis,[.002,.998])
        inferred=float(np.clip(prior[:3,3]@axis,high-HALF[2],low+HALF[2]))
        pose[:3,3]+=axis*(inferred-pose[:3,3]@axis)
    audit.update({k:v for k,v in best.items() if k not in ("pose","covariance")})
    audit.update(accepted=True,axis_uncertainty_deg=max(.5,best["axis_std_deg"]),
        axis_ambiguity_resolved_by_prediction=bool(ambiguous),
        position_observable=bool(position_observable),maximum_alternative_center_shift_m=maximum_shift,
        longitudinal_position_source="measured_surfaces" if position_observable else "bounded_prior_hidden_ends_unobserved",
        bottom_is_shape_inference_not_hidden_surface_measurement=True)
    return pose,audit


def cloud_from_frames(frames,prior):
    from scipy.ndimage import binary_erosion
    from adapters.cup_target_focus import world_points_image
    from adapters.handover.initial_vision import color_masks
    clouds=[]
    audit={}
    for name,frame in frames.items():
        world,valid=world_points_image(frame["depth"],frame["intrinsic"],frame["world_to_camera"])
        red=color_masks(frame["rgb"])[0]
        mask=red&valid&(np.linalg.norm(world-prior[:3,3],axis=2)<.23)
        interior=binary_erosion(mask)
        if interior.sum()>=60:
            mask=interior
        cloud=world[mask]
        audit[name]=dict(red_depth_pixels=int(len(cloud)),mask_eroded=bool(np.array_equal(mask,interior)))
        if len(cloud)>=15:
            clouds.append(cloud)
    if not clouds:
        raise RuntimeError("No measured red target near the tracked handover object")
    return np.concatenate(clouds),audit


class HeldRGBDObserver:
    def __init__(self,task,directory):
        self.task,self.directory=task,Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.calls=0

    def observe(self,prior):
        index=self.calls
        self.calls+=1
        frames={}
        for name in ("head_camera","left_camera","right_camera"):
            value=self.task.get_camera_observation(name)
            frame=dict(rgb=value["rgb"],depth=value["depth"],intrinsic=value["intrinsic_cv"],
                       world_to_camera=value["extrinsic_cv"])
            np.savez_compressed(self.directory/f"observation_{index:03d}_{name}.npz",**frame)
            frames[name]=frame
        points,views=cloud_from_frames(frames,np.asarray(prior))
        from adapters.handover.target_cloud import fit_target_cloud
        pose,audit,selected=fit_target_cloud(points,prior)
        audit.update(views=views,observation_index=index)
        np.savez_compressed(self.directory/f"observation_{index:03d}_fit.npz",
                            points=selected,candidate_points=points,pose=pose,prior=prior)
        return pose,audit
