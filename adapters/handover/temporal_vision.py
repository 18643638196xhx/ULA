"""History-conditioned held-block estimation from RGB-D and robot FK only.

A full visual pose after handover initializes the track. Robot motion predicts
the next pose; surface residuals update only their observable directions. Hidden
coordinates retain propagated uncertainty, rather than become new measurements.
The covariance/process model is an explicit engineering assumption, not a
physical proof of a rigid grasp or a calibrated safety guarantee.
"""
from pathlib import Path
import numpy as np

from adapters.handover.held_vision import HALF, cloud_from_frames, surface_distance
from adapters.handover.target_cloud import connected_clouds, fit_target_cloud


def skew(vector):
    x,y,z=vector
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def perturb(pose,delta):
    from scipy.spatial.transform import Rotation
    result=pose.copy()
    result[:3,:3]=Rotation.from_rotvec(delta[:3]).as_matrix()@pose[:3,:3]
    result[:3,3]+=delta[3:]
    return result


def surface_jacobian(points,pose):
    jac=np.empty((len(points),6))
    for index in range(6):
        step=1e-5 if index<3 else 1e-6
        delta=np.zeros(6);delta[index]=step
        jac[:,index]=(surface_distance(points,perturb(pose,delta))-
                      surface_distance(points,perturb(pose,-delta)))/(2*step)
    return jac


class TemporalHeldTrack:
    """Pose state with [world small-angle, world center] covariance."""
    def __init__(self):
        self.pose=None
        self.hand=None
        self.covariance=None
        self.frame=-1
        self.last_full_frame=-1
        self.distance_since_full=0.
        self.total_motion=0.
        self.sensor_sigma_m=.002
        self.position_floor_m=.0015
        self.angle_floor_rad=np.deg2rad(.3)
        self.full_anchor_seen=False
        self.initial_anchor_source='post_handover_full_RGBD'

    def predict(self,hand):
        from scipy.spatial.transform import Rotation
        hand=np.asarray(hand,dtype=float)
        if hand.shape!=(4,4) or not np.isfinite(hand).all():
            raise ValueError("Finite robot FK required for temporal tracking")
        if self.pose is None:
            raise RuntimeError("No reliable post-handover visual anchor")
        motion=hand@np.linalg.inv(self.hand)
        rotation=motion[:3,:3]
        transform=np.zeros((6,6));transform[:3,:3]=rotation;transform[3:,3:]=rotation
        travel=float(np.linalg.norm(hand[:3,3]-self.hand[:3,3]))
        angle=float(Rotation.from_matrix(rotation).magnitude())
        # Random-walk process uncertainty accounts for imperfect FK/attachment.
        # It cannot detect an unobserved slide along a perfectly uniform side.
        pnoise=.00015+.004*travel+.002*angle
        anoise=np.deg2rad(.04)+.02*angle
        process=np.diag([anoise**2]*3+[pnoise**2]*3)
        self.covariance=transform@self.covariance@transform.T+process
        self.pose=motion@self.pose
        self.hand=hand.copy()
        self.frame+=1
        self.distance_since_full+=travel
        self.total_motion+=travel
        return self.pose.copy()

    def initialize(self,hand,pose,quality):
        if not quality.get('accepted') or not quality.get('position_observable'):
            raise RuntimeError("A reliable full post-handover visual anchor is required")
        self.pose=np.asarray(pose,dtype=float).copy()
        self.hand=np.asarray(hand,dtype=float).copy()
        self.frame=0;self.last_full_frame=0;self.distance_since_full=0.
        self.full_anchor_seen=True
        self.initial_anchor_source='post_handover_full_RGBD'
        pstd=max(self.position_floor_m,float(quality['position_std_m']))
        astd=max(self.angle_floor_rad,np.deg2rad(float(quality['axis_uncertainty_deg'])))
        self.covariance=np.diag([astd**2]*3+[pstd**2]*3)
        audit=self.audit()
        return self.pose.copy(),audit

    def initialize_partial(self,hand,pose,quality):
        """Experimental predicted longitudinal state, NOT a full visual anchor.

        The fitted pose contains a bounded transfer-prediction coordinate for
        the hidden ends. Give it 15 mm longitudinal sigma. Release cannot use
        this state until a later independent full-position measurement arrives.
        """
        if not quality.get('accepted'):
            raise RuntimeError('Partial bootstrap still requires accepted current surfaces')
        pose,hand=np.asarray(pose,dtype=float),np.asarray(hand,dtype=float)
        if pose.shape!=(4,4) or hand.shape!=(4,4) or not np.isfinite(pose).all() or not np.isfinite(hand).all():
            raise ValueError('Finite partial pose and robot FK are required')
        self.pose=pose.copy();self.hand=hand.copy()
        self.frame=0;self.last_full_frame=-1;self.distance_since_full=0.
        self.full_anchor_seen=False
        self.initial_anchor_source='partial_RGBD_and_robot_transfer_prediction'
        axis=pose[:3,2]
        self.covariance=np.zeros((6,6))
        astd=max(self.angle_floor_rad,np.deg2rad(float(quality['axis_uncertainty_deg'])))
        if quality.get('axis_ambiguity_resolved_by_prediction'):
            astd=max(astd,np.deg2rad(15.))
        self.covariance[:3,:3]=np.eye(3)*astd**2
        self.covariance[3:,3:]=np.eye(3)*self.position_floor_m**2+(
            .015**2-self.position_floor_m**2)*np.outer(axis,axis)
        return self.pose.copy(),self.audit()

    def select(self,points):
        """Associate complete components to predicted surfaces, not truth IDs."""
        from scipy.spatial.distance import pdist
        uncertainty=np.sqrt(max(np.linalg.eigvalsh(self.covariance[3:,3:])))
        tolerance=min(.02,.006+3*uncertainty)
        selected=[];records=[]
        for index,component in enumerate(connected_clouds(points)):
            cloud=component['points']
            local=(cloud-self.pose[:3,3])@self.pose[:3,:3]
            inside=float(np.mean(np.all(np.abs(local)<=HALF+.035,axis=1)))
            sample=cloud[np.linspace(0,len(cloud)-1,min(512,len(cloud))).astype(int)]
            diameter=float(np.max(pdist(sample)))
            residual=np.abs(surface_distance(cloud,self.pose))
            fraction=float(np.mean(residual<tolerance))
            use=inside>=.5 and diameter<=np.linalg.norm(2*HALF)+.025 and fraction>=.85
            records.append(dict(component=index,points=len(cloud),selected=bool(use),
                predicted_surface_fraction=fraction,prediction_volume_fraction=inside,
                median_residual_m=float(np.median(residual)),diameter_m=diameter))
            if use:selected.append(cloud)
        if not selected:
            raise RuntimeError('No measured component consistent with anchored track: '+str(records))
        cloud=np.concatenate(selected)
        if len(cloud)<60:
            raise RuntimeError('Insufficient current surfaces; history alone cannot authorize motion')
        return cloud,dict(method='anchored_motion_and_surface_association',
            tolerance_m=tolerance,components=records,input_points=len(points),selected_points=len(cloud))

    def update(self,points):
        from scipy.optimize import least_squares
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<60 or not np.isfinite(points).all():
            raise RuntimeError('Insufficient finite current surfaces for temporal update')
        _,indices=np.unique(np.floor(points/.003).astype(np.int64),axis=0,return_index=True)
        cloud=points[np.sort(indices)]
        if len(cloud)<60:raise RuntimeError('Insufficient independent current surface points')
        if len(cloud)>1500:cloud=cloud[np.linspace(0,len(cloud)-1,1500).astype(int)]
        observed=cloud[np.linspace(0,len(cloud)-1,min(250,len(cloud))).astype(int)]
        prior=self.pose.copy()
        covariance=self.covariance.copy()
        precision=np.linalg.inv(covariance)
        whitening=np.linalg.cholesky(precision).T
        def residual(delta):
            errors=surface_distance(observed,perturb(prior,delta))/self.sensor_sigma_m
            # Robust surface loss, but a genuinely quadratic Gaussian prior.
            robust=np.sign(errors)*np.sqrt(2*(np.sqrt(1+errors**2)-1))
            return np.r_[robust,whitening@delta]
        fit=least_squares(residual,np.zeros(6),max_nfev=60,diff_step=1e-5)
        pose=perturb(prior,fit.x)
        errors=surface_distance(cloud,pose)
        rmse=float(np.sqrt(np.mean(errors**2)))
        overlap=float(np.mean(np.abs(errors)<.006))
        if not fit.success or rmse>=.004 or overlap<=.85:
            raise RuntimeError(f'Current surfaces contradict temporal prediction: rmse={rmse}, overlap={overlap}')
        if np.linalg.norm(fit.x[3:])>.02 or np.linalg.norm(fit.x[:3])>np.deg2rad(10):
            raise RuntimeError('Abrupt object/hand inconsistency; do not silently rebind the grasp')
        jac=surface_jacobian(observed,pose)
        r=surface_distance(observed,pose)/self.sensor_sigma_m
        weights=1/np.sqrt(1+r*r)
        information=jac.T@(weights[:,None]*jac)/self.sensor_sigma_m**2
        posterior=np.linalg.inv(precision+information)
        # Do not average away shared calibration/shape systematics over frames.
        floor=np.diag([self.angle_floor_rad**2]*3+[self.position_floor_m**2]*3)
        posterior+=np.diag(np.maximum(np.diag(floor)-np.diag(posterior),0.))
        self.pose=pose;self.covariance=(posterior+posterior.T)/2
        _,singular,vt=np.linalg.svd(jac,full_matrices=False)
        measurement_cov=vt.T@np.diag((self.sensor_sigma_m/np.maximum(singular,1e-8))**2)@vt
        independent_position_std=float(np.sqrt(max(np.linalg.eigvalsh(measurement_cov[3:,3:]))))
        position_observable=independent_position_std<.01
        if not self.full_anchor_seen:
            # Provisional hypotheses must acquire a larger measured surface
            # before they become full anchors; local covariance alone does
            # not resolve the discrete long-axis ambiguity of a small patch.
            position_observable=position_observable and np.ptp((cloud-pose[:3,3])@pose[:3,2])>.07
        if position_observable:
            self.last_full_frame=self.frame
            self.distance_since_full=0.
            self.full_anchor_seen=True
        audit=self.audit()
        audit.update(current_position_observable=bool(position_observable),
            current_position_std_m=independent_position_std,
            measurement_information_rank=int(np.sum(singular>.001)),
            correction_translation_m=float(np.linalg.norm(fit.x[3:])),
            correction_rotation_deg=float(np.degrees(np.linalg.norm(fit.x[:3]))),
            surface_rmse_m=rmse,inlier_fraction=overlap)
        quality=dict(source='history_conditioned_measured_RGBD_surface_update',accepted=True,
            uses_object_truth=False,uses_robot_identity_mask=False,uses_prepared_CAD=False,
            known_dimensions_m=(2*HALF).tolist(),position_observable=bool(position_observable),
            position_std_m=independent_position_std,axis_uncertainty_deg=audit['axis_95_deg'],
            rmse_m=rmse,inlier_fraction=overlap,measured_points=len(cloud),
            longitudinal_position_source='fused_visual_history_and_robot_motion',
            bottom_is_shape_inference_not_hidden_surface_measurement=True,temporal_tracking=audit)
        return pose.copy(),quality

    def audit(self):
        axis=self.pose[:3,2]
        jac=np.column_stack((.1*skew(axis),np.eye(3)))
        bottom_cov=jac@self.covariance@jac.T+np.eye(3)*.001**2
        xy95=2*float(np.sqrt(max(np.linalg.eigvalsh(bottom_cov[:2,:2]))))
        z95=2*float(np.sqrt(bottom_cov[2,2]))
        projector=np.eye(3)-np.outer(axis,axis)
        axis95=2*float(np.degrees(np.sqrt(max(0.,max(np.linalg.eigvalsh(
            projector@self.covariance[:3,:3]@projector))))))
        age=self.frame-self.last_full_frame
        usable=self.full_anchor_seen and age<=30 and self.distance_since_full<=.5 and max(xy95,z95)<=.005
        return dict(anchor_source=self.initial_anchor_source+'_then_surface_updates',
            full_visual_anchor_seen=self.full_anchor_seen,
            initial_table_pose_alone_can_authorize_release=False,uses_object_truth=False,
            frame=self.frame,last_full_frame=self.last_full_frame,partial_age_frames=age,
            distance_since_full_m=self.distance_since_full,total_robot_motion_m=self.total_motion,
            bottom_xy_95_m=xy95,bottom_z_95_m=z95,axis_95_deg=axis95,
            release_usable=bool(usable),current_visible_surfaces_required=True,
            covariance_6x6=self.covariance.tolist(),confidence_is_engineering_model_not_grip_proof=True,
            unobservable_longitudinal_slip_can_remain_undetected=True)


class TemporalHeldRGBDObserver:
    def __init__(self,task,directory,*,min_initial_visible_span_m=.07,allow_partial_initialization=False,
                 resolve_initial_axis_ambiguity_with_prior=False):
        self.task=task;self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.calls=0;self.track=TemporalHeldTrack()
        self.min_initial_visible_span_m=min_initial_visible_span_m
        self.allow_partial_initialization=allow_partial_initialization
        self.resolve_initial_axis_ambiguity_with_prior=resolve_initial_axis_ambiguity_with_prior

    def observe(self,prior):
        from traditional_place_empty_cup.geometry import pose_wxyz_to_matrix
        index=self.calls;self.calls+=1
        frames={}
        for name in ('head_camera','left_camera','right_camera'):
            value=self.task.get_camera_observation(name)
            frame=dict(rgb=value['rgb'],depth=value['depth'],intrinsic=value['intrinsic_cv'],
                       world_to_camera=value['extrinsic_cv'])
            np.savez_compressed(self.directory/f'observation_{index:03d}_{name}.npz',**frame)
            frames[name]=frame
        hand=pose_wxyz_to_matrix(self.task.get_arm_pose('right'))
        if self.track.pose is None:
            predicted=np.asarray(prior)
            points,views=cloud_from_frames(frames,predicted)
            pose,quality,selected=fit_target_cloud(points,predicted,
                min_visible_span_m=self.min_initial_visible_span_m,
                resolve_axis_ambiguity_with_prior=self.resolve_initial_axis_ambiguity_with_prior)
            if quality.get('position_observable') or not self.allow_partial_initialization:
                pose,history=self.track.initialize(hand,pose,quality)
            else:
                pose,history=self.track.initialize_partial(hand,pose,quality)
                quality['experimental_partial_bootstrap']=True
            quality['temporal_tracking']=history
            quality['axis_uncertainty_deg']=history['axis_95_deg']
        else:
            # Retry at unchanged FK is not another independent prediction step.
            if not np.allclose(hand,self.track.hand,rtol=0,atol=1e-9):
                predicted=self.track.predict(hand)
            else:predicted=self.track.pose.copy()
            points,views=cloud_from_frames(frames,predicted)
            selected,selection=self.track.select(points)
            pose,quality=self.track.update(selected)
            quality['target_selection']=selection
        quality.update(views=views,observation_index=index)
        np.savez_compressed(self.directory/f'observation_{index:03d}_fit.npz',points=selected,
            candidate_points=points,pose=pose,prior=predicted,robot_ee=hand,
            covariance=self.track.covariance)
        return pose,quality
