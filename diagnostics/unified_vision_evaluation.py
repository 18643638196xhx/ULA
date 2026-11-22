"""Post-prediction offline metrics. NEVER imported by the vision process.

Target masks are independent renderer labels from the ORIGINAL saved frame.
No current simulator state or old predicted geometry is treated as truth.
"""
import numpy as np

from adapters.cup_initial_audit import mask_metrics
from adapters.unified_vision.geometry import world_grid,voxelize


def visible_reference(frame,truth_mask,*,voxel_m=.003,min_points=40):
    grid,valid=world_grid(frame)
    truth_mask=np.asarray(truth_mask,bool)
    if truth_mask.shape!=valid.shape:raise ValueError('Evaluation mask and sensor frame differ')
    points=voxelize(grid[valid&truth_mask],voxel_m)
    if len(points)<min_points:return None,len(points)
    return np.quantile(points,[.01,.99],axis=0).mean(0),len(points)


def evaluate_target(target,mask,truth_mask,frame,*,position_tolerance_m=.02,voxel_m=.003):
    """One predeclared identity rule for all cohorts; strict IoU also reported.

    Precision>=0.8 and >=20 intersecting pixels tolerates conservative erosion
    without calling a small correct partial surface a different object. Neither
    metric proves semantic part ownership, a complete shape or grasp success.
    """
    output=bool(target.get('required_available') and target.get('fresh'))
    shape=target.get('geometry',{})
    # This metric remains visible-centre error even when a caller explicitly
    # requests a CAD origin/anchor for control. Never silently score that origin
    # against a visible-surface reference, or rename the old metric as CAD pose.
    position=(shape.get('visible_surface_position_world') if shape.get('position_is_model_inferred')
              else shape.get('position_world'))
    position_available=bool(target.get('fresh') and target.get('usable',{}).get('position') and
        position is not None and np.asarray(position).shape==(3,) and np.isfinite(position).all())
    result=dict(output=output,position_available=position_available,status=target.get('status'),
        independent_label_available=truth_mask is not None,mask_available=mask is not None,
        identity_match=None,strict_mask_match=None,localization_verified=None,
        visible_surface_center_error_m=None,reference_kind='same_frame_target_visible_surface_bounds_centre',
        semantic_CAD_origin_or_hidden_bottom_error=False)
    if truth_mask is None:return result
    truth_mask=np.asarray(truth_mask,bool)
    predicted=np.zeros_like(truth_mask) if mask is None else np.asarray(mask,bool)
    if predicted.shape!=truth_mask.shape:raise ValueError('Prediction/evaluation mask shape differs')
    metrics=mask_metrics(predicted,truth_mask)
    strict=metrics.pop('identity_match')
    identity=bool(metrics['target_visible'] and metrics['intersection_pixels']>=20 and
                  metrics['target_precision']>=.8)
    reference,count=visible_reference(frame,truth_mask,voxel_m=voxel_m)
    result.update(mask_metrics=metrics,identity_match=identity,strict_mask_match=strict,
        reference_position_world=None if reference is None else reference.tolist(),
        reference_points=count,localization_verified=False)
    if reference is not None and position_available:
        error=float(np.linalg.norm(np.asarray(position)-reference))
        result['visible_surface_center_error_m']=error
        result['localization_verified']=bool(output and identity and error<=position_tolerance_m)
    return result


def summarize_target(rows,key,method='unified'):
    metrics=[row.get(method,{}).get(key) for row in rows if row.get('sensor_available')]
    available=[value for value in metrics if value is not None]
    labelled=[value for value in available if value['independent_label_available']]
    errors=[value['visible_surface_center_error_m'] for value in labelled
            if value['identity_match'] and value['position_available'] and
            value['visible_surface_center_error_m'] is not None]
    denominator=len(metrics)
    def rate(value,n):return value/n if n else None
    output=sum(value['output'] for value in available)
    identities=sum(bool(value['identity_match']) for value in labelled)
    strict=sum(bool(value['strict_mask_match']) for value in labelled)
    localized=sum(bool(value['localization_verified']) for value in labelled)
    return dict(sensor_frames=denominator,records_available=len(available),
        output_count=output,output_rate=rate(output,denominator),
        independent_label_count=len(labelled),identity_matches=identities if labelled else None,
        identity_match_rate=rate(identities,len(labelled)),
        strict_IoU_half_matches=strict if labelled else None,strict_IoU_half_rate=rate(strict,len(labelled)),
        localization_verified=localized if labelled else None,localization_rate=rate(localized,len(labelled)),
        selected_wrong_identity_seeds=[row['seed'] for row in rows
            if (row.get(method,{}).get(key) or {}).get('mask_available') and
            (row.get(method,{}).get(key) or {}).get('identity_match') is False],
        no_output_seeds=[row['seed'] for row in rows if row.get('sensor_available') and
            not (row.get(method,{}).get(key) or {}).get('output')],
        target_not_visible_seeds=[row['seed'] for row in rows if
            (row.get(method,{}).get(key) or {}).get('mask_metrics',{}).get('target_visible') is False],
        position_error_condition='identity_matched_current_position_with_valid_visible_depth_reference',
        position_error_samples=len(errors),mean_visible_surface_error_m=float(np.mean(errors)) if errors else None,
        median_visible_surface_error_m=float(np.median(errors)) if errors else None,
        p90_visible_surface_error_m=float(np.quantile(errors,.9)) if errors else None)


def summarize_cohort(rows,keys):
    images=[row for row in rows if row['sensor_available']]
    complete=[row for row in images if row.get('inference_status')=='ok']
    labelled=[row for row in images if all(
        (row.get('unified',{}).get(key) or {}).get('independent_label_available') for key in keys)]
    return dict(archived_seed_directories=len(rows),sensor_frames=len(images),
        archive_missing_sensor_seeds=[row['seed'] for row in rows if not row['sensor_available']],
        completed_inference=len(complete),infrastructure_error_seeds=[row['seed'] for row in images
            if row.get('inference_status')!='ok'],
        both_required_outputs=sum(all((row.get('unified',{}).get(key) or {}).get('output') for key in keys)
                                  for row in images),
        both_independent_labels=len(labelled),
        both_identity_matches=sum(all(row['unified'][key]['identity_match'] for key in keys)
                                  for row in labelled) if labelled else None,
        both_localization_verified=sum(all(row['unified'][key]['localization_verified'] for key in keys)
                                      for row in labelled) if labelled else None,
        targets={key:summarize_target(rows,key) for key in keys},
        baseline_targets={key:summarize_target(rows,key,'baseline') for key in keys}
            if any(row.get('baseline') for row in rows) else None)
