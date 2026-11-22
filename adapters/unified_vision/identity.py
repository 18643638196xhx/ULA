"""Explicit semantic support, never an identity truth label or probability."""
import numpy as np


def cad_color_evidence(observed_rgb, rendered_rgb, selected, min_pixels=40):
    """Compare visible chromaticity distributions, not background or brightness.

    Inputs are RGB floats in [0,1]. Neutral/dark surfaces are deliberately
    inconclusive: color cannot resolve every geometry ambiguity.
    """
    selected=np.asarray(selected,bool)
    a=np.asarray(observed_rgb,float)[selected];b=np.asarray(rendered_rgb,float)[selected]
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3:
        raise ValueError('Expected matching RGB images and selected pixels')
    def histogram(values):
        hi=values.max(1);lo=values.min(1)
        keep=np.isfinite(values).all(1)&(hi>.12)&((hi-lo)/np.maximum(hi,1e-6)>.20)
        values=values[keep]
        if len(values)<min_pixels:return None,len(values)
        normalized=values/np.maximum(values.sum(1,keepdims=True),1e-6)
        h=np.histogramdd(normalized[:,:2],bins=12,range=((0,1),(0,1)))[0]
        # Neighbour bins tolerate modest illumination/white-balance changes.
        from scipy.ndimage import gaussian_filter
        h=gaussian_filter(h,.8);return h/h.sum(),len(values)
    ha,na=histogram(a);hb,nb=histogram(b)
    audit=dict(source='supplied_CAD_texture_vs_measured_mask_RGB',
        observed_chromatic_pixels=na,reference_chromatic_pixels=nb,
        state='unobservable',accepted=True,identity_proven=False)
    if ha is None or hb is None:return audit
    overlap=float(np.minimum(ha,hb).sum())
    audit.update(histogram_overlap=overlap,state='supported' if overlap>=.35 else 'contradicted',
                 accepted=overlap>=.35)
    return audit


def semantic_evidence(mask,detections):
    mask=np.asarray(mask,bool);y,x=np.nonzero(mask)
    best=None
    for score,box,prompt in detections:
        if not np.isfinite(score) or score<=0:continue
        box=np.asarray(box,float)
        fraction=float(np.mean((x>=box[0])&(x<=box[2])&(y>=box[1])&(y<=box[3]))) if len(x) else 0.
        value=dict(prompt=prompt,detector_score=float(score),mask_fraction_in_semantic_box=fraction,
                   supported=fraction>=.8,identity_proven=False,score_is_probability=False)
        area=max(1.,float(max(0,box[2]-box[0])*max(0,box[3]-box[1])))
        intersection=float(np.sum((x>=box[0])&(x<=box[2])&(y>=box[1])&(y<=box[3])))
        value.update(box_xyxy=box.tolist(),mask_box_iou=intersection/max(1.,len(x)+area-intersection))
        if best is None or score*fraction>best['detector_score']*best['mask_fraction_in_semantic_box']:best=value
    return best or dict(supported=False,identity_proven=False,reason='no_overlapping_semantic_detection')


def identity_supported(record):
    evidence=record['mask'].identity_evidence
    # Explicit appearance alone does not name an object. CAD geometry alone
    # cannot turn an unlabelled region into a semantically identified target.
    return bool(evidence.get('supported'))


def identity_ranking(record):
    """Declared semantic/appearance evidence; CAD never increases this score."""
    e=record['mask'].identity_evidence
    support=bool(e.get('supported'))
    semantic=float(np.clip(e.get('detector_score',0.),0,1)) if support else 0.
    compactness=float(np.sqrt(np.clip(e.get('mask_box_iou',0.),0,1)))
    quality=record.get('quality') or {}
    # Absence of a declared color prior is not evidence of matching appearance.
    colors=quality.get('color_fractions') or {}
    appearance=float(np.clip(quality.get('appearance_score',0.),0,1)) if colors else None
    score=semantic*compactness
    if appearance is not None:score*=appearance
    return dict(score=score,semantic_supported=support,semantic_score=semantic,
        box_compactness=compactness,declared_appearance_score=appearance,
        geometry_contributes=False,score_is_probability=False,identity_proven=False)


def selection_cost(record,config):
    base=float(record['cad_verification']['cost'])
    if not config.evidence_ranked_cad:return base
    return base+1.-float(record['identity_ranking']['score'])
