"""Explicit sensor/query contracts. No simulator entities or task identifiers."""
from dataclasses import dataclass,field,asdict
from pathlib import Path
import numpy as np


def rigid(value):
    value=np.asarray(value,dtype=float)
    if value.shape==(3,4):value=np.vstack((value,[0,0,0,1]))
    if (value.shape!=(4,4) or not np.isfinite(value).all() or
        not np.allclose(value[3],[0,0,0,1]) or
        not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-5) or
        not np.isclose(np.linalg.det(value[:3,:3]),1.,atol=1e-5)):
        raise ValueError('Expected a finite rigid world-to-camera transform')
    return value


def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
    return value


@dataclass(frozen=True)
class RGBDFrame:
    camera: str
    capture_id: str
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    depth_unit: str
    timestamp_s: float|None=None

    def __post_init__(self):
        if not self.camera or not self.capture_id or self.depth_unit not in ('m','mm'):
            raise ValueError('Camera, capture identity and explicit m/mm depth unit required')
        rgb=np.asarray(self.rgb);depth=np.asarray(self.depth)
        intrinsic=np.asarray(self.intrinsic,dtype=float)
        if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or depth.shape!=rgb.shape[:2]:
            raise ValueError('Expected uint8 RGB HxWx3 and matching scalar depth')
        if (intrinsic.shape!=(3,3) or not np.isfinite(intrinsic).all() or
            intrinsic[0,0]<=0 or intrinsic[1,1]<=0 or abs(np.linalg.det(intrinsic))<1e-9 or
            not np.allclose(intrinsic[2],[0,0,1])):
            raise ValueError('Invalid OpenCV intrinsic matrix')
        if self.timestamp_s is not None and not np.isfinite(self.timestamp_s):
            raise ValueError('Invalid sensor timestamp')
        # Immutable snapshots; a caller cannot overwrite an old cached image
        # and accidentally make it look like a new synchronous observation.
        for key,value in dict(rgb=rgb,depth=depth,intrinsic=intrinsic,
                              world_to_camera=rigid(self.world_to_camera)).items():
            value=np.array(value,copy=True);value.setflags(write=False)
            object.__setattr__(self,key,value)

    @classmethod
    def from_npz(cls,path,*,camera,capture_id,depth_unit,timestamp_s=None):
        # Whitelist sensor arrays. Never read actor labels, object poses,
        # evaluation_only masks or functional points even if present in NPZ.
        with np.load(Path(path),allow_pickle=False) as raw:
            k='intrinsic' if 'intrinsic' in raw else 'intrinsic_cv'
            t='world_to_camera' if 'world_to_camera' in raw else 'extrinsic_cv'
            return cls(camera,capture_id,raw['rgb'][...,:3],raw['depth'],raw[k],raw[t],
                       depth_unit,timestamp_s)

    @property
    def depth_m(self):
        value=np.asarray(self.depth,float)*(1e-3 if self.depth_unit=='mm' else 1.)
        return np.where(np.isfinite(value)&(value>0),value,0.)


@dataclass(frozen=True)
class VisualHints:
    """Optional, declared evidence; never inferred from a task/class name.

    up_world is a caller-supplied calibrated direction, not an object pose.
    Colors may rank softly or be an explicitly required appearance prior.
    """
    colors: tuple=()
    all_colors: bool=False
    appearance_required: bool=False
    support_relation: str|None=None
    up_world: tuple|None=None
    vertical_aspect_range: tuple|None=None
    min_planar_aspect: float|None=None

    def __post_init__(self):
        palette={'red','orange','yellow','green','blue','purple','white','black','gray'}
        if (not isinstance(self.colors,(list,tuple)) or not set(self.colors)<=palette or
            self.support_relation not in (None,'above','on') or
            self.appearance_required and not self.colors):
            raise ValueError('Invalid explicit appearance/support hint')
        object.__setattr__(self,'colors',tuple(self.colors))
        if self.up_world is not None:
            up=np.asarray(self.up_world,float)
            if up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-9:
                raise ValueError('Expected a finite nonzero calibrated up direction')
            object.__setattr__(self,'up_world',tuple(up/np.linalg.norm(up)))
        if (self.support_relation is not None or self.vertical_aspect_range is not None) and self.up_world is None:
            raise ValueError('Support/upright hints require an explicit up_world direction')
        if self.vertical_aspect_range is not None:
            value=np.asarray(self.vertical_aspect_range,float)
            if value.shape!=(2,) or not np.isfinite(value).all() or value[0]<=0 or value[1]<value[0]:
                raise ValueError('Invalid explicit vertical aspect range')
            object.__setattr__(self,'vertical_aspect_range',tuple(value))
        if self.min_planar_aspect is not None and not 0<self.min_planar_aspect<=1:
            raise ValueError('Invalid planar compactness hint')


@dataclass(frozen=True)
class CadModel:
    """Caller-supplied static CAD in metres; never a live object pose.

    mesh_to_object only fixes the asset coordinate frame. Registration never
    changes scale, and no contact annotations or simulator metadata are read.
    """
    mesh_path: str
    scale: tuple=(1.,1.,1.)
    mesh_to_object: tuple|None=None

    def __post_init__(self):
        if not isinstance(self.mesh_path,str) or not self.mesh_path.strip():
            raise ValueError('An explicit CAD mesh path is required')
        scale=np.asarray(self.scale,float)
        if scale.shape!=(3,) or not np.isfinite(scale).all() or np.any(scale<=0):
            raise ValueError('CAD scale must be three fixed positive lengths')
        object.__setattr__(self,'scale',tuple(float(x) for x in scale))
        if self.mesh_to_object is not None:
            object.__setattr__(self,'mesh_to_object',tuple(tuple(float(x) for x in row)
                for row in rigid(self.mesh_to_object)))


@dataclass(frozen=True)
class CadLimits:
    iterations: int=5
    max_hypotheses: int=8
    max_models: int=4
    depth_tolerance_m: float=.015
    max_median_error_m: float=.008
    min_observed_inliers: float=.65
    min_mask_precision: float=.65
    min_render_coverage: float=.55
    min_visible_fraction: float=.35
    max_free_space_fraction: float=.20
    min_cost_gap: float=.15
    cross_view_translation_m: float=.03

    def __post_init__(self):
        positive=(self.depth_tolerance_m,self.max_median_error_m,self.min_cost_gap,
                  self.cross_view_translation_m)
        fractions=(self.min_observed_inliers,self.min_mask_precision,self.min_render_coverage,
                   self.min_visible_fraction,self.max_free_space_fraction)
        if (not np.isfinite(positive+fractions).all() or min(positive)<=0 or
            any(not 0<x<=1 for x in fractions) or
            any(not isinstance(x,int) or isinstance(x,bool) or x<1
                for x in (self.iterations,self.max_hypotheses,self.max_models))):
            raise ValueError('Invalid shared CAD verification limits')


@dataclass(frozen=True)
class TargetQuery:
    key: str
    prompts: tuple
    attributes: tuple=('position','bounds')
    required: tuple=('position',)
    parent: str|None=None
    search_protrusions: bool=False
    min_extent_m: tuple|None=None
    max_extent_m: tuple|None=None
    hints: VisualHints|dict|None=None
    cad: CadModel|dict|None=None
    position_reference: str='visible_surface'
    anchor_object_m: tuple|None=None

    def __post_init__(self):
        if isinstance(self.hints,dict):object.__setattr__(self,'hints',VisualHints(**self.hints))
        if self.hints is not None and not isinstance(self.hints,VisualHints):
            raise ValueError('Expected optional VisualHints')
        if isinstance(self.cad,dict):object.__setattr__(self,'cad',CadModel(**self.cad))
        if self.cad is not None and not isinstance(self.cad,CadModel):
            raise ValueError('Expected optional explicit CadModel')
        if self.position_reference not in ('visible_surface','cad_origin','cad_anchor'):
            raise ValueError('Unknown position reference')
        if self.position_reference!='visible_surface' and self.cad is None:
            raise ValueError('CAD position reference requires an explicit CAD')
        if self.anchor_object_m is not None:
            anchor=np.asarray(self.anchor_object_m,float)
            if anchor.shape!=(3,) or not np.isfinite(anchor).all():raise ValueError('Expected finite CAD-frame anchor')
            object.__setattr__(self,'anchor_object_m',tuple(float(x) for x in anchor))
        if (self.position_reference=='cad_anchor')!=(self.anchor_object_m is not None):
            raise ValueError('An explicit static anchor is required only for cad_anchor')
        allowed={'position','bounds','plane','axis','structures','motion'}
        if (not isinstance(self.key,str) or not self.key.replace('_','').replace('-','').isalnum() or
            not isinstance(self.prompts,(list,tuple)) or
            any(not isinstance(p,str) or not p.strip() for p in self.prompts) or
            not self.prompts and not (self.parent and self.search_protrusions) or
            not self.attributes or not set(self.attributes)<=allowed or
            not set(self.required)<=set(self.attributes) or self.parent==self.key or
            self.search_protrusions and not self.parent):
            raise ValueError('Invalid target query, attributes or parent relation')
        for value in (self.min_extent_m,self.max_extent_m):
            if value is not None and (np.asarray(value).shape!=(3,) or
                not np.isfinite(value).all() or np.any(np.asarray(value)<0)):
                raise ValueError('Extent priors must be finite positive world-axis lengths')
        if self.min_extent_m is not None and self.max_extent_m is not None:
            if np.any(np.asarray(self.min_extent_m)>self.max_extent_m):raise ValueError('Inverted extent prior')


@dataclass(frozen=True)
class VisionConfig:
    cad_refine_on_demand: bool=False
    evidence_ranked_cad: bool=False
    verify_scene_structure: bool=False
    audit_candidate_selection: bool=False
    body_guided_completion: bool=False
    completion_max_gap_m: float=.02
    verify_cad_color: bool=False
    require_cross_view_cad: bool=False
    require_semantic_identity: bool=False
    complete_visible_surface: bool=False
    recovery_detection_threshold: float=.15
    min_points: int=40
    voxel_m: float=.003
    plane_tolerance_m: float=.004
    plane_min_fraction: float=.55
    max_candidates: int=4
    recovery_max_boxes: int=8
    max_refined_candidates: int=12
    max_masks_per_box: int=3
    max_recovery_tiles: int=4
    max_depth_regions: int=4
    max_views: int=3
    max_sync_skew_s: float=.05
    ambiguity_score_gap: float=.10
    association_distance_m: float=.06
    max_history_age_s: float=2.
    max_missed_captures: int=2
    protrusion_min_m: float=.012
    protrusion_max_m: float=.12
    local_surface_radius_m: float=.045
    attachment_distance_m: float=.03
    cad: CadLimits|dict=field(default_factory=CadLimits)

    def __post_init__(self):
        if any(not isinstance(v,bool) for v in (self.cad_refine_on_demand,self.evidence_ranked_cad,
                                                self.verify_scene_structure,self.audit_candidate_selection)):
            raise ValueError('Expected explicit evidence ranking/scene verification flags')
        if not isinstance(self.body_guided_completion,bool):raise ValueError('Expected explicit body completion flag')
        if self.body_guided_completion and not self.complete_visible_surface:
            raise ValueError('Body completion requires visible surface completion')
        if not np.isfinite(self.completion_max_gap_m) or not 0<=self.completion_max_gap_m<=.05:
            raise ValueError('Completion gap must be finite and between 0 and 5 cm')
        if not isinstance(self.verify_cad_color,bool):raise ValueError('Expected explicit CAD color flag')
        if not isinstance(self.require_cross_view_cad,bool):raise ValueError('Expected explicit cross-view flag')
        if not isinstance(self.require_semantic_identity,bool) or not isinstance(self.complete_visible_surface,bool):
            raise ValueError('Expected explicit identity/completion flags')
        if not np.isfinite(self.recovery_detection_threshold) or not 0<self.recovery_detection_threshold<1:
            raise ValueError('Expected finite recovery detection threshold in (0,1)')
        if isinstance(self.cad,dict):object.__setattr__(self,'cad',CadLimits(**self.cad))
        if not isinstance(self.cad,CadLimits):raise ValueError('Expected shared CadLimits')
        budgets=(self.max_candidates,self.recovery_max_boxes,self.max_refined_candidates,
                 self.max_masks_per_box,self.max_recovery_tiles,self.max_depth_regions)
        if any(not isinstance(value,int) or isinstance(value,bool) or value<1 for value in budgets):
            raise ValueError('Candidate/recovery budgets must be positive integers')
        numbers=[self.voxel_m,self.plane_tolerance_m,self.max_sync_skew_s,
            self.ambiguity_score_gap,self.association_distance_m,self.max_history_age_s,
            self.protrusion_min_m,self.protrusion_max_m,self.local_surface_radius_m,
            self.attachment_distance_m]
        if (not np.isfinite(numbers).all() or min(numbers)<=0 or self.min_points<6 or
            self.max_candidates<1 or self.max_views<1 or self.max_missed_captures<0 or
            not 0<self.plane_min_fraction<=1 or self.protrusion_min_m>=self.protrusion_max_m):
            raise ValueError('Invalid shared visual geometry/retry budget')


@dataclass
class MaskCandidate:
    mask: np.ndarray
    box: np.ndarray
    score: float
    prompt: str
    source: str='semantic_box_SAM'
    identity_evidence: dict=field(default_factory=dict)


@dataclass
class TargetObservation:
    key: str
    status: str
    required_available: bool=False
    fresh: bool=False
    geometry: dict=field(default_factory=dict)
    usable: dict=field(default_factory=dict)
    evidence: list=field(default_factory=list)
    candidates: list=field(default_factory=list)
    reasons: list=field(default_factory=list)
    last_known: dict|None=None
    assignment: dict=field(default_factory=dict)
    points: np.ndarray=field(default_factory=lambda:np.empty((0,3)),repr=False)
    views: dict=field(default_factory=dict,repr=False)
    candidate_records: list=field(default_factory=list,repr=False)

    def to_dict(self):
        return serializable({key:value for key,value in self.__dict__.items()
                             if key not in ('points','views','candidate_records')})


@dataclass
class SceneObservation:
    episode_id: str
    capture_id: str
    targets: dict
    audit: dict

    def to_dict(self):
        return dict(episode_id=self.episode_id,capture_id=self.capture_id,
            targets={key:value.to_dict() for key,value in self.targets.items()},audit=serializable(self.audit))
