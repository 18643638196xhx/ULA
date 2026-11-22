"""Opt-in runtime adapter, leaving the original backend and defaults intact."""
import numpy as np
from pathlib import Path
from adapters.unified_vision.backends import DinoSamBackend
from adapters.fixed_candidate_selection_v1 import projection_visibility,frustum_cost
from adapters.unified_vision.worker import UnifiedVisionWorker


class FrustumSelectionWorker(UnifiedVisionWorker):
    """Explicit opt-in; does not change UnifiedVisionWorker's default entry."""
    def _launch_spec(self):
        command,cwd,environment=super()._launch_spec()
        command[1]=str(self.root/'scripts/run_frustum_selection_v1.py')
        return command,cwd,environment


class FrustumCadBackend(DinoSamBackend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.frustum_vertices={}

    def verify_cad(self,record,specification):
        result=super().verify_cad(record,specification)
        if not result.get('accepted'):return result
        import trimesh
        key=repr(specification)
        if key not in self.frustum_vertices:
            path=Path(specification.mesh_path)
            if not path.is_absolute():path=self.project_root/path
            mesh=trimesh.load(str(path),force='scene')
            mesh=trimesh.util.concatenate(tuple(mesh.dump()))
            mesh.apply_scale(specification.scale)
            if specification.mesh_to_object is not None:mesh.apply_transform(specification.mesh_to_object)
            self.frustum_vertices[key]=np.array(mesh.vertices,copy=True)
        frame=record['frame']
        visibility=projection_visibility(self.frustum_vertices[key],result['object_to_world'],
            frame.intrinsic,frame.world_to_camera,frame.depth.shape)
        return dict(result,geometric_cost=result['cost'],
            cost=frustum_cost(result['cost'],visibility),frustum_prior=visibility,
            frustum_weight=2.,pose_hypotheses_reranked=False)
