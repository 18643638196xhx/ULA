"""3.11 policy client -> existing 3.10 vision environment, one model load.

Reuses the proven bounded JSONL process lifecycle, not task perception logic.
The client accepts RGBDFrame/TargetQuery and returns the same observation type.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np

from adapters.perception import PerceptionWorker
from .contracts import VisionConfig,TargetObservation,SceneObservation,serializable


class UnifiedVisionWorker(PerceptionWorker):
    def __init__(self,ula_root,output_dir,*,gpu=None,config=VisionConfig(),timeout_s=300):
        super().__init__(ula_root,output_dir,backend='pointcloud',timeout_s=timeout_s)
        if not self.output.is_relative_to(self.root):raise ValueError('Worker artifacts must remain under ULA')
        self.gpu,self.geometry_config=gpu,config

    def _launch_spec(self):
        command,cwd,environment=super()._launch_spec()
        config_path=self.output/'geometry_config.json'
        config_path.write_text(json.dumps(asdict(self.geometry_config),indent=2)+'\n')
        command=[command[0],str(self.root/'scripts/run_unified_vision.py'),'--serve',
                 '--geometry-config',str(config_path)]
        cache=self.root/'.runtime_cache'
        environment.update(HF_HOME=str(cache/'hf_home'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TMPDIR=str(cache/'tmp'),MPLCONFIGDIR=str(cache/'matplotlib'),
            CUDA_CACHE_PATH=str(cache/'cuda'),XDG_CACHE_HOME=str(cache/'xdg'),
            TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'))
        for name in ('tmp','matplotlib','cuda','xdg'):(cache/name).mkdir(parents=True,exist_ok=True)
        if self.gpu is not None:environment['CUDA_VISIBLE_DEVICES']=str(self.gpu)
        return command,cwd,environment

    def observe(self,frames,queries,*,episode_id,search_motion=None,inference_seed=None):
        frames=list(frames);queries=list(queries)
        if not frames:raise ValueError('A current RGB-D batch is required')
        # Preserve the individual sensor units. Convert to metres for this wire
        # request only, rather than guessing units from numeric magnitudes.
        with self._lock:
            started=time.perf_counter()
            try:
                self._start();self._stats['requests']+=1
                number=self._stats['requests']
                directory=self.output/f'request_{number:05d}'
                directory.mkdir(exist_ok=False)
                inputs=[]
                for index,frame in enumerate(frames):
                    path=directory/f'camera_{index:02d}.npz'
                    np.savez_compressed(path,rgb=frame.rgb,depth=frame.depth_m,intrinsic=frame.intrinsic,
                                        world_to_camera=frame.world_to_camera)
                    inputs.append(dict(path=str(path),camera=frame.camera,timestamp_s=frame.timestamp_s))
                if len({frame.capture_id for frame in frames})!=1:
                    raise ValueError('Mixed capture groups must not be rewritten into a single observation')
                self._send(dict(op='observe',id=number,episode_id=episode_id,capture_id=frames[0].capture_id,
                    inference_seed=inference_seed,
                    depth_unit='m',queries=[asdict(query) for query in queries],frames=inputs,
                    search_motion=serializable(search_motion),output=str(directory/'prediction')))
                reply=self._receive()
                if reply.get('id')!=number or reply.get('status') not in ('ok','error'):
                    raise RuntimeError('Unexpected unified vision worker reply')
            except BaseException:
                self.close();raise
            if reply['status']=='error':raise RuntimeError(reply['error'])
            raw=json.loads((directory/'prediction/prediction.json').read_text())
            targets={}
            for key,value in raw['targets'].items():
                # Prediction files are worker-owned sensor outputs, not an
                # alternate route to any simulator diagnostic data.
                points=np.load(directory/'prediction'/(key+'_surface.npz'))['points_world']
                targets[key]=TargetObservation(**value,points=points)
            self._stats['completed_requests']+=1
            raw['audit'].update(worker_pid=self.process.pid,worker_request=number,
                                worker_request_wall_s=time.perf_counter()-started)
            return SceneObservation(raw['episode_id'],raw['capture_id'],targets,raw['audit'])
