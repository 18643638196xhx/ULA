"""Standalone opt-in runtime worker for the frustum cost prior."""
import argparse,json,sys,time,os,traceback,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.unified_vision import VisionConfig,UnifiedVision
from adapters.frustum_selection_v1 import FrustumCadBackend
spec=importlib.util.spec_from_file_location('frustum_original_protocol',ROOT/'scripts/run_unified_vision.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def main():
    p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true',required=True)
    p.add_argument('--geometry-config',type=Path,required=True);a=p.parse_args()
    protocol=helpers.isolate_protocol_output();started=time.perf_counter()
    config=VisionConfig(**json.loads(a.geometry_config.read_text()))
    backend=FrustumCadBackend(ROOT/'tra_pipe/RoboTwin',config=config,project_root=ROOT,
        cad_output=a.geometry_config.parent/'cad_runtime')
    engine=UnifiedVision(backend,config)
    print(json.dumps(dict(status='ready',pid=os.getpid(),model_load_s=time.perf_counter()-started)),file=protocol,flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('op') in ('close','shutdown'):break
        try:
            helpers.execute(engine,request);reply=dict(id=request.get('id'),status='ok',output=request['output'])
        except Exception as error:
            traceback.print_exc(file=sys.stderr);reply=dict(id=request.get('id'),status='error',error=f'{type(error).__name__}: {error}')
        print(json.dumps(reply),file=protocol,flush=True)
    protocol.close()

if __name__=='__main__':main()
