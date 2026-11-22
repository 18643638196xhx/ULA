"""Separate JSONL worker. Does not patch the existing worker or task code."""
import argparse,json,sys,time,traceback,os,importlib.util
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.unified_vision import VisionConfig,UnifiedVision
from adapters.cad_prompted_sam_v1.backend import CadPromptedSamBackend
spec=importlib.util.spec_from_file_location('cad_prompt_original_protocol',ROOT/'scripts/run_unified_vision.py')
protocol_helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol_helpers)


def main():
    p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true',required=True)
    p.add_argument('--geometry-config',type=Path,required=True);p.add_argument('--baseline',action='store_true')
    a=p.parse_args();protocol=protocol_helpers.isolate_protocol_output();started=time.perf_counter()
    config=VisionConfig(**json.loads(a.geometry_config.read_text()))
    # Both arms have the SAME base proposal budget. Six guided masks fit in
    # the engine's extra slots; original masks are never truncated to add them.
    base_config=replace(config,max_refined_candidates=12)
    backend=CadPromptedSamBackend(ROOT/'tra_pipe/RoboTwin',config=base_config,project_root=ROOT,
        cad_output=a.geometry_config.parent/'cad_runtime',use_cad_prompts=not a.baseline)
    engine=UnifiedVision(backend,replace(config,max_refined_candidates=18))
    print(json.dumps(dict(status='ready',pid=os.getpid(),model_load_s=time.perf_counter()-started)),file=protocol,flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('op') in ('close','shutdown'):break
        try:
            scene=protocol_helpers.execute(engine,request)
            # New audit is sidecar-only, keeps existing SceneObservation API.
            out=Path(request['output'])
            audit=dict(backend=backend.summary(),last_prompt_audits=getattr(backend,'last_prompt_audits',[]))
            from adapters.unified_vision.contracts import serializable
            (out/'cad_prompt_audit.json').write_text(json.dumps(serializable(audit),indent=2)+'\n')
            reply=dict(id=request.get('id'),status='ok',output=str(out))
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply=dict(id=request.get('id'),status='error',error=f'{type(error).__name__}: {error}')
        print(json.dumps(reply),file=protocol,flush=True)
    protocol.close()

if __name__=='__main__':main()
