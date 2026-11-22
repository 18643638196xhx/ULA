from adapters.unified_vision.worker import UnifiedVisionWorker


class CadPromptedSamWorker(UnifiedVisionWorker):
    def __init__(self,*args,use_cad_prompts=True,**kwargs):
        super().__init__(*args,**kwargs);self.use_cad_prompts=use_cad_prompts

    def _launch_spec(self):
        command,cwd,environment=super()._launch_spec()
        command[1]=str(self.root/'scripts/run_cad_prompted_sam_v1.py')
        if not self.use_cad_prompts:command.append('--baseline')
        return command,cwd,environment
