"""Caption-only cup ablation; no changes to pixel gates or geometry."""
from adapters.cup_initial_perception import InitialCupPerception, InitialCupWorker

GENERIC_CAPTION = "cup . mug ."
BLUE_CAPTION = "blue cup ."


class BlueInitialCupPerception(InitialCupPerception):
    def _run_dino(self, rgb, caption):
        return super()._run_dino(rgb, BLUE_CAPTION if caption == GENERIC_CAPTION else caption)

    def estimate(self, *args, **kwargs):
        result = super().estimate(*args, **kwargs)
        result.observation_audit.update(cup_semantic_caption=BLUE_CAPTION,
            generic_cup_caption_fallback=False, color_is_hard_gate=False,
            changed_component="cup_semantic_caption_only")
        return result


class BlueInitialCupWorker(InitialCupWorker):
    def _launch_spec(self):
        command, cwd, environment = super()._launch_spec()
        command[1] = str(self.root/"scripts/perceive_cup_initial_blue.py")
        return command, cwd, environment
