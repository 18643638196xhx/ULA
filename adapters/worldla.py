"""Reuse the user's WorldLA adapter while keeping runtime writes inside ULA."""

import importlib
import os
import sys
from pathlib import Path


class WorldLA:
    def __init__(self, ula_root, model_dir, asset_checkpoint, *, blend=0.25,
                 replan_mode="tail_blend", inference_seed=20260901, chunk_steps=10,
                 native_pi05_checkpoint=None, native_pi05_config=None):
        self.root = Path(ula_root)
        self.chunk_steps = int(chunk_steps)
        self.args = dict(
            model_dir=str(model_dir), asset_checkpoint=str(asset_checkpoint),
            native_pi05_checkpoint=(str(native_pi05_checkpoint)
                                    if native_pi05_checkpoint else ""),
            native_pi05_config=(str(native_pi05_config) if native_pi05_config
                                else "multi_task_4"),
            condition="wam_stage_dynamic", message_stage_mode="normal",
            inference_seed=inference_seed, pi0_step=chunk_steps,
            diagnostic_replan_mode=replan_mode,
            tail_blend_start=blend, tail_blend_end=blend,
            tail_blend_gripper_mode="switch", tail_blend_gripper_threshold=0.5,
            trajectory_diagnostic_path="", diagnostic_visual_feature=False,
        )
        self.module = self.model = None

    def load(self):
        if self.model is not None:
            return
        workspace = self.root / "external/wam_vla"
        pi05 = self.root / "external/pi05"
        for path in (workspace, pi05, pi05 / "src"):
            sys.path.insert(0, str(path))
        common = importlib.import_module("experiments.oracle_shared_kv_multi_task4.common")
        original = common.configure_pi_runtime

        def local_runtime(*_args, **_kwargs):
            cache = self.root / ".runtime_cache"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["OPENPI_RUNTIME_CACHE_DIR"] = str(cache)
            return cache

        # Both imported WorldLA modules call this helper during import. Override
        # only in this process; no linked source or checkpoint file is changed.
        common.configure_pi_runtime = local_runtime
        try:
            self.module = importlib.import_module(
                "experiments.stage_relation_router_multi_task4.closed_loop_policy"
            )
        finally:
            common.configure_pi_runtime = original
        self.model = self.module.get_model(self.args)

    def reset(self):
        self.module.reset_model(self.model)

    def step(self, task, observation):
        self.module.eval(task, self.model, observation)
