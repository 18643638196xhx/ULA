"""Native pi0.5 policy and fixed-budget pickup rollout; no WorldLA model."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np


class NativePi05Pickup:
    def __init__(self, checkpoint, *, chunk_steps=10, inference_seed=20261107):
        self.checkpoint=Path(checkpoint)
        self.chunk_steps=chunk_steps
        self.inference_seed=inference_seed
        self.policy=self.model=None
        # Compatibility with the existing read-only policy truth guard.
        self.module=SimpleNamespace()

    def load(self):
        if self.policy is not None:
            return
        from openpi.training import config
        from openpi.policies import policy_config
        from openpi.shared import normalize
        assets=sorted(p for p in (self.checkpoint/"assets").iterdir() if p.is_dir())
        if len(assets)!=1 or not (self.checkpoint/"params").is_dir():
            raise ValueError("Expected a complete native pi0.5 checkpoint and one normalization asset")
        self.policy=policy_config.create_trained_policy(config.get_config("multi_task_4"),self.checkpoint,
            norm_stats=normalize.load(assets[0]),sample_kwargs={"num_steps":10})
        self.model=self.policy._model

    def reset(self):
        import jax
        self.load()
        # Same inference noise seed per scene; independent of failed localizations.
        self.policy._rng=jax.random.key(self.inference_seed)

    def step(self,task,observation):
        self.load()
        sample=dict(state=np.asarray(observation["joint_action"]["vector"],np.float32),
            prompt=task.get_instruction(),
            images={name:np.transpose(observation["observation"][camera]["rgb"],(2,0,1))
                    for name,camera in (("cam_high","head_camera"),("cam_left_wrist","left_camera"),
                                        ("cam_right_wrist","right_camera"))})
        actions=np.asarray(self.policy.infer(sample)["actions"])
        if actions.shape!=(50,14) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid native pi0.5 action chunk: {actions.shape}")
        for action in actions[:self.chunk_steps]:
            if task.take_action_cnt>=task.step_lim:
                break
            task.take_action(action)


def rollout_fixed_budget(task,policy,result,*,rise_m=None):
    """No early exit based on hand height, object pose, grasp score or contacts."""
    original=task.take_action
    result.update(robot_endpoint_reached=False,commands=[],uses_object_truth=False,
        traditional_trial_lift=False,robot_rise_stop_m=None,stop_reason="fixed_action_budget",
        replan_mode="native_fresh_chunks_no_tail_blend",chunk_steps=policy.chunk_steps)
    def action(values,*args,**kwargs):
        values=np.asarray(values)
        if values.shape!=(14,) or not np.isfinite(values).all():
            raise ValueError("Expected finite dual-arm 14-D action")
        original(values,*args,**kwargs)
        result["commands"].append(dict(step=int(task.take_action_cnt),
            left=float(task.robot.get_left_gripper_val()),right=float(task.robot.get_right_gripper_val()),
            left_ee_pose=list(task.get_arm_pose("left"))))
    task.take_action=action
    try:
        policy.reset()
        while task.take_action_cnt<task.step_lim:
            before=task.take_action_cnt
            policy.step(task,task.get_obs())
            print(f"[native-pi05-pickup] step={task.take_action_cnt}/{task.step_lim}",flush=True)
            if task.take_action_cnt==before:
                raise RuntimeError("Native pi0.5 returned without executing an action")
    finally:
        task.take_action=original
        result["vla_steps"]=int(task.take_action_cnt)
