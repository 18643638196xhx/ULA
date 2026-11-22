"""Robot-state-only endpoint for the isolated VLA left pickup experiment."""
import numpy as np


class PickupProgress:
    """Hand progress is a stop condition, NOT evidence that the object followed."""
    def __init__(self, rise_m=.08, closed_steps=3, finish_steps=5):
        self.rise_m, self.closed_steps, self.finish_steps = rise_m, closed_steps, finish_steps
        self.anchor_z = None
        self.closed_count = self.finish_count = 0

    def update(self, left_command, hand_z):
        if not 0 <= left_command < .05:
            self.anchor_z = None
            self.closed_count = self.finish_count = 0
            return False
        if self.anchor_z is None:
            self.anchor_z = float(hand_z)
        self.closed_count += 1
        risen = hand_z-self.anchor_z >= self.rise_m
        self.finish_count = self.finish_count+1 if risen and self.closed_count >= self.closed_steps else 0
        return self.finish_count >= self.finish_steps


class PickupEndpoint(Exception):
    pass


def rollout_pickup(task, policy, result, *, rise_m=.08):
    progress = PickupProgress(rise_m=rise_m)
    original = task.take_action
    result.update(robot_endpoint_reached=False, commands=[], uses_object_truth=False,
                  traditional_trial_lift=False, robot_rise_stop_m=rise_m)

    def action(values, *args, **kwargs):
        values = np.asarray(values)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("Expected finite dual-arm 14-D action")
        # A model action chunk must not silently exceed the declared budget.
        if task.take_action_cnt >= task.step_lim:
            raise PickupEndpoint()
        original(values, *args, **kwargs)
        left = float(task.robot.get_left_gripper_val())
        right = float(task.robot.get_right_gripper_val())
        hand = list(task.get_arm_pose("left"))
        result["commands"].append(dict(step=int(task.take_action_cnt), left=left, right=right,
                                        left_ee_pose=hand))
        if progress.update(left, hand[2]):
            result["robot_endpoint_reached"] = True
            raise PickupEndpoint()

    task.take_action = action
    try:
        policy.reset()
        while task.take_action_cnt < task.step_lim:
            before = task.take_action_cnt
            policy.step(task, task.get_obs())
            print(f"[handover-pickup] VLA step={task.take_action_cnt}/{task.step_lim}", flush=True)
            if task.take_action_cnt == before:
                raise RuntimeError("VLA returned without executing an action")
    except PickupEndpoint:
        pass
    finally:
        task.take_action = original
        result.update(vla_steps=int(task.take_action_cnt),
                      stop_reason="robot_hand_rise_not_grasp_proof" if result["robot_endpoint_reached"]
                      else "fixed_action_budget")
