"""Passive simulation evaluator. Its outputs must never enter action selection."""
from collections import deque
import numpy as np


def lowest_surface_z(matrix, half_size=(.03, .03, .1)):
    matrix = np.asarray(matrix)
    return float(matrix[2, 3]-np.abs(matrix[2, :3]) @ np.asarray(half_size))


class PickupAudit:
    def __init__(self, initial_matrix, clearance_m=.03, stable_steps=5):
        self.initial_lowest_z = lowest_surface_z(initial_matrix)
        self.clearance_m, self.stable_steps = clearance_m, stable_steps
        self.window = deque(maxlen=stable_steps)
        self.samples = []
        self.ever_sustained = False

    def observe(self, matrix, left_hand_matrix, left_command, step):
        matrix, hand = np.asarray(matrix), np.asarray(left_hand_matrix)
        clearance = lowest_surface_z(matrix)-self.initial_lowest_z
        relative = np.linalg.solve(hand, matrix)[:3, 3]
        near = float(np.linalg.norm(relative)) <= .20
        candidate = bool(clearance >= self.clearance_m and 0 <= left_command < .05 and near)
        if not candidate:
            self.window.clear()
        else:
            self.window.append(relative.copy())
        drift = (float(np.max(np.linalg.norm(np.asarray(self.window)-self.window[0], axis=1)))
                 if self.window else None)
        sustained = bool(len(self.window) == self.stable_steps and drift <= .02)
        self.ever_sustained |= sustained
        self.samples.append(dict(step=int(step), clearance_m=float(clearance),
            left_command=float(left_command), block_xyz=matrix[:3, 3].tolist(),
            left_ee_to_block_distance_m=float(np.linalg.norm(relative)),
            relative_translation_drift_m=drift, sustained=sustained))

    def report(self):
        final = self.samples[-1] if self.samples else {}
        # Both sustained pickup and a still-raised, closed left hand at the end.
        final_candidate = bool(final.get("clearance_m", -1) >= self.clearance_m
            and 0 <= final.get("left_command", 1) < .05
            and final.get("left_ee_to_block_distance_m", 1) <= .20)
        return dict(source="passive_simulation_evaluator_not_returned_to_controller",
            feeds_back=False, initial_lowest_surface_z_m=self.initial_lowest_z,
            clearance_threshold_m=self.clearance_m, stable_action_samples=self.stable_steps,
            relative_translation_tolerance_m=.02, maximum_hand_distance_m=.20,
            ever_sustained_pickup=self.ever_sustained,
            final_raised_in_closed_left_hand=final_candidate,
            pickup_success=bool(self.ever_sustained and final_candidate),
            maximum_clearance_m=max((s["clearance_m"] for s in self.samples), default=0.),
            final_clearance_m=final.get("clearance_m"), samples=self.samples)
