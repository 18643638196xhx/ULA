import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.task_rules import load_support_placement_rule, load_task_rule
from scripts.run_ula_task import _make_rule_driven_vla_rollout


ROOT = Path(__file__).resolve().parents[1]


class TaskRuleTests(unittest.TestCase):
    def test_place_empty_cup_rule_is_executable_and_validated(self):
        rule = load_support_placement_rule(ROOT / "rules/place_empty_cup.yaml")
        self.assertEqual(rule.task, "place_empty_cup")
        self.assertEqual(rule.candidate_budget, 6)
        self.assertFalse(rule.trial_lift)
        self.assertEqual(rule.desired_orientation, "initial_world")
        self.assertEqual(set(rule.release_requirements), {
            "support_contact_observed", "footprint_contained",
        })
        self.assertFalse(rule.simulator_truth_for_control)

    def test_all_registered_support_placement_rules_parse(self):
        from adapters.ula_task_registry import TASKS

        self.assertEqual(set(TASKS), {
            "place_empty_cup", "move_pillbottle_pad", "place_fan", "place_mouse_pad",
            "move_stapler_pad", "place_object_scale", "place_object_stand",
            "place_shoe", "place_container_plate", "put_object_cabinet",
        })
        for task, spec in TASKS.items():
            with self.subTest(task=task):
                rule = load_task_rule(ROOT / spec["rule"])
                self.assertEqual(rule.task, task)

    def test_unified_rollout_receives_declarative_handoff_settings(self):
        rule = load_support_placement_rule(ROOT / "rules/move_pillbottle_pad.yaml")
        captured = {}

        def original(*args, **kwargs):
            captured.update(kwargs)

        _make_rule_driven_vla_rollout(rule, original)(SimpleNamespace(), key="value")
        self.assertEqual(captured["stable_steps"], 3)
        self.assertEqual(captured["gripper_threshold"], .25)
        self.assertFalse(captured["require_visual_proximity"])
        self.assertEqual(captured["key"], "value")


if __name__ == "__main__":
    unittest.main()
