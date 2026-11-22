import unittest
from pathlib import Path

from perception_cad_search.pillbottle_cad_search import (
    build_object_query,
    build_support_query,
    build_vision_config,
    load_model_spec,
)


ROOT = Path(__file__).resolve().parents[1]


class PillbottleCadSearchTest(unittest.TestCase):
    def test_all_five_exact_models_exist(self):
        for model_id in range(1, 6):
            spec = load_model_spec(ROOT, "080_pillbottle", model_id)
            self.assertEqual(spec["model_id"], model_id)
            self.assertGreater(spec["diameter_m"], .10)
            self.assertTrue((ROOT / spec["mesh_path"]).is_file())

    def test_color_and_cup_aspect_are_not_hard_gates(self):
        query = build_object_query(
            load_model_spec(ROOT, "080_pillbottle", 2),
            "orange pill bottle",
        )
        self.assertIsNotNone(query.cad)
        self.assertEqual(query.position_reference, "cad_origin")
        self.assertFalse(query.hints.appearance_required)
        self.assertEqual(query.hints.colors, ())
        self.assertIsNone(query.hints.vertical_aspect_range)
        self.assertIn("orange pill bottle", query.prompts)
        self.assertIn("medicine bottle", query.prompts)

    def test_search_and_foundationpose_work_are_bounded(self):
        config = build_vision_config(5)
        self.assertTrue(config.cad_refine_on_demand)
        self.assertFalse(config.require_semantic_identity)
        self.assertLessEqual(config.max_candidates, 4)
        self.assertLessEqual(config.max_depth_regions, 4)
        self.assertLessEqual(config.max_refined_candidates, 10)
        self.assertEqual(config.cad.iterations, 5)
        self.assertEqual(config.cad.max_models, 5)

    def test_missing_model_zero_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_model_spec(ROOT, "080_pillbottle", 0)

    def test_support_query_is_color_grounded_but_not_cad_dependent(self):
        query = build_support_query("blue square pad")
        self.assertIsNone(query.cad)
        self.assertTrue(query.hints.appearance_required)
        self.assertEqual(query.hints.colors, ("blue",))
        self.assertIn("blue pad", query.prompts)


if __name__ == "__main__":
    unittest.main()
