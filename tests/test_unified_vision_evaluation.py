import unittest
import numpy as np

from adapters.unified_vision import RGBDFrame
from diagnostics.unified_vision_evaluation import evaluate_target,summarize_cohort,visible_reference


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((40,50),bool);self.mask[10:30,10:30]=True
        self.frame=RGBDFrame('head','initial',np.zeros((40,50,3),np.uint8),np.ones((40,50)),
            np.array([[100.,0,25],[0,100.,20],[0,0,1.]]),np.eye(4),'m')
        centre,_=visible_reference(self.frame,self.mask)
        self.target=dict(required_available=True,fresh=True,status='ok',usable=dict(position=True),
                         geometry=dict(position_world=centre.tolist()))

    def test_no_independent_label_means_unknown_not_correct(self):
        value=evaluate_target(self.target,self.mask,None,self.frame)
        self.assertTrue(value['output'])
        self.assertIsNone(value['identity_match'])
        self.assertIsNone(value['localization_verified'])

    def test_fresh_correct_region_and_position_pass_declared_metric(self):
        value=evaluate_target(self.target,self.mask,self.mask,self.frame)
        self.assertTrue(value['identity_match'])
        self.assertTrue(value['strict_mask_match'])
        self.assertTrue(value['localization_verified'])
        self.assertLess(value['visible_surface_center_error_m'],1e-10)
        self.assertFalse(value['semantic_CAD_origin_or_hidden_bottom_error'])

    def test_partial_correct_region_is_not_wrong_identity_because_of_IoU(self):
        partial=self.mask.copy();partial[18:]=False
        value=evaluate_target(self.target,partial,self.mask,self.frame)
        self.assertTrue(value['identity_match'])
        self.assertFalse(value['strict_mask_match'])

    def test_wrong_region_cannot_pass_because_it_returns_a_position(self):
        wrong=np.zeros_like(self.mask);wrong[:8,:8]=True
        value=evaluate_target(self.target,wrong,self.mask,self.frame)
        self.assertFalse(value['identity_match'])
        self.assertFalse(value['localization_verified'])

    def test_position_error_is_not_cad_origin_or_hidden_support(self):
        target=dict(self.target,geometry=dict(position_world=[0.,0.,0.]))
        value=evaluate_target(target,self.mask,self.mask,self.frame)
        self.assertTrue(value['identity_match'])
        self.assertFalse(value['localization_verified'])
        self.assertGreater(value['visible_surface_center_error_m'],.9)

    def test_missing_output_still_counts_in_labelled_denominator(self):
        missing=evaluate_target({},None,self.mask,self.frame)
        good=evaluate_target(self.target,self.mask,self.mask,self.frame)
        rows=[dict(seed=0,sensor_available=True,inference_status='ok',unified=dict(target=good)),
              dict(seed=1,sensor_available=True,inference_status='error',unified=dict(target=missing)),
              dict(seed=2,sensor_available=False)]
        report=summarize_cohort(rows,['target'])
        self.assertEqual(report['sensor_frames'],2)
        self.assertEqual(report['targets']['target']['identity_match_rate'],.5)
        self.assertEqual(report['targets']['target']['output_rate'],.5)
        self.assertEqual(report['archive_missing_sensor_seeds'],[2])

    def test_unlabelled_cohort_has_no_invented_accuracy(self):
        value=evaluate_target(self.target,self.mask,None,self.frame)
        report=summarize_cohort([dict(seed=0,sensor_available=True,inference_status='ok',unified=dict(target=value))],['target'])
        self.assertIsNone(report['both_identity_matches'])
        self.assertIsNone(report['targets']['target']['identity_match_rate'])
        self.assertEqual(report['targets']['target']['output_rate'],1.)


if __name__=='__main__':unittest.main()
