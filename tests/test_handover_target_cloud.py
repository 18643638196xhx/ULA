from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.handover.target_cloud import connected_clouds,fit_target_cloud


def surfaces(pose, partial=False):
    rng=np.random.default_rng(712)
    a=np.linspace(-.03,.03,24)
    z=np.linspace(-.04 if partial else -.1,.04 if partial else .1,70)
    y,zz=np.meshgrid(a,z)
    sides=[np.c_[np.full(y.size,.03),y.ravel(),zz.ravel()],
           np.c_[y.ravel(),np.full(y.size,-.03),zz.ravel()]]
    if not partial:
        x,y=np.meshgrid(a,a)
        sides.append(np.c_[x.ravel(),y.ravel(),np.full(x.size,.1)])
    p=np.concatenate(sides)
    return p@pose[:3,:3].T+pose[:3,3]+rng.normal(0,.00005,p.shape)


class TargetCloudTests(unittest.TestCase):
    def setUp(self):
        self.pose=np.eye(4)
        self.pose[:3,3]=[.1,.1,1.]
    def test_separates_nearby_red_cap_from_block(self):
        body=surfaces(self.pose)
        rng=np.random.default_rng(3)
        cap=rng.uniform([.22,.11,.89],[.24,.13,.90],(1500,3))
        pose,audit,selected=fit_target_cloud(np.r_[body,cap],self.pose)
        self.assertLess(np.linalg.norm(pose[:3,3]-self.pose[:3,3]),.002)
        self.assertEqual(len(selected),len(body))
        self.assertEqual(audit['target_selection']['rejected_points'],len(cap))
        self.assertTrue(audit['position_observable'])
    def test_does_not_invent_hidden_end_when_only_sides_visible(self):
        _,audit,_=fit_target_cloud(surfaces(self.pose,True),self.pose)
        self.assertFalse(audit['position_observable'])
    def test_allows_tilt_and_imperfect_motion_prediction(self):
        tilted=self.pose.copy()
        tilted[:3,:3]=Rotation.from_euler('y',25,degrees=True).as_matrix()
        prior=self.pose.copy();prior[:3,3]+=[.015,-.01,.005]
        pose,audit,_=fit_target_cloud(surfaces(tilted),prior)
        self.assertLess(np.linalg.norm(pose[:3,3]-tilted[:3,3]),.003)
        self.assertTrue(audit['accepted'])
    def test_large_red_background_is_not_forced_into_block(self):
        x,y=np.meshgrid(np.linspace(-.1,.3,70),np.linspace(-.1,.3,70))
        points=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
        with self.assertRaisesRegex(RuntimeError,'No trustworthy target component'):
            fit_target_cloud(points,self.pose)
    def test_empty_and_nonfinite_input(self):
        self.assertEqual(connected_clouds(np.empty((0,3))),[])
        with self.assertRaises(ValueError):connected_clouds([[np.nan,0,0]])
    def test_small_red_patch_alone_is_not_a_whole_target(self):
        rng=np.random.default_rng(2)
        patch=rng.uniform([.09,.09,.99],[.11,.11,1.01],(1000,3))
        with self.assertRaisesRegex(RuntimeError,'No trustworthy target component'):
            fit_target_cloud(patch,self.pose)
    def test_occlusion_separated_fragments_are_tested_together(self):
        cloud=surfaces(self.pose)
        z=cloud[:,2]-self.pose[2,3]
        cloud=cloud[(z<-.025)|(z>.025)]
        _,audit,selected=fit_target_cloud(cloud,self.pose)
        self.assertTrue(audit['position_observable'])
        self.assertEqual(len(selected),len(cloud))
        self.assertGreaterEqual(len(audit['target_selection']['selected_components']),2)


if __name__=='__main__':unittest.main()
