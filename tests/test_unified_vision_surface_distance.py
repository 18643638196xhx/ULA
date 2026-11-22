"""Exact CAD-surface distances must not depend on sampling or object size."""
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from adapters.unified_vision import CadLimits
from adapters.unified_vision.cad_geometry import TriangleSurface,point_agreement,render_agreement,scene_depth_evidence


def square(size=1.):
    return np.array([[0,0,0],[size,0,0],[size,size,0],[0,size,0.]]),np.array([[0,1,2],[0,2,3]])


class SurfaceDistanceTests(unittest.TestCase):
    def test_large_triangle_interior_has_zero_distance_not_distance_to_vertices(self):
        vertices,faces=square(10.)
        points=np.array([[5.,5.,0.],[4.,3.,0.],[3.,4.,0.]])
        result=point_agreement(points,np.eye(4),TriangleSurface(vertices,faces))
        self.assertTrue(result['accepted'])
        self.assertLess(result['median_point_to_model_m'],1e-12)
        self.assertEqual(result['observed_surface_inliers'],1.)
        self.assertFalse(result['sampled_model_points'])
        json.dumps(result,allow_nan=False)

    def test_small_and_large_surfaces_get_identical_metric_error(self):
        errors=[]
        for size in (.01,.1,1.,10.):
            vertices,faces=square(size);surface=TriangleSurface(vertices,faces)
            points=np.array([[.2,.2],[.3,.4],[.7,.6]])*size
            points=np.c_[points,np.full(3,.006)]
            result=point_agreement(points,np.eye(4),surface)
            self.assertTrue(result['accepted']);errors.append(result['median_point_to_model_m'])
        np.testing.assert_allclose(errors,.006,rtol=0,atol=1e-12)

    def test_tessellation_density_does_not_change_distance(self):
        vertices,faces=square();refined=trimesh.Trimesh(vertices,faces,process=False).subdivide().subdivide()
        points=np.array([[.17,.36,.003],[.5,.5,.009],[.99,.61,-.005]])
        a=TriangleSurface(vertices,faces).distances(points)
        b=TriangleSurface(refined.vertices,refined.faces).distances(points)
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_rigid_rotation_and_translation_do_not_change_distance(self):
        vertices,faces=square();surface=TriangleSurface(vertices,faces)
        points=np.array([[.2,.2,.006],[.3,.4,-.006],[.7,.6,.006]])
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.7,.2,-.4]).as_matrix()
        pose[:3,3]=[3.,-2.,1.]
        world=points@pose[:3,:3].T+pose[:3,3]
        result=point_agreement(world,pose,surface)
        self.assertTrue(result['accepted']);self.assertAlmostEqual(result['median_point_to_model_m'],.006)

    def test_large_asset_coordinate_offset_does_not_lose_surface_precision(self):
        vertices,faces=square();vertices+=np.array([1e6,-1e6,1e6])
        surface=TriangleSurface(vertices,faces)
        points=np.array([[.2,.2,.006],[.3,.4,.006],[.7,.6,.006]])+np.array([1e6,-1e6,1e6])
        np.testing.assert_allclose(surface.distances(points),.006,rtol=0,atol=1e-9)

    def test_distances_to_edges_and_vertices_not_infinite_plane(self):
        vertices,faces=square();surface=TriangleSurface(vertices,faces)
        points=np.array([[.5,-.03,.04],[-.03,-.04,0.],[1.03,1.04,.12]])
        np.testing.assert_allclose(surface.distances(points),[.05,.05,.13],atol=1e-12)

    def test_inside_solid_is_not_automatically_surface_distance_zero(self):
        mesh=trimesh.creation.box(extents=[1,1,1]);surface=TriangleSurface(mesh.vertices,mesh.faces)
        points=np.array([[0.,0.,0.],[.1,0,0],[0,.1,0]])
        result=point_agreement(points,np.eye(4),surface)
        self.assertFalse(result['accepted']);self.assertGreater(result['median_point_to_model_m'],.39)

    def test_original_distance_and_inlier_thresholds_still_apply(self):
        vertices,faces=square();surface=TriangleSurface(vertices,faces)
        for offset,accepted in ((.0079,True),(.0081,False)):
            points=np.array([[.2,.2,offset],[.3,.4,offset],[.7,.6,offset]])
            self.assertEqual(point_agreement(points,np.eye(4),surface)['accepted'],accepted)
        points=np.tile([.4,.4,0.],(10,1));points[6:,2]=.03
        result=point_agreement(points,np.eye(4),surface)
        self.assertFalse(result['accepted']);self.assertEqual(result['observed_surface_inliers'],.6)
        self.assertEqual(asdict(CadLimits())['max_median_error_m'],.008)

    def test_input_mesh_is_copied_and_queries_do_not_sample(self):
        vertices,faces=square();surface=TriangleSurface(vertices,faces)
        vertices[:]=100;faces[:]=0
        state=np.random.get_state()
        with patch.object(trimesh.Trimesh,'sample',side_effect=AssertionError('sampling is forbidden')):
            np.testing.assert_allclose(surface.distances([[.5,.5,.004]]),[.004])
        after=np.random.get_state()
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1])
        self.assertEqual(state[2:],after[2:])

    def test_degenerate_faces_are_ignored_but_invalid_mesh_fails_closed(self):
        vertices,faces=square();faces=np.r_[faces,[[0,0,0]]]
        surface=TriangleSurface(vertices,faces)
        self.assertEqual(surface.degenerate_faces_removed,1)
        self.assertEqual(surface.triangle_count,2)
        for bad in (np.array([[0,0,0]]),np.array([[0,1,99]]),np.array([[0.,1.,2.]])):
            with self.assertRaises(ValueError):TriangleSurface(vertices,bad)
        with self.assertRaises(ValueError):TriangleSurface(vertices*np.nan,faces)

    def test_point_cloud_without_triangles_cannot_silently_restore_old_metric(self):
        with self.assertRaises(ValueError):point_agreement(np.ones((3,3)),np.eye(4),np.ones((100,3)))

    def test_empty_and_invalid_queries(self):
        vertices,faces=square();surface=TriangleSurface(vertices,faces)
        self.assertEqual(surface.distances(np.empty((0,3))).shape,(0,))
        for points in (np.array([1,2,3]),np.full((3,3),np.nan),np.full((3,3),np.inf)):
            with self.assertRaises(ValueError):surface.distances(points)
        with self.assertRaises(ValueError):point_agreement(np.zeros((2,3)),np.eye(4),surface)

    def test_batched_queries_match_independent_brute_force_triangles(self):
        mesh=trimesh.creation.icosphere(subdivisions=1,radius=.3)
        surface=TriangleSurface(mesh.vertices,mesh.faces)
        points=np.random.default_rng(7).uniform(-.4,.4,(150,3))
        expected=trimesh.proximity.closest_point_naive(mesh,points)[1]
        np.testing.assert_allclose(surface.distances(points),expected,atol=1e-12)

    def test_matching_surface_does_not_bypass_rendered_depth_rejection(self):
        vertices,faces=square();surface=TriangleSurface(vertices,faces)
        self.assertTrue(point_agreement(np.array([[.2,.2,0],[.3,.4,0],[.7,.6,0]]),np.eye(4),surface)['accepted'])
        mask=np.ones((10,10),bool)
        result=render_agreement(np.ones((10,10)),mask,np.full((10,10),.9))
        self.assertFalse(result['accepted'])
        self.assertIn('CAD_occupies_observed_empty_space',result['reasons'])

    def test_independent_scene_depth_rejects_wrong_pose_without_second_segmentation(self):
        rendered=np.zeros((30,40));rendered[5:25,10:30]=1.
        silhouette=rendered>0
        self.assertTrue(render_agreement(np.ones((30,40)),silhouette,rendered)['accepted'])
        wrong=np.ones((30,40));wrong[silhouette]=1.1
        self.assertFalse(render_agreement(wrong,silhouette,rendered)['accepted'])
        occluded=np.full((30,40),.8)
        self.assertFalse(render_agreement(occluded,silhouette,rendered)['accepted'])

    def test_scene_depth_distinguishes_hidden_from_contradictory(self):
        rendered=np.ones((20,20))
        self.assertEqual(scene_depth_evidence(np.ones((20,20)),rendered)['state'],'supported')
        self.assertEqual(scene_depth_evidence(np.full((20,20),.8),rendered)['state'],'unobservable')
        self.assertEqual(scene_depth_evidence(np.zeros((20,20)),rendered)['state'],'unobservable')
        self.assertEqual(scene_depth_evidence(np.full((20,20),1.1),rendered)['state'],'contradicted')
        occluded=np.full((20,20),.8);occluded[:3]=1.1
        self.assertEqual(scene_depth_evidence(occluded,rendered)['state'],'contradicted')


if __name__=='__main__':unittest.main()
