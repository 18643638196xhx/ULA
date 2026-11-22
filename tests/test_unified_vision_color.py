import numpy as np
from adapters.unified_vision.identity import cad_color_evidence
from adapters.unified_vision import VisionConfig


def test_same_color_under_brightness_change():
    rgb=np.tile([.10,.20,.80],(10,10,1));mask=np.ones((10,10),bool)
    assert cad_color_evidence(rgb,rgb*.6,mask)['state']=='supported'


def test_different_color_rejected():
    blue=np.tile([.05,.10,.8],(10,10,1));red=blue[...,::-1]
    assert cad_color_evidence(blue,red,np.ones((10,10),bool))['state']=='contradicted'


def test_neutral_and_missing_pixels_are_inconclusive():
    rgb=np.full((10,10,3),.5)
    assert cad_color_evidence(rgb,rgb,np.ones((10,10),bool))['state']=='unobservable'
    assert cad_color_evidence(rgb,rgb,np.zeros((10,10),bool))['accepted']


def test_background_not_compared_and_default_off():
    a=np.tile([.05,.10,.8],(10,10,1));b=a.copy();b[5:]=[.8,.1,.05]
    mask=np.zeros((10,10),bool);mask[:5]=True
    assert cad_color_evidence(a,b,mask)['state']=='supported'
    assert not VisionConfig().verify_cad_color
