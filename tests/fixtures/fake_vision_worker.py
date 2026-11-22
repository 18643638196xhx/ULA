"""CPU-only model double; uses the real worker protocol and NPZ code."""

import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from perceive_rgbd import serve


class FakePerception:
    def __init__(self):
        if "bad-load" in sys.argv:
            raise RuntimeError("fake initialization failure")
        print("Python model initialization log", flush=True)
        os.write(1, b"Native model initialization log\n")
        self.load_timings_s = dict(model_stack_load_total=0.125)

    def estimate(self, rgb, **_):
        marker = float(rgb.flat[0])
        if marker == -1:
            raise ValueError("fake frame failure")
        if marker == -2:
            time.sleep(60)
        if "pointcloud" in sys.argv:
            return SimpleNamespace(cup_bounds_world=np.array([[marker,0,0],[marker+0.1,0.1,0.1]]),
                                   cup_points_world=np.full((40,3),marker),
                                   observation_audit=dict(uses_prepared_object_model=False),
                                   coaster_world_center=np.full(3,marker), timings_s=dict(inference=0.01))
        pose = np.eye(4)
        pose[0, 3] = marker
        return SimpleNamespace(cup_world_pose=pose, coaster_world_center=np.full(3, marker),
                               timings_s=dict(inference=0.01))


raise SystemExit(serve(FakePerception))
