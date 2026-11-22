#!/usr/bin/env python3
"""Load native pi0.5 checkpoint and check inference format, not task success."""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"external/pi05"),str(ROOT/"external/pi05/src")]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--trajectory",type=Path,required=True)
    args=parser.parse_args()
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.shared import normalize
    assets=sorted(p for p in (args.checkpoint/"assets").iterdir() if p.is_dir())
    if len(assets)!=1:
        raise ValueError("Expected one normalization asset in native checkpoint")
    policy=policy_config.create_trained_policy(config.get_config("multi_task_4"),args.checkpoint,
        norm_stats=normalize.load(assets[0]),sample_kwargs={"num_steps":10})
    with h5py.File(args.trajectory) as data:
        sample=dict(state=data["observation.state"][0],prompt=str(data.attrs["instruction"]),
            images={name:np.transpose(data[f"images/{name}"][0],(2,0,1))
                    for name in ("cam_high","cam_left_wrist","cam_right_wrist")})
    actions=np.asarray(policy.infer(sample)["actions"])
    if actions.shape!=(50,14) or not np.isfinite(actions).all():
        raise ValueError(f"Bad pi0.5 output shape or values: {actions.shape}")
    print(json.dumps(dict(checkpoint=str(args.checkpoint),architecture="native_pi0.5",
        action_shape=list(actions.shape),finite=True,WAM=False,
        check="load_and_inference_format_only_not_grasp_quality")),flush=True)


if __name__=="__main__":
    main()
