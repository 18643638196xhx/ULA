"""Read-only planning queries from a saved grasp; never executes trajectories."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "tra_pipe/RoboTwin"
sys.path[:0] = [str(ROOT), str(ROBOTWIN)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    record = json.loads(args.result.read_text())
    os.chdir(ROBOTWIN)
    # The main runner is a script, not the copied RoboTwin scripts package.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ula_runner", ROOT / "scripts/run_place_empty_cup.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    from envs.place_empty_cup import place_empty_cup
    from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz, placement_ee_pose

    task = place_empty_cup()
    config = runner.task_config(SimpleNamespace(arm="left", camera_shader="default"))
    task.setup_demo(now_ep_num=0, seed=record["seed"], **config)
    try:
        arm = record["handoff"]["arm"]
        entity = getattr(task.robot, f"{arm}_entity")
        names = [joint.get_name() for joint in entity.get_active_joints()]
        indices = [names.index(name) for name in getattr(task.robot, f"{arm}_arm_joints_name")]
        qpos = entity.get_qpos().copy()
        qpos[indices] = record["grasp_handoff"]["arm_joint_positions"][:len(indices)]
        planner = getattr(task.robot, f"{arm}_plan_path")
        placement = record["traditional_placement"]
        metadata = json.loads((ROBOTWIN / "assets/objects/021_cup/model_data0.json").read_text())
        functional = np.asarray(metadata["functional_matrix"][0], dtype=float)
        functional[:3, 3] *= metadata["scale"]
        for rotation_name, reference in (
            ("held", placement["cup_world_pose"]),
            ("upright", record["handoff"]["estimated_cup_world_pose"]),
        ):
            for yaw in (0.0, -0.3, 0.3, -0.6, 0.6):
                cup = np.asarray(reference).copy()
                cosine, sine = np.cos(yaw), np.sin(yaw)
                cup[:3, :3] = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]) @ cup[:3, :3]
                desired = placement_ee_pose(cup, placement["ee_to_cup"], functional,
                                            placement["coaster_world_center"])
                desired[2, 3] += 0.05
                result = planner(matrix_to_pose_wxyz(desired), last_qpos=qpos)
                print(json.dumps(dict(rotation=rotation_name, yaw=yaw, status=result["status"],
                                      detail=result.get("planner_status"))), flush=True)
    finally:
        task.close_env()


if __name__ == "__main__":
    main()
