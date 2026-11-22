#!/usr/bin/env python3
"""Independent persistent microwave RGB-D worker in the Python 3.10 environment."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def serve(door_prompt, handle_prompt):
    # Keep native GPU-library stdout away from the JSON response stream.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        def send(value):
            protocol.write(json.dumps(value)+"\n")
            protocol.flush()
        try:
            from adapters.microwave.perception import MicrowaveVision
            root = Path(__file__).resolve().parents[1]/"tra_pipe/RoboTwin"
            vision = MicrowaveVision(root, door_prompt, handle_prompt)
            send(dict(status="ready", pid=os.getpid(),
                      model_load_s=vision.load_timings_s["model_stack_load_total"]))
        except Exception as error:
            traceback.print_exc()
            send(dict(status="error", error=f"{type(error).__name__}: {error}"))
            return 1
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                if request["op"] == "shutdown":
                    return 0
                if request["op"] != "estimate":
                    raise ValueError("Unknown microwave vision operation")
                with np.load(request["input"], allow_pickle=False) as data:
                    result = vision.estimate(**dict(data))
                np.savez(request["output"], door_points=result["door_points"],
                    handle_world=result["handle_world"], camera_world=result["camera_world"],
                    audit_json=np.asarray(json.dumps(result["audit"])),
                    timings_json=np.asarray(json.dumps(result["timings_s"])))
                send(dict(status="ok", id=request["id"]))
            except Exception as error:
                traceback.print_exc()
                send(dict(status="error", id=request.get("id"), error=f"{type(error).__name__}: {error}"))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--door-prompt", default="microwave door")
    parser.add_argument("--handle-prompt", default="microwave handle")
    args = parser.parse_args()
    return serve(args.door_prompt, args.handle_prompt)


if __name__ == "__main__":
    raise SystemExit(main())
