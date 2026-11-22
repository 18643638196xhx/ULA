#!/usr/bin/env python3
"""Known-model-only vision server; the default pointcloud server is unchanged."""
import argparse
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("ula_known_model_transport",
    Path(__file__).with_name("perceive_microwave_rgbd.py"))
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--fp-iterations", type=int, default=5)
    parser.add_argument("--door-prompt", default="microwave door")
    parser.add_argument("--handle-prompt", default="microwave handle")
    args = parser.parse_args()
    # Reuse the audited JSON transport, replacing only its lazily imported class
    # inside this independent child process. No source files are patched.
    import adapters.microwave.perception as perception
    from adapters.microwave.known_model import KnownModelVision
    original = perception.MicrowaveVision
    perception.MicrowaveVision = lambda root, door, handle: KnownModelVision(root, door, handle,
        args.model_directory, args.debug_dir, args.fp_iterations)
    try:
        return transport.serve(args.door_prompt, args.handle_prompt)
    finally:
        perception.MicrowaveVision = original


if __name__ == "__main__":
    raise SystemExit(main())
