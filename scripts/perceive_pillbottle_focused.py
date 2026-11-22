#!/usr/bin/env python3
"""Isolated worker: focused RGB-D frontend, unchanged known-CAD backend."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"tra_pipe/RoboTwin")]


def main():
    import adapters.pillbottle_known_vision as backend
    from adapters.pillbottle_focus.frontend import FocusedBottleFrontend
    # The legacy backend explicitly dispatches through this module-local alias.
    # Rebind only inside this dedicated serial worker process; do not modify the
    # shared PillbottleVision class, source files, or other running workers.
    backend.PillbottleVision = FocusedBottleFrontend
    spec = importlib.util.spec_from_file_location(
        "ula_focused_pill_worker", ROOT/"scripts/perceive_pillbottle_known_model.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    return original.main()


if __name__ == "__main__":
    raise SystemExit(main())
