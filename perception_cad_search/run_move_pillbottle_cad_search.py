#!/usr/bin/env python3
"""Run move_pillbottle_pad with the isolated CAD-ranked perception worker."""

import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tra_pipe/RoboTwin")]


def main():
    from adapters import known_cad_model
    from perception_cad_search.pillbottle_cad_search import PillbottleCadSearchWorker

    # Patch only this process. Source files and the regular registry remain
    # unchanged, so the 35.7% command is still reproducible.
    known_cad_model.KnownCadWorker = PillbottleCadSearchWorker
    os.environ["ULA_PIPELINE"] = "known_cad"

    spec = importlib.util.spec_from_file_location(
        "cad_search_unified_entry", ROOT / "scripts/run_ula_task.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    sys.argv = [str(ROOT / "scripts/run_ula_task.py"),
                "--task", "move_pillbottle_pad", *sys.argv[1:]]
    return entry.main()


if __name__ == "__main__":
    raise SystemExit(main())
