#!/usr/bin/env python3
"""Batch the demo-template localization adapter over 50 x 50 scenes.

The manifest is produced by ``build_demo_clean_pointcloud_templates50.py``.
This coordinator performs a complete preflight before starting any task: every
requested target template must exist and match its recorded array name.  Each
task writes to a new independent directory and existing results are never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.unified_vision.contracts import serializable  # noqa: E402
from diagnostics.vision50_catalog import TASK_TARGETS  # noqa: E402


def dump(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(serializable(value), ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n"
    )
    temporary.replace(path)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_manifest(path: Path, selected: list[str]) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or not isinstance(value.get("tasks"), dict):
        raise ValueError("Unsupported demo-template manifest")
    missing = []
    for task in selected:
        expected = set(TASK_TARGETS[task])
        # Collection selectors such as bread.* are expanded in the clean
        # manifest; require at least one concrete member with that prefix.
        records = value["tasks"].get(task, {})
        for selector in expected:
            if selector.endswith(".*"):
                prefix = selector[:-2] + "_"
                if not any(key.startswith(prefix) for key in records):
                    missing.append(f"{task}:{selector}")
            elif selector not in records:
                missing.append(f"{task}:{selector}")
        for key, record in records.items():
            template = resolve(record["path"])
            array = record["array"]
            if not template.is_file():
                missing.append(f"{task}:{key}:missing_file:{template}")
                continue
            with np.load(template, allow_pickle=False) as arrays:
                if array not in arrays:
                    missing.append(f"{task}:{key}:missing_array:{array}")
    if missing:
        raise RuntimeError("Template preflight failed:\n" + "\n".join(missing))
    return value


def task_command(args, manifest: dict, task: str, destination: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_demo_template_localization.py"),
        "evaluate",
        "--task", task,
        "--episodes", str(args.episodes),
        "--ambiguity-gap", str(args.ambiguity_gap),
        "--source", str(args.source),
        "--output", str(destination),
    ]
    for key, record in manifest["tasks"][task].items():
        command.extend([
            "--template",
            f"{key}={resolve(record['path'])}#{record['array']}",
        ])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/frustum_vision50_2500_20261112_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ambiguity-gap", type=float, default=0.03)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_TARGETS))
    args = parser.parse_args()
    args.manifest = resolve(args.manifest)
    args.source = resolve(args.source)
    args.output = resolve(args.output)
    selected = args.tasks or list(TASK_TARGETS)
    if args.output == ROOT or not args.output.is_relative_to(ROOT):
        parser.error("--output must be a new directory under ULA")
    if args.output.exists():
        parser.error("Refusing to overwrite existing output")
    if not 1 <= args.episodes <= 50 or not 1 <= args.workers <= 16:
        parser.error("episodes must be 1..50 and workers 1..16")
    if not 0 <= args.ambiguity_gap <= 1:
        parser.error("ambiguity gap must be 0..1")
    if args.tasks and len(selected) != len(set(selected)):
        parser.error("Duplicate task")
    manifest = load_manifest(args.manifest, selected)
    args.output.mkdir(parents=True)
    (args.output / "logs").mkdir()
    pending = list(selected)
    active = {}
    completed = {}
    report = {
        "status": "running",
        "manifest": str(args.manifest),
        "source": str(args.source),
        "episodes_per_task": args.episodes,
        "requested_tasks": selected,
        "workers": args.workers,
        "robot_actions": 0,
        "CAD_used_for_selection": False,
        "tasks": {},
    }
    started = time.perf_counter()

    def save() -> None:
        tasks = {}
        for task in selected:
            path = args.output / "tasks" / task / "report.json"
            if path.is_file():
                value = json.loads(path.read_text())
                tasks[task] = {
                    key: value.get(key) for key in ("status", "summary", "error")
                }
            elif task in completed:
                tasks[task] = {
                    "status": "process_error" if completed[task] else "finished_without_report",
                    "exit_code": completed[task],
                }
            else:
                tasks[task] = {"status": "pending"}
        report.update(
            tasks=tasks,
            elapsed_s=time.perf_counter() - started,
            active={str(slot): {"pid": item[0].pid, "task": item[1]}
                    for slot, item in active.items()},
            finished_tasks=sum(value["status"] == "finished" for value in tasks.values()),
            evaluated_scenes=sum((value.get("summary") or {}).get("evaluated_scenes", 0)
                                 for value in tasks.values()),
            all_requested_targets_localized=sum(
                (value.get("summary") or {}).get("all_requested_targets_localized", 0)
                for value in tasks.values()
            ),
        )
        dump(args.output / "report.json", report)

    (args.output / "tasks").mkdir()
    save()
    try:
        while pending or active:
            for slot in range(args.workers):
                if slot in active:
                    process, task, log = active[slot]
                    if process.poll() is None:
                        continue
                    log.close()
                    completed[task] = process.returncode
                    del active[slot]
                if pending:
                    task = pending.pop(0)
                    destination = args.output / "tasks" / task
                    log = (args.output / "logs" / f"{task}.log").open("x")
                    environment = os.environ.copy()
                    environment.update(
                        OMP_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        PYTHONUNBUFFERED="1",
                    )
                    process = subprocess.Popen(
                        task_command(args, manifest, task, destination),
                        cwd=ROOT,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    active[slot] = (process, task, log)
            save()
            if active:
                time.sleep(2)
        failed = [task for task, code in completed.items() if code]
        report.update(
            status="finished" if not failed else "finished_with_errors",
            failed_tasks=failed,
            process_exit_codes=completed,
        )
        save()
    except BaseException as error:
        for process, _task, log in active.values():
            if process.poll() is None:
                process.terminate()
            log.close()
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
