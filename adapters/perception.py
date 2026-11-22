"""Persistent Python 3.10 vision worker, isolated from the 3.11 policy runtime."""

import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np


class PerceptionWorker:
    """One lazy model stack per batch. Failures never trigger hidden reloads."""

    def __init__(self, ula_root, output_dir, fp_iterations=5, timeout_s=300, backend="foundationpose"):
        if backend not in ("foundationpose", "pointcloud"):
            raise ValueError("Unknown perception backend")
        self.root = Path(ula_root).resolve()
        self.robotwin = self.root / "tra_pipe/RoboTwin"
        self.output = Path(output_dir).resolve()
        self.fp_iterations = fp_iterations
        self.backend = backend
        self.timeout_s = timeout_s
        self.process = None
        self._log = None
        self._buffer = b""
        self._closed = False
        self._lock = threading.RLock()
        self._stats = dict(starts=0, requests=0, completed_requests=0,
                           startup_wall_s=0.0, model_load_s=0.0, pid=None)

    def _launch_spec(self):
        environment = os.environ.copy()
        env_root = self.robotwin / ".conda_envs/RoboTwin_yzy"
        fp_root = self.robotwin / "third_party/FoundationPose"
        paths = [self.root, self.robotwin]
        if self.backend == "foundationpose":
            paths.extend((fp_root, fp_root / "mycpp/build"))
        environment["PYTHONPATH"] = os.pathsep.join(map(str, paths))
        environment["LD_LIBRARY_PATH"] = str(env_root / "lib") + ":/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib"
        environment["WARP_CACHE_PATH"] = str(self.root / ".runtime_cache/warp_vision")
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [str(env_root / "bin/python"),
                   str(self.root / "scripts/perceive_rgbd.py"), "--serve",
                   "--debug-dir", str(self.output / "debug"),
                   "--backend", self.backend,
                   "--fp-iterations", str(self.fp_iterations)]
        return command, self.robotwin, environment

    def _receive(self):
        deadline = time.monotonic() + self.timeout_s
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError(f"Vision worker timed out; see {self.output / 'perception.log'}")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"Vision worker exited unexpectedly; see {self.output / 'perception.log'}")
            self._buffer += chunk
            if len(self._buffer) > 1024 * 1024:
                raise RuntimeError("Vision worker protocol message exceeds 1 MiB")
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def _send(self, message):
        self.process.stdin.write(json.dumps(message).encode() + b"\n")
        self.process.stdin.flush()

    def _start(self):
        if self._closed:
            raise RuntimeError("Vision worker is closed (automatic reload is disabled)")
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError("Vision worker died (automatic reload is disabled)")
            return
        self.output.mkdir(parents=True, exist_ok=True)
        command, cwd, environment = self._launch_spec()
        start = time.perf_counter()
        self._log = (self.output / "perception.log").open("wb")
        self.process = subprocess.Popen(command, cwd=cwd, env=environment,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._log, bufsize=0)
        self._stats.update(starts=1, pid=self.process.pid)
        ready = self._receive()
        if ready.get("status") != "ready" or ready.get("pid") != self.process.pid:
            raise RuntimeError(f"Vision worker failed to start: {ready}")
        self._stats.update(startup_wall_s=time.perf_counter() - start,
                           model_load_s=ready["model_load_s"])

    def estimate_files(self, request, result):
        with self._lock:
            start = time.perf_counter()
            first = self.process is None
            try:
                self._start()
                self._stats["requests"] += 1
                request_id = self._stats["requests"]
                self._send(dict(op="estimate", id=request_id,
                                input=str(request), output=str(result)))
                reply = self._receive()
                if reply.get("id") != request_id or reply.get("status") not in ("ok", "error"):
                    raise RuntimeError(f"Invalid vision worker response: {reply}")
            except BaseException:
                self.close()
                raise
            if reply["status"] == "error":
                raise RuntimeError(f"Vision estimate failed: {reply['error']}; see {self.output / 'perception.log'}")
            try:
                with np.load(result, allow_pickle=False) as values:
                    timings = json.loads(str(values["timings_json"]))
                    timings.update(
                        vision_model_load=self._stats["model_load_s"] if first else 0.0,
                        vision_worker_startup=self._stats["startup_wall_s"] if first else 0.0,
                        perception_subprocess_wall=time.perf_counter() - start,
                        vision_worker_pid=self.process.pid, vision_request_id=request_id,
                    )
                    estimate = SimpleNamespace(
                        cup_world_pose=values["cup_world_pose"].copy() if "cup_world_pose" in values else None,
                        cup_bounds_world=values["cup_bounds_world"].copy() if "cup_bounds_world" in values else None,
                        cup_points_world=values["cup_points_world"].copy() if "cup_points_world" in values else None,
                        observation_audit=(json.loads(str(values["observation_audit_json"]))
                                           if "observation_audit_json" in values else None),
                        coaster_world_center=values["coaster_world_center"].copy(),
                        coaster_points_world=(values["coaster_points_world"].copy()
                                              if "coaster_points_world" in values else None),
                        support_world_pose=(values["support_world_pose"].copy()
                                            if "support_world_pose" in values else None),
                        timings_s=timings,
                    )
                self._stats["completed_requests"] += 1
                return estimate
            except BaseException:
                self.close()
                raise

    def summary(self):
        return dict(self._stats, backend=self.backend, closed=self._closed,
                    running=self.process is not None and self.process.poll() is None,
                    log=str(self.output / "perception.log"))

    def close(self):
        with self._lock:
            self._closed = True
            if self.process is not None:
                if self.process.poll() is None:
                    try:
                        self._send(dict(op="shutdown"))
                        self.process.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            self.process.wait()
                self.process.stdin.close()
                self.process.stdout.close()
            if self._log is not None:
                self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ExternalPerception:
    """Per-stage artifacts, with shared worker ownership at batch level."""

    def __init__(self, ula_root, output_dir, fp_iterations=5, *, worker):
        if worker.fp_iterations != fp_iterations:
            raise ValueError("Shared worker fp_iterations mismatch")
        self.output = Path(output_dir).resolve()
        self.worker = worker
        self.reference_path = self.output / "estimate.npz"

    def estimate(self, rgb, depth, intrinsic, world_to_camera):
        self.output.mkdir(parents=True, exist_ok=True)
        request = self.output / "rgbd.npz"
        result = self.output / "estimate.npz"
        np.savez_compressed(request, rgb=rgb, depth=depth, intrinsic=intrinsic,
                            world_to_camera=world_to_camera)
        return self.worker.estimate_files(request, result)

    def supplementary(self, camera, label, *, cup_only=True, refine_cup=False):
        """Cup-only wrist request: coaster target must remain from the head view."""
        directory = self.output / label
        directory.mkdir(parents=True,exist_ok=True)
        request, result = directory / "rgbd.npz", directory / "estimate.npz"
        payload = dict(rgb=camera["rgb"],depth=camera["depth"],
                       intrinsic=camera["intrinsic_cv"],world_to_camera=camera["extrinsic_cv"])
        if cup_only:
            payload["cup_only"] = np.asarray(True)
        if refine_cup:
            payload["refine_cup"] = np.asarray(True)
        np.savez_compressed(request,**payload)
        return self.worker.estimate_files(request,result)

    def support_only(self, camera, label="support_only"):
        """Observe only the placement support after grasp.

        The grasped object is commonly hidden by the gripper.  Requiring it in
        the same post-grasp frame can reject an otherwise usable support
        observation.  The caller must already have an independent hand/object
        transform; this method deliberately returns no object measurement.
        """
        directory = self.output / label
        directory.mkdir(parents=True, exist_ok=True)
        request, result = directory / "rgbd.npz", directory / "estimate.npz"
        np.savez_compressed(
            request,
            rgb=camera["rgb"], depth=camera["depth"],
            intrinsic=camera["intrinsic_cv"],
            world_to_camera=camera["extrinsic_cv"],
            support_only=np.asarray(True),
        )
        return self.worker.estimate_files(request, result)

    def capture_reference_view(self, camera, camera_name):
        """Save a pre-VLA wrist frame, infer it lazily only if needed later."""
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "wrist_reference_frame.npz"
        np.savez_compressed(path, rgb=camera["rgb"], depth=camera["depth"],
            intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"],
            camera_name=np.asarray(camera_name))
        return path

    def select_reference(self, estimate):
        self.reference_path = self.output / "selected_reference.npz"
        payload = dict(
            cup_bounds_world=estimate.cup_bounds_world,
            cup_points_world=estimate.cup_points_world,
            coaster_world_center=estimate.coaster_world_center,
            coaster_points_world=(getattr(estimate, "coaster_points_world", None)
                                  if getattr(estimate, "coaster_points_world", None) is not None
                                  else np.empty((0, 3))),
            observation_audit_json=np.asarray(json.dumps(estimate.observation_audit)),
            timings_json=np.asarray(json.dumps(estimate.timings_s)),
        )
        support_pose = getattr(estimate, "support_world_pose", None)
        if support_pose is not None:
            payload["support_world_pose"] = np.asarray(support_pose, dtype=float)
        np.savez(self.reference_path, **payload)
