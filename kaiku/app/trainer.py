"""Training from the live demo in a process of its own, so the robots keep their 30 Hz.

The local backend runs :mod:`kaiku.policy.fit` here.  The Modal backend runs the two policies
in parallel on the deployed ``kaiku-policy`` functions and copies their small policy files back
into the live container.  Both backends speak the same ``KAIKU {json}`` stdout protocol.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WHICH = ("kaiku", "plain")


class Trainer:
    def __init__(self, data: Path | str | None, out: Path | str, backend: str = "local", steps: int = 30000,
                 scenes: int = 48, last: int = 16):
        self.data = Path(data) if data else None
        self.out = Path(out)
        self.backend, self.steps, self.scenes = backend, steps, scenes
        self.last = last                     # the operator's latest episodes: few enough that the echoes decide it
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.error: str | None = None
        self.finished = False                # the run has ended, well or badly
        self.fresh = False                   # ... and nobody has picked its result up yet

    @property
    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> bool:
        if self.busy or self.data is None:
            return False
        self.out.mkdir(parents=True, exist_ok=True)
        common = [str(self.data), "--out", str(self.out), "--steps", str(self.steps), "--scenes", str(self.scenes),
                  "--last", str(self.last)]
        if self.backend == "modal":
            # The live image has the Modal client.  Calling the deployed function from a process
            # inside the container avoids a second ``modal run`` app and lets the runner stream
            # generator events while both jobs execute.
            cmd = [sys.executable, "-u", "-m", "kaiku.policy.modal_runner", *common]
        else:
            # Below the live demo in the queue for the processor, and on a share of the cores.
            cmd = [*(["nice", "-n", "10"] if shutil.which("nice") else []), sys.executable, "-u", "-m",
                   "kaiku.policy.fit", *common, "--threads", str(max(2, (os.cpu_count() or 4) // 4))]
        with self.lock:
            self.jobs = {w: {"stage": "loading", "progress": 0.0} for w in WHICH}
            self.error, self.finished, self.fresh = None, False, False
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     start_new_session=True)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True, name="kaiku-trainer").start()
        return True

    def _read(self, proc: subprocess.Popen) -> None:
        last = ""
        for line in proc.stdout:
            line = line.strip()
            at = line.find("KAIKU {")
            if at < 0:
                last = line or last
                continue
            try:
                m = json.loads(line[at + 6:])
            except ValueError:
                continue
            with self.lock:
                if m.get("stage") == "error":
                    self.error = str(m.get("text", "Training failed."))
                elif m.get("which") in self.jobs:
                    self.jobs[m["which"]].update(m)
        code = proc.wait()
        with self.lock:
            done = all(j.get("stage") == "done" for j in self.jobs.values())
            if not done and self.error is None:
                self.error = f"Training stopped ({last[-160:] or f'exit code {code}'})."
            self.finished = self.fresh = True

    def status(self) -> dict:
        """Where the training is: ``state`` idle, training, done or failed, and the two jobs."""
        with self.lock:
            jobs = {w: dict(j) for w, j in self.jobs.items()}
            error, finished = self.error, self.finished
        if not jobs:
            return {"state": "idle"}
        state = "failed" if error else ("done" if finished else "training")
        k = jobs["kaiku"]
        return {"state": state, "error": error, "stage": k.get("stage"), "loss": k.get("loss"),
                "progress": round(sum(j.get("progress", 0.0) for j in jobs.values()) / len(jobs), 4), "jobs": jobs}

    def close(self) -> None:
        if self.busy:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)       # its own process group, nobody else's
            except ProcessLookupError:
                pass
