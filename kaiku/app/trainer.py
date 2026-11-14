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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WHICH = ("kaiku", "plain")
MODAL_ABOUT_MIN = 4.0                # what a run on rented GPUs takes, start to result
MODAL_RUNS = int(os.environ.get("KAIKU_MODAL_RUNS", "40"))     # rented runs a demo may start; after that it trains here


class Trainer:
    def __init__(self, data: Path | str | None, out: Path | str, backend: str = "local", steps: int = 30000,
                 scenes: int = 48, last: int = 16, about_min: float | None = None):
        self.data = Path(data) if data else None
        self.out = Path(out)
        self.backend, self.steps, self.scenes = backend, steps, scenes
        self.last = last                     # the operator's latest episodes: few enough that the echoes decide it
        self.about_min = about_min           # what the page may say a run takes here, when somebody has measured it
        self.marks: list[tuple[float, float]] = []      # (time, progress) of the run, to say how long is left
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.error: str | None = None
        self.finished = False                # the run has ended, well or badly
        self.fresh = False                   # ... and nobody has picked its result up yet
        self.using = backend                 # where the run in hand trains: a rented run that fails is done again here
        self.closed = False

    @property
    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def _count(self) -> Path:
        return (self.data or self.out) / ".modal_runs"

    def _rented_left(self) -> int:
        try:
            return MODAL_RUNS - int(self._count.read_text().strip() or 0)
        except (OSError, ValueError):
            return MODAL_RUNS

    @property
    def about(self) -> float | None:
        """Minutes the next run (or the one in hand) takes, for the page to say."""
        rented = self.using == "modal" if self.busy else self.backend == "modal" and self._rented_left() > 0
        return MODAL_ABOUT_MIN if rented else self.about_min

    def start(self) -> bool:
        if self.busy or self.data is None or (self.jobs and not self.finished):
            return False
        self.out.mkdir(parents=True, exist_ok=True)
        self.using = "modal" if self.backend == "modal" and self._rented_left() > 0 else "local"
        if self.using == "modal":
            try:
                self._count.write_text(str(MODAL_RUNS - self._rented_left() + 1))
            except OSError:
                pass
        self._launch()
        return True

    def _cmd(self) -> list[str]:
        common = [str(self.data), "--out", str(self.out), "--steps", str(self.steps), "--scenes", str(self.scenes),
                  "--last", str(self.last)]
        if self.using == "modal":
            # The two policies train on rented GPUs, in parallel. In a Modal container the
            # demonstrations are on the volume already; anywhere else they are sent there first.
            cmd = [sys.executable, "-u", "-m", "kaiku.policy.modal_runner", *common,
                   *([] if os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_IS_REMOTE") else ["--upload"])]
        else:
            # Below the live demo in the queue for the processor, and on a share of the cores.
            cmd = [*(["nice", "-n", "10"] if shutil.which("nice") else []), sys.executable, "-u", "-m",
                   "kaiku.policy.fit", *common, "--threads", str(max(2, (os.cpu_count() or 4) // 4))]
        return cmd

    def _launch(self) -> None:
        cmd = self._cmd()
        with self.lock:
            self.jobs = {w: {"stage": "loading", "progress": 0.0} for w in WHICH}
            self.error, self.finished, self.fresh = None, False, False
            self.marks = []
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     start_new_session=True)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True, name="kaiku-trainer").start()

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
                    self.marks.append((time.monotonic(), sum(j.get("progress", 0.0) for j in self.jobs.values())
                                       / len(self.jobs)))
                    del self.marks[:-600]
        code = proc.wait()
        with self.lock:
            done = all(j.get("stage") == "done" for j in self.jobs.values())
            rented_failed = not done and self.using == "modal" and not self.closed
        if rented_failed:                    # no credit, no network, no GPU to rent: the same run, here
            print(f"kaiku trainer: the rented run failed ({self.error or last[-160:] or code}); training here", flush=True)
            self.using = "local"
            self._launch()
            return
        with self.lock:
            if not done and self.error is None:
                self.error = f"Training stopped ({last[-160:] or f'exit code {code}'})."
            self.finished = self.fresh = True

    def status(self) -> dict:
        """Where the training is: ``state`` idle, training, done or failed, and the two jobs."""
        with self.lock:
            jobs = {w: dict(j) for w, j in self.jobs.items()}
            error, finished = self.error, self.finished
            marks = self.marks[-300:]
        if not jobs:
            return {"state": "idle"}
        state = "failed" if error else ("done" if finished else "training")
        k = jobs["kaiku"]
        # What is left, from how fast the last stretch went. Reading the files and training do not go
        # at the same pace, so nothing is said until the pace has been steady for a while.
        eta = None
        if len(marks) > 20 and marks[-1][1] > marks[0][1] + 0.01:
            (t0, p0), (t1, p1) = marks[0], marks[-1]
            eta = round((1.0 - p1) * (t1 - t0) / (p1 - p0))
        return {"state": state, "error": error, "stage": k.get("stage"), "loss": k.get("loss"), "eta_s": eta,
                "progress": round(sum(j.get("progress", 0.0) for j in jobs.values()) / len(jobs), 4), "jobs": jobs}

    def close(self) -> None:
        self.closed = True
        if self.busy:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)       # its own process group, nobody else's
            except ProcessLookupError:
                pass
