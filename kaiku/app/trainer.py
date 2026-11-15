"""Training from the live demo in a process of its own, so the robots keep their 30 Hz.

The local backend runs :mod:`kaiku.policy.fit` here.  The Modal backend runs the two policies
in parallel on the deployed ``kaiku-policy`` functions and copies their small policy files back
into the live container.  Both backends speak the same ``KAIKU {json}`` stdout protocol.
"""
from __future__ import annotations

import json
import math
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
WAIT_S = 60.0                        # a run still at nothing done after this long is waiting for its machines
SMOOTH_S = 45.0                      # how slowly the time left may change its mind
MODAL_ABOUT_MIN = 6.0                # what a run on rented GPUs takes, start to result (5.9 measured on the hosted demo)
RENTED_KEPT = 5                      # rented runs whose times are kept: the page promises the slowest of them
MODAL_RUNS = int(os.environ.get("KAIKU_MODAL_RUNS", "40"))     # rented runs a demo may start; after that it trains here


class Trainer:
    def __init__(self, data: Path | str | None, out: Path | str, backend: str = "local", steps: int = 30000,
                 scenes: int = 48, last: int = 16, about_min: float | None = None):
        self.data = Path(data) if data else None
        self.out = Path(out)
        self.backend, self.steps, self.scenes = backend, steps, scenes
        self.last = last                     # the operator's latest episodes: few enough that the echoes decide it
        self.clock = time.monotonic          # a test hands in its own
        # What the page may say a run takes: what somebody measured (--train-about), or else what the
        # last run took. This machine and rented GPUs are timed apart: one is six times the other.
        # A rented run takes four minutes when its machines are warm and six when they are not. The
        # page promises the slowest of the last few, so the number does not change with every run.
        self.about_min = about_min
        self.rented_runs = [MODAL_ABOUT_MIN]
        try:
            timed = json.loads((self.out / "timing.json").read_text())
            if about_min is None and timed.get("minutes"):
                self.about_min = float(timed["minutes"])
            if timed.get("rented_runs"):
                self.rented_runs = [float(m) for m in timed["rented_runs"]][-RENTED_KEPT:]
            elif timed.get("rented_minutes"):
                self.rented_runs.append(float(timed["rented_minutes"]))
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        self.rented_min = max(self.rented_runs)
        self.started = 0.0                   # when the run began
        self.total_s: float | None = None    # how long the whole run looks like taking
        self.total_at = 0.0                  # ... as of when
        self.wait_s = 0.0                    # how long the run sat at nothing done (machines still starting)
        self.took_s: float | None = None     # how long the last finished run took
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
    def rented(self) -> bool:
        """Whether the next run (or the one in hand) trains on rented GPUs rather than on this machine."""
        if self.busy or (self.jobs and not self.finished):
            return self.using == "modal"
        return self.backend == "modal" and self._rented_left() > 0

    @property
    def about(self) -> float | None:
        """Minutes the next run (or the one in hand) takes, for the page to say."""
        return self.rented_min if self.rented else self.about_min

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
            self.started, self.total_s, self.took_s = self.clock(), None, None
            self.wait_s = 0.0
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
            if done and self.error is None:
                self._measured(self.clock() - self.started)
            self.finished = self.fresh = True

    def _measured(self, seconds: float) -> None:
        """A whole run took this long where it ran: the next visitor is told so before pressing Train."""
        self.took_s = seconds
        if seconds < 30.0:                   # nothing trains that fast: not a time to promise anybody
            return
        if self.using == "modal":
            self.rented_runs = (self.rented_runs + [round(seconds / 60.0, 1)])[-RENTED_KEPT:]
            self.rented_min = max(self.rented_runs)
        else:
            self.about_min = round(seconds / 60.0, 1)
        try:
            (self.out / "timing.json").write_text(json.dumps({"minutes": self.about_min, "rented_minutes": self.rented_min,
                                                               "rented_runs": self.rented_runs, "steps": self.steps}))
        except OSError:
            pass

    def _left(self, progress: float) -> float | None:
        """Seconds still to go, or None while there is nothing honest to say.

        The whole run is taken to last what was measured before where it runs (this machine, or rented
        GPUs), and the further it gets the more
        its own pace counts instead: elapsed time over the share done. That never jumps the way the
        pace of the last minute does (reading files, training and testing each go at their own). What
        is said then changes its mind slowly, so the time left drifts and does not leap."""
        now = self.clock()
        elapsed = now - self.started
        about = self.rented_min if self.using == "modal" else self.about_min
        prior = about * 60.0 if about else None
        if progress < 0.02:
            # Nothing has started where it runs (a rented GPU can take minutes to come): that wait says
            # nothing about the pace of the run, and past the first minute there is no honest time to give.
            self.wait_s = elapsed
            if elapsed > WAIT_S:
                self.total_s = None
                return None
        if progress >= 0.05 and elapsed >= 20.0:
            own = self.wait_s + (elapsed - self.wait_s) / progress
            if prior is None:
                total = own
            else:
                # A run going slower than it was said to is believed sooner: the clock already shows it.
                # One that looks faster is not, since the first stretch (reading files) always is.
                weight = progress if own <= prior else min(1.0, progress * (1.0 + 3.0 * math.log(own / prior)))
                total = (1.0 - weight) * prior + weight * own
        elif prior is not None:
            total = prior + (self.wait_s if self.wait_s > WAIT_S else 0.0)
        else:
            return None
        if self.total_s is None:
            self.total_s = total
        else:
            self.total_s += (total - self.total_s) * (1.0 - math.exp(-(now - self.total_at) / SMOOTH_S))
        self.total_at = now
        return max(0.0, self.total_s - elapsed)

    def status(self) -> dict:
        """Where the training is: ``state`` idle, training, done or failed, and the two jobs."""
        with self.lock:
            jobs = {w: dict(j) for w, j in self.jobs.items()}
            error, finished = self.error, self.finished
            if not jobs:
                return {"state": "idle"}
            state = "failed" if error else ("done" if finished else "training")
            progress = sum(j.get("progress", 0.0) for j in jobs.values()) / len(jobs)
            eta = self._left(progress) if state == "training" else None
            took = self.took_s
        k = jobs["kaiku"]
        # The two train side by side and one can be minutes behind (a cold machine): what the page says is
        # being done is what the slower one is doing, or "Finishing" stands over half a bar.
        behind = min(jobs.values(), key=lambda j: j.get("progress", 0.0))
        stage = behind.get("stage") or k.get("stage")
        if state == "training" and progress < 0.02 and self.clock() - self.started > WAIT_S:
            stage = "starting"
        return {"state": state, "error": error, "stage": stage, "loss": k.get("loss"),
                "eta_s": None if eta is None else round(eta), "took_s": None if took is None else round(took),
                "progress": round(progress, 4), "jobs": jobs}

    def close(self) -> None:
        self.closed = True
        if self.busy:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)       # its own process group, nobody else's
            except ProcessLookupError:
                pass
