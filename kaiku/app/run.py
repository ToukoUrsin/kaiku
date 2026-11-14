"""The live demo: one operator, many robots, in the browser.

    python -m kaiku.app.run --n 48 --port 8651 [--parts auto|3|4|5] [--record DIR] [--seed 0]

One loop at the challenge's 30 Hz: the operator (gloves, or the stand-in) moves the lead robot,
the echo carries the motion to every other robot's own scene, the safety filter and the physics
judge all of them, the recorder keeps the clean demonstrations, the browser shows it live.
"""
from __future__ import annotations

import argparse
import queue
import signal
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from kaiku.app.autopilot import Autopilot
from kaiku.app.session import FAILED, IDLE, LEAD, RUNNING, SUCCESS, Session
from kaiku.data.recorder import Recorder
from kaiku.sim.twin import CONTROL_HZ
from kaiku.web.server import ViewerServer

DT = 1.0 / CONTROL_HZ
HUD_EVERY = 3                 # ticks between HUD updates (10 Hz)
MODES = ("autopilot", "gloves", "replay", "policy")


class Runner:
    def __init__(self, n: int = 48, port: int = 8651, host: str = "0.0.0.0", seed: int = 0, parts: str | int = "auto",
                 record: Path | str | None = None, hold_s: float = 1.6, lead_timeout_s: float = 90.0,
                 threads: int | None = None):
        self.s = Session(n, seed=seed, threads=threads)
        self.n = n
        self.parts = parts
        self.hold_s, self.lead_timeout = hold_s, int(lead_timeout_s * CONTROL_HZ)
        self.rng = np.random.default_rng(seed + 1)             # the scene sampler's own, used off-thread
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="kaiku-scenes")
        self.viewer = ViewerServer(self.s.twin.model, n, host=host, port=port, title="Kaiku")
        self.commands: queue.Queue = queue.Queue()
        self.viewer.on_message(self.commands.put)
        self.rec = Recorder(self.s, record)
        self.op = Autopilot(self.s)
        self.mode = "autopilot"
        self.paused = False
        self.stop = False
        self.phase = "run"                                     # run -> hold -> run ...
        self.hold_until = 0.0
        self.states = np.full(n, RUNNING, np.uint8)
        self.states[0] = LEAD
        self.sorted = np.zeros(n, int)
        self.done_ticks = np.zeros(n, int)
        self.history: deque = deque(maxlen=60)
        self.tot = {"episodes": 0, "human": 0, "robot": 0, "attempts": 0, "clipped": 0, "ticks": 0}
        self.walls: deque = deque(maxlen=90)                   # wall-clock time of recent ticks
        self.tick_ms: deque = deque(maxlen=90)
        self.t_start = time.time()
        self.hud_tick = 0
        self.next = self.pool.submit(self._sample)

    # ------------------------------------------------------------------ scenes
    def _sample(self):
        n = self.parts
        if n == "auto":
            n = int(self.rng.choice([3, 4, 5], p=[0.35, 0.40, 0.25]))
        return self.s.sample_scene(int(n), self.rng)

    def _begin_episode(self) -> None:
        lead, followers = self.next.result()
        self.next = self.pool.submit(self._sample)
        self.s.new_episode(lead, followers)
        self.op.begin()
        self.rec.begin()
        self.states[:] = IDLE
        self.states[0] = LEAD
        self.sorted[:] = 0
        self.done_ticks[:] = 0
        self.phase = "run"

    def _end_episode(self, record: bool = True) -> None:
        s = self.s
        r = s.result()
        ok = r.success & ~r.aborted
        self.states[1:] = np.where(ok[1:], SUCCESS, FAILED)
        self.states[0] = LEAD
        self.sorted = s.progress()
        kept = self.rec.end(r) if record else {"kept": 0, "human": 0, "dropped": 0}
        if not record:
            self.rec.begin()
        clean = ok & (r.clipped == 0)
        t = self.tot
        t["episodes"] += 1
        t["human"] += int(clean[0])
        t["robot"] += int(clean[1:].sum())
        t["attempts"] += self.n - 1
        t["clipped"] += int(r.clipped.sum())
        t["ticks"] += s.tick_no
        self.history.append({"episode": t["episodes"], "parts": len(r.layouts[0].placements),
                             "ok": int(clean[1:].sum()), "robots": self.n - 1, "lead_ok": bool(clean[0]),
                             "saved": kept["kept"], "seconds": round(s.tick_no * DT, 1)})
        self.phase = "hold"
        self.hold_until = time.perf_counter() + self.hold_s

    # ------------------------------------------------------------------ the loop
    def _commands(self) -> None:
        while True:
            try:
                m = self.commands.get_nowait()
            except queue.Empty:
                return
            if m.get("type") != "cmd":
                continue
            name, value = m.get("name"), m.get("value")
            if name == "pause":
                self.paused = bool(value)
            elif name == "next_scene":
                if self.phase == "run":
                    self._end_episode(record=False)
                self.hold_until = 0.0
            elif name == "parts":
                self.parts = value if value == "auto" else int(value)
                self.next = self.pool.submit(self._sample)       # the scene already drawn no longer fits
            elif name == "record":
                self.rec.enabled = bool(value) and self.rec.root is not None
                if value and self.rec.root is None:
                    self.toast("Start with --record DIR to save demonstrations.", "warn")
            elif name == "mode":
                if value == self.mode:
                    continue
                self.toast(f"{str(value).capitalize()} mode is not connected in this build.", "warn")
            elif name == "calibrate":
                self.toast("Calibration needs the gloves. The stand-in operator is driving.", "info")
            elif name == "train":
                self.toast("Training is not connected in this build.", "warn")

    def toast(self, text: str, level: str = "info") -> None:
        self.viewer.send_event({"type": "toast", "text": text, "level": level})

    def _step(self) -> None:
        s = self.s
        if self.phase == "hold":
            if time.perf_counter() >= self.hold_until and self.next.done():
                self._begin_episode()
            return
        t0 = time.perf_counter()
        obs = s.observe()
        cmd = self.op.update(obs)
        if self.op.done or s.tick_no >= self.lead_timeout:
            s.finish_lead()
        s.tick(cmd)
        self.rec.add(obs, *s.last_action)
        if s.tick_no % 6 == 0:
            self._refresh()
        self.tick_ms.append(1000.0 * (time.perf_counter() - t0))
        self.walls.append(time.perf_counter())
        if s.settled:
            self._end_episode()

    def _refresh(self) -> None:
        """Per-robot state for the viewer: who is working, who has finished, who has failed."""
        s = self.s
        self.sorted = s.progress()
        parts = len(s.layouts[0].placements)
        if self.n > 1:
            done = s.echo.done
            self.done_ticks[1:] = np.where(done, self.done_ticks[1:] + 6, 0)
            started = s.echo.tau > 1.0
            st = np.where(started, RUNNING, IDLE)
            st = np.where(self.sorted[1:] >= parts, SUCCESS, st)
            st = np.where((self.sorted[1:] < parts) & (self.done_ticks[1:] > 45), FAILED, st)
            self.states[1:] = np.where(s.safety.aborted[1:], FAILED, st)
        self.states[0] = LEAD

    def hud(self) -> dict:
        s, t = self.s, self.tot
        F = self.n - 1
        parts = len(s.layouts[0].placements)
        lag = s.echo.lag if F else np.zeros(0)
        working = self.phase == "run"
        status = "paused" if self.paused else ("running" if working and not s.lead_done else "settling")
        minutes = (t["ticks"] + (s.tick_no if working else 0)) * DT / 60.0
        demos = t["human"] + t["robot"]
        hz = 0.0
        if len(self.walls) > 1 and self.walls[-1] > self.walls[0]:
            hz = (len(self.walls) - 1) / (self.walls[-1] - self.walls[0])
        return {
            "mode": self.mode,
            "status": status,
            "operator": {"left": {"connected": False, "hz": 0.0}, "right": {"connected": False, "hz": 0.0},
                         "calibrated": False, "stand_in": True, "hands": None},
            "episode": {"index": t["episodes"] + 1, "parts": parts, "sorted": int(self.sorted[0]),
                        "elapsed_s": round(s.tick_no * DT, 1), "step": self.op.label},
            "echo": {"robots": F, "done": int(s.echo.done.sum()) if F else 0,
                     "ok": int((self.states[1:] == SUCCESS).sum()), "failed": int((self.states[1:] == FAILED).sum()),
                     "lag_s": round(float(lag.mean()), 2) if F else 0.0,
                     "lag_max_s": round(float(lag.max()), 2) if F else 0.0},
            "envs": {"lag_s": [0.0] + [round(float(v), 2) for v in lag], "sorted": [int(v) for v in self.sorted],
                     "clipped": [int(v) for v in s.clipped]},
            "totals": {"episodes": t["episodes"], "human_demos": t["human"], "robot_demos": t["robot"],
                       "amplification": round(demos / t["human"], 1) if t["human"] else 0.0,
                       "yield": round(t["robot"] / t["attempts"], 4) if t["attempts"] else 0.0,
                       "demos_per_min": round(demos / minutes, 1) if minutes > 0.05 else 0.0,
                       "clipped_steps": t["clipped"], "minutes": round(minutes, 2)},
            "history": list(self.history),
            "dataset": {"episodes": self.rec.demos, "frames": self.rec.frames,
                        "path": str(self.rec.root) if self.rec.root else None, "recording": self.rec.enabled},
            "policy": {"state": "none", "progress": 0.0, "loss": None, "success": None, "curve": []},
            "perf": {"realtime": round(hz / CONTROL_HZ, 2), "tick_ms": round(float(np.mean(self.tick_ms)), 1)
                     if self.tick_ms else 0.0, "hz": CONTROL_HZ},
        }

    def run(self) -> None:
        self.viewer.start()
        print(f"Kaiku: {self.viewer.url}  ({self.n} robots)", flush=True)
        nxt = time.perf_counter()
        try:
            while not self.stop:
                self._commands()
                if not self.paused:
                    self._step()
                self.hud_tick += 1
                hud = self.hud() if self.hud_tick % HUD_EVERY == 0 else None
                self.viewer.publish(self.s.sim.qpos(), self.states, hud)
                nxt += DT
                wait = nxt - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    nxt = time.perf_counter()                  # running behind: never try to catch up
        finally:
            self.viewer.stop()
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.s.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48, help="robots, the operator's included")
    ap.add_argument("--port", type=int, default=8651)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", default="auto", help="auto, 3, 4 or 5 parts per scene")
    ap.add_argument("--record", default=None, help="folder to save the demonstrations in")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    r = Runner(a.n, a.port, a.host, a.seed, a.parts if a.parts == "auto" else int(a.parts), a.record,
               threads=a.threads)

    def quit_(*_):
        r.stop = True

    signal.signal(signal.SIGINT, quit_)
    signal.signal(signal.SIGTERM, quit_)
    r.run()


if __name__ == "__main__":
    main()
