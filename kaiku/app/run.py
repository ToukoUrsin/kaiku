"""The live demo: one operator, many robots, in the browser.

    python -m kaiku.app.run --n 48 --port 8651 [--parts auto|3|4|5] [--record DIR] [--seed 0]
                            [--gloves HOST[:PORT]] [--mode gloves|autopilot|policy]
                            [--policy FILE] [--train local|modal]

Without ``--gloves`` the operator is the stand-in: a scripted person wearing emulated Smartgloves
II, sent through the same RGMP socket, hand solver, calibration and retargeting as real gloves.
``--mode autopilot`` skips the gloves and commands the lead robot directly. In ``policy`` mode
nobody drives: a policy trained on the demonstrations gives every robot a scene of its own.

One loop at the challenge's 30 Hz: the operator (gloves, or the stand-in) moves the lead robot,
the echo carries the motion to every other robot's own scene, the safety filter and the physics
judge all of them, the recorder keeps the clean demonstrations, the browser shows it live.
"""
from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from kaiku.app.autopilot import Autopilot
from kaiku.app.sampler import Sampler
from kaiku.app.session import FAILED, IDLE, LEAD, RUNNING, SUCCESS, Session
from kaiku.app.trainer import Trainer
from kaiku.data.recorder import Recorder
from kaiku.sim.fetch import ensure
from kaiku.sim.twin import CONTROL_HZ
from kaiku.teleop.operator import GloveOperator
from kaiku.web.server import ViewerServer

DT = 1.0 / CONTROL_HZ
HUD_EVERY = 3                 # ticks between HUD updates (10 Hz)
MODES = ("gloves", "autopilot", "policy")
HERE = Path(__file__).resolve().parents[1]
PRETRAINED = HERE / "policy" / "pretrained" / "kaiku.pt"
STUDY = HERE / "policy" / "pretrained" / "study.json"


class Runner:
    def __init__(self, n: int = 48, port: int = 8651, host: str = "0.0.0.0", seed: int | None = None,
                 parts: str | int = "auto",
                 record: Path | str | None = None, hold_s: float = 1.6, lead_timeout_s: float = 90.0,
                 threads: int | None = None, mode: str = "gloves", gloves: str | None = None,
                 policy: Path | str | None = None, train: str = "local", train_steps: int = 30000,
                 policy_timeout_s: float = 80.0, stall_s: float = 15.0):
        if seed is None:                                       # other scenes every time it is started
            seed = int(np.random.SeedSequence().generate_state(1)[0] % 2 ** 30)
        self.s = Session(n, seed=seed, threads=threads)
        self.n = n
        self.parts = parts
        self.hold_s, self.lead_timeout = hold_s, int(lead_timeout_s * CONTROL_HZ)
        self.sampler = Sampler(n, seed + 1, self.s.spread, self.s.vary_shape)
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="kaiku-load")
        self.viewer = ViewerServer(self.s.twin.model, n, host=host, port=port, title="Kaiku")
        self.commands: queue.Queue = queue.Queue()
        self.viewer.on_message(self.commands.put)
        self.gloves = gloves                                   # HOST[:PORT] of a Rokoko driver, or None: the stand-in
        self.ops: dict = {}
        self.mode = mode if mode in ("gloves", "autopilot") else "gloves"
        self.want_mode = mode if mode in MODES else "gloves"
        self.op = self._operator(self.mode)
        self.rec = Recorder(self.s, record, operator=self._who())
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
        self.next = self.sampler.scene(self.parts)
        # The policy: trained here (the Train button) or handed in, and what it has done on its own.
        self.policy = self.pilot = self.next_free = self.loading = None
        self.plain: dict = {}
        self.policy_timeout, self.stall = int(policy_timeout_s * CONTROL_HZ), int(stall_s * CONTROL_HZ)
        self.want = np.zeros(n, int)
        self.last_gain = 0
        self.pol = {"rounds": 0, "scenes": 0, "ok": 0, "parts": 0, "sorted": 0, "clipped": 0}
        self.trained: dict | None = None                       # what the last press of Train came to
        self.study = json.loads(STUDY.read_text()) if STUDY.exists() else {}
        out = Path(record) / "policies" if record else None
        self.trainer = Trainer(record, out or "policies", train, train_steps)
        policy = policy or (PRETRAINED if PRETRAINED.exists() else None)
        if policy:
            self.loading = self.pool.submit(self._load, Path(policy))
        elif self.want_mode == "policy":
            self.want_mode = self.mode

    # ------------------------------------------------------------------ operators
    def _operator(self, mode: str):
        if mode not in self.ops:
            if mode == "autopilot":
                self.ops[mode] = Autopilot(self.s)
            else:
                host, _, port = (self.gloves or "").partition(":")
                self.ops[mode] = GloveOperator(self.s, host or None, int(port) if port else None)
        return self.ops[mode]

    def _who(self) -> str:
        if self.mode == "autopilot":
            return "autopilot"
        return "smartgloves" if self.gloves else "stand-in (emulated smartgloves)"

    # ------------------------------------------------------------------ scenes
    def _load(self, path: Path):
        from kaiku.policy.model import Policy             # torch: only when there is a policy to run
        from kaiku.policy.rollout import Pilot
        policy = Policy.load(path)
        plain = path.with_name("plain.pt")       # trained beside it on the operator's demonstrations alone
        other = Policy.load(plain).info if plain.exists() else {"success": policy.info.get("plain_success")}
        return policy, Pilot(self.s, policy), other

    def _ready(self) -> bool:
        """The next scene has been drawn (and, for the policy, the policy is in)."""
        if self.want_mode == "policy":
            return self.pilot is not None and self.next_free is not None and self.next_free.done()
        return self.next.done()

    def _begin_episode(self) -> None:
        if self.want_mode != self.mode:
            self.mode = self.want_mode
            if self.mode != "policy":
                self.op = self._operator(self.mode)
                self.rec.operator = self._who()
        if self.mode == "policy":
            layouts = self.next_free.result()
            self.next_free = self.sampler.free(self.parts)
            self.s.free_episode(layouts)
            self.pilot.begin(layouts)
            self.want = self.pilot.present.sum(axis=1)
            self.states[:] = RUNNING
            self.sorted[:] = 0
            self.last_gain = 0
            self.phase = "run"
            return
        lead, followers = self.next.result()
        self.next = self.sampler.scene(self.parts)
        self.s.new_episode(lead, followers)
        self.want[:] = len(lead.placements)
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

    def _end_policy_episode(self, count: bool = True) -> None:
        s = self.s
        r = s.result()
        clean = r.success & ~r.aborted & (r.clipped == 0)
        self.states[:] = np.where(clean, SUCCESS, FAILED)
        self.sorted = s.progress()
        if count:
            p = self.pol
            p["rounds"] += 1
            p["scenes"] += self.n
            p["ok"] += int(clean.sum())
            p["parts"] += int(self.want.sum())
            p["sorted"] += int(self.sorted.sum())
            p["clipped"] += int(r.clipped.sum())
            self.history.append({"episode": p["rounds"], "parts": int(self.want.sum()), "ok": int(clean.sum()),
                                 "robots": self.n, "lead_ok": None, "saved": 0, "policy": True,
                                 "seconds": round(s.tick_no * DT, 1)})
        self.phase = "hold"
        self.hold_until = time.perf_counter() + self.hold_s + 1.0

    def _policy_step(self) -> None:
        s = self.s
        obs = s.observe()
        s.drive(*self.pilot.step(obs))
        if s.tick_no % 6:
            return
        got = s.progress()
        if (got > self.sorted).any():
            self.last_gain = s.tick_no
        self.sorted = got
        done = got >= self.want
        self.states[:] = np.where(s.safety.aborted, FAILED, np.where(done, SUCCESS, RUNNING))
        home = np.linalg.norm(obs.pinch - s.home_pinch, axis=-1).max(axis=1) < 0.03
        # Over when everybody has sorted everything and is back home, or when those still at it
        # have not sorted a part for a while.
        if (done & home).all() or s.tick_no - self.last_gain > self.stall or s.tick_no >= self.policy_timeout:
            self._end_policy_episode()

    def _train(self) -> None:
        if self.rec.root is None:
            self.toast("Start with --record DIR: training uses the demonstrations saved there.", "warn")
        elif self.trainer.busy:
            self.toast("Already training.", "info")
        elif not self.rec.demos:
            self.toast("Nothing to train on yet. Let a few scenes finish first.", "warn")
        elif self.trainer.start():
            self.toast(f"Training two policies on {self.rec.demos} demonstrations: one with the echoes, "
                       "one with the operator's own alone.", "info")

    def _policy_news(self) -> None:
        """Pick up a policy that has finished loading, and a training run that has ended."""
        if self.loading is not None and self.loading.done():
            try:
                self.policy, self.pilot, self.plain = self.loading.result()
                if self.next_free is None:
                    self.next_free = self.sampler.free(self.parts)
            except Exception as e:                            # a file from another version, a missing torch
                self.toast(f"The policy could not be loaded: {e}", "warn")
                if self.want_mode == "policy" and self.mode != "policy":
                    self.want_mode = self.mode
            self.loading = None
        st = self.trainer.status()
        if self.trainer.fresh:
            self.trainer.fresh = False
            if st["state"] == "done":
                k, p = st["jobs"]["kaiku"], st["jobs"]["plain"]
                new, old = k.get("success", 0.0), (self.policy.info.get("success") or 0.0) if self.policy else -1.0
                # The one that sorts more drives. A training run of a few minutes can come out
                # worse than the policy that was already here; then that one stays.
                keep = new + 0.02 >= old
                self.trained = {"kaiku": new, "plain": p.get("success", 0.0), "humans": k.get("humans"),
                                "demos": k.get("demos"), "scenes": k.get("scenes"), "in_use": keep}
                text = (f"Trained. With the echoes it sorts {new:.0%} of new scenes, "
                        f"with the operator's demonstrations alone {p.get('success', 0.0):.0%}.")
                if keep:
                    self.loading = self.pool.submit(self._load, self.trainer.out / "kaiku.pt")
                else:
                    text += f" The policy already here sorts {old:.0%}, so it keeps driving."
                self.toast(text, "info")
            else:
                self.toast(st["error"] or "Training failed.", "warn")

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
                    self._end_policy_episode(False) if self.mode == "policy" else self._end_episode(record=False)
                self.hold_until = 0.0
            elif name == "parts":
                self.parts = value if value == "auto" else int(value)
                self.next = self.sampler.scene(self.parts)       # the scenes already drawn no longer fit
                if self.next_free is not None:
                    self.next_free = self.sampler.free(self.parts)
            elif name == "record":
                self.rec.enabled = bool(value) and self.rec.root is not None
                if value and self.rec.root is None:
                    self.toast("Start with --record DIR to save demonstrations.", "warn")
            elif name == "mode":
                if value == "policy" and self.policy is None and self.loading is None:
                    self.toast("No policy yet. Train one in the Policy tab first.", "warn")
                elif value in MODES:
                    self.want_mode = value
                    if "policy" in (value, self.mode) and value != self.mode:
                        # Nobody waits a whole scene to see the robots work alone, or to take them back.
                        if self.phase == "run":
                            self._end_policy_episode(False) if self.mode == "policy" else self._end_episode(False)
                        self.hold_until = 0.0
                    elif value != self.mode:
                        self.toast(f"{str(value).capitalize()} takes over at the next scene.", "info")
                else:
                    self.toast(f"{str(value).capitalize()} mode is not connected in this build.", "warn")
            elif name == "calibrate":
                if isinstance(self.op, GloveOperator) and not self.op.person:
                    ok = self.op.calibrate()
                    self.toast("Calibrated. The robot follows your hands." if ok else
                               "Hold both hands flat over the Coil Pro, then calibrate again.", "info" if ok else "warn")
                else:
                    self.toast("The stand-in operator calibrates itself. Connect gloves to calibrate yours.", "info")
            elif name == "train":
                self._train()

    def toast(self, text: str, level: str = "info") -> None:
        self.viewer.send_event({"type": "toast", "text": text, "level": level})

    def _step(self) -> None:
        s = self.s
        if self.phase == "hold":
            if time.perf_counter() >= self.hold_until and self._ready():
                self._begin_episode()
            return
        t0 = time.perf_counter()
        if self.mode == "policy":
            self._policy_step()
            self.tick_ms.append(1000.0 * (time.perf_counter() - t0))
            self.walls.append(time.perf_counter())
            return
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
        alone = self.mode == "policy"
        F = self.n - 1
        parts = len(s.layouts[0].placements)
        lag = s.echo.lag if F and not alone else np.zeros(F)
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
            "operator": self.op.hud() if isinstance(self.op, GloveOperator) and not alone else
            {"left": {"connected": False, "hz": 0.0}, "right": {"connected": False, "hz": 0.0},
             "calibrated": False, "stand_in": not alone, "source": "policy" if alone else "autopilot", "hands": None},
            "episode": {"index": self.pol["rounds"] + 1, "parts": int(self.want.sum()), "sorted": int(self.sorted.sum()),
                        "elapsed_s": round(s.tick_no * DT, 1), "step": "Policy"} if alone else
            {"index": t["episodes"] + 1, "parts": parts, "sorted": int(self.sorted[0]),
             "elapsed_s": round(s.tick_no * DT, 1), "step": self.op.label},
            "echo": {"robots": self.n, "done": int((self.states != RUNNING).sum()),
                     "ok": int((self.states == SUCCESS).sum()), "failed": int((self.states == FAILED).sum()),
                     "lag_s": 0.0, "lag_max_s": 0.0} if alone else
            {"robots": F, "done": int(s.echo.done.sum()) if F else 0,
                     "ok": int((self.states[1:] == SUCCESS).sum()), "failed": int((self.states[1:] == FAILED).sum()),
                     "lag_s": round(float(lag.mean()), 2) if F else 0.0,
                     "lag_max_s": round(float(lag.max()), 2) if F else 0.0},
            "envs": {"lag_s": [0.0] + [round(float(v), 2) for v in lag], "sorted": [int(v) for v in self.sorted],
                     "parts": [int(v) for v in self.want], "clipped": [int(v) for v in s.clipped]},
            "totals": {"episodes": t["episodes"], "human_demos": t["human"], "robot_demos": t["robot"],
                       "amplification": round(demos / t["human"], 1) if t["human"] else 0.0,
                       "yield": round(t["robot"] / t["attempts"], 4) if t["attempts"] else 0.0,
                       "demos_per_min": round(demos / minutes, 1) if minutes > 0.05 else 0.0,
                       "clipped_steps": t["clipped"], "minutes": round(minutes, 2)},
            "history": list(self.history),
            "dataset": {"episodes": self.rec.demos, "frames": self.rec.frames,
                        "path": str(self.rec.root) if self.rec.root else None, "recording": self.rec.enabled},
            "policy": self._policy_hud(),
            "perf": {"realtime": round(hz / CONTROL_HZ, 2), "tick_ms": round(float(np.mean(self.tick_ms)), 1)
                     if self.tick_ms else 0.0, "hz": CONTROL_HZ},
        }

    def _policy_hud(self) -> dict:
        st, p = self.trainer.status(), self.pol
        info = self.policy.info if self.policy is not None else {}
        out = {"state": "none", "progress": 0.0, "stage": None, "loss": info.get("loss"),
               "success": info.get("success"), "plain": self.plain.get("success"),
               "humans": info.get("humans"), "demos": info.get("demos"), "tested_on": info.get("scenes"),
               "curve": self.study.get("curve", []), "study": self.study.get("about"), "trained": self.trained,
               "error": st.get("error") if st["state"] == "failed" else None,
               "alone": {"rounds": p["rounds"], "scenes": p["scenes"], "ok": p["ok"], "parts": p["parts"],
                         "sorted": p["sorted"], "clipped": p["clipped"]}}
        if st["state"] == "training":
            out.update(state="training", progress=st["progress"], stage=st["stage"], loss=st["loss"])
        elif self.policy is not None:
            out["state"] = "running" if self.mode == "policy" else "ready"
        return out

    def run(self) -> None:
        # The loop gives the interpreter up many times a tick (numpy, the physics). Whoever else
        # holds it then (the web server, the glove socket) hands it back within a millisecond, not five.
        sys.setswitchinterval(0.001)
        self.viewer.start()
        print(f"Kaiku: {self.viewer.url}  ({self.n} robots)", flush=True)
        nxt = time.perf_counter()
        try:
            while not self.stop:
                self._commands()
                self._policy_news()
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
            self.trainer.close()
            self.sampler.close()
            self.pool.shutdown(wait=False, cancel_futures=True)
            for op in self.ops.values():
                if hasattr(op, "close"):
                    op.close()
            self.s.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48, help="robots, the operator's included")
    ap.add_argument("--port", type=int, default=8651)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--seed", type=int, default=None, help="for the same scenes again (default: new ones every run)")
    ap.add_argument("--parts", default="auto", help="auto, 3, 4 or 5 parts per scene")
    ap.add_argument("--record", default=None, help="folder to save the demonstrations in")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--mode", default="gloves", choices=MODES)
    ap.add_argument("--policy", default=None, help="a trained policy (default: the one shipped with Kaiku)")
    ap.add_argument("--train", default="local", choices=("local", "modal"),
                    help="where the Train button trains: this machine, or a rented GPU through Modal")
    ap.add_argument("--train-steps", type=int, default=30000)
    ap.add_argument("--gloves", default=None, metavar="HOST[:PORT]",
                    help="RGMP address of the Rokoko driver; without it the stand-in operator drives")
    a = ap.parse_args(argv)
    ensure()                                                   # the robot model, fetched on the first run
    r = Runner(a.n, a.port, a.host, a.seed, a.parts if a.parts == "auto" else int(a.parts), a.record,
               threads=a.threads, mode=a.mode, gloves=a.gloves, policy=a.policy, train=a.train,
               train_steps=a.train_steps)

    def quit_(*_):
        r.stop = True

    signal.signal(signal.SIGINT, quit_)
    signal.signal(signal.SIGTERM, quit_)
    r.run()


if __name__ == "__main__":
    main()
