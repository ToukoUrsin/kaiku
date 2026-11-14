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
import tempfile
import threading
import time
import socket
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from kaiku.app.autopilot import Autopilot
from kaiku.app.sampler import Sampler
from kaiku.app.session import FAILED, IDLE, LEAD, RUNNING, SUCCESS, Session
from kaiku.app.trainer import Trainer
from kaiku.app.visitor import VisitorOperator
from kaiku.data.recorder import Recorder
from kaiku.sim.fetch import ensure
from kaiku.sim.twin import CONTROL_HZ
from kaiku.teleop.fingers import GloveFingers
from kaiku.teleop.operator import GloveOperator, ReplayOperator
from kaiku.fanout.echo import sample_followers
from kaiku.web.server import ViewerServer

DT = 1.0 / CONTROL_HZ
HUD_EVERY = 3                 # ticks between HUD updates (10 Hz)
PAUSE_S = 20.0                # how long a pause lasts on a session that everybody shares
MODES = ("gloves", "autopilot", "policy", "visitor")
TURN_S, IDLE_S = 240.0, 45.0                      # a visitor's turn at the robots, and how long it may sit untouched
HELD_BY_COMPARISON = ("pause", "next_scene", "parts", "mode", "drive_start")   # what waits while a comparison runs
COPY_EVERY_S = 30.0                               # everybody shares the page: the scene without the echo is not the only one shown
HERE = Path(__file__).resolve().parents[1]
PRETRAINED = HERE / "policy" / "pretrained" / "kaiku.pt"
STUDY = HERE / "policy" / "pretrained" / "study.json"
BASELINES = HERE / "policy" / "pretrained" / "baselines.json"
BASELINE_ARMS = ("plain", "mimicgen", "kaiku")      # what the page compares: the 16 alone, a replay's survivors, the echoes


def baselines(path: Path = BASELINES) -> dict | None:
    """The measured comparison with a MimicGen-style replay, cut down to what the page shows.
    Read from the file the measurement wrote: no number of it lives in the page."""
    try:
        b = json.loads(Path(path).read_text())
        policy, arms = b["policy"], {}
        for name in BASELINE_ARMS:
            a = policy["arms"][name]
            arms[name] = {"clean": int(a["clean"]), "scenes": int(a["scenes"]), "yield": round(float(a["yield"]), 4),
                          "interval95": [round(float(v), 4) for v in a["interval95"]],
                          "low": round(float(a["min_yield"]), 4), "high": round(float(a["max_yield"]), 4)}
        made = {name: {"clean": int(g["clean"]), "attempts": int(g["attempts"])}
                for name, g in b.get("generation", {}).items() if name in ("mimicgen", "kaiku")}
        return {"arms": arms, "tables": int(policy["heldout_scenes"]), "runs": int(policy["seeds_per_arm"]),
                "demos": int(b["generation"]["kaiku"]["source_scenes"]), "made": made}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _has(module: str) -> bool:
    from importlib.util import find_spec
    return find_spec(module) is not None


class Runner:
    def __init__(self, n: int = 48, port: int = 8651, host: str = "0.0.0.0", seed: int | None = None,
                 parts: str | int = "auto",
                 record: Path | str | None = None, hold_s: float = 1.6, lead_timeout_s: float = 90.0,
                 threads: int | None = None, mode: str = "gloves", gloves: str | None = None,
                 hand_solver: str | None = None, glove_fingers: str | None = None,
                 policy: Path | str | None = None, train: str = "local", train_steps: int = 30000,
                 policy_timeout_s: float = 80.0, stall_s: float = 15.0, shared: bool = False,
                 train_about: float | None = None, target_scenes: int | None = None):
        if seed is None:                                       # other scenes every time it is started
            seed = int(np.random.SeedSequence().generate_state(1)[0] % 2 ** 30)
        self.s = Session(n, seed=seed, threads=threads)
        self.n, self.seed = n, seed
        if glove_fingers and parts == "auto":
            parts = 3                                          # one hand, one part at a time: short scenes
        self.target_scenes = target_scenes
        self.parts = parts
        self.hold_s, self.lead_timeout = hold_s, int(lead_timeout_s * CONTROL_HZ)
        self.sampler = Sampler(n, seed + 1, self.s.spread, self.s.vary_shape)
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="kaiku-load")
        self.viewer = ViewerServer(self.s.twin.model, n, host=host, port=port, title="Kaiku")
        self.commands: queue.Queue = queue.Queue()
        self.viewer.on_message(self.commands.put)
        self.gloves = gloves                                   # HOST[:PORT] of a Rokoko driver, or None: the stand-in
        self.session_dir = None
        self.hand_solver = hand_solver                         # HOST:PORT of rkk-hand-solver, "off", or None: the default
        self.glove_fingers = glove_fingers                     # HOST[:PORT] of a driver with one glove on it: a visitor's fingers
        self.ops: dict = {}
        self.visitor: VisitorOperator | None = None
        self.driver_client: int | None = None
        self.turn_started = 0.0
        self.last_visitor_command = 0.0
        self._visitor_expired = False
        self.copy_next = False                                 # the next scene is the comparison: the echo is off
        self.again = None                                      # its tables, for the same scene once more with the echo
        self.replaying = False                                 # this scene is that second half
        self.copy_at = 0.0                                     # when the last comparison ended
        self.compare: dict | None = None                       # what it came to: without the echo, and with it
        self.scene_no = 1                                      # the number of the scene on the tables now
        self.scene_end: str | None = None                      # how it ended: "sorted", "cut" (nobody's result), or not yet
        self.lead = None                                       # kaiku.app.g1.Lead while a challenge runner follows
        self.mode = mode if mode in ("gloves", "autopilot") else "gloves"
        self.want_mode = mode if mode in MODES else "gloves"
        self.op = self._operator(self.mode)
        self.rec = Recorder(self.s, record, operator=self._who())
        if self.rec.root is not None:
            meta = self.rec.root / "session.json"
            if not meta.exists():
                source = "smartgloves" if self.gloves else "emulated"
                meta.write_text(json.dumps({"format": 1, "source": source, "start_time": time.time(),
                                            "host": socket.gethostname()}, indent=2) + "\n")
        self.paused = False
        self.shared = shared                                   # one session for everybody who opens the page
        self._zip, self._zip_lock = None, threading.Lock()     # (scene file, zip) of the last download built
        self.download = self.rec.root is not None and _has("pyarrow")
        if self.download:
            self.viewer.dataset = self._dataset_zip
        self.resume_at: float | None = None
        self.stop = False
        self.phase = "run"                                     # run -> hold -> run ...
        self.hold_until = 0.0
        if self.glove_fingers and self.want_mode == "visitor":
            self.phase = "hold"                                # a glove here drives from the first scene on:
                                                               # no stand-in's scene is recorded ahead of the person's
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
        self.baselines = baselines()
        self.policy_path: Path | None = None                   # the file the policy that drives came from
        out = Path(record) / "policies" if record else None
        self.trainer = Trainer(record, out or "policies", train, train_steps, about_min=train_about)
        try:                                                   # the last result outlives a restart
            self.trained = json.loads((self.trainer.out / "last.json").read_text())
        except (OSError, ValueError):
            pass
        mine = self.trainer.out / "kaiku.pt"                   # the one Train made here, if it was the one driving
        if not policy and self.trained and self.trained.get("in_use") and mine.exists():
            policy = mine
        policy = policy or (PRETRAINED if PRETRAINED.exists() else None)
        if policy:
            self.loading = self.pool.submit(self._load, Path(policy))
        elif self.want_mode == "policy":
            self.want_mode = self.mode
        self.home_mode = self.want_mode

    # ------------------------------------------------------------------ operators
    def _operator(self, mode: str):
        if mode not in self.ops:
            if mode == "autopilot":
                self.ops[mode] = Autopilot(self.s)
            elif mode == "visitor":
                glove = None
                if self.glove_fingers:                         # one Smartglove on the visitor's hand
                    host, _, port = self.glove_fingers.partition(":")
                    glove = GloveFingers(host or "127.0.0.1", int(port) if port else 12276, solver=self.hand_solver)
                self.ops[mode] = VisitorOperator(self.s, glove=glove)
                self.visitor = self.ops[mode]
            else:
                if self.session_dir:
                    self.ops[mode] = ReplayOperator(self.s, self.session_dir)
                else:
                    host, _, port = (self.gloves or "").partition(":")
                    self.ops[mode] = GloveOperator(self.s, host or None, int(port) if port else None,
                                                   solver=self.hand_solver)
        return self.ops[mode]

    def _who(self) -> str:
        if self.mode == "visitor":
            return self.visitor.who() if self.visitor is not None else "visitor"
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
        return policy, Pilot(self.s, policy), other, path

    def _ready(self) -> bool:
        """The next scene has been drawn (and, for the policy, the policy is in)."""
        if self.want_mode == "policy":
            return self.pilot is not None and self.next_free is not None and self.next_free.done()
        return self.next.done() if not isinstance(self.op, ReplayOperator) else bool(self.op.rows)

    def _begin_episode(self) -> None:
        self._visitor_expired = False
        self.scene_end = None
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
            self.scene_no = self.pol["rounds"] + 1
            self.phase = "run"
            return
        if self.mode == "visitor":                             # a visitor's turn is not the comparison
            self.copy_next, self.again = False, None
        self.replaying = self.again is not None
        if self.replaying:                                     # the same tables as the scene without the echo
            (lead, followers), copy, self.again = self.again, False, None
        elif isinstance(self.op, ReplayOperator):              # a recorded person leads: their scene, new tables around it
            lead = self.op.scene()
            followers = sample_followers(self.s.scene, lead, self.s.rng, self.n - 1, self.s.spread,
                                         self.s.vary_shape, self.s.reach)
            copy, self.copy_next = self.copy_next, False
            self.again = (lead, followers) if copy else None
        else:
            lead, followers = self.next.result()
            self.next = self.sampler.scene(self.parts)
            copy, self.copy_next = self.copy_next, False
            self.again = (lead, followers) if copy else None
        self.s.new_episode(lead, followers, copy=copy)
        if copy:
            self.toast(f"Kaiku is off for this scene. All {self.n - 1} robots get the same joint commands as "
                       "the operator's robot.", "warn")
        self.want[:] = len(lead.placements)
        self.op.begin()
        self.rec.begin()
        self.states[:] = IDLE
        self.states[0] = LEAD
        self.sorted[:] = 0
        self.done_ticks[:] = 0
        # The scene's number is given when it starts and kept until the next one starts: the count of
        # finished scenes moves on the moment one ends, while its result is still what the page shows.
        self.scene_no = self.tot["episodes"] + 1
        self.phase = "run"

    def _end_visitor_turn(self, reason: str = "") -> None:
        if self.driver_client is None:
            return
        self.viewer.send_event({"type": "toast", "text": "Your turn has ended. The stand-in is driving again.", "level": "info"})
        self.driver_client = None
        self.turn_started = self.last_visitor_command = 0.0
        self.want_mode = "visitor" if self.glove_fingers else "gloves"   # a glove on this machine keeps the robots
        if self.phase == "run":
            self._end_episode(record=False)
        self.hold_until = 0.0

    def _start_visitor(self, client: int) -> None:
        if self.driver_client is not None:
            self.toast("A visitor is driving. Please wait for the next turn.", "info")
            return
        self.driver_client = int(client)
        self.turn_started = self.last_visitor_command = time.monotonic()
        self.want_mode = "visitor"
        if self.phase == "run":
            self._end_episode(record=False)
        self.hold_until = 0.0
        self.toast("Your turn has started. Press on a part, drag it to the bin of its colour, let go.", "info")

    def _reset(self) -> None:
        """Back to the start, for filming: the first scenes again, this session's counts at zero."""
        if self.shared:
            self.toast("Everybody who opens this page shares it, so it is not started over.", "info")
            return
        if self.driver_client is not None:
            self._end_visitor_turn("reset")
        self.sampler.close()
        self.sampler = Sampler(self.n, self.seed + 1, self.s.spread, self.s.vary_shape)
        self.parts = "auto"
        self.next = self.sampler.scene(self.parts)
        if self.next_free is not None:
            self.next_free = self.sampler.free(self.parts)
        self.want_mode = self.home_mode if self.home_mode != "policy" or self.policy is not None else "gloves"
        self.paused, self.resume_at = False, None
        self.copy_next, self.again, self.replaying, self.compare, self.copy_at = False, None, False, None, 0.0
        self.tot = {k: 0 for k in self.tot}
        self.pol = {k: 0 for k in self.pol}
        self.history.clear()
        if self.phase == "run":
            self._end_policy_episode(False) if self.mode == "policy" else self._end_episode(record=False)
        self.hold_until = 0.0
        self.toast("Started over.", "info")

    @property
    def comparing(self) -> bool:
        """A comparison is on its way: asked for, running without the echo, or running again with it."""
        return bool(self.copy_next or self.s.copy or self.again is not None or (self.replaying and self.phase == "run"))

    def _copy_scene(self) -> None:
        """Somebody asked to see it without the echo: the next scene, every robot is sent what the lead is sent."""
        if self.mode not in ("gloves", "autopilot") or self.want_mode != self.mode:
            self.toast("That comparison runs while an operator drives. Switch to Operator first.", "info")
        elif self.s.copy or self.copy_next or self.again is not None or self.replaying:
            self.toast("That comparison is running now.", "info")
        elif time.monotonic() - self.copy_at < COPY_EVERY_S:
            self.toast("That was just shown. It can run again in half a minute.", "info")
        else:
            self.copy_next = True
            # A scene with every part it can have: with one part alone, half the copies get lucky.
            self.next = self.sampler.scene(3 if self.parts == "auto" else self.parts)
            if self.phase == "run":                            # nobody waits a whole scene for it
                self._end_episode(record=False)
            self.hold_until = 0.0

    def _end_copy(self, r) -> None:
        """The scene without the echo is over: say what came of it. It is nobody's demonstration."""
        ok = int((r.success & ~r.aborted)[1:].sum())
        self.compare = {"copy": ok, "robots": self.n - 1, "echo": None, "parts": len(r.layouts[0].placements),
                        "at": round(time.time(), 1)}
        self.toast(f"Without Kaiku {ok} of {self.n - 1} robots sorted their table. Now the same tables with Kaiku.",
                   "info")
        self.copy_at = time.monotonic()
        self.rec.begin()
        self.scene_end = "cut"                                 # a comparison, not a scene of the session
        self.phase = "hold"
        self.hold_until = time.perf_counter() + self.hold_s + 3.0     # long enough to read the red tables

    def _end_episode(self, record: bool = True) -> None:
        s = self.s
        r = s.result()
        ok = r.success & ~r.aborted
        self.states[1:] = np.where(ok[1:], SUCCESS, FAILED)
        self.states[0] = LEAD
        self.sorted = s.progress()
        if s.copy and record:
            return self._end_copy(r)
        if not record:                                         # cut short by a visitor: nobody's result, nothing to count
            self.again, self.replaying = None, False
            self.rec.begin()
            self.states[1:] = IDLE
            self.scene_end = "cut"                             # the page must not say it was sorted
            self.phase = "hold"
            self.hold_until = time.perf_counter() + self.hold_s
            return
        if self.mode == "visitor":                             # what drove is known once it has driven
            self.rec.operator = self._who()
        kept = self.rec.end(r)
        clean = ok & (r.clipped == 0)
        if record:
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
            if self.target_scenes and t["human"] >= self.target_scenes:
                self.toast(f"You have {t['human']} of {self.target_scenes} clean scenes. You can stop with Ctrl-C.", "ok")
                self.stop = True
        self.scene_end = "sorted"
        self.phase = "hold"
        self.hold_until = time.perf_counter() + self.hold_s
        if self.replaying and self.compare is not None:        # the comparison's second half: the same tables, echoed
            c = self.compare
            c["echo"], c["done"] = int(ok[1:].sum()), round(time.time(), 1)
            self.toast(f"Same tables with Kaiku: {c['echo']} of {c['robots']} robots sorted theirs. "
                       f"Without: {c['copy']} of {c['robots']}.", "info")
            self.copy_at = time.monotonic()
            self.hold_until += 3.0

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
        self.scene_end = "sorted" if count else "cut"
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
            about = self.trainer.about_min
            self.toast(f"Training two policies on the last {self.trainer.last} demonstrations: one with their "
                       "echoes, one without."
                       + (f" It takes about {about:.0f} minutes on this machine." if about and about >= 1.5 else ""), "info")

    def _policy_news(self) -> None:
        """Pick up a policy that has finished loading, and a training run that has ended."""
        if self.loading is not None and self.loading.done():
            try:
                self.policy, self.pilot, self.plain, self.policy_path = self.loading.result()
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
                                "demos": k.get("demos"), "scenes": k.get("scenes"), "in_use": keep,
                                "minutes": round(st["took_s"] / 60.0, 1) if st.get("took_s") else None,
                                "at": round(time.time(), 1)}
                try:
                    (self.trainer.out / "last.json").write_text(json.dumps(self.trained))
                except OSError:
                    pass
                text = (f"Trained on the last {k.get('humans') or 0} demonstrations. With their echoes the policy "
                        f"sorts {new:.0%} of new scenes, without them {p.get('success', 0.0):.0%}.")
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
                if m.get("type") == "disconnect" and m.get("client") == self.driver_client:
                    self._end_visitor_turn("disconnect")
                continue
            name, value = m.get("name"), m.get("value")
            client = m.get("client")
            if self.comparing and name in HELD_BY_COMPARISON and not (name == "pause" and not value):
                # Everybody shares the page: one visitor's button must not end what another asked to see.
                self.toast("The comparison finishes first, in about a minute. Then that works again.", "info")
            elif name == "drive_start":
                self._start_visitor(int(client))
            elif name == "drive_end" and client == self.driver_client:
                self._end_visitor_turn("handback")
            elif name == "hand" and client == self.driver_client and self.mode == "visitor":
                was = (self.visitor.command.copy(), self.visitor.grip, self.visitor.own)
                self.visitor.set_command(value if isinstance(value, dict) else {})
                # An open page keeps saying where the hand is. Only a hand that does something keeps the turn.
                if self.visitor.grip != was[1] or self.visitor.own != was[2] \
                        or float(np.abs(self.visitor.command - was[0]).max()) > 1e-4:
                    self.last_visitor_command = time.monotonic()
            elif self.driver_client is not None and client != self.driver_client:
                continue                                       # a visitor has the robots: nobody else's buttons count
            elif name == "pause":
                self.paused = bool(value)
                # One page for everybody: a visitor who pauses and leaves must not leave it paused.
                self.resume_at = time.perf_counter() + PAUSE_S if self.paused and self.shared else None
                if self.resume_at:
                    self.toast(f"Paused for everyone. It carries on by itself in {PAUSE_S:.0f} seconds.", "info")
            elif name == "next_scene":
                if self.phase == "run":
                    self._end_policy_episode(False) if self.mode == "policy" else self._end_episode(record=False)
                self.hold_until = 0.0
            elif name == "parts":
                self.parts = value if value == "auto" else int(value)
                self.next = self.sampler.scene(self.parts)       # the scenes already drawn no longer fit
                if self.next_free is not None:
                    self.next_free = self.sampler.free(self.parts)
            elif name == "record" and self.shared:
                self.toast("Recording stays on here. Everybody who opens this page shares the session.", "info")
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
                        # Every visitor shares this session: the others see why the scene changed.
                        self.toast("Switched to Policy. Nobody is driving now." if value == "policy" else
                                   "Switched to Operator. One operator drives, the others repeat.", "info")
                    elif value != self.mode:
                        who = "The operator" if value == "gloves" else "The autopilot"
                        self.toast(f"{who} takes over at the next scene.", "info")
                else:
                    self.toast(f"{str(value).capitalize()} mode is not connected in this build.", "warn")
            elif name == "calibrate":
                if self.mode == "visitor" and self.visitor is not None and self.visitor.glove is not None:
                    self.visitor.recentre()
                    self.toast("Hold your gloved hand flat and still. Where you hold it is the middle of the table.", "info")
                elif isinstance(self.op, GloveOperator) and not self.op.person:
                    ok = self.op.calibrate()
                    self.toast("Calibrated. The robot follows your hands." if ok else
                               "Hold both hands flat over the Coil Pro, then calibrate again.", "info" if ok else "warn")
                else:
                    self.toast("The stand-in operator calibrates itself. Connect gloves to calibrate yours.", "info")
            elif name == "train":
                self._train()
            elif name == "copy_scene":
                self._copy_scene()
            elif name == "reset":
                self._reset()

    def _publish(self) -> None:
        if self.lead is not None:                              # the lead robot's command, for a real G1 to follow
            arm, hand = self.s.last_action
            self.lead.put(arm[0], hand[0])

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
            self._publish()
            self.tick_ms.append(1000.0 * (time.perf_counter() - t0))
            self.walls.append(time.perf_counter())
            return
        if self.driver_client is not None:
            now = time.monotonic()
            # a glove on this machine is one person's sitting, not a queue of visitors: it has no clock
            if not self.glove_fingers and (now - self.last_visitor_command > IDLE_S or now - self.turn_started > TURN_S):
                self._end_visitor_turn("timeout")
                return
        obs = s.observe()
        if isinstance(self.op, GloveOperator) and not self.op.person and not self.op.healthy:
            # A radio drop must never become a misleading demonstration. Keep the scene on screen
            # long enough for the banner to explain it, then discard it and draw a fresh table.
            reason = self.op.hud().get("message") or "The glove stream stopped."
            self.toast(f"Scene dropped: {reason}", "warn")
            self._end_episode(record=False)
            self.hold_until = time.perf_counter() + 1.0
            return
        cmd = self.op.update(obs)
        limit = int(150 * CONTROL_HZ) if self.mode == "visitor" else self.lead_timeout
        if self.op.done or s.tick_no >= limit:
            if self.mode == "visitor" and s.tick_no >= limit and not self.op.done:
                self._visitor_expired = True
            s.finish_lead()
        s.tick(cmd)
        if not s.copy:
            points = self.op.hand_points() if hasattr(self.op, "hand_points") else None
            self.rec.add(obs, *s.last_action, command=cmd, hands=points)
        self._publish()
        if s.tick_no % 6 == 0:
            self._refresh()
        self.tick_ms.append(1000.0 * (time.perf_counter() - t0))
        self.walls.append(time.perf_counter())
        if s.settled:
            self._end_episode(record=not self._visitor_expired)

    def _refresh(self) -> None:
        """Per-robot state for the viewer: who is working, who has finished, who has failed."""
        s = self.s
        self.sorted = s.progress()
        parts = len(s.layouts[0].placements)
        if self.n > 1 and s.copy:
            # No echo to ask: a robot whose table is behind the lead's for a while has missed its part.
            behind = self.sorted[1:] < self.sorted[0]
            self.done_ticks[1:] = np.where(behind, self.done_ticks[1:] + 6, 0)
            st = np.where(self.sorted[1:] >= parts, SUCCESS, RUNNING)
            st = np.where(behind & (self.done_ticks[1:] > 30), FAILED, st)
            self.states[1:] = np.where(s.safety.aborted[1:], FAILED, st)
        elif self.n > 1:
            done = s.echo.done
            self.done_ticks[1:] = np.where(done, self.done_ticks[1:] + 6, 0)
            started = s.echo.tau > 1.0
            st = np.where(started, RUNNING, IDLE)
            st = np.where(self.sorted[1:] >= parts, SUCCESS, st)
            st = np.where((self.sorted[1:] < parts) & (self.done_ticks[1:] > 45), FAILED, st)
            self.states[1:] = np.where(s.safety.aborted[1:], FAILED, st)
        self.states[0] = LEAD

    def _drive(self, operator: dict) -> str:
        """What moves the lead robot right now, as one word the page turns into its own. It is read from
        what is driving, never from what could be: a pointer is not a glove, a stand-in is not a person."""
        if self.mode == "policy":
            return "policy"
        if self.mode == "autopilot":
            return "autopilot"
        if self.mode == "visitor":
            fingers, position = operator.get("fingers"), operator.get("input")
            if fingers == "glove":
                return "glove-coil" if position == "coil" else f"glove-{'camera' if position == 'camera' else 'pointer'}"
            return "camera" if fingers == "camera" or position == "camera" else "pointer"
        return str(operator.get("source") or "stand-in")       # stand-in, gloves, emulated, recorded, emulated-recorded

    def hud(self) -> dict:
        s, t = self.s, self.tot
        alone = self.mode == "policy"
        F = self.n - 1
        parts = len(s.layouts[0].placements)
        lag = s.echo.lag if F and not alone and not s.copy else np.zeros(F)
        working = self.phase == "run"
        status = "paused" if self.paused else ("running" if working and not s.lead_done else "settling")
        minutes = (t["ticks"] + (s.tick_no if working else 0)) * DT / 60.0
        demos = t["human"] + t["robot"]
        hz = 0.0
        if len(self.walls) > 1 and self.walls[-1] > self.walls[0]:
            hz = (len(self.walls) - 1) / (self.walls[-1] - self.walls[0])
        operator = self.op.hud() if (isinstance(self.op, (GloveOperator, ReplayOperator)) or self.mode == "visitor") and not alone else \
            {"left": {"connected": False, "hz": 0.0}, "right": {"connected": False, "hz": 0.0},
             "calibrated": False, "stand_in": not alone, "source": "policy" if alone else "autopilot", "hands": None}
        operator["drive"] = self._drive(operator)
        if self.driver_client is not None:                     # whose turn it is, and how long it still lasts
            operator["driver"] = self.driver_client
            operator["starting"] = self.mode != "visitor"      # asked for, and the tables are still being laid
            if not self.glove_fingers:
                operator["turn_left_s"] = round(max(0.0, TURN_S - (time.monotonic() - self.turn_started)))
        return {
            "mode": self.mode,
            "status": status,
            "shared": self.shared,
            "operator": operator,
            # The comparison: is the echo off in this scene, may it be asked for, what the last one came to.
            "copy": {"on": bool(s.copy), "next": self.copy_next, "again": self.replaying, "over": self.phase == "hold",
                     "busy": self.comparing,                    # other buttons are held until it has finished
                     "free": self.mode in ("gloves", "autopilot") and self.want_mode == self.mode,
                     "last": self.compare and dict(self.compare, age_s=round(time.time() - self.compare.get(
                         "done", self.compare["at"])))},
            "episode": {"index": self.scene_no, "parts": int(self.want.sum()), "sorted": int(self.sorted.sum()),
                        "elapsed_s": round(s.tick_no * DT, 1), "step": "Policy", "end": self.scene_end} if alone else
            {"index": self.scene_no, "parts": parts, "sorted": int(self.sorted[0]),
             "elapsed_s": round(s.tick_no * DT, 1), "step": self.op.label, "target": self.target_scenes,
             "end": self.scene_end},
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
                       # Demonstrations saved for every one the operator performed, clean or not: never
                       # more than there are robots.
                       "amplification": round(demos / t["episodes"], 1) if t["episodes"] else 0.0,
                       "yield": round(t["robot"] / t["attempts"], 4) if t["attempts"] else 0.0,
                       "demos_per_min": round(demos / minutes, 1) if minutes > 0.05 else 0.0,
                       "clipped_steps": t["clipped"], "minutes": round(minutes, 2)},
            "history": list(self.history),
            "dataset": {"episodes": self.rec.demos, "frames": self.rec.frames,
                        "path": str(self.rec.root) if self.rec.root else None, "recording": self.rec.enabled,
                        "download": self.download and self.rec.demos > 0},
            "policy": self._policy_hud(),
            "perf": {"realtime": round(hz / CONTROL_HZ, 2), "tick_ms": round(float(np.mean(self.tick_ms)), 1)
                     if self.tick_ms else 0.0, "hz": CONTROL_HZ},
        }

    def _dataset_zip(self) -> tuple[Path, str]:
        """The last saved scene as LeRobot and GR00T, zipped; built once a scene, on the server's thread pool."""
        from kaiku.data.export import bundle, latest_rows
        with self._zip_lock:
            rows = latest_rows(self.rec.root)
            if not rows:
                raise FileNotFoundError("no scene saved yet")
            scene = Path(rows[-1]["file"]).stem
            name = f"kaiku-{scene}-{len(rows)}-demonstrations.zip"
            if self._zip is None or self._zip[0] != scene or not self._zip[1].exists():
                out = Path(tempfile.gettempdir()) / f"kaiku-download-{self.viewer.port}" / f"{scene}.zip"
                bundle(self.rec.root, out, rows=rows)
                if self._zip is not None and self._zip[1] != out:
                    self._zip[1].unlink(missing_ok=True)
                self._zip = (scene, out)
            return self._zip[1], name

    def _policy_hud(self) -> dict:
        st, p = self.trainer.status(), self.pol
        info = self.policy.info if self.policy is not None else {}
        out = {"state": "none", "progress": 0.0, "stage": None, "loss": info.get("loss"),
               "success": info.get("success"), "plain": self.plain.get("success"),
               "humans": info.get("humans"), "demos": info.get("demos"), "tested_on": info.get("scenes"),
               "curve": self.study.get("curve", []), "study": self.study.get("about"), "trained": self.trained,
               "baselines": self.baselines,
               # Which policy this is: the one shipped with Kaiku, or the one Train made in this session.
               "origin": None if self.policy is None else "session" if self.policy_path == self.trainer.out / "kaiku.pt"
               else "shipped" if self.policy_path == PRETRAINED else "file",
               "error": st.get("error") if st["state"] == "failed" else None,
               "alone": {"rounds": p["rounds"], "scenes": p["scenes"], "ok": p["ok"], "parts": p["parts"],
                         "sorted": p["sorted"], "clipped": p["clipped"]}}
        if self.trained:                                       # the page says a result out loud while it is news
            out["trained"] = dict(self.trained, age_s=round(time.time() - self.trained["at"], 1))
        out["about_min"] = self.trainer.about_min
        if st["state"] == "training":
            out.update(state="training", progress=st["progress"], stage=st["stage"], loss=st["loss"],
                       eta_s=st.get("eta_s"), last=self.trainer.last)
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
                if self.paused and self.resume_at and time.perf_counter() > self.resume_at:
                    self.paused, self.resume_at = False, None
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
            self.close()

    def close(self) -> None:
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
    ap.add_argument("--train-about", type=float, default=None, metavar="MINUTES",
                    help="how long the Train button takes on this machine, for the page to say")
    ap.add_argument("--shared", action="store_true",
                    help="a public page: a pause ends by itself and recording cannot be switched off")
    ap.add_argument("--gloves", default=None, metavar="HOST[:PORT]",
                    help="RGMP address of the Rokoko driver; without it the stand-in operator drives")
    ap.add_argument("--hand-solver", default=None, metavar="HOST:PORT|off",
                    help="where rkk-hand-solver serves solved hands (default: port 12277 on the gloves' host); "
                         "the fingers come from it while it runs, from kaiku's own solver otherwise")
    ap.add_argument("--one-glove", "--glove-fingers", dest="glove_fingers", default=None, metavar="HOST[:PORT]",
                    help="one Smartglove on whoever drives: its fingers drive the robot's. With a Coil Pro the "
                         "glove also says where the hand is; without one, the camera or the pointer does")
    ap.add_argument("--session", default=None, metavar="DIR", help="replay a recorded human session")
    ap.add_argument("--target", type=int, default=None, help="stop after this many clean lead scenes")
    ap.add_argument("--g1", type=int, default=None, metavar="PORT",
                    help="serve the lead robot's commands to the humanoid challenge's runner (its policy-server "
                         "protocol, port 8000 there), so a real G1 follows the operator; see kaiku/app/g1.py")
    a = ap.parse_args(argv)
    ensure()                                                   # the robot model, fetched on the first run
    r = Runner(a.n, a.port, a.host, a.seed, a.parts if a.parts == "auto" else int(a.parts), a.record,
               threads=a.threads, mode=a.mode, gloves=a.gloves, hand_solver=a.hand_solver,
               glove_fingers=a.glove_fingers, policy=a.policy, train=a.train,
               train_steps=a.train_steps, shared=a.shared, train_about=a.train_about, target_scenes=a.target)
    # A real glove session, once one exists (python -m kaiku.app.sitting puts it here), is what plays
    # when nobody drives: a person's own demonstrations instead of the scripted stand-in's.
    human = Path(__file__).resolve().parents[1] / "sessions" / "human"
    if a.session is None and not a.gloves and not a.glove_fingers and a.mode == "gloves" \
            and (human / "session.json").exists() and (human / "episodes.jsonl").exists():
        a.session = str(human)
    if a.session and a.session != "off":
        r.session_dir = Path(a.session)
        r.mode = r.want_mode = "gloves"
        old = r.ops.pop("gloves", None)
        if old is not None:
            old.close()
        r.op = r._operator("gloves")
    if a.g1:
        from kaiku.app import g1
        r.lead = g1.Lead()
        g1.start(r.lead, a.g1, a.host)

    def quit_(*_):
        r.stop = True

    signal.signal(signal.SIGINT, quit_)
    signal.signal(signal.SIGTERM, quit_)
    r.run()


if __name__ == "__main__":
    main()
