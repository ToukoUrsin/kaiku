"""The lead robot's commands, served to the humanoid challenge's own runner.

In the Aaltonomous humanoid challenge the runner owns the G1 and its safety filter, and asks a
policy server for joint targets 30 times a second (websocket + msgpack, port 8000, protocol 1).
This is such a server with a person in it: every ``act`` is answered from the command Kaiku's
lead robot was last given, which is already in the challenge's joint order.

    python -m kaiku.app.run --gloves 127.0.0.1 --g1 8000                # Kaiku, with the server in it
    python -m aaltonomous.deploy.runner --backend real --policy ws://KAIKU:8000 --team kaiku

The runner sends its camera images with every request, about 3 MB, too much for wifi 30 times a
second. Next to the runner, ``python g1.py --lead ws://KAIKU:8000`` serves it on that machine and
fetches only the 28 numbers from Kaiku. This file needs numpy, msgpack and websockets, nothing
of Kaiku, so it can be copied there alone.

The real robot does not start where the lead robot is, and the two loops do not share a clock,
so an answer is never the lead's command itself. It is the next steps from the last answer toward
it, inside the safety filter's rate limits, which leaves the filter nothing to clip. When the lead
has said nothing for half a second the robot holds.

Where the challenge's safety filter is at hand (next to its runner, or the copy in Kaiku's
third_party) every step is put through it before it is sent. The runner's filter then gets what
its own code has already passed: a lead that reaches into the counter makes the robot stop at the
edge instead of collecting clipped commands until the runner gives up.

An answer holds two ticks of targets. The runner asks for the next ones while the last of them
runs, so a request may take a whole tick (33 ms) before the robot has to wait; with one target
an answer it would have half of that, and the camera images alone take about 20 ms to arrive.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path

import msgpack
import numpy as np

log = logging.getLogger("kaiku.g1")

VERSION, PORT = 1, 8000
HZ = 30.0
HEADROOM = 0.97                                   # of the filter's 1 rad/s (arms) and 3 rad/s (hands)
ARM_STEP, HAND_STEP = HEADROOM * 1.0 / HZ, HEADROOM * 3.0 / HZ
STALE_S = 0.5
HORIZON = 2                                       # ticks of targets in one answer
N = 14


# ----------------------------------------------------------------------------- the wire
def _enc(o):
    if isinstance(o, np.ndarray):
        return {"__nd__": 1, "dtype": o.dtype.str, "shape": list(o.shape), "data": np.ascontiguousarray(o).tobytes()}
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"cannot serialise {type(o)}")


def _dec(d):
    if "__nd__" in d:
        return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"])     # read-only, not copied
    return d


def pack(msg) -> bytes:
    return msgpack.packb(msg, default=_enc, use_bin_type=True)


def unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_dec, raw=False)


# ----------------------------------------------------------------------------- the lead
class Lead:
    """What the lead robot was last commanded. Kaiku's loop writes it, the server's threads read it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._arm = self._hand = None
        self._at = 0.0

    def put(self, arm: np.ndarray, hand: np.ndarray) -> None:
        arm, hand = np.array(arm, float), np.array(hand, float)
        with self._lock:
            self._arm, self._hand, self._at = arm, hand, time.monotonic()

    def get(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Arm and hand targets (14 each) and how many seconds old they are; None before the first."""
        with self._lock:
            if self._arm is None:
                return None
            return self._arm, self._hand, time.monotonic() - self._at


class Remote:
    """The lead of a Kaiku on another machine: its 28 numbers, fetched a hundred times a second on a
    thread of its own, so that answering the runner never waits for the network."""

    def __init__(self, url: str, every_s: float = 0.01) -> None:
        self.url, self.every_s = url, every_s
        self._lock = threading.Lock()
        self._got, self._at = None, 0.0
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="g1-lead").start()

    def _run(self) -> None:
        from websockets.sync.client import connect
        while not self._stop.is_set():
            try:
                with connect(self.url, max_size=None, compression=None, open_timeout=2) as ws:
                    unpack(ws.recv(timeout=2))                        # its metadata
                    log.info("following %s", self.url)
                    while not self._stop.is_set():
                        ws.send(pack({"type": "lead"}))
                        got = unpack(ws.recv(timeout=1))
                        with self._lock:
                            self._got, self._at = got, time.monotonic()
                        self._stop.wait(self.every_s)
            except Exception as e:                                    # no Kaiku, no motion: the robot holds
                log.warning("lead %s: %s", self.url, e)
                self._stop.wait(0.5)

    def get(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        with self._lock:
            got, waited = self._got, time.monotonic() - self._at
        if got is None or got.get("age") is None:
            return None
        return got["arm"], got["hand"], got["age"] + waited

    def close(self) -> None:
        self._stop.set()


def official_filter():
    """The humanoid challenge's safety filter with its own limits, never giving up: from its package
    when this runs next to the runner, else the copy in Kaiku's third_party. None when neither is here."""
    try:
        try:
            from aaltonomous.interface import safety
            spec = safety.JointSpec()
        except ImportError:
            here = Path(__file__).resolve().parents[2] / "third_party" / "aaltonomous_safety"
            name = "kaiku_official_safety"
            safety = sys.modules.get(name)
            if safety is None:
                found = importlib.util.spec_from_file_location(name, here / "safety.py")
                safety = importlib.util.module_from_spec(found)
                sys.modules[name] = safety
                found.loader.exec_module(safety)
                safety.WorkspaceChecker.__init__.__defaults__ = (here / "g1_body29_hand14.urdf",)
                safety.log.setLevel(logging.ERROR)                    # a step it shortens is no news here
            spec = safety.JointSpec(here / "joint_spec.yaml")
        return safety.SafetyFilter(spec, safety.SafetyConfig(dt=1.0 / HZ, max_clipped_steps=10 ** 9))
    except Exception as e:
        log.warning("the challenge's safety filter is not here (%s): steps are rate-limited only", e)
        return None


class Follow:
    """Answers for one runner: a step a tick from where the robot is toward the lead."""

    def __init__(self, lead, horizon: int = HORIZON, check=official_filter) -> None:
        self.lead, self.horizon = lead, horizon
        self.filter = check() if check else None
        self.arm = self.hand = None
        self.ticks = 0

    def reset(self) -> None:
        self.arm = self.hand = None

    def _step(self, arm: np.ndarray, hand: np.ndarray) -> None:
        # An arm's step is scaled as a whole, so a slowed arm keeps to the lead's path.
        step = (arm - self.arm).reshape(2, 7)
        biggest = np.abs(step).max(axis=1, keepdims=True)
        arm = self.arm + (step * np.minimum(1.0, ARM_STEP / np.maximum(biggest, 1e-12))).reshape(N)
        hand = self.hand + np.clip(hand - self.hand, -HAND_STEP, HAND_STEP)
        if self.filter is not None:
            self.ticks += 1
            passed = self.filter.filter({"arm_q_target": arm, "hand_q_target": hand}, self.ticks / HZ)
            arm, hand = passed.arm_q_cmd, passed.hand_q_cmd
        self.arm, self.hand = arm, hand

    def act(self, obs: dict) -> dict:
        if self.arm is None:                                          # an episode starts where the robot is
            self.arm, self.hand = np.array(obs["arm_q"], float), np.array(obs["hand_q"], float)
            if self.filter is not None:
                self.ticks = 0
                self.filter.reset(self.arm, self.hand, 0.0)
        got = self.lead.get()
        live = got is not None and got[2] < STALE_S and np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
        arms, hands = [], []
        for _ in range(self.horizon):
            if live:
                self._step(got[0], got[1])
            arms.append(self.arm.copy())
            hands.append(self.hand.copy())
        if self.horizon == 1:
            return {"arm_q_target": arms[0], "hand_q_target": hands[0]}
        return {"arm_q_target": np.stack(arms), "hand_q_target": np.stack(hands)}


# ----------------------------------------------------------------------------- the server
def serve(lead, port: int = PORT, host: str = "0.0.0.0", ready: threading.Event | None = None,
          horizon: int = HORIZON) -> None:
    """Blocking. ``ready`` is set once the socket listens and gets the server as ``ready.server``."""
    from websockets.sync.server import serve as ws_serve

    meta = {"type": "metadata", "version": VERSION, "name": "kaiku-teleop", "action_horizon": horizon}

    def answer(follow: Follow, msg) -> dict:
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "reset":
            follow.reset()
            return {"type": "reset_ok"}
        if kind == "act":
            return {"type": "action", "action": follow.act(msg["obs"])}
        if kind == "lead":
            got = lead.get()
            return {"type": "lead", "age": None} if got is None else \
                {"type": "lead", "arm": got[0], "hand": got[1], "age": float(got[2])}
        return {"type": "error", "message": f"unknown message type {kind!r}"}

    def conn(ws) -> None:
        follow = Follow(lead, horizon)
        ws.send(pack(meta))
        for raw in ws:
            try:
                out = answer(follow, unpack(raw))
            except Exception as e:                                    # tell the runner, keep serving
                log.exception("request failed")
                out = {"type": "error", "message": f"{type(e).__name__}: {e}"}
            ws.send(pack(out))

    with ws_serve(conn, host, port, max_size=None, compression=None) as server:
        log.info("lead robot served to the challenge runner on %s:%d", host, server.socket.getsockname()[1])
        if ready is not None:
            ready.server = server
            ready.set()
        server.serve_forever()


def start(lead, port: int = PORT, host: str = "0.0.0.0", horizon: int = HORIZON):
    """The server on a thread of its own; returns it once it listens (``.shutdown()`` stops it)."""
    ready = threading.Event()
    threading.Thread(target=serve, args=(lead, port, host, ready, horizon), daemon=True, name="g1").start()
    if not ready.wait(10.0):
        raise RuntimeError(f"could not listen on {host}:{port}")
    return ready.server


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lead", required=True, metavar="ws://KAIKU:8000", help="the Kaiku to follow")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--horizon", type=int, default=HORIZON, help="ticks of targets in one answer")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    serve(Remote(a.lead), a.port, a.host, horizon=a.horizon)


if __name__ == "__main__":
    main()
