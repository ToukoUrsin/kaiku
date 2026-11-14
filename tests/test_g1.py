"""The lead robot served to the humanoid challenge's runner: the wire, the ramp, and the runner itself."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from kaiku.app import g1
from kaiku.app.session import HOME_ARM, HOME_HAND

HOST = "127.0.0.1"
REF = Path(__file__).resolve().parents[1] / "ref" / "aaltonomous" / "repo"
URDF = Path(__file__).resolve().parents[1] / "third_party" / "aaltonomous_safety" / "g1_body29_hand14.urdf"


def obs_at(arm, hand) -> dict:
    return {"arm_q": np.asarray(arm, np.float32), "hand_q": np.asarray(hand, np.float32), "instruction": "", "t": 0.0}


def test_an_episode_starts_where_the_robot_is_and_never_steps_past_the_rate_limits():
    lead = g1.Lead()
    follow = g1.Follow(lead, horizon=1)
    start_arm, start_hand = np.full(14, 0.4), np.full(14, 0.2)
    first = follow.act(obs_at(start_arm, start_hand))                 # no lead yet: the robot stays put
    np.testing.assert_allclose(first["arm_q_target"], start_arm, atol=1e-6)
    np.testing.assert_allclose(first["hand_q_target"], start_hand, atol=1e-6)
    goal_arm, goal_hand = np.linspace(-0.6, 0.9, 14), np.linspace(0.0, 1.0, 14)
    lead.put(goal_arm, goal_hand)
    arm, hand = first["arm_q_target"], first["hand_q_target"]
    for _ in range(60):
        out = follow.act(obs_at(arm, hand))
        assert np.abs(out["arm_q_target"] - arm).max() <= 1.0 / g1.HZ
        assert np.abs(out["hand_q_target"] - hand).max() <= 3.0 / g1.HZ
        arm, hand = out["arm_q_target"], out["hand_q_target"]
    np.testing.assert_allclose(arm, goal_arm, atol=1e-9)
    np.testing.assert_allclose(hand, goal_hand, atol=1e-9)
    follow.reset()                                                    # the next episode starts from the robot again
    np.testing.assert_allclose(follow.act(obs_at(start_arm, start_hand))["arm_q_target"][0],
                               start_arm[0] + np.sign(goal_arm[0] - start_arm[0]) * g1.ARM_STEP, atol=1e-6)


def test_an_arm_is_slowed_as_a_whole_so_it_keeps_to_the_leads_path():
    lead = g1.Lead()
    follow = g1.Follow(lead, horizon=1)
    goal = np.zeros(14)
    goal[:7] = [0.7, 0.35, 0.0, -0.7, 0.07, 0.0, 0.0]
    lead.put(goal, np.zeros(14))
    step = follow.act(obs_at(np.zeros(14), np.zeros(14)))["arm_q_target"]
    np.testing.assert_allclose(step[:7] / g1.ARM_STEP, goal[:7] / 0.7, atol=1e-9)
    np.testing.assert_allclose(step[7:], 0.0)


def test_the_robot_holds_when_the_lead_goes_quiet(monkeypatch):
    lead = g1.Lead()
    follow = g1.Follow(lead, horizon=1)
    lead.put(np.ones(14), np.ones(14))
    moving = follow.act(obs_at(np.zeros(14), np.zeros(14)))
    assert moving["arm_q_target"].max() > 0.0
    monkeypatch.setattr(g1, "STALE_S", 0.0)
    held = follow.act(obs_at(np.zeros(14), np.zeros(14)))
    np.testing.assert_array_equal(held["arm_q_target"], moving["arm_q_target"])
    lead.put(np.full(14, np.nan), np.ones(14))                        # and never passes on what is not a number
    monkeypatch.setattr(g1, "STALE_S", 0.5)
    np.testing.assert_array_equal(follow.act(obs_at(np.zeros(14), np.zeros(14)))["arm_q_target"],
                                  moving["arm_q_target"])


def test_the_wire_is_the_challenges_protocol_and_a_relay_follows_over_it():
    from websockets.sync.client import connect

    lead = g1.Lead()
    server = g1.start(lead, 0, HOST)
    port = server.socket.getsockname()[1]
    relay = None
    try:
        with connect(f"ws://{HOST}:{port}", max_size=None, compression=None) as ws:
            meta = g1.unpack(ws.recv(timeout=5))
            assert (meta["type"], meta["version"], meta["action_horizon"]) == ("metadata", 1, 2)
            ws.send(g1.pack({"type": "reset", "task": "sort_fixed", "instruction": "sort", "seed": 0}))
            assert g1.unpack(ws.recv(timeout=5)) == {"type": "reset_ok"}
            ws.send(g1.pack({"type": "lead"}))
            assert g1.unpack(ws.recv(timeout=5)) == {"type": "lead", "age": None}
            lead.put(np.full(14, 0.5), np.full(14, 0.5))
            ws.send(g1.pack({"type": "act", "obs": obs_at(np.zeros(14), np.zeros(14))}))
            action = g1.unpack(ws.recv(timeout=5))["action"]
            assert action["arm_q_target"].shape == action["hand_q_target"].shape == (2, 14)      # two ticks
            np.testing.assert_allclose(action["arm_q_target"][:, 0], [g1.ARM_STEP, 2 * g1.ARM_STEP])
            np.testing.assert_allclose(action["hand_q_target"][:, 0], [g1.HAND_STEP, 2 * g1.HAND_STEP])
            ws.send(g1.pack({"type": "nonsense"}))
            assert g1.unpack(ws.recv(timeout=5))["type"] == "error"
        # The same lead, followed from another machine: only the 28 numbers cross.
        remote = g1.Remote(f"ws://{HOST}:{port}")
        relay = g1.start(remote, 0, HOST, horizon=1)
        with connect(f"ws://{HOST}:{relay.socket.getsockname()[1]}", max_size=None, compression=None) as ws:
            g1.unpack(ws.recv(timeout=5))
            deadline = time.monotonic() + 5.0
            while remote.get() is None:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            ws.send(g1.pack({"type": "act", "obs": obs_at(np.zeros(14), np.zeros(14))}))
            np.testing.assert_allclose(g1.unpack(ws.recv(timeout=5))["action"]["arm_q_target"], g1.ARM_STEP)
    finally:
        server.shutdown()
        if relay is not None:
            remote.close()
            relay.shutdown()


@pytest.mark.skipif(not (REF / "aaltonomous" / "deploy" / "runner.py").exists() or not URDF.exists(),
                    reason="the humanoid challenge's repository is not checked out under ref/")
def test_the_challenges_own_runner_follows_the_lead_without_one_clipped_command(tmp_path, monkeypatch):
    pytest.importorskip("pinocchio")
    monkeypatch.syspath_prepend(str(REF))
    from aaltonomous.deploy import runner
    from aaltonomous.interface import safety
    from aaltonomous.interface.env import AaltoEnv

    monkeypatch.setattr(safety.WorkspaceChecker.__init__, "__defaults__", (URDF,))
    lead = g1.Lead()
    server = g1.start(lead, 0, HOST)
    stop = threading.Event()

    def operator() -> None:
        """A lead at Kaiku's home pose that lifts its forearms, turns its wrists and closes its hands, all
        faster than the real robot may move and on a clock of its own."""
        t0 = time.monotonic()
        while not stop.is_set():
            t = time.monotonic() - t0
            arm = HOME_ARM.copy()
            arm[[3, 10]] -= 0.3 * (0.5 - 0.5 * np.cos(5.0 * t))       # elbows: the hands rise
            arm[[4, 11]] += 0.8 * np.sin(7.0 * t)                     # wrist roll
            hand = HOME_HAND * (0.5 + 0.5 * np.sin(9.0 * t))
            lead.put(arm, hand)
            time.sleep(1.0 / 60.0)

    thread = threading.Thread(target=operator, daemon=True)
    thread.start()
    try:
        env = AaltoEnv(backend="mock", task="sort_fixed", seed=0, time_limit=6.0, realtime=True)
        summaries, _ = runner.run("mock", f"ws://{HOST}:{server.socket.getsockname()[1]}", "kaiku", 1,
                                  out=str(tmp_path), time_limit=6.0, env=env)
    finally:
        stop.set()
        thread.join(2.0)
        server.shutdown()
        for name in [m for m in sys.modules if m == "aaltonomous" or m.startswith("aaltonomous.")]:
            del sys.modules[name]
    (summary,) = summaries
    assert summary["end_reason"] == "time_limit"
    assert summary["safety"]["clipped_steps"] == 0 and summary["safety"]["mode"] == "ok"
    assert summary["command_hz"] >= 27.0                              # the challenge's own gate
    assert summary["safety"]["steps"] > 150
