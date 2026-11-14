"""Self-contained viewer test bed: ``python -m kaiku.web.demo --n 64 --port 8650``."""
from __future__ import annotations
import argparse, math, signal, time
from pathlib import Path
import mujoco
import numpy as np
from .server import ViewerServer, STATE_FAILED, STATE_IDLE, STATE_LEAD, STATE_RUNNING, STATE_SUCCESS

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "third_party/mujoco_menagerie/unitree_g1/g1_with_hands.xml"

def load_model(kind: str):
    if kind == "twin":
        try:
            from kaiku.sim.twin import build_twin
            return build_twin().model
        except Exception as exc:
            print(f"twin unavailable ({exc}); using menagerie model")
    return mujoco.MjModel.from_xml_path(str(XML))

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--port", type=int, default=8650)
    ap.add_argument("--model", choices=("g1", "twin"), default="g1")
    args = ap.parse_args(argv)
    n = max(1, min(65535, args.n)); model = load_model(args.model)
    q0 = np.asarray(model.qpos0, dtype=float).copy()
    # The menagerie asset has a free root. Keep the base fixed as the challenge viewer expects.
    if model.nq >= 7 and int(model.jnt_type[0]) == 0:
        q0[:7] = model.qpos0[:7]
    q = np.tile(q0, (n, 1)); states = np.full(n, STATE_IDLE, np.uint8); states[0] = STATE_LEAD
    movable = [int(model.jnt_qposadr[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == 3]
    # Smooth, deterministic echo variation makes the live stream visibly active.
    phase = np.linspace(0, math.tau, max(1, n), endpoint=False)
    paused = False; mode = "autopilot"; parts = "auto"; recording = True; training = False; train_progress = 0.0; cyc = 0
    episode = 1; started = time.monotonic(); last_hud = 0.0; history = []
    def on_message(msg):
        nonlocal paused, mode, parts, recording, training
        if msg.get("type") != "cmd": return
        name, value = msg.get("name"), msg.get("value")
        if name == "pause": paused = bool(value)
        elif name == "mode": mode = str(value)
        elif name == "parts": parts = value
        elif name == "record": recording = bool(value)
        elif name == "train": training = True
        elif name == "next_scene":
            nonlocal_vars["next"] = True
        elif name == "calibrate": nonlocal_vars["calibrated"] = True
    nonlocal_vars = {"next": False, "calibrated": False}
    v = ViewerServer(model, n, host="0.0.0.0", port=args.port, title="Kaiku demo", max_hz=60)
    v.on_message(on_message); v.start()
    stop = False
    def quit_(*_):
        nonlocal stop; stop = True
    signal.signal(signal.SIGINT, quit_); signal.signal(signal.SIGTERM, quit_)
    print(f"Kaiku demo: {v.url} ({n} environments)")
    tick = 0
    try:
        while not stop:
            now = time.monotonic(); t = now - started
            if nonlocal_vars.pop("next", False):
                episode += 1; history.append({"episode": episode - 1, "parts": 4, "ok": int(n * .78), "robots": n, "lead_ok": True})
            if not paused:
                for i in range(n):
                    qi = q[i]
                    for k, a in enumerate(movable):
                        amp = 0.06 if k < 16 else 0.025
                        qi[a] = q0[a] + amp * math.sin(t * (0.8 + (k % 5) * .07) + phase[i] + k * .31)
                    if i and model.nq >= 7 and int(model.jnt_type[0]) == 0: qi[:7] = q0[:7]
                cyc = int(t // 5.0) % 4
                states[:] = STATE_RUNNING
                if cyc == 2: states[:] = STATE_SUCCESS
                elif cyc == 3: states[:] = STATE_FAILED
                states[0] = STATE_LEAD
            if training: train_progress = min(1.0, train_progress + .003); training = train_progress < 1
            hud = None
            if now - last_hud >= .1:
                elapsed = t % 18.0; sorted_ = int(min(4, elapsed / 3.5)); done = n if int(t // 5) % 2 else int(n * .65)
                lag = np.maximum(0, np.linspace(0, 1.7, n) + .18 * np.sin(t + np.arange(n) * .4))
                hud = {
                    "mode": mode, "status": "paused" if paused else ("settling" if cyc == 2 else "running"),
                    "operator": {"source": "autopilot" if mode == "autopilot" else "stand-in", "stand_in": mode != "autopilot",
                                 "left": {"connected": True, "hz": 98.2 + .7 * math.sin(t), "status": "tracking"}, "right": {"connected": True, "hz": 97.9 + .5 * math.cos(t), "status": "tracking"}, "calibrated": nonlocal_vars["calibrated"],
                                 "hands": {"left": [[0.01 * j, 0.002 * math.sin(t + j), .02 + .001 * j] for j in range(21)], "right": [[-.01 * j, 0.002 * math.cos(t + j), .02 + .001 * j] for j in range(21)]}},
                    "episode": {"index": episode, "parts": 4 if parts == "auto" else int(parts), "sorted": sorted_, "elapsed_s": elapsed, "step": ["approach", "descend", "grasp", "lift", "carry", "release"][int(t * .6) % 6]},
                    "echo": {"robots": n, "done": done, "ok": int(done * .92), "failed": max(0, int(done * .08)), "lag_s": float(np.median(lag)), "lag_max_s": float(lag.max())},
                    "envs": {"lag_s": lag.tolist(), "sorted": [sorted_] * n, "clipped": [0] * n},
                    "totals": {"episodes": episode, "human_demos": episode, "robot_demos": episode * n + done, "amplification": round(n + .2, 1), "yield": .91 + .04 * math.sin(t * .2), "demos_per_min": 120 + 8 * math.sin(t), "clipped_steps": int(abs(math.sin(t * .2)) * 3), "minutes": t / 60},
                    "history": history[-60:], "dataset": {"episodes": episode * n + done, "frames": tick * n, "path": "data/demo-run", "recording": recording},
                    "policy": {"state": "training" if training else "ready", "progress": train_progress, "loss": .012, "success": .83, "curve": [{"demos": 1, "plain": .1, "kaiku": .6}, {"demos": 20, "plain": .34, "kaiku": .8}, {"demos": 80, "plain": .53, "kaiku": .9}]},
                    "perf": {"realtime": 1.0, "tick_ms": 16.7, "hz": 60},
                }
                last_hud = now
            v.publish(q, states, hud)
            tick += 1; time.sleep(max(0, 1 / 60 - (time.monotonic() - now)))
    finally: v.stop()

if __name__ == "__main__": main()
