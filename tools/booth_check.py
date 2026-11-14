"""What a booth recording says about a person's hand: how the gloved fingers came through Kaiku.

    python tools/booth_check.py /bulk3/runs/robotteleop/sessions/booth-YYYYMMDD-HHMMSS [--speed 2]

Plays the session's raw and solved recordings back down the path a live glove takes (the solver, the
filters, the gap range, the retargeting) and says what the robot's fingers were asked to do: how
often each finger closed, how often the grasp fired, whether a fist and a flat hand read as such.
Light: two robots, no page.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaiku.app.session import Session                         # noqa: E402
from kaiku.app.visitor import VisitorOperator                 # noqa: E402
from kaiku.rgmp.replay import replay                          # noqa: E402
from kaiku.rgmp.server import RgmpServer                      # noqa: E402
from kaiku.teleop.fingers import GloveFingers                 # noqa: E402
from kaiku.teleop.retarget import closure_of_posture          # noqa: E402

CLOSED, OPEN = 0.6, 0.3            # of a robot finger's way closed: counted as closed, and as open again


def times(flags: np.ndarray) -> int:
    """How often a closure went up through CLOSED after having been below OPEN."""
    count, armed = 0, True
    for value in flags:
        if armed and value >= CLOSED:
            count, armed = count + 1, False
        elif value <= OPEN:
            armed = True
    return count


def longest(mask: np.ndarray, t: np.ndarray) -> float:
    """The longest run of True, in seconds of the recording."""
    best, since = 0.0, None
    for value, at in zip(mask, t):
        since = (at if since is None else since) if value else None
        if since is not None:
            best = max(best, float(at - since))
    return best


def check(session: Path, speed: float = 2.0) -> dict:
    raw, solved = session / "raw.rgmp", session / "solved.rgmp"
    if not raw.exists():
        raise SystemExit(f"no raw.rgmp in {session}")
    stop = threading.Event()
    servers = [RgmpServer("127.0.0.1", 0, send_timeout_s=1.0).start()]
    players = [threading.Thread(target=replay, args=(raw, servers[0]), kwargs={"speed": speed, "stop": stop})]
    solver = "off"
    if solved.exists() and solved.stat().st_size > 0:
        servers.append(RgmpServer("127.0.0.1", 0, send_timeout_s=1.0).start())
        players.append(threading.Thread(target=replay, args=(solved, servers[1]), kwargs={"speed": speed, "stop": stop}))
        solver = f"127.0.0.1:{servers[1].port}"
    s = Session(2, seed=1)
    layout, followers = s.sample_scene(3)
    s.new_episode(layout, followers)
    glove = GloveFingers("127.0.0.1", servers[0].port, solver=solver)
    v = VisitorOperator(s, glove=glove)
    v.begin()
    rows, t0, step = [], time.monotonic(), 1.0 / (50.0 * speed)
    try:
        while not servers[0].wait_for_client(0.1) and time.monotonic() - t0 < 10:
            pass
        for player in players:
            player.start()
        t0 = nxt = time.monotonic()
        while any(player.is_alive() for player in players):
            s.tick(v.update(s.observe()))
            if v.posture is not None and v.human is not None:
                middle, index = closure_of_posture(v.posture)
                hand = glove.hand(glove.side, time.monotonic()) if glove.side else None
                gap = hand.pinch[:2] if hand is not None else (np.nan, np.nan)
                rows.append([(time.monotonic() - t0) * speed, index, middle, float(v.pinched), float(gap[0]), float(gap[1]),
                             *(float(c) for c in v.human.curl[:3])])
            nxt += step
            time.sleep(max(0.0, nxt - time.monotonic()))
    finally:
        stop.set()
        info = glove.hud()
        v.close()
        s.close()
        for server in servers:
            server.stop()
    if not rows:
        return {"seconds": 0, "said": "No glove frame came through: the recording holds no hand."}
    a = np.array(rows)
    index, middle = np.clip(a[:, 1], 0, 1), np.clip(a[:, 2], 0, 1)
    out = {"seconds": round(float(a[-1, 0]), 1), "side": info.get("side"), "solver": info.get("solver"), "coil": info.get("coil"),
           "index_closed_times": times(index), "middle_closed_times": times(middle), "grasp_fired_times": times(a[:, 3]),
           "fist_s": round(longest((index > 0.8) & (middle > 0.8), a[:, 0]), 1),
           "flat_s": round(longest((index < 0.1) & (middle < 0.1), a[:, 0]), 1),
           "thumb_index_gap_mm": [round(1000 * float(np.nanmin(a[:, 4]))), round(1000 * float(np.nanmax(a[:, 4])))],
           "thumb_middle_gap_mm": [round(1000 * float(np.nanmin(a[:, 5]))), round(1000 * float(np.nanmax(a[:, 5])))],
           "curl_thumb_index_middle_max": [round(float(a[:, k].max()), 2) for k in (6, 7, 8)]}
    moved = out["thumb_index_gap_mm"][1] - out["thumb_index_gap_mm"][0] >= 5      # a hand that held one pose did nothing
    out["said"] = (f"{out['seconds']:.0f} s of a {out['side']} glove ({out['solver']} solver). The robot's index finger closed "
                   f"{out['index_closed_times']} times, its middle finger {out['middle_closed_times']} times, the grasp fired "
                   f"{out['grasp_fired_times']} times. Fist held {out['fist_s']} s, flat hand {out['flat_s']} s."
                   if moved else
                   f"{out['seconds']:.0f} s of a {out['side']} glove, and its fingers never opened and closed: "
                   f"thumb to index stayed between {out['thumb_index_gap_mm'][0]} and {out['thumb_index_gap_mm'][1]} mm.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--speed", type=float, default=2.0, help="how much faster than it was recorded")
    args = ap.parse_args()
    result = check(args.session, args.speed)
    (args.session / "check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
