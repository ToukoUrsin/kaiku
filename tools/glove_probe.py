"""Scratch: the stand-in operator through the whole glove path (emulated Smartgloves over RGMP).

    python tools/glove_probe.py [N=8] [episodes=3] [seed=0]     env NOBJ=k, PLAIN=1 (autopilot, no gloves)
"""
import os
import sys
import time

import numpy as np

from kaiku.app.autopilot import Autopilot
from kaiku.app.session import Session
from kaiku.teleop.operator import GloveOperator

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
E = int(sys.argv[2]) if len(sys.argv) > 2 else 3
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NOBJ = int(os.environ.get("NOBJ", 0))
s = Session(N, seed=seed)
op = Autopilot(s) if os.environ.get("PLAIN") else GloveOperator(s, seed=seed)
tot = np.zeros(4, int)
for e in range(E):
    s.new_episode(s.sample_lead(NOBJ or None))
    op.begin()
    nxt = time.perf_counter()
    err = []
    while not s.settled and s.tick_no < 30 * 120:
        obs = s.observe()
        cmd = op.update(obs)
        if op.done:
            s.finish_lead()
        s.tick(cmd)
        err.append(np.linalg.norm(cmd.pinch - obs.pinch[0], axis=-1).max())
        nxt += 1 / 30
        d = nxt - time.perf_counter()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.perf_counter()
    r = s.result()
    n = len(r.layouts[0].placements)
    tot += [int(r.success[0]), 1, int(r.success[1:].sum()), N - 1]
    print(f"ep {e}: {n} parts, lead {'ok' if r.success[0] else 'FAIL ' + str(r.outcomes[0])}, followers "
          f"{int(r.success[1:].sum())}/{N - 1}, clipped lead {r.clipped[0]} all {int(r.clipped.sum())}, {s.tick_no / 30:.1f}s, "
          f"label {op.label}", flush=True)
print(f"lead {tot[0]}/{tot[1]}  followers {tot[2]}/{tot[3]}")
if hasattr(op, "hud"):
    h = op.hud()
    print({k: v for k, v in h.items() if k != "hands"}, "hand points", {k: len(v) for k, v in (h["hands"] or {}).items()})
    op.close()
s.close()
