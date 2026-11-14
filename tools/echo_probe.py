"""Scratch: yield of the live fan-out. Lead sorted by the autopilot, followers by the echo.

    python tools/echo_probe.py [N=32] [episodes=4] [spread=0.08] [seed=0]
"""
import collections
import os
import sys
import time

import numpy as np

from kaiku.app.autopilot import Autopilot
from kaiku.app.session import Session

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
E = int(sys.argv[2]) if len(sys.argv) > 2 else 4
spread = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

NOBJ = int(os.environ.get("NOBJ", 0))
s = Session(N, seed=seed, spread=spread)
sc = s.scene
tot = collections.Counter()
by_shape = collections.defaultdict(list)
t_wall = time.perf_counter()
ticks = 0
for e in range(E):
    if e or NOBJ:
        s.new_episode(s.sample_lead(NOBJ) if NOBJ else None)
    ap = Autopilot(s)
    while not s.settled and s.tick_no < 30 * 90:
        obs = s.observe()
        cmd = ap.update(obs)
        if ap.done:
            s.finish_lead()
        s.tick(cmd)
    ticks += s.tick_no
    r = s.result()
    n_obj = len(r.layouts[0].placements)
    per_obj = [sum(v == "correct" for v in o.values()) for o in r.outcomes]
    tot["lead_ok"] += int(r.success[0]); tot["episodes"] += 1
    tot["fol_ok"] += int(r.success[1:].sum()); tot["fol"] += N - 1
    tot["fol_obj_ok"] += sum(per_obj[1:]); tot["fol_obj"] += n_obj * (N - 1)
    tot["clipped"] += int(r.clipped.sum()); tot["aborted"] += int(r.aborted.sum())
    for i in range(1, N):
        for name, v in r.outcomes[i].items():
            by_shape[sc.by_name[name].shape].append(v == "correct")
    print(f"ep {e}: {n_obj} parts, lead {'ok' if r.success[0] else 'FAIL ' + str(r.outcomes[0])}, followers "
          f"{int(r.success[1:].sum())}/{N-1} full, parts {sum(per_obj[1:])}/{n_obj*(N-1)}, "
          f"lag max {r.lag.max() if len(r.lag) else 0:.1f}s, clipped {int(r.clipped.sum())}, {s.tick_no/30:.1f}s", flush=True)
    if "-v" in sys.argv:
        for i in range(1, N):
            if not r.success[i]:
                bad = {k: v for k, v in r.outcomes[i].items() if v != "correct"}
                lp = {q.name: (l.pos, l.yaw, sc.by_name[l.name].shape) for q, l in zip(r.layouts[i].placements, r.layouts[0].placements)}
                fp = {q.name: q for q in r.layouts[i].placements}
                for k, v in bad.items():
                    q = fp[k]; fin = s.sim.obj_pos()[i, sc.by_name[k].index]
                    print(f"   env{i:3d} {sc.by_name[k].shape:8s}<-{lp[k][2]:8s} {v:7s} at ({q.pos[0]:.3f},{q.pos[1]:+.3f}) yaw {np.degrees(q.yaw):+4.0f}"
                          f"  lead ({lp[k][0][0]:.3f},{lp[k][0][1]:+.3f}) yaw {np.degrees(lp[k][1]):+4.0f}  final ({fin[0]:.3f},{fin[1]:+.3f},{fin[2]:.3f}) lag {r.lag[i-1]:.1f} clip {r.clipped[i]}")
wall = time.perf_counter() - t_wall
print(f"\nlead {tot['lead_ok']}/{tot['episodes']}  followers full {tot['fol_ok']}/{tot['fol']} = {100*tot['fol_ok']/max(tot['fol'],1):.1f}%"
      f"  parts {100*tot['fol_obj_ok']/max(tot['fol_obj'],1):.1f}%  clipped {tot['clipped']} aborted {tot['aborted']}")
print("by follower shape:", {k: f"{100*np.mean(v):.0f}% n={len(v)}" for k, v in by_shape.items()})
print(f"realtime factor {ticks/30/wall:.2f}x; per tick ms:", {k: round(1000 * v / ticks, 2) for k, v in s.timing.items()})
