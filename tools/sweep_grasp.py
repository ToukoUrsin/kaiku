"""Scratch: success rate of planned top-down pick-and-place, one part per env, direct IK control."""
import sys, collections
import numpy as np
from kaiku.sim.batch import BatchSim, HOME_ARM
from kaiku.sim.grasp import plan_pick_place, reachable_direction, palm_position, palm_rotation, hand14, HAND_OPEN, NATURAL_DIR, grasp_offset
from kaiku.sim.ik import ArmIK
from kaiku.sim.motion import Follower
from kaiku.sim.scene import Layout, Placement, SHAPES, BINS
from kaiku.sim.twin import build_twin, CONTROL_HZ

import os, mujoco
import kaiku.sim.grasp as G
N = int(sys.argv[1]) if len(sys.argv) > 1 else 96
speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
tw = build_twin(); sc = tw.scene; ik = ArmIK()
if os.environ.get("NOMULTICCD"): tw.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
ONLY = os.environ.get("ONLY")
rng = np.random.default_rng(1)
b = BatchSim(tw, N)
lo, hi = sc.region
cases = []
for i in range(N):
    shape = ONLY or SHAPES[i % 3]; color = ["red", "yellow", "blue"][(i // 3) % 3]
    ob = next(o for o in sc.objects if o.shape == shape and o.color == color)
    p = np.r_[rng.uniform(lo, hi), sc.rest_z]; yaw = rng.uniform(-np.pi/4, np.pi/4)
    bin_name = {"red": "left", "yellow": "center", "blue": "right"}[color]
    side = {"left": "left", "right": "right", "center": "left" if p[1] > 0 else "right"}[bin_name]
    b.reset(i, Layout([Placement(ob.name, p, yaw)]), hand=hand14(HAND_OPEN, HAND_OPEN))
    f, ferr = reachable_direction(ik, sc, side, shape, p, yaw)
    wps = plan_pick_place(sc, side, shape, p, yaw, bin_name, speed=speed, finger_dir=f)
    P, R = ik.fk(side, HOME_ARM[:7] if side == "left" else HOME_ARM[7:])
    traj = Follower(P + R @ grasp_offset(side), R, HAND_OPEN, wps)
    cases.append(dict(shape=shape, color=color, side=side, bin=bin_name, traj=traj, p=p, yaw=yaw, maxerr=0.0, ferr=ferr, f=f))
arm = np.tile(HOME_ARM, (N, 1)); hand = np.tile(hand14(HAND_OPEN, HAND_OPEN), (N, 1))
qik = arm.copy()
T = 16.0
da, dh = 1.0 / CONTROL_HZ, 3.0 / CONTROL_HZ
for k in range(int(T * CONTROL_HZ)):
    t = k / CONTROL_HZ
    for i, c in enumerate(cases):
        sl = slice(0, 7) if c["side"] == "left" else slice(7, 14)
        gs = b.data[i].site_xpos[tw.site[c["side"] + "_grasp"]]; pr = b.data[i].site_xmat[tw.site[c["side"] + "_palm"]].reshape(3, 3)
        hq = b.data[i].qpos[tw.hand_qadr[:7]] if c["side"] == "left" else G.left_to_right(b.data[i].qpos[tw.hand_qadr[7:]])
        s = c["traj"].update(1.0 / CONTROL_HZ, gs, pr)
        if s.done and "tdone" not in c: c["tdone"] = t
        q, ep, er = ik.solve(c["side"], qik[i, sl], palm_position(c["side"], s.rot, s.point), s.rot)
        qik[i, sl] = q
        if not s.done: c["maxerr"] = max(c["maxerr"], ep)
        if s.label in ("grasp",): c["graspsite"] = np.linalg.norm(b.site_pos(i, c["side"] + "_grasp") - s.point)
        z = b.data[i].qpos[tw.obj_qadr[sc.by_name[b.layouts[i].placements[0].name].index] + 2]
        c.setdefault("zmax", {}); c["zmax"][s.label] = max(c["zmax"].get(s.label, -9), z)
        h = hand14(s.hand, HAND_OPEN) if c["side"] == "left" else hand14(HAND_OPEN, s.hand)
        arm[i] += np.clip(qik[i] - arm[i], -da, da); hand[i] += np.clip(h - hand[i], -dh, dh)
    b.step(arm, hand)
res = collections.defaultdict(list)
for i, c in enumerate(cases):
    out = list(b.outcomes(i).values())[0]
    c["out"] = out
    for key in (("shape", c["shape"]), ("side", c["side"]), ("bin", c["bin"]), ("all", "")):
        res[key].append(out == "correct")
for k, v in sorted(res.items()): print(f"{k[0]:6s}{k[1]:9s} {np.mean(v)*100:5.1f}%  n={len(v)}")
print("max IK pos err (mm): median %.1f  p90 %.1f  max %.1f" % tuple(np.percentile([c["maxerr"]*1000 for c in cases], [50, 90, 100])))
print("episode time s: median %.1f max %.1f; unfinished %d" % (np.median([c.get("tdone", 99) for c in cases]), max(c.get("tdone", 99) for c in cases), sum("tdone" not in c for c in cases)))
import json; json.dump([dict(shape=c["shape"], side=c["side"], bin=c["bin"], ok=c["out"] == "correct", t=c.get("tdone", 99), ferr=c["ferr"]) for c in cases], open("/tmp/sweep.json", "w"))
print("unreachable plans:", sum(c["ferr"] > 0.004 for c in cases))
fails = [c for c in cases if c["out"] != "correct"]
for c in fails[:25]:
    print(f"  FAIL {c['shape']:9s}{c['side']:6s}->{c['bin']:7s} p=({c['p'][0]:.3f},{c['p'][1]:+.3f}) yaw={np.degrees(c['yaw']):+4.0f} f={np.degrees(c['f']):+4.0f} ferr={c['ferr']*1000:4.1f} gsite={c.get('graspsite',0)*1000:4.1f}mm out={c['out']} z: " + " ".join(f"{k}={v:.2f}" for k, v in c["zmax"].items()))
