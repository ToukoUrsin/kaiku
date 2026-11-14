"""A stand-in operator: sorts the lead's scene with planned top-down pinches.

It produces exactly what a gloved person produces after retargeting (a pinch-point target, a
palm orientation and Dex3 finger targets per hand), so it can drive the lead robot directly,
or be turned back into human wrist and finger motion and sent through the whole glove
pipeline. It looks at the scene the way a person does: each part is planned from where it is
at that moment, and the hand waits for the robot before it closes.
"""
from __future__ import annotations

import numpy as np

from kaiku.app.session import LeadCommand, Observation, Session
from kaiku.sim.grasp import HAND_OPEN, Waypoint, arm_for, hand14, pinch_shift, plan_pick_place
from kaiku.fanout.echo import SHAPE_CODE
from kaiku.sim.ik import SIDES
from kaiku.sim.motion import Follower
from kaiku.sim.reach import part_radius
from kaiku.sim.twin import CONTROL_HZ

DT = 1.0 / CONTROL_HZ


class Autopilot:
    def __init__(self, session: Session, speed: float = 1.0, rng: np.random.Generator | None = None):
        self.s = session
        self.scene = session.scene
        self.speed = speed
        self.rng = rng or np.random.default_rng(0)
        self.begin()

    def begin(self) -> None:
        """Plan the order for the session's current episode: nearest part to the robot first."""
        lay = self.s.layouts[0]
        order = sorted(range(len(lay.placements)), key=lambda k: lay.placements[k].pos[0])
        self.queue = list(order)
        self.slots = {b: 0 for b in self.scene.bin_centers}
        self.job: tuple[int, Follower] | None = None          # (side index, motion)
        self.homing: dict[int, Follower] = {}
        self.hand = [HAND_OPEN.copy(), HAND_OPEN.copy()]      # left-order posture per side
        self.pinch = self.s.home_pinch.copy()
        self.rot = self.s.home_rot.copy()
        self.label = "idle"
        self.target = -1

    @property
    def done(self) -> bool:
        return not self.queue and self.job is None and not self.homing

    def _start_next(self, obs: Observation) -> None:
        sc, lay = self.scene, self.s.layouts[0]
        k = self.queue.pop(0)
        ob = sc.by_name[lay.placements[k].name]
        pos, yaw = obs.obj_pos[0, ob.index], float(obs.obj_yaw[0, ob.index])
        bin_name = lay.mapping[ob.color]
        side = arm_for(bin_name, pos[1])
        s = SIDES.index(side)
        # Parts still on the mat are in the way of the open hand; the grasp is chosen around them.
        rest = [sc.by_name[lay.placements[j].name] for j in self.queue]
        others = np.array([obs.obj_pos[0, o.index, :2] for o in rest]).reshape(-1, 2)
        radius = part_radius(np.array([SHAPE_CODE[o.shape] for o in rest], int), sc.size)
        f = float(self.s.reach.best(np.array(SHAPE_CODE[ob.shape]), np.array(yaw), pos[:2], s, others=others,
                                    radius=radius)[0])
        wps = plan_pick_place(sc, side, ob.shape, pos, yaw, bin_name, slot=self.slots[bin_name], speed=self.speed,
                              finger_dir=f)
        self.slots[bin_name] += 1
        self.homing.pop(s, None)
        self.job = (s, Follower(self.pinch[s], self.rot[s], self.hand[s], wps))
        self.target = k
        self.aim = (ob.index, ob.shape, f, wps[1].point[:2].copy())
        other = 1 - s
        if np.linalg.norm(self.pinch[other] - self.s.home_pinch[other]) > 0.02:
            self._send_home(other)

    def _send_home(self, s: int) -> None:
        d = float(np.linalg.norm(self.pinch[s] - self.s.home_pinch[s]))
        wp = Waypoint("home", self.s.home_pinch[s], self.s.home_rot[s], HAND_OPEN, max(0.6, 2.5 * d) / self.speed)
        self.homing[s] = Follower(self.pinch[s], self.rot[s], self.hand[s], [wp], pos_tol=0.02, rot_tol=0.3)

    def update(self, obs: Observation) -> LeadCommand:
        if self.job is None and self.queue:
            self._start_next(obs)
        if self.job is not None:
            s, f = self.job
            if f.label in ("approach", "descend"):
                # Like a person, keep looking at the part: if it was nudged, aim where it is now.
                k, shape, direction, aimed = self.aim
                now = obs.obj_pos[0, k, :2] + pinch_shift(shape, float(obs.obj_yaw[0, k]), direction, self.scene.size)
                for w in f.wps[:4]:
                    w.point[:2] += now - aimed
                self.aim = (k, shape, direction, now)
            st = f.update(DT, obs.pinch[0, s], obs.rot[0, s])
            self.pinch[s], self.rot[s], self.hand[s], self.label = st.point, st.rot, st.hand, st.label
            if f.done:
                self.job = None
                if not self.queue:
                    self._send_home(s)
        for s, f in list(self.homing.items()):
            st = f.update(DT, obs.pinch[0, s], obs.rot[0, s])
            self.pinch[s], self.rot[s], self.hand[s] = st.point, st.rot, st.hand
            if f.done:
                del self.homing[s]
        if self.job is None and not self.queue:
            self.label = "home" if self.homing else "idle"
        return LeadCommand(self.pinch.copy(), self.rot.copy(), hand14(self.hand[0], self.hand[1]))
