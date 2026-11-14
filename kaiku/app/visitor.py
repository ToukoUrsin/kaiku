"""Pointer driven visitor operator.

The browser only chooses a point and whether the hand is pinched.  This operator turns that
small command into the same top down grasp primitives as the scripted operator, so the lead and
the echo path remain identical.
"""
from __future__ import annotations

import time

import numpy as np

from kaiku.app.session import LeadCommand, Observation, Session
from kaiku.fanout.echo import SHAPE_CODE
from kaiku.sim.grasp import HAND_CLOSED, HAND_OPEN, HAND_RELEASE, Waypoint, arm_for, hand14, palm_rotation, plan_pick_place, drop_point, drop_direction
from kaiku.sim.ik import SIDES
from kaiku.sim.motion import Follower
from kaiku.sim.reach import part_radius
from kaiku.sim.twin import CONTROL_HZ

DT = 1.0 / CONTROL_HZ


class VisitorOperator:
    """An operator controlled by ``{x, y, grip}`` pointer commands."""

    source = "visitor"
    label = "pointer"

    def __init__(self, session: Session):
        self.s = session
        self.scene = session.scene
        self.command = np.array([0.28, 0.0], float)
        self.grip = False
        self.last_command = 0.0
        self.begin()

    def begin(self) -> None:
        self.command[:] = [0.28, 0.0]
        self.grip = False
        self.last_command = time.monotonic()
        self.pinch = self.s.home_pinch.copy()
        self.rot = self.s.home_rot.copy()
        self.hand = [HAND_OPEN.copy(), HAND_OPEN.copy()]
        self.job: Follower | None = None
        self.side = 0
        self.held = -1
        self.held_shape = "cube"
        self.held_bin = "center"
        self.held_finger = 0.0
        self.slots = {b: 0 for b in self.scene.bin_centers}
        self.phase = "idle"
        self.release_target = None
        self.release_bin = None

    @property
    def done(self) -> bool:
        return self.held < 0 and not self.job and self.s.progress()[0] >= len(self.s.layouts[0].placements)

    def set_command(self, value: dict) -> None:
        try:
            x, y = float(value["x"]), float(value["y"])
        except (KeyError, TypeError, ValueError):
            return
        if not np.isfinite([x, y]).all():
            return
        lo, hi = self.scene.region
        # The pointer may reach bins just outside the mat, but never the robot or the room.
        self.command[:] = np.clip([x, y], lo - 0.18, hi + 0.18)
        self.grip = bool(value.get("grip", False))
        self.last_command = time.monotonic()

    def _near_part(self, obs: Observation):
        idx = self.s.echo.lead_idx
        if len(idx) == 0:
            return None
        p = obs.obj_pos[0, idx]
        d = np.linalg.norm(p[:, :2] - self.command[None], axis=1)
        k = int(np.argmin(d))
        if d[k] > 0.05 or not self.s._on_mat(p[k:k + 1], 0.03)[0]:
            return None
        return int(idx[k]), p[k].copy(), self.s.layouts[0].placements[k]

    def _start_pick(self, obs: Observation) -> bool:
        found = self._near_part(obs)
        if found is None:
            return False
        index, pos, placement = found
        ob = self.scene.by_name[placement.name]
        side_name = arm_for(self.s.layouts[0].mapping[ob.color], pos[1])
        self.side = SIDES.index(side_name)
        rest = [self.scene.by_name[p.name] for p in self.s.layouts[0].placements if p.name != placement.name]
        others = np.array([obs.obj_pos[0, o.index, :2] for o in rest]).reshape(-1, 2)
        radius = part_radius(np.array([SHAPE_CODE[o.shape] for o in rest], int), self.scene.size)
        finger = float(self.s.reach.best(np.array(SHAPE_CODE[ob.shape]), np.array(float(obs.obj_yaw[0, index])),
                                         pos[:2], self.side, others=others, radius=radius)[0])
        wps = plan_pick_place(self.scene, side_name, ob.shape, pos, float(obs.obj_yaw[0, index]),
                              self.s.layouts[0].mapping[ob.color], slot=self.slots[self.s.layouts[0].mapping[ob.color]],
                              speed=1.0, finger_dir=finger)
        self.job = Follower(self.pinch[self.side], self.rot[self.side], self.hand[self.side], wps[:4])
        self.held, self.held_shape, self.held_bin, self.held_finger = index, ob.shape, self.s.layouts[0].mapping[ob.color], finger
        self.phase = "pick"
        return True

    def _bin_at(self, xy: np.ndarray) -> str | None:
        for name in self.scene.bin_centers:
            lo, hi = self.scene.bin_inner(name)
            if np.all(xy >= lo[:2]) and np.all(xy <= hi[:2]):
                return name
        return None

    def _start_release(self) -> None:
        side = SIDES[self.side]
        target_bin = self._bin_at(self.command)
        if target_bin is not None:
            point = drop_point(self.scene, side, target_bin, self.slots[target_bin])
            self.release_bin = target_bin
        else:
            point = np.r_[self.command, self.scene.rest_z + 0.004]
            self.release_bin = None
        carry_z = max(self.pinch[self.side][2], point[2] + 0.03)
        rot = self.rot[self.side]
        if self.release_bin is not None:
            rot = palm_rotation(side, drop_direction(side, target_bin))
        self.release_target = point
        self.job = Follower(self.pinch[self.side], self.rot[self.side], self.hand[self.side], [
            Waypoint("lower", point, rot, HAND_CLOSED, 0.5, hold=0.1, tol=0.012),
            Waypoint("release", point, rot, HAND_RELEASE, 0.4, hold=0.3, tol=0.05),
            Waypoint("retreat", np.r_[point[:2], carry_z + 0.02], rot, HAND_OPEN, 0.6, tol=0.05),
        ])
        self.phase = "release"

    def update(self, obs: Observation) -> LeadCommand:
        s = self.side
        if self.phase == "idle":
            # Keep a hand over the pointer even before the first pinch.  The other rests at home.
            near = self._near_part(obs)
            s = self.side if near is None else SIDES.index(arm_for(self.s.layouts[0].mapping[self.scene.by_name[near[2].name].color], near[1][1]))
            self.side = s
            z = self.scene.rest_z + 0.14
            self.pinch[s] = np.r_[self.command, z]
            self.rot[s] = palm_rotation(SIDES[s], 0.0)
            self.hand[s] = HAND_OPEN.copy()
            if self.grip:
                self._start_pick(obs)
        if self.phase == "pick" and self.job is not None:
            st = self.job.update(DT, obs.pinch[0, self.side], obs.rot[0, self.side], obs.hand_q[0, self.side * 7:(self.side + 1) * 7])
            self.pinch[self.side], self.rot[self.side], self.hand[self.side] = st.point, st.rot, st.hand
            if self.job.done:
                self.job = None
                self.phase = "carry"
        elif self.phase == "carry":
            self.pinch[self.side] = np.r_[self.command, self.scene.rest_z + 0.14]
            self.hand[self.side] = HAND_CLOSED.copy()
            if not self.grip:
                self._start_release()
        elif self.phase == "release" and self.job is not None:
            st = self.job.update(DT, obs.pinch[0, self.side], obs.rot[0, self.side], obs.hand_q[0, self.side * 7:(self.side + 1) * 7])
            self.pinch[self.side], self.rot[self.side], self.hand[self.side] = st.point, st.rot, st.hand
            if self.job.done:
                if self.release_bin is not None:
                    self.slots[self.release_bin] += 1
                self.job = None
                self.held = -1
                self.phase = "idle"
                self.hand[self.side] = HAND_OPEN.copy()
        if self.phase == "pick" and not self.grip and self.job is not None:
            # A pinch released before lift is simply cancelled.
            self.job = None
            self.phase = "idle"
            self.held = -1
            self.hand[self.side] = HAND_OPEN.copy()
        other = 1 - self.side
        self.pinch[other] = self.s.home_pinch[other]
        self.rot[other] = self.s.home_rot[other]
        self.hand[other] = HAND_OPEN.copy()
        return LeadCommand(self.pinch.copy(), self.rot.copy(), hand14(self.hand[0], self.hand[1]))

    def hud(self) -> dict:
        # A compact synthetic hand stream.  The browser uses it exactly like glove points.
        return {"calibrated": True, "source": self.source, "stand_in": False,
                "driving": True, "hands": None, "left": {"connected": True, "hz": 30.0, "status": "tracking"},
                "right": {"connected": True, "hz": 30.0, "status": "tracking"}}

