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
from kaiku.sim.grasp import HAND_CLOSED, HAND_OPEN, HAND_RELEASE, Waypoint, arm_for, hand14, palm_rotation, plan_pick_place
from kaiku.sim.ik import SIDES
from kaiku.sim.motion import Follower
from kaiku.sim.reach import part_radius
from kaiku.sim.twin import CONTROL_HZ

DT = 1.0 / CONTROL_HZ
HOVER = 0.11                  # m above the mat the open hand rides while it follows the visitor
SPEED = 0.30                  # m/s the hand's target may travel after the visitor's point: faster than
                              # this and the echoes, each with its own way to go, lose the part they carry
ACCEL = 1.2                   # m/s2 it may gain or lose: a hand, not a cursor
PACE = 1.25                   # the grasp itself, against the stand-in's unhurried 1.0


class VisitorOperator:
    """An operator controlled by ``{x, y, grip}`` commands: a pointer, or a hand in front of a camera.

    The visitor says where the hand is and when it pinches. The pinch itself, and the way into a
    bin, are the stand-in's own motions, so a visitor's demonstration is as clean as a scripted one
    and the echoes can redo it."""

    source = "visitor"
    label = "pointer"

    def __init__(self, session: Session, pace: float = PACE, speed: float = SPEED, hover: float = HOVER,
                 approach_s: tuple[float, float] = (0.35, 1.2)):
        self.s = session
        self.scene = session.scene
        self.pace, self.speed, self.hover, self.approach_s = pace, speed, hover, approach_s
        self.command = np.array([0.28, 0.0], float)
        self.grip = False
        self.last_command = 0.0
        self.begin()

    def begin(self) -> None:
        self.command[:] = [0.28, 0.0]
        self.grip = False
        self.last_command = time.monotonic()
        self.pinch = self.s.home_pinch.copy()
        self.vel = np.zeros((2, 3))                           # of each hand's target while it follows the visitor
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
        self.plan = None
        self.release_target = None
        self.release_bin = None
        self.carry_z = self.scene.rest_z + self.hover
        self.input = "pointer"                                # or "camera": what the visitor drives with
        self.points = None                                    # the visitor's hand as the camera saw it, for the page

    def _toward(self, s: int, target: np.ndarray) -> None:
        """Move one hand's target after ``target`` at a hand's pace, not a pointer's."""
        step = target - self.pinch[s]
        far = float(np.linalg.norm(step))
        # As fast as it can still stop in the way that is left, never faster than the echoes can follow.
        want = step / max(far, 1e-9) * min(self.speed, np.sqrt(2.0 * ACCEL * far))
        change = want - self.vel[s]
        size = float(np.linalg.norm(change))
        self.vel[s] += change if size <= ACCEL * DT else change * (ACCEL * DT / size)
        self.pinch[s] = self.pinch[s] + self.vel[s] * DT

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
        self.input = "camera" if value.get("input") == "camera" else "pointer"
        points = value.get("points")
        self.points = points if isinstance(points, list) and len(points) <= 42 else None
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
                              speed=self.pace, finger_dir=finger)
        # The hand is already over the part: the way down is short.
        wps[0].duration = float(np.clip(3.0 * np.linalg.norm(wps[0].point - self.pinch[self.side]), *self.approach_s))
        self.carry_z = float(wps[3].point[2])
        self.plan = (side_name, ob.shape, pos.copy(), float(obs.obj_yaw[0, index]), finger)
        self.vel[self.side] = 0.0
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
        self.release_bin = target_bin
        if target_bin is not None:
            # Over a bin: over its free slot, turn the palm, down, let go, up. The stand-in's own way.
            name, shape, pos, yaw, finger = self.plan
            wps = plan_pick_place(self.scene, name, shape, pos, yaw, target_bin, slot=self.slots[target_bin],
                                  speed=self.pace, finger_dir=finger)[4:]
            wps[0].duration = float(np.clip(4.0 * np.linalg.norm(wps[0].point - self.pinch[self.side]), 0.5, 1.3))
            self.release_target = wps[1].point
        else:
            # Anywhere else: put it down where the hand is.
            point = np.r_[self.command, self.scene.rest_z + 0.004]
            rot, up = self.rot[self.side], max(self.pinch[self.side][2], point[2] + 0.03) + 0.02
            self.release_target = point
            wps = [Waypoint("lower", point, rot, HAND_CLOSED, 0.6, hold=0.1, tol=0.012),
                   Waypoint("release", point, rot, HAND_RELEASE, 0.4, hold=0.3, tol=0.05),
                   Waypoint("retreat", np.r_[point[:2], up], rot, HAND_OPEN, 0.6, tol=0.05)]
        self.vel[self.side] = 0.0
        self.job = Follower(self.pinch[self.side], self.rot[self.side], self.hand[self.side], wps)
        self.phase = "release"

    def update(self, obs: Observation) -> LeadCommand:
        s = self.side
        if self.phase == "idle":
            # Keep a hand over the pointer even before the first pinch.  The other rests at home.
            near = self._near_part(obs)
            s = self.side if near is None else SIDES.index(arm_for(self.s.layouts[0].mapping[self.scene.by_name[near[2].name].color], near[1][1]))
            self.side = s
            self._toward(s, np.r_[self.command, self.scene.rest_z + self.hover])
            self.rot[s] = palm_rotation(SIDES[s], 0.0)
            self.hand[s] = HAND_OPEN.copy()
            if self.grip:
                self._start_pick(obs)
        if self.phase == "pick" and self.job is not None:
            st = self.job.update(DT, obs.pinch[0, self.side], obs.rot[0, self.side])
            self.pinch[self.side], self.rot[self.side], self.hand[self.side] = st.point, st.rot, st.hand
            if self.job.done:
                self.job = None
                self.phase = "carry"
        elif self.phase == "carry":
            self._toward(self.side, np.r_[self.command, self.carry_z])
            self.hand[self.side] = HAND_CLOSED.copy()
            if not self.grip:
                self._start_release()
        elif self.phase == "release" and self.job is not None:
            st = self.job.update(DT, obs.pinch[0, self.side], obs.rot[0, self.side])
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
        self._toward(other, self.s.home_pinch[other])
        self.rot[other] = self.s.home_rot[other]
        self.hand[other] = HAND_OPEN.copy()
        return LeadCommand(self.pinch.copy(), self.rot.copy(), hand14(self.hand[0], self.hand[1]))

    def hud(self) -> dict:
        # A compact synthetic hand stream.  The browser uses it exactly like glove points.
        return {"calibrated": True, "source": self.source, "stand_in": False, "driving": True, "hands": None,
                "input": self.input, "points": self.points, "phase": self.phase,
                "holding": self.held >= 0 and self.phase in ("carry", "release"),
                "over": self._bin_at(self.command),
                "left": {"connected": True, "hz": 30.0, "status": "tracking"},
                "right": {"connected": True, "hz": 30.0, "status": "tracking"}}

