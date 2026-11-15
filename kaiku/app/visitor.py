"""A visitor drives the lead robot: with a pointer, with a hand in front of a camera, with one glove.

Assisted (the default): the visitor says where the hand is and when it pinches, and the pinch itself
and the way into a bin are the stand-in's own motions. While the hand is free the robot's fingers
already follow the visitor's.

Their own grasp (``own``): nothing is scripted. The hand goes where the visitor's hand is, comes down
as far as the visitor lowers theirs, and its fingers close as far as the visitor's fingers do. The
fingers come from the camera's 21 landmarks or from one Smartglove (``kaiku.teleop.fingers``), through
the gloves' filters and retargeting. The robot still chooses the arm and turns its palm to the part.

The grasp is the visitor's choice and only theirs: when nothing sees their hand while they do their own
grasp, the robot keeps what it has where it has it, and the page says the hand is not seen.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from kaiku.app.session import LeadCommand, Observation, Session
from kaiku.fanout.echo import SHAPE_CODE
from kaiku.hand import Calibration, Teleop
from kaiku.hand.geometry import quat_rotate
from kaiku.sim.grasp import (HAND_CLOSED, HAND_OPEN, HAND_RELEASE, NATURAL_DIR, Waypoint, arm_for, drop_direction,
                             drop_point, hand14, palm_rotation, plan_pick_place)
from kaiku.sim.ik import SIDES
from kaiku.sim.motion import Follower
from kaiku.sim.reach import part_radius
from kaiku.sim.twin import CONTROL_HZ
from kaiku.teleop.fingers import LANDMARKS, GloveFingers, solve_landmarks
from kaiku.teleop.retarget import Retarget, closure_of_posture

DT = 1.0 / CONTROL_HZ
HOVER = 0.11                  # m above the mat the open hand rides while it follows the visitor
SPEED = 0.30                  # m/s the hand's target may travel after the visitor's point: faster than
                              # this and the echoes, each with its own way to go, lose the part they carry
ACCEL = 1.2                   # m/s2 it may gain or lose: a hand, not a cursor
PACE = 1.25                   # the grasp itself, against the stand-in's unhurried 1.0
FRESH_S = 0.5                 # landmarks older than this are no hand
DEPTH_FULL = 0.30             # ln of how much larger the hand is in the picture when it is all the way down
TURN = 2.5                    # rad/s the palm may turn after the part it is over
PINCH_ON, PINCH_OFF = 0.045, 0.065   # m between a gloved thumb and index finger: a pinch, and open again
LIFT_AFTER = 0.3              # s the fingers stay closed before the hand lifts what they hold
MISSED_S = 1.5                # s the page says a pinch took nothing, after the pinch
DROPPED_S = 4.0               # s it says a part fell out of the hand
LANDED_S = (2.5, 6.0)         # s it says where a part that left the hand came to rest: in its bin, anywhere else
REST = 0.02                   # m/s under which a part that was let go lies still
REST_TICKS, REST_S = 5, 1.5   # for this many ticks in a row; or it is judged where it is after this long
BESIDE = 0.05                 # m around a bin that still reads as "beside the bin"
ANCHOR_S = 1.5                # s a gloved hand is held flat and still before where it is counts as the middle
ABOVE = 0.10                  # m over the hover height a hand on a Coil Pro may rise


class VisitorOperator:
    """An operator controlled by ``{x, y, grip}`` commands: a pointer, or a hand in front of a camera.

    The visitor says where the hand is and when it pinches. The pinch itself, and the way into a
    bin, are the stand-in's own motions, so a visitor's demonstration is as clean as a scripted one
    and the echoes can redo it."""

    source = "visitor"
    label = "pointer"

    def __init__(self, session: Session, pace: float = PACE, speed: float = SPEED, hover: float = HOVER,
                 approach_s: tuple[float, float] = (0.35, 1.2), glove: GloveFingers | None = None):
        self.s = session
        self.scene = session.scene
        self.glove = glove                                    # one Smartglove's fingers, when there is one
        # With a Coil Pro that glove also has a place: where it was held flat and still is the middle of the table.
        self.anchor = None                                    # (position, turn about the vertical) of that pose
        self.still_since = None
        self.last_wrist = None
        self.height = 0.0                                     # m the hand is above that pose, in the robot's metres
        self.flat_s = 0.0
        self.own = False                                      # the visitor does the grasp themselves
        self.asked = False                                    # has a page said how it wants the grasp done
        self.rt = Retarget()
        self.pace, self.speed, self.hover, self.approach_s = pace, speed, hover, approach_s
        # Where the visitor's point may be: over the parts and over the bins, never at the robot or off the table.
        lo, hi = self.scene.region
        inner = [self.scene.bin_inner(b) for b in self.scene.bin_centers]
        far = max(float(c[0]) for c in self.scene.bin_centers.values())
        self.reach = (np.array([lo[0] - 0.05, min(float(a[1]) for a, _ in inner) + 0.01]),
                      np.array([far + 0.01, max(float(b[1]) for _, b in inner) - 0.01]))
        self.command = np.array([0.28, 0.0], float)
        self.grip = False
        self.last_command = 0.0
        self.begin()

    def turn(self) -> None:
        """Another visitor has the robots: what the one before chose and showed is not theirs."""
        if self.glove is None:                                # a glove on this machine is one person's sitting
            self.own, self.asked = False, False
        self.grip, self.world, self.points = False, None, None

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
        self.seen: Observation | None = None                  # the last tick's, for what the page is told
        # The visitor's own fingers: where they come from, and the gloves' filters and ranges for them.
        self.world = None                                     # (21, 3) landmarks in metres, as the camera saw them
        self.world_at = 0.0
        self.right = True                                     # the hand in the picture is a right hand
        self.depth = 0.0                                      # ln of how much nearer the hand is than where it started
        self.fingers = None                                   # "camera", "glove" or None: whose fingers these are
        self.human = None                                     # the solved hand, for the page and the recording
        self.lean = 0.0
        still = {side: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])) for side in SIDES}
        self.filter = Teleop(Calibration(0.0, self.rt.scale, still, self.rt.ready()))
        self.range = self.rt.gap_range()
        self.dir = [0.0, 0.0]                                 # the way each palm's fingers point now
        self.pinched = False                                  # the solved hand's own pinch
        self.posture = None                                   # the Dex3 posture their fingers ask for
        self.closed_at = None                                 # since when the fingers have been closed, their own grasp
        self.at_part = False                                  # ... around a part, lifted yet or not
        self.was_own = False
        self.drove = None                                     # whose fingers drove in this scene, for its label
        self.grasped_own = False                              # and whether a grasp in it was the visitor's own
        # What the page tells the visitor and nothing else reads: a pinch with no part under it, a part that fell.
        self.missed_at = -1e9
        self.dropped_at = -1e9
        self.again = False                                    # a grasp took nothing: the next one starts with a new press
        self.fell: dict[int, int] = {}                        # times the helped grasp has come up without each part
        self.fell_n = 0                                       # ... the part it last came up without
        self.away = 0                                         # control steps the part being carried has not been at the hand
        self.held_since = None                                # since when their own fingers have held a part
        self.lost = False                                     # their own grasp, and nothing sees the hand: it holds
        self.let_go = None                                    # the part that just left the hand, until it lies still
        self.landed = None                                    # where it came to rest, for the page to say
        self.there = None                                     # since when the hand has been as near the pointer as it gets

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

    def close(self) -> None:
        if self.glove is not None:
            self.glove.close()

    def who(self) -> str:
        """Who drove, for the recording."""
        position = "camera" if self.input == "camera" else "pointer"
        fingers = self.fingers or self.drove
        grasp = ", own grasp" if self.grasped_own else ", assisted grasp"
        if fingers == "glove" and self.input == "coil":
            return f"visitor (one Smartglove with a Coil Pro: fingers and position{grasp})"
        if fingers == "glove":
            return f"visitor (one Smartglove's fingers, position from the {position}{grasp})"
        return f"visitor (a hand seen by a camera{grasp})" if fingers == "camera" else f"visitor ({position})"

    def hand_points(self) -> np.ndarray:
        """The visitor's solved hand, wrist-relative, the way the gloves' hands are saved."""
        points = np.full((2, 21, 3), np.nan, dtype=np.float32)
        if self.human is not None:
            points[self.side, 0] = 0.0
            points[self.side, 1:] = self.rt.models[SIDES[self.side]].points(self.human.joints).reshape(20, 3)
        return points

    def set_command(self, value: dict) -> None:
        try:
            x, y = float(value["x"]), float(value["y"])
        except (KeyError, TypeError, ValueError):
            return
        if not np.isfinite([x, y]).all():
            return
        if self.input != "coil":                              # a hand on a Coil Pro says where it is itself
            self.command[:] = np.clip([x, y], *self.reach)
            self.input = "camera" if value.get("input") == "camera" else "pointer"
        self.grip = bool(value.get("grip", False))
        points = value.get("points")
        self.points = points if isinstance(points, list) and len(points) <= 42 else None
        self.own = bool(value.get("own", False))
        self.asked = True
        self.last_command = time.monotonic()
        world = value.get("world")
        if isinstance(world, list) and len(world) == 3 * LANDMARKS:
            try:
                world = np.asarray(world, float).reshape(LANDMARKS, 3)
            except (TypeError, ValueError):
                return
            if np.isfinite(world).all() and np.abs(world).max() < 1.0:
                self.world, self.world_at = world, self.last_command
                self.right = bool(value.get("right", True))
                try:
                    depth = float(value.get("depth", 0.0))
                except (TypeError, ValueError):
                    depth = 0.0
                self.depth = float(np.clip(depth, -1.0, 1.0)) if np.isfinite(depth) else 0.0

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

    def _over(self) -> str | None:
        """The bin the hand is over, for the page's "let go". The hand follows the pointer at its own speed:
        while it carries a part and is still on its way, it is not over the bin the pointer is over."""
        name = self._bin_at(self.command)
        if name is None or self.seen is None or self.phase != "carry":
            self.there = None
            return name
        if np.linalg.norm(self.seen.pinch[0, self.side, :2] - self.command) < 0.05:
            return name
        # The arm does not reach every corner of a bin. Once the hand has gone as far as it gets, letting go
        # still puts the part in that bin, so the page says so and does not ask for more.
        now = time.monotonic()
        arrived = np.linalg.norm(self.pinch[self.side][:2] - self.command) < 0.01
        self.there = (self.there or now) if arrived else None
        return name if self.there is not None and now - self.there > 1.0 else None

    @property
    def busy(self) -> bool:
        """A part is in the hand or on its way into a bin: not the moment to deal a new scene."""
        return (self.held >= 0 and self.phase in ("pick", "carry", "release")) or self.held_since is not None

    def _start_release(self) -> None:
        side = SIDES[self.side]
        target_bin = self._bin_at(self.command)
        self.release_bin = target_bin
        if target_bin is not None:
            # Over a bin: over its free slot, turn the palm, down, let go, up. The stand-in's own way.
            name, shape, pos, yaw, finger = self.plan
            wps = plan_pick_place(self.scene, name, shape, pos, yaw, target_bin, slot=self.slots[target_bin],
                                  speed=self.pace, finger_dir=finger)[4:]
            wps[0].duration = float(np.clip(4.0 * np.linalg.norm(wps[0].point - self.pinch[self.side]), 0.5, 2.5))
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

    # ------------------------------------------------------------------ the visitor's own fingers
    def _fingers(self, side: int, now: float) -> np.ndarray | None:
        """The Dex3 posture (7, left-hand order) the visitor's fingers ask of the ``side`` hand, or None
        when nothing sees their fingers. A glove counts before the camera: it measures them."""
        name, hand, source = SIDES[side], None, None
        if self.glove is not None:
            hand = self.glove.hand(name, now)
            source = "glove" if hand is not None else None
        if hand is None and self.world is not None and now - self.world_at < FRESH_S:
            hand, self.lean = solve_landmarks(name, self.world, self.right, self.rt.models[name], now)
            source = "camera"
        if source != self.fingers:                             # another sensor, another hand: its own range
            self.range = self.rt.gap_range()
        self.fingers, self.human = source, hand
        self.drove = source or self.drove
        if source == "glove" and hand.has_position:
            self._place(hand, now)
        elif self.input == "coil":                             # the coil lost the hand: the page's point again
            self.input, self.anchor, self.still_since = "pointer", None, None
        if hand is None:
            self.pinched, self.posture = False, None
            return None
        hand = self.filter.update({name: hand}, now).hands[name] or hand      # the gloves' own filters
        # How wide this hand opens and how far it closes, brought onto what the robot's fingers are driven by.
        self.range.open(hand.pinch if self.range.high is None else np.maximum(self.range.high, hand.pinch))
        gaps = self.range(hand.pinch)
        self.human = replace(hand, pinch=gaps, gaps=gaps)
        self.pinched = float(gaps[0]) < (PINCH_OFF if self.pinched else PINCH_ON)
        self.posture = self.rt.fingers(self.human, curl=True)
        return self.posture

    def _place(self, hand, now: float) -> None:
        """A gloved hand on a Coil Pro: where it is, is where the robot's hand is. The pose it is first
        held in, flat and still, is the middle of the table at the height the open hand rides."""
        self.input = "coil"
        if not self.asked:
            self.own = True                                    # nobody at a page: the hand does all of it
        back = quat_rotate(hand.wrist_quat, (0.0, 0.0, 1.0))
        if self.anchor is None:
            last = self.last_wrist
            still = last is not None and np.linalg.norm(hand.wrist_pos - last[0]) < 0.006 \
                and abs(float(np.dot(hand.wrist_quat, last[1]))) > 0.999
            flat = abs(float(back[2])) > 0.3 and abs(float(back[0])) < 0.45 and abs(float(back[1])) < 0.45
            self.last_wrist = (hand.wrist_pos.copy(), hand.wrist_quat.copy())
            if not (still and flat):
                self.still_since = None
            elif self.still_since is None:
                self.still_since = now
            self.flat_s = 0.0 if self.still_since is None else now - self.still_since
            if self.flat_s >= ANCHOR_S:
                if back[2] < 0.0:                              # palm down reads as palm up: the coil is the other way up
                    self.glove.flip()
                    self.still_since, self.last_wrist = None, None
                    return
                ahead = quat_rotate(hand.wrist_quat, (1.0, 0.0, 0.0))
                self.anchor = (hand.wrist_pos.copy(), -float(np.arctan2(ahead[1], ahead[0])))
                self.command[:], self.height = [0.28, 0.0], 0.0
            return
        origin, yaw = self.anchor
        c, sn = np.cos(yaw), np.sin(yaw)
        d = (hand.wrist_pos - origin) * self.rt.scale
        self.command[:] = np.clip([0.28 + c * d[0] - sn * d[1], sn * d[0] + c * d[1]], *self.reach)
        self.height = float(d[2])
        self.last_command = now

    def recentre(self) -> None:
        """Where the gloved hand is held flat and still next is the middle of the table again."""
        self.anchor, self.still_since, self.last_wrist = None, None, None

    @property
    def gripping(self) -> bool:
        """Is the visitor pinching: the page's word for it, or a gloved hand's own thumb and finger."""
        if self.fingers != "glove":
            return self.grip
        return self.pinched if self.input in ("camera", "coil") else (self.grip or self.pinched)

    def _floor(self, xy: np.ndarray) -> tuple[float, float]:
        """How low the pinch point may go at ``xy`` and where it rides: onto the mat from the hover height,
        or over a bin from just above where a part is let go into it."""
        for name in self.scene.bin_centers:
            lo, hi = self.scene.bin_inner(name)
            if np.all(xy >= lo[:2] - 0.04) and np.all(xy <= hi[:2] + 0.04):
                drop = float(drop_point(self.scene, SIDES[self.side], name)[2])
                return drop, drop + 0.02
        return self.scene.rest_z + 0.004, self.scene.rest_z + self.hover

    def _own(self, obs: Observation, posture: np.ndarray, now: float) -> None:
        """Nothing scripted: the hand is where the visitor's is, as low as theirs, as closed as theirs.
        The robot chooses the arm, turns its palm to the part under it and lifts what the fingers hold."""
        u = closure_of_posture(posture)[1]                      # the index finger: 0 open, 1 the pinch that holds
        if u > 0.8:
            self.closed_at = self.closed_at or now
        elif u < 0.3:
            self.closed_at = None
        free = u < 0.35
        near = self._near_part(obs) if free else None
        if near is not None:
            self.side = SIDES.index(arm_for(self.s.layouts[0].mapping[self.scene.by_name[near[2].name].color], near[1][1]))
        s, name = self.side, SIDES[self.side]
        # How far down: a hand pushed towards the camera; with a pointer, the press.
        down = float(np.clip(self.depth / DEPTH_FULL, 0.0, 1.0)) if self.input == "camera" else float(self.grip)
        floor, top = self._floor(self.command)
        if self.input == "coil":                               # as high as the hand itself: nothing lifts for it
            z = float(np.clip(self.scene.rest_z + self.hover + self.height, floor, self.scene.rest_z + self.hover + ABOVE))
        else:
            if self.closed_at is not None and now - self.closed_at > LIFT_AFTER:
                down = 0.0                                     # closed on it: up it comes
                top = max(top, self.scene.rest_z + self.hover)
            z = top - down * (top - floor)
        self._toward(s, np.r_[self.command, z])
        # Which way the palm points: across the part under it, into the bin over it, otherwise the arm's own way.
        want, over = NATURAL_DIR[name], self._bin_at(self.command)
        if over is not None:
            want = drop_direction(name, over)
        elif near is not None:
            index, pos, placement = near
            ob = self.scene.by_name[placement.name]
            want = float(self.s.reach.best(np.array(SHAPE_CODE[ob.shape]), np.array(float(obs.obj_yaw[0, index])),
                                           pos[:2], s)[0])
        if free or over is not None:
            step = (want - self.dir[s] + np.pi) % (2 * np.pi) - np.pi
            self.dir[s] += float(np.clip(step, -TURN * DT, TURN * DT))
        self.rot[s] = palm_rotation(name, self.dir[s])
        self.hand[s] = posture
        # What the page says next: is a part in the hand, and which.
        was = self.held
        self.held, self.phase = -1, "idle"
        self.at_part = False
        idx = self.s.echo.lead_idx
        if not free and len(idx):
            p = obs.obj_pos[0, idx]
            d = np.linalg.norm(p - obs.pinch[0, s], axis=1)
            k = int(np.argmin(d))
            self.at_part = bool(d[k] < 0.06)
            if d[k] < 0.06 and p[k, 2] > self.scene.rest_z + 0.02:
                ob = self.scene.by_name[self.s.layouts[0].placements[k].name]
                self.held, self.phase = int(idx[k]), "carry"
                self.grasped_own = True
                self.held_shape, self.held_bin = ob.shape, self.s.layouts[0].mapping[ob.color]
        self._own_news(obs, was, now)

    def _own_news(self, obs: Observation, was: int, now: float) -> None:
        """Their own grasp, for the page only: fingers closed a while on nothing, or a part that left the hand
        anywhere but over a bin. Nothing here moves the robot."""
        if self.held >= 0:
            self.held_since = self.held_since or now
            return
        if was >= 0 and self.held_since is not None and now - self.held_since > 0.3 \
                and self._bin_at(obs.obj_pos[0, was, :2]) is None:
            self.dropped_at = now
        self.held_since = None
        if was >= 0 and self.closed_at is not None:
            self.closed_at = now                               # fingers that just let a part go have not closed on nothing
        # Closed around a part, the fingers get the time the lift takes before the page calls it a miss.
        wait = 2.5 if self.at_part else 1.2
        if self.closed_at is not None and now - self.closed_at > wait and now - self.dropped_at > DROPPED_S:
            self.missed_at = now

    def _left_hand(self, obs: Observation, index: int, now: float) -> None:
        """A part the hand carried is no longer in it: watch it until it lies still."""
        self.let_go = {"index": index, "at": now, "last": obs.obj_pos[0, index].copy(), "still": 0}
        self.landed = None

    def _rest(self, obs: Observation, now: float) -> None:
        """Where the part that left the hand came to rest. For the page only: nothing here moves the robot."""
        go = self.let_go
        if go is None:
            return
        pos = obs.obj_pos[0, go["index"]]
        if pos[2] > self.scene.rest_z + 0.02 and np.linalg.norm(pos - obs.pinch[0, self.side]) < 0.06:
            go["at"], go["still"] = now, 0                     # off the table and at the fingers: not let go yet
            return
        go["still"] = go["still"] + 1 if np.linalg.norm(pos - go["last"]) < REST * DT else 0
        go["last"] = pos.copy()
        if go["still"] < REST_TICKS and now - go["at"] < REST_S:
            return
        self.let_go = None
        self.landed = dict(self.where(go["index"], pos), at=now)

    def where(self, index: int, pos: np.ndarray) -> dict:
        """Where a part lies, in the page's words: ``in`` its own bin, in the ``wrong`` one, ``beside`` a bin,
        elsewhere on the ``table``, or ``off`` it. ``reach``: the hand can take it again from there."""
        pos = np.asarray(pos, float)
        ob = next(o for o in self.scene.objects if o.index == index)
        belongs = self.s.layouts[0].mapping.get(ob.color)
        inside = self.scene.bin_of(pos)
        if not self.scene.on_counter(pos):
            where = "off"
        elif inside is not None:
            where = "in" if inside == belongs else "wrong"
        else:
            half = self.scene.bin_outer[::-1] / 2 + BESIDE
            near = any(np.all(np.abs(pos[:2] - c[:2]) <= half) for c in self.scene.bin_centers.values())
            where = "beside" if near else "table"
        return {"where": where, "color": ob.color, "bin": belongs,
                "reach": bool(where in ("beside", "table") and self.s._on_mat(pos[None], 0.03)[0])}

    def update(self, obs: Observation) -> LeadCommand:
        now = time.monotonic()
        carried = self.held if self.phase in ("carry", "release") else -1
        command = self._update(obs, now)
        if carried >= 0 and self.held != carried:
            self._left_hand(obs, carried, now)
        elif self.held >= 0:
            self.let_go = self.landed = None                   # another part is in hand: the last one is old news
        self._rest(obs, now)
        return command

    def _update(self, obs: Observation, now: float) -> LeadCommand:
        s = self.side
        self.seen = obs
        posture = self._fingers(s, now)
        # Their own grasp and nothing sees their hand: the robot keeps what it has, where it has it. It does
        # not take the grasp over. The grasp changes when the visitor changes it, never by itself.
        self.lost = self.own and posture is None and self.job is None
        if self.lost:
            self.was_own = True
            self._toward(s, self.pinch[s].copy())              # comes to rest; the fingers stay as they were
            other = 1 - s
            self._toward(other, self.s.home_pinch[other])
            self.rot[other] = self.s.home_rot[other]
            self.hand[other] = HAND_OPEN.copy()
            return LeadCommand(self.pinch.copy(), self.rot.copy(), hand14(self.hand[0], self.hand[1]))
        own = self.own and posture is not None and self.job is None
        if own or self.was_own:
            if not own:                                        # back to the assisted grasp: nothing is in the hand
                self.phase, self.held, self.closed_at = "idle", -1, None
            self.was_own = own
        if own:
            self._own(obs, posture, now)
            other = 1 - self.side
            self._toward(other, self.s.home_pinch[other])
            self.rot[other] = self.s.home_rot[other]
            self.hand[other] = HAND_OPEN.copy()
            return LeadCommand(self.pinch.copy(), self.rot.copy(), hand14(self.hand[0], self.hand[1]))
        grip = self.gripping
        if self.phase == "idle":
            # Keep a hand over the pointer even before the first pinch.  The other rests at home.
            near = self._near_part(obs)
            s = self.side if near is None else SIDES.index(arm_for(self.s.layouts[0].mapping[self.scene.by_name[near[2].name].color], near[1][1]))
            self.side = s
            self._toward(s, np.r_[self.command, self.scene.rest_z + self.hover])
            self.rot[s] = palm_rotation(SIDES[s], 0.0)
            self.dir[s] = 0.0
            # The robot's fingers already follow the visitor's while the hand is free.
            self.hand[s] = HAND_OPEN.copy() if posture is None else posture
            if not grip:
                self.again = False
            elif not self.again and not self._start_pick(obs):
                self.missed_at = now                           # nothing under the hand: the page says so
        if self.phase == "pick" and self.job is not None:
            st = self.job.update(DT, obs.pinch[0, self.side], obs.rot[0, self.side])
            self.pinch[self.side], self.rot[self.side], self.hand[self.side] = st.point, st.rot, st.hand
            if self.job.done:
                self.job = None
                self.phase, self.away = "carry", 0
        elif self.phase == "carry":
            self._toward(self.side, np.r_[self.command, self.carry_z])
            self.hand[self.side] = HAND_CLOSED.copy()
            away = self.held < 0 or np.linalg.norm(obs.obj_pos[0, self.held] - obs.pinch[0, self.side]) > 0.08
            self.away = self.away + 1 if away else 0
            if self.away * DT > 0.4:
                # The hand is empty: the grasp took nothing or the part slipped out. The page says so, the hand
                # opens, and nothing is carried to a bin that is not in it.
                self.dropped_at, self.again = now, True
                if self.held >= 0:                             # the second time from the same place the page says what else to do
                    self.fell_n = self.fell[self.held] = self.fell.get(self.held, 0) + 1
                self.phase, self.held, self.away = "idle", -1, 0
                self.hand[self.side] = HAND_OPEN.copy()
            elif not grip:
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
        if self.phase == "pick" and not grip and self.job is not None:
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

    def _table(self) -> dict:
        """What the visitor's page draws on the table: the parts still to sort, each with its bin, and the bins."""
        lay, parts = self.s.layouts[0], []
        for p in lay.placements:
            ob = self.scene.by_name[p.name]
            at = p.pos if self.seen is None else self.seen.obj_pos[0, ob.index]
            parts.append({"x": round(float(at[0]), 3), "y": round(float(at[1]), 3), "color": ob.color,
                          "bin": lay.mapping[ob.color], "held": ob.index == self.held,
                          "on_mat": bool(self.s._on_mat(np.asarray(at, float)[None], 0.03)[0])})
        bins = {}
        for name in self.scene.bin_centers:
            lo, hi = self.scene.bin_inner(name)
            bins[name] = [round(float(v), 3) for v in (lo[0], lo[1], hi[0], hi[1])]
        return {"parts": parts, "bins": bins, "reach": [round(float(v), 3) for v in (*self.reach[0], *self.reach[1])]}

    def _landed(self) -> dict | None:
        """Where the part that last left the hand lies, while that is news: {where: in | wrong | beside |
        table | off, color, bin (where it belongs), reach}. {where: None} while it has not come to rest."""
        if self.let_go is not None:
            return {"where": None}
        got = self.landed
        if got is None or time.monotonic() - got["at"] > LANDED_S[0 if got["where"] == "in" else 1]:
            return None
        return {k: got[k] for k in ("where", "color", "bin", "reach")}

    def hud(self) -> dict:
        held = self.held >= 0 and self.phase in ("pick", "carry", "release")
        hands = None
        if self.human is not None:
            p = self.rt.models[SIDES[self.side]].points(self.human.joints).reshape(20, 3)
            hands = {SIDES[self.side]: [[0.0, 0.0, 0.0]] + [[round(float(v), 4) for v in row] for row in p]}
        gap = None if self.human is None else round(float(self.human.pinch[0]) * 1000.0)
        closure = None
        if self.posture is not None:
            middle, index = closure_of_posture(self.posture)
            closure = {"index": round(float(np.clip(index, 0.0, 1.0)), 2), "middle": round(float(np.clip(middle, 0.0, 1.0)), 2)}
        # No glove rows here: a pointer or a camera is not a glove, and the page must not say it is.
        return {"source": self.source, "stand_in": False, "driving": True, "hands": hands,
                # Where each part of the command comes from, for the page to say.
                # ``own`` is the grasp the visitor chose. ``lost``: they do it themselves and nothing sees their
                # hand, so the robot holds what it has. The page says so; the grasp is never changed for them.
                "fingers": self.fingers, "own": bool(self.own), "lost": bool(self.lost), "gap_mm": gap,
                # How far the person's own index and middle finger close the robot's, 0 open to 1 closed.
                "closure": closure,
                "glove": self.glove.hud() if self.glove is not None else None,
                "placed": self.anchor is not None, "flat_s": round(float(self.flat_s), 2),
                "input": self.input, "points": self.points, "phase": self.phase,
                # A pinch that took nothing, a part that fell out of the hand: just now, for the page to say.
                "missed": time.monotonic() - self.missed_at < MISSED_S and not held,
                "dropped": time.monotonic() - self.dropped_at < DROPPED_S and not held,
                "dropped_n": int(self.fell_n),
                "holding": self.held >= 0 and self.phase in ("carry", "release"),
                "over": self._over(),
                # A part being let go: over which bin (None: onto the table). Then where it came to rest.
                "into": self.release_bin if self.phase == "release" else None,
                "landed": self._landed(),
                "want": self.held_bin if held else None,            # the bin the part in the hand belongs in
                "table": self._table()}

