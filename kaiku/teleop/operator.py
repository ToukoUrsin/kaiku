"""Who moves the lead robot.

    GloveOperator   Smartgloves over RGMP -> solved hands -> calibration and filters -> retarget.
                    The gloves are the Rokoko driver's, a replayed recording's, or the stand-in's.
    StandIn         a scripted person. It wants what the autopilot wants, moves a human hand model
                    there the way a person would (smoothly, with tremor) and wears emulated
                    Smartgloves II. Everything after the glove sensors is the real pipeline, over
                    a real RGMP socket.
"""
from __future__ import annotations

import threading
import time
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from kaiku.app.autopilot import Autopilot
from kaiku.app.session import LeadCommand, Observation, Session
from kaiku.hand import Calibration, GloveNoise, GloveSolver, Teleop, hand_to_glove_frame
from kaiku.hand.geometry import Pose, quat_from_rotvec, quat_mul, quat_rotate, quat_slerp
from kaiku.hand.solved import with_fingers
from kaiku.hand.synth import _TREMOR_BANDS, _TREMOR_JOINT, _TREMOR_POS, _TREMOR_ROT, DEFAULT_HOME
from kaiku.rgmp import GloveStream
from kaiku.rgmp.emulator import Emulator
from kaiku.rgmp.solved import SOLVED_PORT, SolvedStream
from kaiku.sim.grasp import HAND_OPEN, hand14
from kaiku.teleop.retarget import Retarget
from kaiku.types import SIDES, GloveFrame
from kaiku.sim.scene import Layout, Placement


class StandIn:
    """The emulator's glove source: a person's two hands going where they are asked to.

    ``want`` (any thread) sets where each hand should be; ``sample`` (the emulator's thread, 83 Hz)
    moves the hands there with a short lag and a speed limit, adds a person's sway and tremor,
    and reads the gloves' sensors.
    """

    def __init__(self, retarget: Retarget, home: dict[str, Pose] = DEFAULT_HOME, noise: GloveNoise | None = GloveNoise(),
                 tremor: float = 1.0, lag_s: float = 0.03, max_speed: float = 1.5, seed: int = 0):
        self.rt = retarget
        self.noise, self.lag, self.max_speed = noise, lag_s, max_speed
        flat = {side: retarget.poses[side].open for side in SIDES}
        self._target = {side: (np.array(home[side][0], float), np.array(home[side][1], float), flat[side]) for side in SIDES}
        self._state = dict(self._target)
        self._t = {side: None for side in SIDES}
        self._rng = np.random.default_rng(seed)
        self._phase = {side: self._rng.uniform(0.0, 2.0 * np.pi, (len(_TREMOR_BANDS), 26)) for side in SIDES}
        self._size = tremor * np.concatenate([np.full(3, _TREMOR_POS), np.full(3, _TREMOR_ROT), np.full(20, _TREMOR_JOINT)])
        self.truth: dict[str, tuple] = {}                 # what the hands really did last (for tests)

    def want(self, side: str, pinch: np.ndarray, rot: np.ndarray, posture: np.ndarray, calibration: Calibration) -> None:
        """Ask for the hand pose that makes the robot's hand pinch at ``pinch`` with palm ``rot``
        and Dex3 ``posture`` (7, left-hand order), under the calibration in force."""
        wrist, joints = self.rt.human(side, pinch, rot, posture)
        pos, quat = calibration.to_tracking(side, *wrist)
        self._target[side] = (pos, quat, joints)

    def sample(self, side: str, t: float) -> GloveFrame:
        pos_t, quat_t, joints_t = self._target[side]
        pos, quat, joints = self._state[side]
        last = self._t[side]
        dt = 0.012 if last is None else max(t - last, 1e-4)
        self._t[side] = t
        a = 1.0 - np.exp(-dt / self.lag)
        step = a * (pos_t - pos)
        far = np.linalg.norm(step)
        if far > self.max_speed * dt:
            step *= self.max_speed * dt / far
        pos = pos + step
        quat = quat_slerp(quat, quat_t, a)
        joints = joints + a * (joints_t - joints)
        self._state[side] = (pos, quat, joints)
        freq, gain = _TREMOR_BANDS[:, :1], _TREMOR_BANDS[:, 1:]
        wob = self._size * np.sum(gain * np.sin(2.0 * np.pi * freq * t + self._phase[side]), axis=0)
        model = self.rt.models[side]
        shaky = (pos + wob[:3], quat_mul(quat_from_rotvec(wob[3:6]), quat))
        shaky_joints = model.clamp(joints + wob[6:].reshape(5, 4))
        self.truth[side] = (shaky, shaky_joints)
        return hand_to_glove_frame(side, shaky_joints, shaky, t, self.noise, rng=self._rng, hand_length=model.hand_length)


class GloveOperator:
    """Drives the lead robot from Smartgloves. With no ``host`` it brings its own: the stand-in
    person wearing emulated gloves, steered by the autopilot's intent.

    With real gloves the fingers come from Rokoko's own solver whenever ``rkk-hand-solver`` is up on
    ``solver`` (HOST:PORT, by default port 12277 on the gloves' host; "off" for never), and from kaiku's
    otherwise. The wrists always come from the driver's stream."""

    def __init__(self, session: Session, host: str | None = None, port: int | None = None, seed: int = 0,
                 noise: GloveNoise | None = GloveNoise(), tremor: float = 1.0, solver: str | None = None):
        self.s = session
        self.rt = Retarget()
        self.solvers = {side: GloveSolver(side) for side in SIDES}
        self.person: StandIn | None = None
        self.pilot: Autopilot | None = None
        self.emulator: Emulator | None = None
        if host is None:
            self.person = StandIn(self.rt, noise=noise, tremor=tremor, seed=seed)
            self.emulator = Emulator(self.person, "127.0.0.1", 0).start()
            host, port = "127.0.0.1", self.emulator.port
            self.pilot = Autopilot(session)
        self.source = "stand-in" if self.person else "gloves"
        self.stream = GloveStream(host, port or 12276).start()
        self.solved: SolvedStream | None = None
        self.ranges = None                                           # per side, what this operator's gloves read
        if self.person is None:
            self.ranges = {side: self.rt.gap_range() for side in SIDES}
            if solver != "off":
                shost, _, sport = (solver or "").partition(":")
                self.solved = SolvedStream(shost or host, int(sport) if sport else SOLVED_PORT).start()
        self.fingers = {side: "kaiku" for side in SIDES}             # whose solver the fingers are from
        self.teleop: Teleop | None = None
        self.hands: dict = {side: None for side in SIDES}
        self.posture = np.stack([HAND_OPEN, HAND_OPEN])              # left-hand order, per side
        self.pinch, self.rot = session.home_pinch.copy(), session.home_rot.copy()
        self.ready = False            # the stand-in has brought the robot to where the autopilot starts
        self.settle = 0
        self.label = "connecting"
        self._lock = threading.Lock()
        self._flat_since: float | None = None
        self._still_since: float | None = None
        self._last_pose: dict[str, tuple[np.ndarray, np.ndarray] | None] = {side: None for side in SIDES}
        self._closed = {side: False for side in SIDES}
        self._capture_at = 1.5

    # ------------------------------------------------------------------ operator interface
    def begin(self) -> None:
        if self.pilot:
            self.pilot.begin()

    @property
    def done(self) -> bool:
        if self.pilot:
            return self.ready and self.pilot.done
        lay = self.s.layouts[0]
        return all(v == "correct" for v in self.s.sim.outcomes(0).values()) and len(lay.placements) > 0 \
            and bool((self.posture[:, 5] > -0.6).all())

    def calibrate(self) -> bool:
        """Take the operator's pose now as the calibration pose: flat hands, pointing ahead."""
        left, right = self.hands["left"], self.hands["right"]
        if left is None or right is None or not (left.has_position and right.has_position):
            return False
        if not all(self._is_flat(hand) for hand in (left, right)):
            backs = [quat_rotate(hand.wrist_quat, (0.0, 0.0, 1.0))[2] for hand in (left, right)]
            if self.person is None and max(backs) < -0.3:
                frames = {side: self.stream.latest(side) for side in SIDES}
                if None in frames.values():
                    return False
                for side in SIDES:
                    self.solvers[side].flip_vertical()
                    self.hands[side] = self._solve(side, frames[side])
                left, right = self.hands["left"], self.hands["right"]
            if not all(self._is_flat(hand) for hand in (left, right)):
                return False
        if self.person is None:
            # The palms face down in this pose. Backs of both hands pointing down in the tracking frame
            # means the Coil Pro's Z is the other way up than assumed: turn the frame over and solve again.
            back = [quat_rotate(hand.wrist_quat, (0.0, 0.0, 1.0))[2] for hand in (left, right)]
            if min(back) < 0.3:
                return False                                  # not flat, or the two hands disagree
        try:
            cal = Calibration.capture(left, right, robot_home=self.rt.ready(), reach_scale=self.rt.scale)
        except ValueError:
            return False
        self.teleop = Teleop(cal)
        if self.ranges:
            for side, hand in (("left", left), ("right", right)):
                self.ranges[side].open(hand.pinch)
        return True

    @staticmethod
    def _is_flat(hand) -> bool:
        """A level hand has its dorsal axis up in the tracking frame."""
        if hand is None or not hand.has_position:
            return False
        dorsal = quat_rotate(hand.wrist_quat, (0.0, 0.0, 1.0))
        return bool(dorsal[2] > 0.3 and abs(float(dorsal[0])) < 0.45 and abs(float(dorsal[1])) < 0.45)

    def _calibration_state(self, now: float) -> tuple[bool, float]:
        hands = [self.hands[side] for side in SIDES]
        flat = all(self._is_flat(hand) for hand in hands)
        still = flat
        for side, hand in zip(SIDES, hands):
            previous = self._last_pose[side]
            if hand is None or not hand.has_position or previous is None:
                still = False
            else:
                p, q = previous
                still &= np.linalg.norm(hand.wrist_pos - p) < 0.006
                still &= abs(float(np.dot(hand.wrist_quat, q))) > 0.999
            if hand is not None and hand.has_position:
                self._last_pose[side] = (hand.wrist_pos.copy(), hand.wrist_quat.copy())
            else:
                self._last_pose[side] = None
        if flat and self._flat_since is None:
            self._flat_since = now
        if not flat:
            self._flat_since = self._still_since = None
        elif still and self._still_since is None:
            self._still_since = now
        elif not still:
            self._still_since = None
        held = 0.0 if self._still_since is None else max(0.0, now - self._still_since)
        return flat and still, held

    def _solve(self, side: str, frame: GloveFrame | None):
        """One glove frame to a hand: the wrist from the driver, the fingers from Rokoko's solver when it runs."""
        if frame is None:
            self.fingers[side] = "kaiku"
            return None
        hand = self.solvers[side].solve(frame)
        theirs = self.solved.latest(side) if self.solved else None
        self.fingers[side] = "kaiku" if theirs is None else "rokoko"
        return hand if theirs is None else with_fingers(hand, self.rt.models[side], theirs.points)

    def update(self, obs: Observation) -> LeadCommand:
        for side in SIDES:
            self.hands[side] = self._solve(side, self.stream.latest(side))
        now = time.monotonic()
        ready, held = self._calibration_state(now)
        if self.teleop is None:
            self.label = "calibrating" if self.stream.connected else "connecting"
            if self.person and self.stream.rate_hz("left") > 40 and self.stream.rate_hz("right") > 40:
                self.calibrate()
            elif ready and held >= self._capture_at:
                self.calibrate()
            return self.s.home_command()
        if self.person:
            if self.ready:
                want = self.pilot.update(obs)
                self.label = self.pilot.label
                posture = self.pilot.hand
            else:
                # Bring the hands from the calibration pose to where the robot is resting.
                want, posture = self.s.home_command(), [HAND_OPEN, HAND_OPEN]
                self.label = "calibrating"
                near = np.linalg.norm(obs.pinch[0] - self.s.home_pinch, axis=-1).max() < 0.012
                self.settle = self.settle + 1 if near else 0
                self.ready = self.settle > 10
            for s, side in enumerate(SIDES):
                self.person.want(side, want.pinch[s], want.rot[s], posture[s], self.teleop.calibration)
        else:
            self.label = "teleop"
        cmd = self.teleop.update(self.hands, now)
        if self.ranges:
            for side in SIDES:
                hand = cmd.hands[side]
                if hand is not None:
                    cmd.hands[side] = replace(hand, pinch=self.ranges[side](hand.pinch))
        self.pinch, self.rot, self.posture = self.rt.command(cmd, (self.pinch, self.rot, self.posture))
        for side in SIDES:
            hand = self.hands[side]
            if hand is not None and float(np.mean(hand.curl)) > 0.62:
                self._closed[side] = True
        return LeadCommand(self.pinch.copy(), self.rot.copy(), hand14(self.posture[0], self.posture[1]))

    # ------------------------------------------------------------------ for the viewer
    def hud(self) -> dict:
        out = {"calibrated": self.teleop is not None, "source": self.source, "stand_in": self.person is not None}
        out["flat_s"] = 0.0 if self._flat_since is None else round(max(0.0, time.monotonic() - self._flat_since), 2)
        out["still_s"] = 0.0 if self._still_since is None else round(max(0.0, time.monotonic() - self._still_since), 2)
        pts = {}
        for side in SIDES:
            hz = self.stream.rate_hz(side)
            hand = self.hands[side]
            flat = self._is_flat(hand)
            position = bool(hand is not None and hand.has_position)
            still_s = 0.0 if self._still_since is None else max(0.0, time.monotonic() - self._still_since)
            out[side] = {"connected": hz > 1.0, "hz": round(hz, 1), "fingers": self.fingers[side],
                         "status": self.teleop.status[side] if self.teleop else "uncalibrated",
                         "seen": hand is not None, "position": position, "has_position": position,
                         "flat": flat, "is_flat": flat, "still": flat and still_s > 0,
                         "still_s": round(still_s, 2), "closed": self._closed[side]}
            hand = self.hands[side]
            if hand is not None:
                p = self.rt.models[side].points(hand.joints).reshape(20, 3)
                pts[side] = [[0.0, 0.0, 0.0]] + [[round(float(v), 4) for v in row] for row in p]
        out["hands"] = pts or None
        return out

    def hand_points(self) -> np.ndarray:
        """The same wrist-relative points sent to the viewer, for a replay."""
        points = np.full((2, 21, 3), np.nan, dtype=np.float32)
        for s, side in enumerate(SIDES):
            hand = self.hands[side]
            if hand is not None:
                points[s, 0] = 0.0
                points[s, 1:] = self.rt.models[side].points(hand.joints).reshape(20, 3)
        return points

    def close(self) -> None:
        self.stream.stop()
        if self.solved:
            self.solved.stop()
        if self.emulator:
            self.emulator.stop()


class ReplayOperator:
    """Drive the lead from command arrays saved by ``Recorder``."""

    def __init__(self, session: Session, root: str | Path):
        self.s = session
        self.root = Path(root)
        self.rows = []
        side = self.root / "episodes.jsonl"
        if side.exists():
            for line in side.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("env") == 0 and row.get("source") == "human":
                    self.rows.append(row)
        self.index = -1
        self.current = None
        self.tick = 0
        self.label = "replay"
        self.source = "recorded"
        self.person = None
        self.hands = {side: None for side in SIDES}
        self.calibrated = True

    @property
    def done(self) -> bool:
        return self.current is not None and self.tick >= len(self.current["lead_pinch"])

    def begin(self) -> None:
        if not self.rows:
            self.current = None
            return
        self.index = (self.index + 1) % len(self.rows)
        row = self.rows[self.index]
        data = np.load(self.root / row["file"])
        self.current = {k: data[k] for k in ("lead_pinch", "lead_rot", "lead_hand") if k in data}
        self._points = None
        if "hands" in data:
            self.current["hands"] = data["hands"]
        if "lead_pinch" not in self.current and "lead_command" in data:
            packed = data["lead_command"]
            self.current.update({"lead_pinch": packed[:, :6].reshape(-1, 2, 3),
                                 "lead_rot": packed[:, 6:24].reshape(-1, 2, 3, 3),
                                 "lead_hand": packed[:, 24:38]})
        if "lead_pinch" not in self.current:
            i = int(row.get("row", 0))
            self.current = {"lead_pinch": np.tile(self.s.home_pinch, (len(data["action"][i]), 1, 1)),
                            "lead_rot": np.tile(self.s.home_rot, (len(data["action"][i]), 1, 1)),
                            "lead_hand": data["action"][i, :, 14:]}
        self.tick = 0

    def scene(self) -> Layout:
        if not self.rows:
            return self.s.sample_lead()
        row = self.rows[(self.index + 1) % len(self.rows)]
        return Layout([Placement(p["name"], np.asarray(p["pos"], float), float(p["yaw"])) for p in row.get("layout", [])],
                      dict(row.get("mapping") or {}))

    def update(self, obs: Observation) -> LeadCommand:
        if self.current is None:
            return self.s.home_command()
        k = min(self.tick, len(self.current["lead_pinch"]) - 1)
        self.tick += 1
        if "hands" in self.current:
            self._points = self.current["hands"][k]
        return LeadCommand(np.array(self.current["lead_pinch"][k]), np.array(self.current["lead_rot"][k]),
                           np.array(self.current["lead_hand"][k]))

    def hud(self) -> dict:
        row = self.rows[self.index] if self.rows and self.index >= 0 else {}
        points = getattr(self, "_points", None)
        return {"calibrated": True, "source": "recorded", "stand_in": False, "date": row.get("time"),
                "left": {"connected": True, "hz": 0.0, "status": "recorded"},
                "right": {"connected": True, "hz": 0.0, "status": "recorded"},
                "hands": None if points is None else np.asarray(points).tolist()}

    def close(self) -> None:
        pass
