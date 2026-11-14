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

import numpy as np

from kaiku.app.autopilot import Autopilot
from kaiku.app.session import LeadCommand, Observation, Session
from kaiku.hand import Calibration, GloveNoise, GloveSolver, Teleop, hand_to_glove_frame
from kaiku.hand.geometry import Pose, quat_from_rotvec, quat_mul, quat_slerp
from kaiku.hand.synth import _TREMOR_BANDS, _TREMOR_JOINT, _TREMOR_POS, _TREMOR_ROT, DEFAULT_HOME
from kaiku.rgmp import GloveStream
from kaiku.rgmp.emulator import Emulator
from kaiku.sim.grasp import HAND_OPEN, hand14
from kaiku.teleop.retarget import Retarget
from kaiku.types import SIDES, GloveFrame


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
    person wearing emulated gloves, steered by the autopilot's intent."""

    def __init__(self, session: Session, host: str | None = None, port: int | None = None, seed: int = 0,
                 noise: GloveNoise | None = GloveNoise(), tremor: float = 1.0):
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
        self.teleop: Teleop | None = None
        self.hands: dict = {side: None for side in SIDES}
        self.posture = np.stack([HAND_OPEN, HAND_OPEN])              # left-hand order, per side
        self.pinch, self.rot = session.home_pinch.copy(), session.home_rot.copy()
        self.ready = False            # the stand-in has brought the robot to where the autopilot starts
        self.settle = 0
        self.label = "connecting"
        self._lock = threading.Lock()

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
        try:
            cal = Calibration.capture(left, right, robot_home=self.rt.ready(), reach_scale=self.rt.scale)
        except ValueError:
            return False
        self.teleop = Teleop(cal)
        return True

    def update(self, obs: Observation) -> LeadCommand:
        for side in SIDES:
            frame = self.stream.latest(side)
            self.hands[side] = self.solvers[side].solve(frame) if frame is not None else None
        if self.teleop is None:
            self.label = "calibrating" if self.stream.connected else "connecting"
            if self.person and self.stream.rate_hz("left") > 40 and self.stream.rate_hz("right") > 40:
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
        cmd = self.teleop.update(self.hands, time.monotonic())
        self.pinch, self.rot, self.posture = self.rt.command(cmd, (self.pinch, self.rot, self.posture))
        return LeadCommand(self.pinch.copy(), self.rot.copy(), hand14(self.posture[0], self.posture[1]))

    # ------------------------------------------------------------------ for the viewer
    def hud(self) -> dict:
        out = {"calibrated": self.teleop is not None, "source": self.source, "stand_in": self.person is not None}
        pts = {}
        for side in SIDES:
            hz = self.stream.rate_hz(side)
            out[side] = {"connected": hz > 1.0, "hz": round(hz, 1),
                         "status": self.teleop.status[side] if self.teleop else "uncalibrated"}
            hand = self.hands[side]
            if hand is not None:
                p = self.rt.models[side].points(hand.joints).reshape(20, 3)
                pts[side] = [[0.0, 0.0, 0.0]] + [[round(float(v), 4) for v in row] for row in p]
        out["hands"] = pts or None
        return out

    def close(self) -> None:
        self.stream.stop()
        if self.emulator:
            self.emulator.stop()
