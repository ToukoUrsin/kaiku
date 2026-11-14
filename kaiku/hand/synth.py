"""Forward model of the glove, and a scripted operator that plays hand motion through it.

``hand_to_glove_frame`` is the exact inverse of ``solve.GloveSolver``: given a hand state it produces what
the sensors would read, under the assumptions of ``glove_model.py``. ``ScriptedOperator`` turns a short
declarative script (move the wrist here, open, close to this aperture, pinch that finger, hold) into smooth
two-handed motion and samples it as glove frames, so everything downstream can be built and tested without
hardware.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike

from kaiku.hand import glove_model as gm
from kaiku.hand.geometry import (IDENTITY_QUAT, Pose, mat_to_quat, pose_inv, pose_mul, quat_conj,
                                 quat_from_rotvec, quat_from_yaw, quat_mul, quat_slerp, rot_y, rot_z)
from kaiku.hand.layout import GloveLayout, glove_layout
from kaiku.hand.model import DEFAULT_HAND_LENGTH, HandModel
from kaiku.hand.poses import HandPoses
from kaiku.types import FINGERS, SIDES, GloveFrame, SensorPose

DEVICE_IDS = {"left": 1, "right": 2}
"""RGMP device ids the synthetic gloves announce themselves with."""


# ----------------------------------------------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GloveNoise:
    """Sensor imperfections, one standard deviation each. The defaults are a plausible well-behaved glove.

    Noise is white from frame to frame; a real glove's fused output is smoother than that, so this is on
    the harsh side for the downstream filters.
    """
    hub_pos: float = 0.0005     # m, finger and arm sensor position relative to the hub (short-range EMF)
    hub_rot: float = 0.005      # rad per axis, finger and arm sensor orientation relative to the hub
    coil_pos: float = 0.001     # m, arm sensor position relative to the Coil Pro (long-range EMF)
    coil_rot: float = 0.005     # rad per axis, arm sensor orientation relative to the Coil Pro
    dropout: float = 0.0        # chance per frame, per finger sensor, of being flagged invalid


@lru_cache(maxsize=None)
def _hand(side: str, hand_length: float) -> tuple[HandModel, GloveLayout]:
    model = HandModel(side, hand_length)
    return model, glove_layout(model)


def _noisy(pose: Pose, pos_sigma: float, rot_sigma: float, rng: np.random.Generator) -> Pose:
    """The pose as a sensor with the given noise would report it."""
    return (pose[0] + rng.normal(0.0, pos_sigma, 3),
            quat_mul(pose[1], quat_from_rotvec(rng.normal(0.0, rot_sigma, 3))))


def hand_to_glove_frame(
    side: str,
    joints: ArrayLike,
    wrist_pose_in_coil: Pose,
    t: float,
    noise: GloveNoise | None = None,
    *,
    rng: np.random.Generator | None = None,
    hand_length: float = DEFAULT_HAND_LENGTH,
    wrist_bend: tuple[float, float] = (0.0, 0.0),
    coil_present: bool = True,
    north_yaw: float = 0.0,
) -> GloveFrame:
    """What one glove reads for a given hand state.

    Args:
        side: "left" or "right".
        joints: (5, 4) joint values in the ``HumanHand.joints`` layout.
        wrist_pose_in_coil: ``(position, quaternion wxyz)`` of the hand frame in the tracking frame, which is
            the Coil Pro frame made z-up (``glove_model.TRACKING_FROM_COIL``; the same frame by default).
        t: time of the sample in seconds; becomes both the device and the host timestamp.
        noise: sensor noise and dropouts; ``None`` for an ideal glove.
        rng: random source for the noise (a fresh unseeded one if omitted).
        hand_length: wrist joint to middle fingertip, metres.
        wrist_bend: ``(flexion, radial deviation)`` of the hand against the forearm, radians. It moves the
            arm sensor relative to the hub, as on a real wrist, and must not affect the solved hand.
        coil_present: when False the glove reports no Coil Pro (no ``arm_in_coil``, flags cleared).
        north_yaw: heading of the tracking frame against magnetic north: the rotation about up that takes
            tracking-frame vectors into the north-west-up frame. Arbitrary per session on real hardware.
    """
    model, layout = _hand(side, hand_length)
    axes = layout.hand_from_glove
    hub_pos, hub_quat = layout.hand_from_hub
    hub_from_hand = pose_inv(layout.hand_from_hub)
    if noise is not None and rng is None:
        rng = np.random.default_rng()

    # Finger sensors: each rides on one bone, at a fixed offset, with glove-style axes.
    joints = np.asarray(joints, dtype=float)
    bone_rot = model.bone_frames(joints)[:, gm.SENSOR_BONE]
    bone_origin = model.points(joints)[:, gm.SENSOR_BONE]
    sensor_pos = bone_origin + (bone_rot @ layout.sensor_offset[:, :, None])[:, :, 0]
    pos_in_hub = (sensor_pos - hub_pos) @ axes
    rot_in_hub = axes.T @ bone_rot @ axes
    fingers = {}
    for i, name in enumerate(FINGERS):
        pose, valid = (pos_in_hub[i], mat_to_quat(rot_in_hub[i])), True
        if noise is not None:
            pose = _noisy(pose, noise.hub_pos, noise.hub_rot, rng)
            valid = bool(rng.random() >= noise.dropout)
        fingers[name] = SensorPose(pose[0], pose[1], valid)

    # Arm sensor: fixed on the forearm, which the wrist joint connects to the hand.
    flexion, deviation = wrist_bend
    forearm_from_hand = (np.zeros(3), mat_to_quat(rot_z(model.mirror * deviation) @ rot_y(flexion)))
    hand_from_arm = pose_mul(pose_inv(forearm_from_hand), (layout.arm_sensor_pos, hub_quat))
    arm_in_hub = pose_mul(hub_from_hand, hand_from_arm)
    wrist = (np.asarray(wrist_pose_in_coil[0], dtype=float), np.asarray(wrist_pose_in_coil[1], dtype=float))
    coil_from_tracking = (np.zeros(3), quat_conj(mat_to_quat(gm.TRACKING_FROM_COIL)))
    arm_in_coil = pose_mul(coil_from_tracking, pose_mul(wrist, hand_from_arm))
    if noise is not None:
        arm_in_hub = _noisy(arm_in_hub, noise.hub_pos, noise.hub_rot, rng)
        arm_in_coil = _noisy(arm_in_coil, noise.coil_pos, noise.coil_rot, rng)

    # Hub orientation against gravity and magnetic north.
    ned_from_tracking = quat_mul(quat_conj(mat_to_quat(gm.TRACKING_FROM_NED)), quat_from_yaw(north_yaw))
    return GloveFrame(
        side=side,
        device_id=DEVICE_IDS[side],
        t_us=round(t * 1e6),
        t_recv=t,
        hub_quat_ned=quat_mul(ned_from_tracking, quat_mul(wrist[1], hub_quat)),
        fingers=fingers,
        arm_in_hub=SensorPose(*arm_in_hub),
        arm_in_coil=SensorPose(*arm_in_coil) if coil_present else None,
        coil_present=coil_present,
        coil_flip_decided=coil_present,
        heading_reliable=True,
    )


# ----------------------------------------------------------------------------------------------------------
# Script vocabulary
# ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Open:
    """Flat open hand, fingers slightly fanned."""


@dataclass(frozen=True)
class Relax:
    """Resting hand, fingers loosely curled."""


@dataclass(frozen=True)
class Grip:
    """Grasp posture with ``aperture`` metres between the thumb tip and the index fingertip."""
    aperture: float


@dataclass(frozen=True)
class Pinch:
    """Thumb tip on the tip of ``finger`` ("index", "middle", "ring" or "little")."""
    finger: str


FingerTarget = Open | Relax | Grip | Pinch


@dataclass(frozen=True)
class Step:
    """One movement of one hand, lasting ``duration`` seconds.

    The hand travels from wherever the previous step left it to the targets given here along a minimum-jerk
    profile, so it starts and stops at rest like a human point-to-point movement. A target left as ``None``
    keeps its previous value; a step with no targets is a hold.
    """
    duration: float
    pos: ArrayLike | None = None          # wrist position in the coil (tracking) frame, metres
    quat: ArrayLike | None = None         # wrist orientation in the coil (tracking) frame, wxyz
    fingers: FingerTarget | None = None


Script = Mapping[str, Sequence[Step]]
"""Steps per side. A side that is absent has no glove: the operator returns ``None`` for it."""

DEFAULT_HOME: dict[str, Pose] = {
    "left": (np.array([-0.80, 0.20, 0.10]), IDENTITY_QUAT),
    "right": (np.array([-0.80, -0.20, 0.10]), IDENTITY_QUAT),
}
"""Where the scripted operator rests its hands, in the coil frame: flat, palm down, pointing along coil +x,
shoulder width apart and 0.8 m from the coil (outside the roughly 0.5 m zone where the coil saturates)."""


# ----------------------------------------------------------------------------------------------------------
# Scripted operator
# ----------------------------------------------------------------------------------------------------------

# Involuntary motion added on top of the script. Rows: slow postural sway (two components), then
# physiological tremor (two components near 10 Hz), as (frequency Hz, relative size).
_TREMOR_BANDS = np.array([[0.23, 1.0], [0.61, 0.5], [8.7, 0.10], [10.9, 0.06]])
_TREMOR_POS = 0.0012      # m, size of the slow sway of the wrist position
_TREMOR_ROT = 0.006       # rad, same for the wrist orientation
_TREMOR_JOINT = 0.008     # rad, same for every finger joint


def _minimum_jerk(u: float) -> float:
    """Fraction of the way travelled at fraction ``u`` of the time, for the smoothest possible movement."""
    return u * u * u * (10.0 + u * (6.0 * u - 15.0))


class _Track:
    """Keyframes of one hand: its state at the end of every step, with the start state first."""

    def __init__(self, steps: Sequence[Step], home: Pose, poses: HandPoses) -> None:
        pos, quat, joints = np.asarray(home[0], dtype=float), np.asarray(home[1], dtype=float), poses.relaxed
        self.times = [0.0]
        self.pos, self.quat, self.joints = [pos], [quat], [joints]
        for step in steps:
            if step.duration <= 0.0:
                raise ValueError(f"step duration must be positive, got {step.duration}")
            if step.pos is not None:
                pos = np.asarray(step.pos, dtype=float)
            if step.quat is not None:
                quat = np.asarray(step.quat, dtype=float)
            if step.fingers is not None:
                joints = _finger_posture(step.fingers, poses)
            self.times.append(self.times[-1] + step.duration)
            self.pos.append(pos)
            self.quat.append(quat)
            self.joints.append(joints)

    def at(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wrist position, wrist quaternion and joints at ``time`` seconds into the script."""
        k = min(bisect_right(self.times, time), len(self.times) - 1)
        if k == 0:
            return self.pos[0], self.quat[0], self.joints[0]
        u = min((time - self.times[k - 1]) / (self.times[k] - self.times[k - 1]), 1.0)
        s = _minimum_jerk(u)
        return (self.pos[k - 1] + s * (self.pos[k] - self.pos[k - 1]),
                quat_slerp(self.quat[k - 1], self.quat[k], s),
                self.joints[k - 1] + s * (self.joints[k] - self.joints[k - 1]))


def _finger_posture(target: FingerTarget, poses: HandPoses) -> np.ndarray:
    if isinstance(target, Open):
        return poses.open
    if isinstance(target, Relax):
        return poses.relaxed
    if isinstance(target, Grip):
        return poses.grip(target.aperture)
    return poses.pinch(target.finger)


class ScriptedOperator:
    """A glove source that plays a script: ``sample(side, t)`` returns the glove frame at time ``t``.

    ``sample`` is a pure function of its arguments (noise included), so frames can be requested in any
    order and a run can be replayed exactly.

    Args:
        script: steps per side; both hands start at ``home`` with relaxed fingers.
        home: start pose of each wrist in the coil frame.
        hand_length: operator's hand length, metres.
        noise: sensor noise applied to the frames; ``None`` for an ideal glove.
        tremor: scale of the involuntary sway and tremor added to the scripted motion; 0 turns it off.
        loop: repeat the script forever. The script should then end in the pose it starts in, as all the
            stock scripts do; otherwise the hands jump when it wraps. Without ``loop`` the final pose is held.
        seed: fixes the tremor phases and the noise.
    """

    def __init__(self, script: Script, *, home: Mapping[str, Pose] = DEFAULT_HOME,
                 hand_length: float = DEFAULT_HAND_LENGTH, noise: GloveNoise | None = None,
                 tremor: float = 1.0, loop: bool = True, seed: int = 0) -> None:
        unknown = set(script) - set(SIDES)
        if unknown:
            raise ValueError(f"script has unknown sides {sorted(unknown)}")
        self.home = dict(home)
        self.hand_length = hand_length
        self.noise = noise
        self.tremor = tremor
        self.loop = loop
        self.seed = seed
        self._model = HandModel("right", hand_length)   # joint values and limits are the same for both sides
        poses = HandPoses(self._model)
        self._tracks = {side: _Track(steps, self.home[side], poses) for side, steps in script.items()}
        self.duration = max((track.times[-1] for track in self._tracks.values()), default=0.0)
        phase_rng = np.random.default_rng(seed)
        self._tremor_phase = {side: phase_rng.uniform(0.0, 2.0 * np.pi, (len(_TREMOR_BANDS), 26))
                              for side in SIDES}
        self._tremor_size = tremor * np.concatenate([np.full(3, _TREMOR_POS), np.full(3, _TREMOR_ROT),
                                                     np.full(20, _TREMOR_JOINT)])

    def state(self, side: str, t: float) -> tuple[Pose, np.ndarray]:
        """True hand state at time ``t``: ``((wrist position, wrist quaternion), joints)``, tremor included."""
        time = t % self.duration if self.loop and self.duration > 0.0 else min(t, self.duration)
        pos, quat, joints = self._tracks[side].at(time)
        freq, gain = _TREMOR_BANDS[:, :1], _TREMOR_BANDS[:, 1:]
        wobble = self._tremor_size * np.sum(gain * np.sin(2.0 * np.pi * freq * t + self._tremor_phase[side]), axis=0)
        joints = self._model.clamp(joints + wobble[6:].reshape(5, 4))
        return (pos + wobble[:3], quat_mul(quat_from_rotvec(wobble[3:6]), quat)), joints

    def sample(self, side: str, t: float) -> GloveFrame | None:
        """Glove frame of ``side`` at time ``t`` seconds, or ``None`` if the script has no such hand."""
        if side not in self._tracks:
            return None
        wrist, joints = self.state(side, t)
        rng = None
        if self.noise is not None:
            rng = np.random.default_rng([self.seed, SIDES.index(side), round(t * 1e6)])
        return hand_to_glove_frame(side, joints, wrist, t, self.noise, rng=rng, hand_length=self.hand_length)


# ----------------------------------------------------------------------------------------------------------
# Stock scripts
# ----------------------------------------------------------------------------------------------------------

def idle_script(home: Mapping[str, Pose] = DEFAULT_HOME) -> dict[str, list[Step]]:
    """Both hands resting at home, rising and sinking a few millimetres with the operator's breathing."""
    up = {side: home[side][0] + np.array([0.0, 0.0, 0.006]) for side in SIDES}
    return {
        "left": [Step(2.2, pos=up["left"]), Step(2.2, pos=home["left"][0])],
        "right": [Step(0.4), Step(2.2, pos=up["right"]), Step(1.8, pos=home["right"][0])],
    }


def open_close_script(cycles: int = 3, aperture: float = 0.01) -> dict[str, list[Step]]:
    """Both hands open flat and close to a tight grip, ``cycles`` times, the left a beat behind the right."""
    cycle = [Step(0.6, fingers=Open()), Step(0.3), Step(0.7, fingers=Grip(aperture)), Step(0.3)]
    return {
        "left": [Step(0.25)] + cycle * cycles + [Step(0.55, fingers=Relax())],
        "right": cycle * cycles + [Step(0.8, fingers=Relax())],
    }


def pinch_script() -> dict[str, list[Step]]:
    """Both hands touch the thumb to each fingertip in turn, index to little."""
    steps = [Step(0.5, fingers=Open())]
    for finger in FINGERS[1:]:
        steps += [Step(0.5, fingers=Pinch(finger)), Step(0.25), Step(0.45, fingers=Open()), Step(0.15)]
    steps.append(Step(0.7, fingers=Relax()))
    return {"left": list(steps), "right": list(steps)}


@dataclass(frozen=True)
class ReachWaypoints:
    """Wrist positions of a pick-and-place, in the coil (tracking) frame, metres."""
    pregrasp: ArrayLike     # hand open, short of the object
    grasp: ArrayLike        # fingers around the object
    lift: ArrayLike         # object clear of the surface
    carry: ArrayLike        # above the place position
    place: ArrayLike        # object set down

    @classmethod
    def in_front_of(cls, start: ArrayLike, towards_centre: float = 0.20) -> ReachWaypoints:
        """A plausible pick 30 cm ahead of ``start`` (+x) and a place ``towards_centre`` metres along +y of it.

        For demos without a scene; a simulator passes waypoints derived from its objects instead.
        """
        grasp = np.asarray(start, dtype=float) + np.array([0.30, 0.0, -0.04])
        place = grasp + np.array([0.0, towards_centre, 0.0])
        rise = np.array([0.0, 0.0, 0.10])
        return cls(grasp + np.array([-0.08, 0.0, 0.03]), grasp, grasp + rise, place + rise, place)


def reach_script(side: str = "right", waypoints: ReachWaypoints | None = None, *,
                 quat: ArrayLike | None = None, open_aperture: float = 0.11, grip_aperture: float = 0.05,
                 home: Mapping[str, Pose] = DEFAULT_HOME, speed: float = 0.30) -> dict[str, list[Step]]:
    """One hand reaches, grasps, lifts, carries, sets down and releases; the other rests at home.

    Args:
        side: the hand that does the work.
        waypoints: wrist positions in the coil frame; a default pick-and-place in front of home if omitted.
        quat: wrist orientation to hold from pre-grasp to release (default: the home orientation).
        open_aperture: thumb-to-index opening while approaching and after releasing, metres.
        grip_aperture: thumb-to-index opening when holding the object, metres.
        home: rest pose of each wrist; the working hand starts and ends there.
        speed: average hand speed between waypoints, m/s (the peak is about twice that).
    """
    start, home_quat = np.asarray(home[side][0], dtype=float), home[side][1]
    grasp_quat = home_quat if quat is None else quat
    if waypoints is None:
        waypoints = ReachWaypoints.in_front_of(start, towards_centre=0.20 if side == "right" else -0.20)
    pregrasp, grasp, lift, carry, place = (np.asarray(p, dtype=float) for p in (
        waypoints.pregrasp, waypoints.grasp, waypoints.lift, waypoints.carry, waypoints.place))

    def leg(a: np.ndarray, b: np.ndarray) -> float:
        """Seconds to move the hand from a to b: distance over speed, but never rushed."""
        return max(0.45, float(np.linalg.norm(b - a)) / speed)

    steps = [
        Step(leg(start, pregrasp), pos=pregrasp, quat=grasp_quat, fingers=Grip(open_aperture)),   # reach, pre-shaping
        Step(leg(pregrasp, grasp), pos=grasp),                                                    # approach
        Step(0.5, fingers=Grip(grip_aperture)),                                                   # grasp
        Step(0.25),
        Step(leg(grasp, lift), pos=lift),                                                         # lift
        Step(leg(lift, carry), pos=carry),                                                        # carry
        Step(leg(carry, place), pos=place),                                                       # set down
        Step(0.4, fingers=Grip(open_aperture)),                                                   # release
        Step(0.2),
        Step(leg(place, carry), pos=carry),                                                       # retreat upwards
        Step(leg(carry, start), pos=start, quat=home_quat, fingers=Relax()),                      # return home
    ]
    other = SIDES[1 - SIDES.index(side)]
    return {side: steps, other: [Step(sum(step.duration for step in steps))]}


SCRIPTS: dict[str, Callable[[], Script]] = {
    "idle": idle_script,
    "open_close": open_close_script,
    "pinch": pinch_script,
    "reach": reach_script,
}
"""Stock scripts by name; each entry builds its script with default arguments."""
