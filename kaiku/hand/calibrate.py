"""Operator calibration and command building: solved human hands in, a ``TeleopCommand`` for the robot out.

``Calibration`` relates the tracking (Coil Pro) frame to the robot's pelvis frame. Both are z-up, so the
relation is a turn about the vertical, a per-hand anchor (this point of the operator's space is that point of
the robot's) and one scale for how far a hand movement carries the robot's hand.

``Teleop`` applies it every frame, smooths the result with One-Euro filters, and keeps the robot's hands
still whenever the operator's are not to be trusted: while the clutch is held, when a glove stops sending,
when the coil loses the hand.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from kaiku.hand.geometry import (IDENTITY_QUAT, Pose, quat_conj, quat_from_yaw, quat_mul, quat_rotate,
                                 quat_slerp, quat_to_rotvec)
from kaiku.hand.model import DEFAULT_HAND_LENGTH, HandModel
from kaiku.types import SIDES, HumanHand, TeleopCommand

DEFAULT_ROBOT_HOME: dict[str, Pose] = {
    "left": (np.array([0.28, 0.20, 0.12]), IDENTITY_QUAT),
    "right": (np.array([0.28, -0.20, 0.12]), IDENTITY_QUAT),
}
"""Wrist poses of the robot in its pelvis frame that the operator's home pose maps to: forearms level,
hands flat, palms down, in front of the hips. A guess for the G1; the simulator passes its own."""

DEFAULT_REACH_SCALE = 0.8
"""Robot hand travel per metre of operator hand travel. The G1's arm is roughly 0.8 of an adult's, so at this
scale the operator's comfortable reach is about the robot's."""

# Per-hand tracking status, reported in ``Teleop.status``.
TRACKING = "tracking"          # wrist and fingers live (the wrist command still holds while the clutch is engaged)
NO_POSITION = "no_position"    # glove data but no Coil Pro position: wrist held, fingers live
DROPPED = "dropped"            # no glove data: the last command is held
MISSING = "missing"            # never seen: the command has None for this hand


# ----------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """Map from the tracking frame to the robot's pelvis frame, per hand.

    A hand at ``track_anchor[side]`` commands the robot's hand to ``robot_anchor[side]``. Away from the
    anchor, displacements are turned by ``yaw`` about the vertical and scaled by ``reach_scale``; rotations
    are turned by ``yaw`` and applied unscaled. Because the turn is about z only, up stays up: the robot's
    hand is level exactly when the operator's is.

    Attributes:
        yaw: radians about +z taking tracking-frame directions into the pelvis frame.
        reach_scale: robot hand travel per metre of operator hand travel.
        track_anchor: per side, a wrist pose in the tracking frame.
        robot_anchor: per side, the wrist pose in the pelvis frame that it maps to.
    """
    yaw: float
    reach_scale: float
    track_anchor: Mapping[str, Pose]
    robot_anchor: Mapping[str, Pose]

    @classmethod
    def capture(cls, left: HumanHand, right: HumanHand, *,
                robot_home: Mapping[str, Pose] = DEFAULT_ROBOT_HOME,
                reach_scale: float = DEFAULT_REACH_SCALE) -> Calibration:
        """Calibrate from one sample of the operator's home pose.

        The operator holds both hands flat, palm down, fingers pointing straight ahead, wherever is
        comfortable. "Ahead" is taken as the mean horizontal direction of the two hands and becomes the
        robot's +x. Each captured wrist pose is anchored to that side's ``robot_home`` pose, which also
        cancels whatever tilt the glove has on the hand: by definition of the pose, the hand was level.

        Raises ``ValueError`` if a hand has no Coil Pro position or the hands give no usable heading.
        """
        hands = {"left": left, "right": right}
        heading = np.zeros(2)
        for side, hand in hands.items():
            if hand.side != side:
                raise ValueError(f"expected the {side} hand, got a {hand.side} one")
            if not hand.has_position:
                raise ValueError(f"the {side} hand has no Coil Pro position; calibration needs both wrists placed")
            heading += quat_rotate(hand.wrist_quat, (1.0, 0.0, 0.0))[:2]
        if np.linalg.norm(heading) < 0.5:
            raise ValueError("hands do not point in a common horizontal direction; hold them flat and forward")
        return cls(
            yaw=-math.atan2(heading[1], heading[0]),
            reach_scale=reach_scale,
            track_anchor={side: (hand.wrist_pos.copy(), hand.wrist_quat.copy()) for side, hand in hands.items()},
            robot_anchor={side: (np.array(robot_home[side][0], dtype=float), np.array(robot_home[side][1], dtype=float))
                          for side in SIDES},
        )

    def to_robot(self, side: str, pos: np.ndarray, quat: np.ndarray) -> Pose:
        """Robot wrist pose in the pelvis frame for an operator wrist pose in the tracking frame."""
        turn = quat_from_yaw(self.yaw)
        (track_pos, track_quat), (robot_pos, robot_quat) = self.track_anchor[side], self.robot_anchor[side]
        # The hand's rotation away from its anchor, re-expressed in the pelvis frame, applied to the robot's anchor.
        away = quat_mul(turn, quat_mul(quat_mul(quat, quat_conj(track_quat)), quat_conj(turn)))
        return (robot_pos + self.reach_scale * quat_rotate(turn, pos - track_pos), quat_mul(away, robot_quat))

    def to_tracking(self, side: str, pos: np.ndarray, quat: np.ndarray) -> Pose:
        """Inverse of ``to_robot``: where the operator's wrist must be for the robot's wrist to be at a pose.

        This is how a simulator turns object positions into waypoints for a scripted operator.
        """
        turn = quat_from_yaw(self.yaw)
        (track_pos, track_quat), (robot_pos, robot_quat) = self.track_anchor[side], self.robot_anchor[side]
        away = quat_mul(quat_conj(turn), quat_mul(quat_mul(quat, quat_conj(robot_quat)), turn))
        return (track_pos + quat_rotate(quat_conj(turn), pos - robot_pos) / self.reach_scale, quat_mul(away, track_quat))

    def rebased(self, side: str, track_pose: Pose, robot_pose: Pose) -> Calibration:
        """A calibration with the same turn and scale whose ``side`` anchor is moved: ``track_pose`` now maps to ``robot_pose``."""
        return replace(self, track_anchor={**self.track_anchor, side: track_pose},
                       robot_anchor={**self.robot_anchor, side: robot_pose})


# ----------------------------------------------------------------------------------------------------------
# One-Euro filters
# ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterParams:
    """Tuning of a One-Euro filter.

    Attributes:
        min_cutoff: low-pass cutoff in Hz while the signal is still. Lower removes more jitter at rest.
        beta: Hz of extra cutoff per unit of signal speed (units per second). Higher means less lag in motion.
        d_cutoff: cutoff in Hz of the low-pass on the speed estimate itself.
    """
    min_cutoff: float
    beta: float
    d_cutoff: float = 1.5


@dataclass(frozen=True)
class TeleopFilters:
    """Filter tuning for 83 Hz glove data: quiet at rest, a few milliseconds behind in motion.

    Speeds are in m/s, rad/s and rad/s. A wrist reaching at 0.3 m/s runs at about 14 Hz cutoff (11 ms time
    constant); a still hand at 1.5 Hz, which removes most sensor jitter and physiological tremor.
    """
    position: FilterParams = FilterParams(min_cutoff=1.5, beta=40.0)
    orientation: FilterParams = FilterParams(min_cutoff=1.5, beta=5.0)
    fingers: FilterParams = FilterParams(min_cutoff=2.5, beta=3.0)


FILTER_RESET_GAP = 0.2
"""Seconds without a sample after which a filter starts afresh rather than blend across the gap."""


def _alpha(dt: float, cutoff: float | np.ndarray) -> float | np.ndarray:
    """Weight of a new sample in an exponential low-pass with the given cutoff (Hz) and sample spacing (s)."""
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


class OneEuro:
    """One-Euro filter (Casiez, Roussel and Vogel, 2012) for a vector such as a position.

    An exponential low-pass whose cutoff rises with the signal's speed: heavy smoothing while the signal
    rests, where jitter is visible, and almost none while it moves, where lag is. One cutoff is used for the
    whole vector, from the length of its velocity, so all axes lag alike and a straight movement stays
    straight. Call it with each sample and its time; it returns the filtered sample.
    """

    def __init__(self, params: FilterParams) -> None:
        self.params = params
        self._state: np.ndarray | None = None
        self._raw: np.ndarray | None = None
        self._rate: np.ndarray | float = 0.0
        self._t = 0.0

    def __call__(self, sample: np.ndarray, t: float) -> np.ndarray:
        dt = t - self._t
        if self._state is None or dt > FILTER_RESET_GAP:
            self._state, self._raw, self._rate, self._t = sample, sample, 0.0, t
        elif dt > 0.0:
            rate = self._step(self._raw, sample) / dt
            self._rate = self._rate + _alpha(dt, self.params.d_cutoff) * (rate - self._rate)
            cutoff = self.params.min_cutoff + self.params.beta * self._speed(self._rate)
            self._state = self._blend(self._state, sample, _alpha(dt, cutoff))
            self._raw, self._t = sample, t
        return self._state

    def _step(self, previous: np.ndarray, sample: np.ndarray) -> np.ndarray:
        """Change from one raw sample to the next."""
        return sample - previous

    def _speed(self, rate: np.ndarray | float) -> np.ndarray | float:
        """Speed that sets the cutoff, from the smoothed rate of change."""
        return float(np.linalg.norm(rate))

    def _blend(self, state: np.ndarray, sample: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
        """Move the filter state towards the new sample by the fraction ``alpha``."""
        return state + alpha * (sample - state)


class OneEuroElementwise(OneEuro):
    """One-Euro filter for an array of independent values (joint angles): each element has its own cutoff."""

    def _speed(self, rate: np.ndarray | float) -> np.ndarray | float:
        return np.abs(rate)


class OneEuroQuat(OneEuro):
    """One-Euro filter for an orientation quaternion (wxyz): speed is angular speed, blending is slerp."""

    def _step(self, previous: np.ndarray, sample: np.ndarray) -> np.ndarray:
        return quat_to_rotvec(quat_mul(quat_conj(previous), sample))

    def _blend(self, state: np.ndarray, sample: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
        return quat_slerp(state, sample, float(alpha))


# ----------------------------------------------------------------------------------------------------------
# Teleop
# ----------------------------------------------------------------------------------------------------------

@dataclass
class _HandState:
    """Everything Teleop remembers about one hand between updates."""
    model: HandModel
    position: OneEuro
    orientation: OneEuroQuat
    joints: OneEuroElementwise
    wrist: Pose                           # last commanded robot wrist pose
    hand: HumanHand | None = None         # last filtered hand
    frozen_at: float | None = None        # when the wrist command stopped following the operator, if it has
    rebase: bool = False                  # re-anchor when the wrist command resumes


class Teleop:
    """Turns solved hands into robot commands: calibrate, filter, and hold still when in doubt.

    Wrist. While a hand is tracked with a position and the clutch is off, its filtered wrist pose goes
    through the calibration. Otherwise the last commanded wrist pose is held. When following resumes after
    the clutch, or after more than ``rebase_after`` seconds without a usable wrist, the calibration anchor
    of that hand is moved so that the operator's current pose maps to the held command: the robot continues
    from where it is instead of jumping to wherever the hand went meanwhile. Shorter gaps resume as they
    were, so brief dropouts do not let the anchor wander.

    Fingers. Joint angles are filtered and the fingertips, curl and pinch distances recomputed from them, so
    the command stays self-consistent. Fingers stay live through the clutch and a lost coil: neither affects
    what the finger sensors see, and the operator may need the grip while re-centring an arm.

    Attributes:
        calibration: the mapping currently in force (the captured one, with anchors moved by re-basing).
        status: per side, one of TRACKING, NO_POSITION, DROPPED, MISSING.

    Args:
        calibration: from ``Calibration.capture``.
        hand_length: the operator's hand length given to ``GloveSolver``, metres.
        filters: One-Euro tuning.
        rebase_after: seconds a wrist may be lost and still resume without re-anchoring.
    """

    def __init__(self, calibration: Calibration, *, hand_length: float = DEFAULT_HAND_LENGTH,
                 filters: TeleopFilters = TeleopFilters(), rebase_after: float = 0.3) -> None:
        self.calibration = calibration
        self.rebase_after = rebase_after
        self.status = {side: MISSING for side in SIDES}
        self._clutch = False
        self._hands = {
            side: _HandState(
                model=HandModel(side, hand_length),
                position=OneEuro(filters.position),
                orientation=OneEuroQuat(filters.orientation),
                joints=OneEuroElementwise(filters.fingers),
                wrist=calibration.robot_anchor[side],
            )
            for side in SIDES
        }

    def set_clutch(self, engaged: bool) -> None:
        """Engage the clutch to hold the robot's wrists while the operator repositions; release to resume.

        On release each hand is re-anchored, so the robot's hands do not move until the operator's do.
        """
        self._clutch = engaged

    def update(self, hands: Mapping[str, HumanHand | None], t: float) -> TeleopCommand:
        """Command for time ``t`` (host monotonic seconds) from the latest solved hands; ``None`` or absent = no data."""
        for side in SIDES:
            self._update_hand(side, hands.get(side), t)
        seen = {side: state for side, state in self._hands.items() if state.hand is not None}
        return TeleopCommand(
            t=t,
            wrist_pos={side: seen[side].wrist[0] if side in seen else None for side in SIDES},
            wrist_quat={side: seen[side].wrist[1] if side in seen else None for side in SIDES},
            hands={side: self._hands[side].hand for side in SIDES},
            clutch=self._clutch,
        )

    def _update_hand(self, side: str, hand: HumanHand | None, t: float) -> None:
        state = self._hands[side]
        if hand is None:
            if state.hand is not None:
                self.status[side] = DROPPED
            self._hold(state, t)
            return

        joints = state.joints(hand.joints, t)
        tips = state.model.tips(joints)
        state.hand = replace(hand, joints=joints, tips=tips, curl=state.model.curl(joints),
                             pinch=np.linalg.norm(tips[1:] - tips[0], axis=1))
        if not hand.has_position:
            self.status[side] = NO_POSITION
            self._hold(state, t)
            return

        # The filters run in the tracking frame and keep running under the clutch, so the pose used for
        # re-anchoring on release is the same smooth signal the command is built from.
        track = (state.position(hand.wrist_pos, t), state.orientation(hand.wrist_quat, t))
        state.hand = replace(state.hand, wrist_pos=track[0], wrist_quat=track[1])
        self.status[side] = TRACKING
        if self._clutch:
            self._hold(state, t)
            return
        if state.rebase or (state.frozen_at is not None and t - state.frozen_at > self.rebase_after):
            self.calibration = self.calibration.rebased(side, track, state.wrist)
        state.frozen_at, state.rebase = None, False
        state.wrist = self.calibration.to_robot(side, *track)

    def _hold(self, state: _HandState, t: float) -> None:
        """Keep the wrist command where it is, noting since when and whether the clutch asked for it."""
        if state.frozen_at is None:
            state.frozen_at = t
        state.rebase = state.rebase or self._clutch
