"""From a person's hands to the robot's: where the Dex3 pinches, how its palm is turned, how far
it closes.

A human hand has five fingers and the Dex3 three, so joint angles do not carry over. What does
carry over is what the hand is doing to an object:

  * where: the point between the thumb tip and the index tip in a grip. The robot's pinch point
    (between its thumb and its two fingers) goes there, so the operator aims with the fingertips,
    not with the wrist.
  * which way: the palm turns with the operator's palm, one to one.
  * how wide: the gap between the operator's thumb and index finger sets how far the robot's
    index finger closes; the gap to the middle finger sets its other finger. A gap of 35 mm, as
    if holding the part, is the robot's grip on a 45 mm part; closing further squeezes; a flat
    hand straightens the robot's fingers so a part falls freely.

Everything here is a pure function of one command, in both directions: ``Retarget.command``
turns a ``TeleopCommand`` into the session's lead command, and ``Retarget.human`` gives the hand
pose a person would need for a wanted robot pose (how the stand-in operator moves its hands).
"""
from __future__ import annotations

import numpy as np

from kaiku.hand.geometry import Pose, mat_to_quat, quat_to_mat
from kaiku.hand.model import DEFAULT_HAND_LENGTH, HandModel
from kaiku.hand.poses import HandPoses
from kaiku.sim.grasp import HAND_CLOSED, HAND_OPEN, HAND_RELEASE
from kaiku.sim.reach import level_palm
from kaiku.types import SIDES, HumanHand, TeleopCommand

# Gap between thumb tip and fingertip (m) against the robot finger's closure u:
# -1 fingers straight (lets go), 0 open claw, 1 the pinch that holds a 45 mm part, beyond: squeeze.
GAP_KNOTS = np.array([0.012, 0.035, 0.080, 0.125])
U_KNOTS = np.array([1.2, 1.0, 0.0, -1.0])
HOLD_GAP = 0.035
# A finger's own curl (``HandModel.curl``, 0 straight to 1 fully curled), measured on real hands seen by
# a camera: a flat or relaxed hand reads up to 0.25, a fingertip on the thumb 0.45, a fist 0.6 and more.
CURL_BENT, CURL_OPEN, CURL_CLOSED = 0.25, 0.30, 0.60
READY_PINCH = {"left": np.array([0.28, 0.14, 0.17]), "right": np.array([0.28, -0.14, 0.17])}
"""Where the robot's pinch points go when the operator calibrates (flat hands, pointing ahead):
level, above the mat, in reach of every part."""


class GapRange:
    """What one operator's glove reads with thumb and finger together and with the hand flat, learned while
    they work, and their gaps brought onto the range the robot's fingers are driven by.

    A solved hand is a generic skeleton on somebody's real hand: fingertips that touch may read 25 mm apart,
    a flat hand 70 mm instead of the model's 125. Left alone, that operator could never squeeze or never
    let go. ``open`` takes the flat hand of the calibration pose; the closed end is the smallest gap seen
    so far, and until the operator has pinched it is taken to be the hold gap, so closing the hand grips
    from the first try.
    """

    MOST = 3.0          # how far a short range is stretched at most

    def __init__(self, closed: np.ndarray, flat: np.ndarray) -> None:
        self.closed, self.flat = np.asarray(closed, dtype=float), np.asarray(flat, dtype=float)   # the model's
        self.low = np.full(4, HOLD_GAP)
        self.high: np.ndarray | None = None

    def open(self, pinch: np.ndarray) -> None:
        """The operator holds the hand flat now."""
        self.high = np.asarray(pinch, dtype=float).copy()

    def __call__(self, pinch: np.ndarray) -> np.ndarray:
        pinch = np.asarray(pinch, dtype=float)
        self.low = np.minimum(self.low, np.maximum(pinch, GAP_KNOTS[0]))
        if self.high is None:
            scale = 1.0
        else:
            scale = np.clip((self.flat - self.closed) / np.maximum(self.high - self.low, 1e-3), 1.0, self.MOST)
        return self.closed + (pinch - self.low) * scale


def closure_from_gap(gap: np.ndarray | float) -> np.ndarray:
    return np.interp(gap, GAP_KNOTS, U_KNOTS)


def gap_from_closure(u: np.ndarray | float) -> np.ndarray:
    return np.interp(u, U_KNOTS[::-1], GAP_KNOTS[::-1])


def dex3_posture(u_thumb: float, u_middle: float, u_index: float) -> np.ndarray:
    """Dex3 joints (7, left-hand order: thumb 0-2, middle 0-1, index 0-1) for the three closures."""
    def finger(u: float, lo: int) -> np.ndarray:
        o, c, r = HAND_OPEN[lo:lo + 2], HAND_CLOSED[lo:lo + 2], HAND_RELEASE[lo:lo + 2]
        return o + u * (c - o) if u >= 0.0 else o - u * (r - o)

    t = float(np.clip(u_thumb, 0.0, U_KNOTS[0]))
    return np.concatenate([HAND_OPEN[:3] + t * (HAND_CLOSED[:3] - HAND_OPEN[:3]), finger(u_middle, 3), finger(u_index, 5)])


def closure_of_posture(posture: np.ndarray) -> tuple[float, float]:
    """The (middle, index) closures of a Dex3 posture in left-hand order; inverse of ``dex3_posture``."""
    def one(v: float, lo: int) -> float:
        o, c, r = HAND_OPEN[lo], HAND_CLOSED[lo], HAND_RELEASE[lo]
        return (v - o) / (c - o) if v <= o else -(v - o) / (r - o)

    return one(float(posture[3]), 3), one(float(posture[5]), 5)


class Retarget:
    def __init__(self, hand_length: float = DEFAULT_HAND_LENGTH, reach_scale: float = 0.8):
        self.scale = reach_scale
        self.models = {side: HandModel(side, hand_length) for side in SIDES}
        self.poses = {side: HandPoses(m) for side, m in self.models.items()}
        self.palm = dict(zip(SIDES, level_palm(np.zeros(2))))            # robot palm axes for a flat, forward hand
        # The grip centre in the hand frame: between the thumb tip and the index tip while holding a part.
        self.centre = {}
        for side in SIDES:
            tips = self.models[side].tips(self.poses[side].grip(HOLD_GAP))
            self.centre[side] = 0.5 * (tips[0] + tips[1])
        # In a grip the middle finger sits further from the thumb than the index finger does. The
        # middle finger's gap is read against that, so a plain grip closes both robot fingers alike.
        gaps = np.linspace(0.0, 0.14, 57)
        mid = []
        for g in gaps:
            tips = self.models["right"].tips(self.poses["right"].grip(g))
            mid.append(np.linalg.norm(tips[0] - tips[2]))
        self._index_gap, self._middle_gap = gaps, np.array(mid)

    # ------------------------------------------------------------------ person -> robot
    def gap_range(self) -> GapRange:
        """A fresh ``GapRange`` for one glove: the closed and the flat end of every thumb-to-finger gap as
        the robot's fingers read them (the middle finger's against its own, wider, scale)."""
        ends = np.array([GAP_KNOTS[0], GAP_KNOTS[-1]])
        middle = np.interp(ends, self._index_gap, self._middle_gap)
        closed, flat = np.column_stack([ends, middle, ends, ends])
        return GapRange(closed, flat)

    def fingers(self, hand: HumanHand, curl: bool = False) -> np.ndarray:
        """Dex3 posture (7, left-hand order) for a solved human hand.

        The thumb-to-finger gaps close the robot's fingers, as a pinch does. With ``curl`` each finger's
        own bend counts as well: an index finger curled alone closes the robot's index finger, a fist
        closes the hand. And a gap only closes a finger whose own joints, or the thumb's, are bent: a
        camera that sees a flat hand from an odd side can read its thumb and finger as near each other,
        and a flat hand is not a pinch."""
        u_index = float(closure_from_gap(hand.pinch[0]))
        as_index = float(np.interp(hand.pinch[1], self._middle_gap, self._index_gap))
        u_middle = float(closure_from_gap(as_index))
        if curl:
            thumb, index, middle = (float(c) for c in hand.curl[:3])
            own = np.clip((np.array([index, middle]) - CURL_OPEN) / (CURL_CLOSED - CURL_OPEN), 0.0, 1.0)
            if max(thumb, index) < CURL_BENT:
                u_index = min(u_index, 0.0)
            if max(thumb, middle) < CURL_BENT:
                u_middle = min(u_middle, 0.0)
            if own[0] > 0.0:
                u_index = max(u_index, float(own[0]))
            if own[1] > 0.0:
                u_middle = max(u_middle, float(own[1]))
        return dex3_posture(max(u_index, u_middle), u_middle, u_index)

    def pose(self, side: str, wrist_pos: np.ndarray, wrist_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Robot pinch point (3,) and palm rotation (3, 3) for the operator's wrist, both in the pelvis frame."""
        R = quat_to_mat(wrist_quat)
        return np.asarray(wrist_pos) + self.scale * R @ self.centre[side], R @ self.palm[side]

    def command(self, cmd: TeleopCommand, hold: tuple[np.ndarray, np.ndarray, np.ndarray]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pinch (2, 3), palm rotation (2, 3, 3), Dex3 postures (2, 7), each in left-hand order)
        for a teleop command. A hand that is not tracked keeps what ``hold`` (the same triple)
        has for it."""
        pinch, rot, posture = hold[0].copy(), hold[1].copy(), hold[2].copy()
        for s, side in enumerate(SIDES):
            if cmd.wrist_pos[side] is not None:
                pinch[s], rot[s] = self.pose(side, cmd.wrist_pos[side], cmd.wrist_quat[side])
            if cmd.hands[side] is not None:
                posture[s] = self.fingers(cmd.hands[side])
        return pinch, rot, posture

    def ready(self) -> dict[str, Pose]:
        """Robot-frame wrist poses the operator's calibration pose is anchored to: they put the
        pinch points at ``READY_PINCH`` with the palms level and the fingers pointing ahead."""
        eye = np.array([1.0, 0.0, 0.0, 0.0])
        return {side: (READY_PINCH[side] - self.scale * self.centre[side], eye) for side in SIDES}

    # ------------------------------------------------------------------ robot -> person
    def human(self, side: str, pinch: np.ndarray, rot: np.ndarray, posture: np.ndarray) -> tuple[Pose, np.ndarray]:
        """The hand a person needs for the robot's ``side`` hand to pinch at ``pinch`` with palm
        ``rot`` and Dex3 ``posture`` (7, left-hand order): wrist pose in the pelvis frame, as the
        calibration maps it, and finger joints (5, 4)."""
        R = rot @ self.palm[side].T
        wrist = (np.asarray(pinch) - self.scale * R @ self.centre[side], mat_to_quat(R))
        _, u_index = closure_of_posture(posture)
        return wrist, self.poses[side].grip(float(gap_from_closure(u_index)))
