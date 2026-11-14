"""One hand's fingers, from whichever sensor sees them: a camera's hand landmarks or one Smartglove.

Both end where two gloves end: a solved human hand (``HumanHand``) in kaiku's hand model, ready for the
gloves' own filters (``Teleop``) and retargeting (``Retarget``) onto the Dex3.

    camera   21 landmarks of a hand (the layout MediaPipe gives: wrist, then four per digit) in metres.
             The hand's own frame is read from the palm, the joints from where every bone points
             (``joints_from_points``, the reader used for Rokoko's solved hands), the thumb-to-finger
             gaps from the fingertips. The same landmarks say which way the fingers lean in the
             picture and how large the hand is in it, which is how near it is.
    glove    one Smartglove over RGMP, left or right, with or without a Coil Pro: kaiku's solver, or
             Rokoko's ``rkk-hand-solver`` when it runs. Whichever glove streams is the hand. With a
             Coil Pro the hand also has a place, and that drives where the robot's hand is.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from kaiku.hand import GloveSolver
from kaiku.hand.model import HandModel
from kaiku.hand.solved import hand_length, joints_from_points, with_fingers
from kaiku.rgmp import GloveStream
from kaiku.rgmp.solved import SOLVED_PORT, SolvedStream
from kaiku.types import SIDES, HumanHand

LANDMARKS = 21
_DIGITS = np.arange(1, LANDMARKS).reshape(5, 4)          # thumb, index, middle, ring, little: joint 0, 1, 2, tip
_NO_TURN = np.array([1.0, 0.0, 0.0, 0.0])
STALE_S = 0.5                                            # a glove silent this long is not on a hand


def landmark_frame(world: np.ndarray, right: bool) -> np.ndarray:
    """The hand frame (columns x to the fingers, y, z out of the back of the hand) in the camera's frame,
    from the palm: wrist, the middle knuckle, and the line from the little finger's knuckle to the index's."""
    w = np.asarray(world, dtype=float) - world[0]
    x = w[9] / max(np.linalg.norm(w[9]), 1e-9)
    across = w[5] - w[17]                                # towards the thumb side
    across = across - (across @ x) * x
    across /= max(np.linalg.norm(across), 1e-9)
    y = across if right else -across                     # the thumb is on +y of a right hand, -y of a left
    return np.stack([x, y, np.cross(x, y)], axis=1)


def solve_landmarks(side: str, world: np.ndarray, right: bool, model: HandModel, t: float) -> tuple[HumanHand, float]:
    """A camera's 21 landmarks (metres, camera axes: x right in the picture as taken, y down, z away) as
    the ``side`` hand of ``model``, and how far the fingers lean to the right of straight up in the
    mirrored picture (radians).

    ``right`` says whether the person's hand is a right hand; one of the other side is mirrored onto
    ``side``, so either hand can drive either arm. The hand has no position: where it is on the table
    is the caller's to say."""
    world = np.asarray(world, dtype=float).reshape(LANDMARKS, 3)
    frame = landmark_frame(world, right)
    points = (world - world[0]) @ frame                  # in the hand's own frame
    # Fingers curl towards the palm, never over the back of the hand: if these do, the camera called a
    # left hand right or the other way round.
    if float((points[_DIGITS[1:, 3], 2] - points[_DIGITS[1:, 0], 2]).sum()) > 0.03:
        right = not right
        frame = landmark_frame(world, right)
        points = (world - world[0]) @ frame
    if (side == "right") != right:
        points[:, 1] *= -1.0
    chains = points[_DIGITS]                             # (5, 4, 3), the layout of HandModel.points
    joints = joints_from_points(model, chains)
    tips = chains[:, 3] * (model.hand_length / max(hand_length(chains), 1e-6))
    gaps = np.linalg.norm(tips[1:] - tips[0], axis=1)
    lean = float(np.arctan2(-frame[0, 0], -frame[1, 0]))
    hand = HumanHand(side=side, t=t, wrist_pos=np.zeros(3), wrist_quat=_NO_TURN.copy(), has_position=False,
                     tips=model.tips(joints), joints=joints, curl=model.curl(joints), pinch=gaps, gaps=gaps)
    return hand, lean


def picture_scale(world: np.ndarray, image: np.ndarray) -> float:
    """How large the hand is in the picture: picture units per metre, from the landmarks' own shape, so a
    turned or closed hand reads the same as a flat one. Twice as large is half as far."""
    a = np.asarray(world, dtype=float)[:, :2]
    b = np.asarray(image, dtype=float)
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    return float((a * b).sum() / max((a * a).sum(), 1e-9))


class GloveFingers:
    """The fingers of whichever Smartglove streams on ``host:port``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 12276, solver: str | None = None):
        self.stream = GloveStream(host, port).start()
        self.solved: SolvedStream | None = None
        if solver != "off":
            shost, _, sport = (solver or "").partition(":")
            self.solved = SolvedStream(shost or host, int(sport) if sport else SOLVED_PORT).start()
        self.solvers = {side: GloveSolver(side) for side in SIDES}
        self.models = {side: HandModel(side) for side in SIDES}
        self.side: str | None = None                     # the glove that is on a hand now
        self.fingers = "kaiku"                           # whose solver the fingers are from
        self.coil = False

    def hand(self, side: str, now: float) -> HumanHand | None:
        """The gloved hand as the ``side`` hand, or None while no glove streams."""
        frames = [(f.t_recv, s, f) for s in SIDES if (f := self.stream.latest(s)) is not None and now - f.t_recv < STALE_S]
        if not frames:
            self.side = None
            return None
        _, worn, frame = max(frames, key=lambda item: item[0])
        self.side, self.coil = worn, bool(frame.coil_present)
        hand = self.solvers[worn].solve(frame)
        theirs = self.solved.latest(worn) if self.solved else None
        self.fingers = "kaiku" if theirs is None else "rokoko"
        if theirs is not None:
            hand = with_fingers(hand, self.models[worn], theirs.points)
        if worn == side:
            return hand
        # The other hand's glove: the same joints on the mirrored hand, which is the same gaps. Where the
        # hand is stays where the Coil Pro says it is.
        return replace(hand, side=side, tips=self.models[side].tips(hand.joints))

    def flip(self) -> None:
        """The Coil Pro's Z points the other way than assumed: turn the tracking frame over."""
        for solver in self.solvers.values():
            solver.flip_vertical()

    def hud(self) -> dict:
        side = self.side
        return {"side": side, "hz": round(self.stream.rate_hz(side), 1) if side else 0.0,
                "connected": self.stream.connected, "solver": self.fingers, "coil": self.coil}

    def close(self) -> None:
        self.stream.stop()
        if self.solved:
            self.solved.stop()
