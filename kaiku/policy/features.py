"""What the policy sees and what it commands, in the task's own terms.

The demonstrations hold joint angles (``observation.state``) and joint targets (``action``).
A policy that maps one to the other has to learn the arm's kinematics before it learns the
task, and needs far more data for it. The kinematics are known, so they are used on both sides:

    sees      where each pinch point and palm is, how the fingers stand, what it asked for on
              the last tick, and the part to sort next: where it is from each hand, its shape
              and turn, its bin, what lies around it
    commands  for the next ``HORIZON`` ticks, where each pinch point should go (from where it
              was last told to go), how the palm should be turned and the 14 finger targets

Its own last command is the policy's only memory. A hand resting above a part looks the same a
moment before it goes down and a moment after it came up; what was last asked of it tells the
two apart.

The command becomes the challenge's 28 joint targets through the same inverse kinematics and
safety filter as the operator's. Parts come in the order they are to be sorted, each with the
arm that takes it: the plan. In a demonstration the plan is read off what happened; when the
policy runs alone it is the operator's habit (nearest part first, the arm on the bin's side).
"""
from __future__ import annotations

import numpy as np

from kaiku.sim.scene import BINS, Scene

MAX_PARTS = 5
HORIZON = 16
SYMMETRY = np.array([4, 0, 3])          # equivalent turns of a cube, a cylinder, a triangle (echo.SHAPE_CODE order)
ACT_DIM = 32                            # per tick: 2 x pinch move (3), 2 x palm (6), fingers (14)
LIFTED = 0.015                          # m above its resting height: a part is off the mat


def rot6(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two columns, enough to rebuild the rotation."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def from_rot6(v: np.ndarray) -> np.ndarray:
    a, b = v[..., :3], v[..., 3:]
    x = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b - (x * b).sum(-1, keepdims=True) * x
    y = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def quat_yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Featurizer:
    """Turns measured state into the policy's input, and its output back into targets."""

    def __init__(self, scene: Scene):
        self.rest_z = scene.rest_z
        self.bin_c = np.stack([scene.bin_centers[b] for b in BINS])                       # (3, 3)
        inner = [scene.bin_inner(b) for b in BINS]
        self.bin_lo = np.stack([lo for lo, _ in inner])
        self.bin_hi = np.stack([hi for _, hi in inner])
        self.bin_hi[:, 2] += 0.005
        # Where things sit in the input vector; training moves the hands about in it (see train.jitter).
        self.slices = {"has": slice(0, 1), "pinch": slice(1, 7), "palm": slice(7, 19), "fingers": slice(19, 33),
                       "part": slice(33, 36), "to_part": slice(36, 42), "turn": slice(42, 44),
                       "shape": slice(44, 47), "bin": slice(47, 50), "arm": slice(50, 51),
                       "to_bin": slice(51, 57), "slot": slice(57, 58), "around": slice(58, 70),
                       "left": slice(70, 71), "lead": slice(71, 77), "asked": slice(77, 91)}
        self.dim = 91

    def in_bin(self, pos: np.ndarray) -> np.ndarray:
        """(..., 3) -> bool: the centre is inside some bin, below its rim."""
        p = pos[..., None, :]
        return ((p >= self.bin_lo) & (p <= self.bin_hi)).all(-1).any(-1)

    def observe(self, pinch: np.ndarray, rot: np.ndarray, fingers: np.ndarray, pos: np.ndarray, yaw: np.ndarray,
                shape: np.ndarray, bins: np.ndarray, side: np.ndarray, present: np.ndarray,
                asked_pinch: np.ndarray, asked_fingers: np.ndarray) -> np.ndarray:
        """Batch of B. ``pinch`` (B, 2, 3), ``rot`` (B, 2, 3, 3), ``fingers`` (B, 14) measured;
        parts in plan order, padded to ``MAX_PARTS``: ``pos`` (B, P, 3), ``yaw`` (B, P),
        ``shape``, ``bins`` (0 left, 1 centre, 2 right), ``side`` (0 left arm, 1 right) ints,
        ``present`` bool; ``asked_pinch`` (B, 2, 3) and ``asked_fingers`` (B, 14): the command
        of the tick before. Returns (B, dim) float32."""
        B = len(pinch)
        rows = np.arange(B)
        todo = present & ~self.in_bin(pos)
        has = todo.any(axis=1)
        t = np.argmax(todo, axis=1)                                   # the first part still to sort
        tp = np.where(has[:, None], pos[rows, t], 0.0)
        k = SYMMETRY[shape[rows, t]]
        turn = np.stack([np.cos(k * yaw[rows, t]), np.sin(k * yaw[rows, t])], axis=1) * (k > 0)[:, None]
        tbin = bins[rows, t]
        slot = (present & ~todo & (bins == tbin[:, None])).sum(axis=1)
        # The other parts still on the mat, nearest first, from the part to sort.
        other = todo.copy()
        other[rows, t] = False
        rel = pos[:, :, :2] - tp[:, None, :2]
        dist = np.where(other, np.linalg.norm(rel, axis=-1), np.inf)
        near = np.argsort(dist, axis=1)[:, :4]
        around = np.where(np.isfinite(np.take_along_axis(dist, near, 1))[..., None],
                          np.concatenate([np.take_along_axis(rel, near[..., None], 1), np.ones((B, 4, 1))], -1), 0.0)
        f = np.zeros((B, self.dim), np.float32)
        s = self.slices
        f[:, s["has"]] = has[:, None]
        f[:, s["pinch"]] = pinch.reshape(B, 6)
        f[:, s["palm"]] = rot6(rot).reshape(B, 12)
        f[:, s["fingers"]] = fingers
        f[:, s["part"]] = tp
        f[:, s["to_part"]] = (tp[:, None] - pinch).reshape(B, 6)
        f[:, s["turn"]] = turn
        f[rows, s["shape"].start + shape[rows, t]] = 1.0
        f[rows, s["bin"].start + tbin] = 1.0
        f[:, s["arm"]] = (1.0 - 2.0 * side[rows, t])[:, None]
        f[:, s["to_bin"]] = (self.bin_c[tbin][:, None] - pinch).reshape(B, 6)
        f[:, s["slot"]] = slot[:, None]
        f[:, s["around"]] = around.reshape(B, 12)
        f[:, s["left"]] = todo.sum(axis=1)[:, None]
        f[:, s["lead"]] = (asked_pinch - pinch).reshape(B, 6)
        f[:, s["asked"]] = asked_fingers
        gone = ~has
        for name in ("to_part", "turn", "shape", "bin", "arm", "to_bin", "slot", "around"):
            f[gone, s[name]] = 0.0
        return f

    @staticmethod
    def command(pinch: np.ndarray, rot: np.ndarray, fingers: np.ndarray, asked: np.ndarray) -> np.ndarray:
        """Targets -> the policy's output for one tick. ``pinch`` (..., 2, 3) and ``rot``
        (..., 2, 3, 3) commanded, ``fingers`` (..., 14); ``asked`` (..., 2, 3) the pinch points
        last asked for, which the move is counted from."""
        lead = pinch.shape[:-2]
        return np.concatenate([(pinch - asked).reshape(*lead, 6), rot6(rot).reshape(*lead, 12), fingers], axis=-1)

    @staticmethod
    def targets(y: np.ndarray, asked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The policy's output (..., ACT_DIM) -> pinch (..., 2, 3), palm (..., 2, 3, 3), fingers (..., 14)."""
        lead = y.shape[:-1]
        pinch = y[..., :6].reshape(*lead, 2, 3) + asked
        return pinch, from_rot6(y[..., 6:18].reshape(*lead, 2, 6)), y[..., 18:]


def plan_for(pos: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The operator's habit for a scene with parts at ``pos`` (n, 3): nearest part first, the
    arm on the bin's side, and for the centre bin the arm nearer the part. Returns (order (n,),
    side (n,) in that order)."""
    order = np.argsort(pos[:, 0], kind="stable")
    side = np.where(bins == 0, 0, np.where(bins == 2, 1, np.where(pos[:, 1] > 0, 0, 1)))
    return order, side[order]
