"""Turn a list of waypoints into a smooth palm trajectory, sampled at any time.

Positions follow minimum-jerk profiles (the velocity bell of human reaching), orientations
slerp on the same profile and hand postures blend linearly. The scripted operator samples
this at the glove rate; tests sample it at the control rate and feed the arm IK directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from kaiku.sim.grasp import Waypoint
from kaiku.sim.ik import mat_to_quat, quat_to_mat


def min_jerk(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


@dataclass
class PalmState:
    point: np.ndarray     # (3,) pinch point
    rot: np.ndarray       # (3, 3)
    hand: np.ndarray      # (7,) left-order posture
    label: str
    done: bool


class Trajectory:
    """Piecewise minimum-jerk path through waypoints, starting from a given palm state."""

    def __init__(self, start_point: np.ndarray, start_rot: np.ndarray, start_hand: np.ndarray,
                 waypoints: list[Waypoint]):
        self.wps = waypoints
        self.points = [np.asarray(start_point, float)] + [w.point for w in waypoints]
        self.quats = [mat_to_quat(start_rot)] + [mat_to_quat(w.rot) for w in waypoints]
        self.hands = [np.asarray(start_hand, float)] + [w.hand for w in waypoints]
        self.t_end = np.cumsum([w.duration + w.hold for w in waypoints])
        self.duration = float(self.t_end[-1]) if waypoints else 0.0

    def sample(self, t: float) -> PalmState:
        if not self.wps:
            return PalmState(self.points[0], quat_to_mat(self.quats[0]), self.hands[0], "idle", True)
        k = int(np.searchsorted(self.t_end, t, side="right"))
        if k >= len(self.wps):
            return PalmState(self.points[-1], quat_to_mat(self.quats[-1]), self.hands[-1], self.wps[-1].label, True)
        w = self.wps[k]
        t0 = self.t_end[k - 1] if k else 0.0
        s = min_jerk((t - t0) / w.duration) if w.duration > 0 else 1.0
        point = self.points[k] + s * (self.points[k + 1] - self.points[k])
        quat = slerp(self.quats[k], self.quats[k + 1], s)
        hand = self.hands[k] + s * (self.hands[k + 1] - self.hands[k])
        return PalmState(point, quat_to_mat(quat), hand, w.label, False)


class Follower:
    """Plays waypoints the way a person does: move, look, and only go on once the hand is there.

    Each segment is a minimum-jerk move; at its end the follower waits until the measured pinch
    point is within tolerance of the waypoint (or a timeout passes), then holds for the
    waypoint's dwell. Without this the fingers would close while a rate-limited arm is still
    travelling. The scripted operator and the reference controller both use it.
    """

    def __init__(self, start_point: np.ndarray, start_rot: np.ndarray, start_hand: np.ndarray,
                 waypoints: list[Waypoint], pos_tol: float = 0.006, rot_tol: float = 0.10,
                 patience: float = 2.0):
        self.wps = list(waypoints)
        self.pos_tol, self.rot_tol, self.patience = pos_tol, rot_tol, patience
        self.k = 0                      # current waypoint
        self.t = 0.0                    # time inside the current waypoint
        self.waited = 0.0               # time spent waiting for the robot at the end of the move
        self.held = 0.0
        self._from = (np.asarray(start_point, float), mat_to_quat(start_rot), np.asarray(start_hand, float))

    @property
    def done(self) -> bool:
        return self.k >= len(self.wps)

    @property
    def label(self) -> str:
        return self.wps[min(self.k, len(self.wps) - 1)].label if self.wps else "idle"

    def update(self, dt: float, point: np.ndarray, rot: np.ndarray, hand: np.ndarray | None = None) -> PalmState:
        """Advance by ``dt`` given where the robot's pinch point, palm (and working hand) are now."""
        if self.done:
            p, q, h = self._from
            return PalmState(p, quat_to_mat(q), h, self.label, True)
        w = self.wps[self.k]
        p0, q0, h0 = self._from
        q1 = mat_to_quat(w.rot)
        self.t += dt
        s = min_jerk(self.t / w.duration) if w.duration > 0 else 1.0
        state = PalmState(p0 + s * (w.point - p0), quat_to_mat(slerp(q0, q1, s)), h0 + s * (w.hand - h0),
                          w.label, False)
        if self.t >= w.duration:
            tol = self.pos_tol if w.tol is None else w.tol
            near = (np.linalg.norm(point - w.point) <= tol
                    and np.linalg.norm(_rotvec(w.rot, rot)) <= self.rot_tol * max(1.0, tol / self.pos_tol)
                    and (hand is None or np.abs(hand - w.hand).max() <= 0.15))
            if not near and self.waited < self.patience:
                self.waited += dt
            else:
                self.held += dt
                if self.held >= w.hold:
                    self._from = (w.point, q1, w.hand)
                    self.k, self.t, self.waited, self.held = self.k + 1, 0.0, 0.0, 0.0
        return state


def _rotvec(R_target: np.ndarray, R_now: np.ndarray) -> np.ndarray:
    q, v = np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(q, (R_target @ R_now.T).reshape(9))
    mujoco.mju_quat2Vel(v, q if q[0] >= 0 else -q, 1.0)
    return v
