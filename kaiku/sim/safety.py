"""The challenge's safety filter, for N robots at once.

A re-implementation of the rule set every action passes through on the real G1 in the
Aaltonomous challenge (``aaltonomous/interface/safety.py``, MIT): joint limits with a margin,
a cap on how fast a command may change, and a workspace box that the wrist and palm frames
may not leave, with the counter as a keep-out volume. A tick on which anything had to be
changed counts as clipped; more than ``max_clipped_steps`` of them aborts the episode.

Kaiku runs every commanded action through it, for the operator's robot and every follower,
so a recorded demonstration is by construction one the real robot would have accepted.
:class:`Shaper` sits in front: it turns raw targets into a stream that already respects the
rate limits, which is why a clean teleoperation session shows zero clipped steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kaiku.sim.ik import BatchArmIK
from kaiku.sim.twin import CONTROL_HZ, PALM_IN_WRIST, Twin


@dataclass
class SafetyConfig:
    dt: float = 1.0 / CONTROL_HZ
    arm_pos_margin: float = 0.05
    hand_pos_margin: float = 0.02
    arm_max_vel: float = 1.0                  # rad/s of command change
    hand_max_vel: float = 3.0
    workspace_min: tuple = (0.0, -0.70, -0.25)
    workspace_max: tuple = (0.80, 0.70, 0.60)
    counter_keepout_min: tuple = (0.14, -0.55, -1.0)
    counter_keepout_max: tuple = (1.20, 0.55, 0.03)
    workspace_tol: float = 1e-4
    bisect_iters: int = 8
    max_clipped_steps: int = 30
    eps: float = 1e-6


def _shrink(lo: np.ndarray, hi: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    m = np.minimum(margin, 0.25 * (hi - lo))
    return lo + m, hi - m


class SafetyFilter:
    def __init__(self, twin: Twin, n_envs: int, kin: BatchArmIK | None = None, cfg: SafetyConfig | None = None):
        self.cfg = cfg or SafetyConfig()
        self.n = n_envs
        self.kin = kin or BatchArmIK()
        self.arm_urdf = twin.arm_range.copy()
        self.hand_urdf = twin.hand_range.copy()
        self.arm_lo, self.arm_hi = _shrink(self.arm_urdf[:, 0], self.arm_urdf[:, 1], self.cfg.arm_pos_margin)
        self.hand_lo, self.hand_hi = _shrink(self.hand_urdf[:, 0], self.hand_urdf[:, 1], self.cfg.hand_pos_margin)
        self.arm_cmd = np.zeros((n_envs, 14))
        self.hand_cmd = np.zeros((n_envs, 14))
        self.clipped_steps = np.zeros(n_envs, int)
        self.aborted = np.zeros(n_envs, bool)
        self.last_reasons: list[set[str]] = [set() for _ in range(n_envs)]

    def reset(self, i: int, arm: np.ndarray, hand: np.ndarray) -> None:
        """Start an episode from the measured pose: the first command is where the robot is."""
        self.arm_cmd[i], self.hand_cmd[i] = arm, hand
        self.clipped_steps[i] = 0
        self.aborted[i] = False

    def violation(self, arm: np.ndarray) -> np.ndarray:
        """Metres by which wrist and palm frames are outside the workspace box or inside the
        counter keep-out, summed over the four frames. ``arm`` (N, 14) -> (N,)."""
        c = self.cfg
        palm, _, wrist = self.kin.fk(arm.reshape(-1, 2, 7))
        # The filter's palm frame is the URDF palm link origin; the kinematics' palm site is the same point.
        p = np.concatenate([wrist, palm], axis=1)                                   # (N, 4, 3)
        lo, hi = np.asarray(c.workspace_min), np.asarray(c.workspace_max)
        out = np.linalg.norm(np.maximum(lo - p, 0.0) + np.maximum(p - hi, 0.0), axis=-1).sum(axis=1)
        klo, khi = np.asarray(c.counter_keepout_min), np.asarray(c.counter_keepout_max)
        depth = np.minimum(p - klo, khi - p).min(axis=-1)
        return out + np.maximum(depth, 0.0).sum(axis=1)

    def filter(self, arm_target: np.ndarray, hand_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One tick for every env: (N, 14) targets in, the commands to send and a (N,) clipped flag out."""
        c = self.cfg
        reasons: list[set[str]] = [set() for _ in range(self.n)]

        def clamp(v, lo, hi, reason):
            out = np.clip(v, lo, hi)
            for i in np.flatnonzero((np.abs(out - v) > c.eps).any(axis=1)):
                reasons[i].add(reason)
            return out

        bad = ~np.isfinite(arm_target).all(axis=1) | ~np.isfinite(hand_target).all(axis=1)
        arm = np.where(bad[:, None], self.arm_cmd, arm_target)
        hand = np.where(bad[:, None], self.hand_cmd, hand_target)
        for i in np.flatnonzero(bad):
            reasons[i].add("invalid")
        # Beyond the URDF limit is a clip; inside it but within the margin is clamped silently.
        arm = np.clip(clamp(arm, self.arm_urdf[:, 0], self.arm_urdf[:, 1], "arm_position"), self.arm_lo, self.arm_hi)
        hand = np.clip(clamp(hand, self.hand_urdf[:, 0], self.hand_urdf[:, 1], "hand_position"),
                       self.hand_lo, self.hand_hi)
        da, dh = c.arm_max_vel * c.dt, c.hand_max_vel * c.dt
        arm = clamp(arm, self.arm_cmd - da, self.arm_cmd + da, "arm_velocity")
        hand = clamp(hand, self.hand_cmd - dh, self.hand_cmd + dh, "hand_velocity")

        # Workspace: a step that would leave the box (or go further out) is shortened by bisection.
        allowed = np.maximum(self.violation(self.arm_cmd), c.workspace_tol)
        over = self.violation(arm) > allowed
        if over.any():
            idx = np.flatnonzero(over)
            base, step = self.arm_cmd[idx], arm[idx] - self.arm_cmd[idx]
            lo_s, hi_s = np.zeros(len(idx)), np.ones(len(idx))
            for _ in range(c.bisect_iters):
                mid = 0.5 * (lo_s + hi_s)
                ok = self.violation(base + mid[:, None] * step) <= allowed[idx]
                lo_s, hi_s = np.where(ok, mid, lo_s), np.where(ok, hi_s, mid)
            arm[idx] = base + lo_s[:, None] * step
            for i in idx:
                reasons[i].add("workspace")

        hold = self.aborted
        arm = np.where(hold[:, None], self.arm_cmd, arm)
        hand = np.where(hold[:, None], self.hand_cmd, hand)
        clipped = np.array([bool(r) for r in reasons]) & ~hold
        self.clipped_steps += clipped
        self.aborted |= self.clipped_steps > c.max_clipped_steps
        self.arm_cmd, self.hand_cmd, self.last_reasons = arm, hand, reasons
        return arm.copy(), hand.copy(), clipped


class Shaper:
    """Makes a target stream the safety filter accepts without clipping: joint limits and a
    rate limit slightly inside the filter's, tracked from the last shaped command.

    An arm's step is scaled as a whole rather than joint by joint, so a rate-limited arm moves
    along the path the inverse kinematics asked for, only slower; clipping joints separately
    bends that path and swings the hand sideways.
    """

    def __init__(self, safety: SafetyFilter, headroom: float = 0.97):
        self.s = safety
        self.da = safety.cfg.arm_max_vel * safety.cfg.dt * headroom
        self.dh = safety.cfg.hand_max_vel * safety.cfg.dt * headroom

    def shape(self, arm_target: np.ndarray, hand_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.s
        n = len(arm_target)
        step = (np.clip(arm_target, s.arm_lo, s.arm_hi) - s.arm_cmd).reshape(n, 2, 7)
        biggest = np.abs(step).max(axis=-1, keepdims=True)
        arm = s.arm_cmd + (step * np.minimum(1.0, self.da / np.maximum(biggest, 1e-12))).reshape(n, 14)
        hand = np.clip(hand_target, s.hand_lo, s.hand_hi)
        hand = s.hand_cmd + np.clip(hand - s.hand_cmd, -self.dh, self.dh)
        return arm, hand


__all__ = ["SafetyConfig", "SafetyFilter", "Shaper", "PALM_IN_WRIST"]
