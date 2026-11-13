"""Arm inverse kinematics: palm pose -> the 7 arm joints of one side.

Damped least squares on the bare-robot model, a few iterations per control tick, warm-started
from the last command. That is what teleoperation needs: it tracks a moving target smoothly,
never jumps between elbow configurations, and degrades gracefully (holds the closest reachable
pose) when the operator reaches outside the workspace. A null-space term pulls the elbow
towards a rest posture so the redundant 7th degree of freedom does not drift.

One instance owns one scratch ``MjData``; it is cheap (about 20 microseconds per iteration),
so the fan-out solves every follower robot's arms on each tick with the same code.
"""
from __future__ import annotations

import mujoco
import numpy as np

from kaiku.sim.twin import Twin, build_twin

SIDES = ("left", "right")
# Arm home pose of the challenge (elbows bent, hands above the counter), per side.
ARM_HOME = {"left": np.array([0.0, 0.35, 0.0, -0.5, 0.0, 0.0, 0.0]),
            "right": np.array([0.0, -0.35, 0.0, -0.5, 0.0, 0.0, 0.0])}


def quat_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def mat_to_quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def rotation_error(R_target: np.ndarray, R_now: np.ndarray) -> np.ndarray:
    """World-frame rotation vector that takes ``R_now`` to ``R_target``."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R_target @ R_now.T).reshape(9))
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class ArmIK:
    def __init__(self, robot: Twin | None = None, rest: dict[str, np.ndarray] | None = None,
                 limit_margin: float = 0.06):
        self.robot = robot or build_twin(with_scene=False)
        self.model = self.robot.model
        self.data = mujoco.MjData(self.model)
        self.rest = rest or ARM_HOME
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self.idx = {s: (self.robot.arm_qadr[i * 7:(i + 1) * 7], self.robot.arm_dadr[i * 7:(i + 1) * 7])
                    for i, s in enumerate(SIDES)}
        self.lim = {s: (self.robot.arm_range[i * 7:(i + 1) * 7, 0] + limit_margin,
                        self.robot.arm_range[i * 7:(i + 1) * 7, 1] - limit_margin) for i, s in enumerate(SIDES)}
        self.site = {s: self.robot.site[f"{s}_palm"] for s in SIDES}

    def fk(self, side: str, q: np.ndarray, site: str = "palm") -> tuple[np.ndarray, np.ndarray]:
        """Position and rotation matrix of ``<side>_<site>`` for arm joints ``q`` (7,)."""
        qadr, _ = self.idx[side]
        self.data.qpos[qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        sid = self.robot.site[f"{side}_{site}"]
        return self.data.site_xpos[sid].copy(), self.data.site_xmat[sid].reshape(3, 3).copy()

    def solve(self, side: str, q0: np.ndarray, pos: np.ndarray, rot: np.ndarray, iters: int = 3,
              rot_weight: float = 0.35, damping: float = 0.03, max_step: float = 0.25,
              rest_gain: float = 0.02) -> tuple[np.ndarray, float, float]:
        """Move arm joints ``q0`` towards palm position ``pos`` (3,) and rotation ``rot`` (3,3).

        ``rot_weight`` (metres per radian) trades orientation against position when both cannot
        be met. Returns (q, position error in m, orientation error in rad) after ``iters`` steps.
        """
        qadr, dadr = self.idx[side]
        lo, hi = self.lim[side]
        sid = self.site[side]
        q = np.clip(np.asarray(q0, float), lo, hi)
        ep = er = 0.0
        for i in range(iters + 1):
            self.data.qpos[qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            dp = pos - self.data.site_xpos[sid]
            dr = rotation_error(rot, self.data.site_xmat[sid].reshape(3, 3))
            ep, er = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
            if i == iters:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, sid)
            J = np.vstack([self._jacp[:, dadr], rot_weight * self._jacr[:, dadr]])
            e = np.r_[dp, rot_weight * dr]
            JJt = J @ J.T + damping ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # Null-space pull to the rest posture: uses only motion that leaves the palm where it is.
            pull = rest_gain * (self.rest[side] - q)
            dq += pull - J.T @ np.linalg.solve(JJt, J @ pull)
            n = np.abs(dq).max()
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, lo, hi)
        return q, ep, er

    def solve_full(self, side: str, pos: np.ndarray, rot: np.ndarray, seed: np.ndarray | None = None,
                   iters: int = 120, **kw) -> tuple[np.ndarray, float, float]:
        """Converged solution from a seed (default: rest posture); for planning, not for tracking."""
        q = self.rest[side] if seed is None else seed
        return self.solve(side, q, pos, rot, iters=iters, **kw)
