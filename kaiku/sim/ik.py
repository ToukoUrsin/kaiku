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


class BatchArmIK:
    """The same damped-least-squares step as :class:`ArmIK`, for N robots x 2 arms at once.

    The fan-out retargets every follower robot on every tick, so the per-robot cost has to be
    microseconds. Each arm is a chain of seven hinges; its forward kinematics and Jacobian are
    written out in numpy over the batch dimension instead of calling MuJoCo per robot.
    The chain constants are read from the model, so this stays in sync with the twin.
    """

    def __init__(self, robot: Twin | None = None, rest: dict[str, np.ndarray] | None = None,
                 limit_margin: float = 0.06):
        robot = robot or build_twin(with_scene=False)
        m = robot.model
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        self.rest = np.stack([(rest or ARM_HOME)[s] for s in SIDES])                     # (2, 7)
        self.lo = robot.arm_range[:, 0].reshape(2, 7) + limit_margin
        self.hi = robot.arm_range[:, 1].reshape(2, 7) - limit_margin
        self.base_p = np.zeros((2, 3))          # frame the first arm body is attached to (the torso)
        self.base_R = np.zeros((2, 3, 3))
        self.link_p = np.zeros((2, 7, 3))       # body offset in its parent, per joint
        self.link_R = np.zeros((2, 7, 3, 3))
        self.axis_K = np.zeros((2, 7, 3, 3))    # cross-product matrix of each joint axis (body frame)
        self.axis = np.zeros((2, 7, 3))
        self.palm = np.zeros((2, 3))            # palm site in the last body
        self.palm_R = np.zeros((2, 3, 3))
        for s, side in enumerate(SIDES):
            for k, name in enumerate(robot.arm_names[s * 7:(s + 1) * 7]):
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                b = m.jnt_bodyid[j]
                assert np.allclose(m.jnt_pos[j], 0.0), "arm joints are expected at their body origin"
                if k == 0:
                    parent = m.body_parentid[b]
                    self.base_p[s], self.base_R[s] = d.xpos[parent], d.xmat[parent].reshape(3, 3)
                self.link_p[s, k], self.link_R[s, k] = m.body_pos[b], quat_to_mat(m.body_quat[b])
                a = m.jnt_axis[j]
                self.axis[s, k] = a
                self.axis_K[s, k] = [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]
            sid = robot.site[f"{side}_palm"]
            self.palm[s], self.palm_R[s] = m.site_pos[sid], quat_to_mat(m.site_quat[sid])

    def fk(self, q: np.ndarray, with_jacobian: bool = False):
        """``q`` (N, 2, 7) -> palm position (N, 2, 3), rotation (N, 2, 3, 3), wrist origin (N, 2, 3)
        and, if asked, the 6 x 7 Jacobian (N, 2, 6, 7) of [position; rotation]."""
        n = q.shape[0]
        R = np.broadcast_to(self.base_R, (n, 2, 3, 3)).copy()
        p = np.broadcast_to(self.base_p, (n, 2, 3)).copy()
        sin, cos = np.sin(q), np.cos(q)
        K = self.axis_K
        KK = K @ K
        eye = np.eye(3)
        joint_p, joint_a = [], []
        for k in range(7):
            p = p + np.einsum("nsij,sj->nsi", R, self.link_p[:, k])
            R = R @ self.link_R[:, k]
            joint_p.append(p)
            joint_a.append(np.einsum("nsij,sj->nsi", R, self.axis[:, k]))
            Rq = eye + sin[:, :, k, None, None] * K[:, k] + (1.0 - cos[:, :, k, None, None]) * KK[:, k]
            R = R @ Rq
        wrist = p
        palm_p = p + np.einsum("nsij,sj->nsi", R, self.palm)
        palm_R = R @ self.palm_R
        if not with_jacobian:
            return palm_p, palm_R, wrist
        a = np.stack(joint_a, axis=-1)                                     # (N, 2, 3, 7)
        r = palm_p[..., None] - np.stack(joint_p, axis=-1)                 # (N, 2, 3, 7)
        J = np.concatenate([np.cross(a, r, axis=2), a], axis=2)            # (N, 2, 6, 7)
        return palm_p, palm_R, wrist, J

    @staticmethod
    def _rotvec(R_target: np.ndarray, R_now: np.ndarray) -> np.ndarray:
        """Batched world-frame rotation vector from ``R_now`` to ``R_target`` (..., 3)."""
        E = R_target @ np.swapaxes(R_now, -1, -2)
        v = 0.5 * np.stack([E[..., 2, 1] - E[..., 1, 2], E[..., 0, 2] - E[..., 2, 0], E[..., 1, 0] - E[..., 0, 1]], -1)
        s = np.linalg.norm(v, axis=-1)
        c = np.clip(0.5 * (np.trace(E, axis1=-2, axis2=-1) - 1.0), -1.0, 1.0)
        angle = np.arctan2(s, c)
        scale = np.where(s > 1e-9, angle / np.maximum(s, 1e-9), 1.0)
        return v * scale[..., None]

    def solve(self, q0: np.ndarray, pos: np.ndarray, rot: np.ndarray, iters: int = 3,
              rot_weight: float = 0.35, damping: float = 0.03, max_step: float = 0.25,
              rest_gain: float = 0.02, rest: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``q0`` (N, 14) arm joints, ``pos`` (N, 2, 3) and ``rot`` (N, 2, 3, 3) palm targets for
        left and right. ``rest`` (N, 2, 7) is the posture the spare degree of freedom is drawn
        to (default: the home posture). Returns (q (N, 14), position error (N, 2) m, orientation
        error (N, 2) rad)."""
        n = q0.shape[0]
        rest = self.rest if rest is None else rest
        q = np.clip(q0.reshape(n, 2, 7), self.lo, self.hi)
        W = np.r_[1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight]
        eye6 = damping ** 2 * np.eye(6)
        for _ in range(iters):
            p, R, _, J = self.fk(q, with_jacobian=True)
            e = np.concatenate([pos - p, rot_weight * self._rotvec(rot, R)], axis=-1)       # (N, 2, 6)
            J = J * W[:, None]
            JJt = J @ np.swapaxes(J, -1, -2) + eye6
            pull = rest_gain * (rest - q)
            rhs = np.stack([e, np.einsum("nsij,nsj->nsi", J, pull)], axis=-1)                  # (N, 2, 6, 2)
            sol = np.linalg.solve(JJt, rhs)
            dq = np.einsum("nsji,nsj->nsi", J, sol[..., 0] - sol[..., 1]) + pull
            big = np.abs(dq).max(axis=-1, keepdims=True)
            dq = dq * np.minimum(1.0, max_step / np.maximum(big, 1e-9))
            q = np.clip(q + dq, self.lo, self.hi)
        p, R, _ = self.fk(q)
        return q.reshape(n, 14), np.linalg.norm(pos - p, axis=-1), np.linalg.norm(self._rotvec(rot, R), axis=-1)
