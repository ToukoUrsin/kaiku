"""N independent copies of the task on one shared model, stepped in parallel on CPU threads.

MuJoCo releases the GIL inside ``mj_step``, so a plain thread pool scales across cores: each
environment advances one 30 Hz control tick (16 physics steps) per call. Environments differ
only in their ``MjData``: which objects are on the mat, where, and what each robot is told to do.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from kaiku.sim.ik import ARM_HOME
from kaiku.sim.scene import Layout
from kaiku.sim.twin import SUBSTEPS, Twin, build_twin

HOME_ARM = np.r_[ARM_HOME["left"], ARM_HOME["right"]]
HOME_HAND = np.zeros(14)


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class BatchSim:
    def __init__(self, twin: Twin | None = None, n_envs: int = 1, threads: int | None = None):
        self.twin = twin or build_twin()
        self.model = self.twin.model
        self.n = int(n_envs)
        self.data = [mujoco.MjData(self.model) for _ in range(self.n)]
        self.layouts: list[Layout | None] = [None] * self.n
        self.active = np.zeros((self.n, self.twin.n_objects), bool)   # object is part of the episode
        self.running = np.ones(self.n, bool)                          # env is stepped
        self.ticks = np.zeros(self.n, int)
        threads = threads or min(self.n, max(1, (os.cpu_count() or 2) - 2))
        self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="kaiku-sim")
        self._chunks = [list(c) for c in np.array_split(np.arange(self.n), min(threads, self.n)) if len(c)]
        tw = self.twin
        self._weight = -self.model.opt.gravity[2] * self.model.body_mass[tw.obj_body]
        self._obj_pos_idx = tw.obj_qadr[:, None] + np.arange(3)
        self._obj_quat_idx = tw.obj_qadr[:, None] + 3 + np.arange(4)
        self._obj_vel_idx = tw.obj_dadr[:, None] + np.arange(3)
        for i in range(self.n):
            self.reset(i, None)

    # ------------------------------------------------------------------ episodes
    def reset(self, i: int, layout: Layout | None, arm: np.ndarray | None = None,
              hand: np.ndarray | None = None) -> None:
        """Start an episode in env ``i``: robot at the home pose, ``layout`` objects on the mat,
        every other object parked weightless under the floor."""
        tw, d, sc = self.twin, self.data[i], self.twin.scene
        mujoco.mj_resetData(self.model, d)
        arm = HOME_ARM if arm is None else arm
        hand = HOME_HAND if hand is None else hand
        d.qpos[tw.arm_qadr], d.qpos[tw.hand_qadr] = arm, hand
        d.ctrl[tw.arm_act], d.ctrl[tw.hand_act] = arm, hand
        self.active[i] = False
        d.xfrc_applied[tw.obj_body, 2] = self._weight
        for p in (layout.placements if layout else ()):
            k = sc.by_name[p.name].index
            a = tw.obj_qadr[k]
            d.qpos[a:a + 3] = p.pos + [0.0, 0.0, 0.001]
            d.qpos[a + 3:a + 7] = _quat_yaw(p.yaw)
            d.xfrc_applied[tw.obj_body[k], 2] = 0.0
            self.active[i, k] = True
        self.layouts[i] = layout
        self.ticks[i] = 0
        mujoco.mj_forward(self.model, d)

    # ------------------------------------------------------------------ stepping
    def step(self, arm_cmd: np.ndarray, hand_cmd: np.ndarray) -> None:
        """One control tick for every running env. ``arm_cmd`` and ``hand_cmd`` are (N, 14)
        absolute joint targets in the challenge's order."""
        tw = self.twin
        for i in np.flatnonzero(self.running):
            d = self.data[i]
            d.ctrl[tw.arm_act] = arm_cmd[i]
            d.ctrl[tw.hand_act] = hand_cmd[i]
        list(self._pool.map(self._step_chunk, self._chunks))
        self.ticks[self.running] += 1

    def _step_chunk(self, idx: list[int]) -> None:
        for i in idx:
            if self.running[i]:
                mujoco.mj_step(self.model, self.data[i], SUBSTEPS)

    # ------------------------------------------------------------------ state
    def qpos(self) -> np.ndarray:
        return np.stack([d.qpos for d in self.data])

    def arm_q(self) -> np.ndarray:
        return np.stack([d.qpos[self.twin.arm_qadr] for d in self.data])

    def hand_q(self) -> np.ndarray:
        return np.stack([d.qpos[self.twin.hand_qadr] for d in self.data])

    def obj_pos(self) -> np.ndarray:
        """(N, 18, 3) object centres."""
        return np.stack([d.qpos[self._obj_pos_idx] for d in self.data])

    def obj_quat(self) -> np.ndarray:
        """(N, 18, 4) object orientations, wxyz."""
        return np.stack([d.qpos[self._obj_quat_idx] for d in self.data])

    def obj_speed(self) -> np.ndarray:
        """(N, 18) speed of each object's centre, m/s."""
        return np.stack([np.linalg.norm(d.qvel[self._obj_vel_idx], axis=-1) for d in self.data])

    def site_pos(self, i: int, name: str) -> np.ndarray:
        return self.data[i].site_xpos[self.twin.site[name]]

    # ------------------------------------------------------------------ scoring
    def outcomes(self, i: int) -> dict[str, str]:
        """The challenge's verdict for each active object of env ``i`` as things stand now."""
        sc, d, lay = self.twin.scene, self.data[i], self.layouts[i]
        out = {}
        if lay is None:
            return out
        for p in lay.placements:
            ob = sc.by_name[p.name]
            pos = d.qpos[self._obj_pos_idx[ob.index]]
            speed = float(np.linalg.norm(d.qvel[self._obj_vel_idx[ob.index]]))
            out[p.name] = sc.classify(pos, speed, ob.color, lay.mapping)
        return out

    def score(self, i: int) -> int:
        return self.twin.scene.score(self.outcomes(i).values())

    def close(self) -> None:
        self._pool.shutdown(wait=False)
