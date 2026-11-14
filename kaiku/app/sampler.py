"""Scenes drawn in processes of their own.

Finding a scene every arm can reach takes many tries, all of them Python. In a thread next to
the control loop that work holds the interpreter lock and the robots fall to a third of real
time; in another process it costs them nothing.
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor

import numpy as np

_W: dict = {}


def _start() -> None:
    from kaiku.sim.ik import BatchArmIK
    from kaiku.sim.reach import ReachMap
    from kaiku.sim.twin import build_twin
    twin = build_twin()
    _W.update(scene=twin.scene, reach=ReachMap(twin.scene, BatchArmIK(), twin=twin))


def _count(parts, rng) -> int:
    return int(rng.choice([3, 4, 5], p=[0.35, 0.40, 0.25])) if parts == "auto" else int(parts)


def _scene(seed: int, parts, followers: int, spread: float, vary_shape: bool):
    from kaiku.app.session import draw_scene
    rng = np.random.default_rng(seed)
    return draw_scene(_W["scene"], _W["reach"], rng, followers, _count(parts, rng), spread, vary_shape)


def _free(seed: int, parts, n: int):
    from kaiku.app.session import draw_lead
    rng = np.random.default_rng(seed)
    return [draw_lead(_W["scene"], _W["reach"], rng, _count(parts, rng)) for _ in range(n)]


class Sampler:
    def __init__(self, n: int, seed: int = 0, spread: float = 0.08, vary_shape: bool = True, workers: int = 2):
        self.n, self.spread, self.vary_shape = n, spread, vary_shape
        self.rng = np.random.default_rng(seed)
        self.pool = ProcessPoolExecutor(workers, mp.get_context("spawn"), initializer=_start)

    def _seed(self) -> int:
        return int(self.rng.integers(2 ** 31))

    def scene(self, parts="auto") -> Future:
        """The operator's next scene and its variations: (lead, followers)."""
        return self.pool.submit(_scene, self._seed(), parts, self.n - 1, self.spread, self.vary_shape)

    def free(self, parts="auto") -> Future:
        """A scene of its own for every robot."""
        return self.pool.submit(_free, self._seed(), parts, self.n)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
