"""Recorded demonstrations -> training pairs for the policy.

Reads a recorder folder (``kaiku.data.recorder``): for every demonstration, every tick becomes
one pair of what the policy would have seen and the next ``HORIZON`` commands that were sent.
The plan the policy is given (which part next, with which arm) is read off the demonstration
itself: the order in which the parts left the mat, and the hand that was at each one.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kaiku.fanout.echo import SHAPE_CODE
from kaiku.policy.features import HORIZON, LIFTED, MAX_PARTS, Featurizer, quat_yaw
from kaiku.sim.grasp import grasp_offset
from kaiku.sim.ik import BatchArmIK
from kaiku.sim.scene import BINS, Scene


@dataclass
class Pairs:
    x: np.ndarray           # (M, dim) what the policy saw
    y: np.ndarray           # (M, HORIZON, ACT_DIM) what was commanded next
    episode: np.ndarray     # (M,) the operator's episode each pair comes from
    human: np.ndarray       # (M,) bool: the operator's own robot (else an echo)
    demos: int = 0
    human_demos: int = 0

    def pick(self, episodes, echoes: bool) -> "Pairs":
        """The pairs of the given operator episodes: the operator's own demonstrations, and with
        ``echoes`` the echoes of them too."""
        keep = np.isin(self.episode, np.asarray(list(episodes)))
        if not echoes:
            keep &= self.human
        return Pairs(self.x[keep], self.y[keep], self.episode[keep], self.human[keep])

    def sample(self, most: int, seed: int = 0) -> "Pairs":
        """At most ``most`` pairs, drawn evenly: what fits beside the network on one GPU."""
        if len(self.x) <= most:
            return self
        keep = np.sort(np.random.default_rng(seed).choice(len(self.x), most, replace=False))
        return Pairs(self.x[keep], self.y[keep], self.episode[keep], self.human[keep], self.demos, self.human_demos)


def rows_of(root: Path | str) -> list[dict]:
    side = Path(root) / "episodes.jsonl"
    return [json.loads(line) for line in side.read_text().splitlines() if line.strip()]


class Loader:
    def __init__(self, scene: Scene | None = None, kin: BatchArmIK | None = None):
        self.scene = scene or Scene()
        self.kin = kin or BatchArmIK()
        self.feat = Featurizer(self.scene)
        self.off = np.stack([grasp_offset("left"), grasp_offset("right")])

    def pinch(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(M, 14) arm joints -> pinch points (M, 2, 3) and palm rotations (M, 2, 3, 3)."""
        out_p, out_R = [], []
        for a in range(0, len(arm_q), 65536):
            p, R, _ = self.kin.fk(arm_q[a:a + 65536].reshape(-1, 2, 7).astype(float))
            out_p.append(p + np.einsum("nsij,sj->nsi", R, self.off))
            out_R.append(R)
        return np.concatenate(out_p), np.concatenate(out_R)

    def demo(self, state: np.ndarray, action: np.ndarray, parts: np.ndarray, layout: list[dict],
             mapping: dict[str, str], span=None, kinematics=None) -> tuple[np.ndarray, np.ndarray] | None:
        """One demonstration (T ticks) -> (x (T', dim), y (T', HORIZON, ACT_DIM))."""
        T, n = parts.shape[:2]
        if kinematics is None:
            pinch, rot = self.pinch(state[:, :14])
            cmd_pinch, cmd_rot = self.pinch(action[:, :14])
        else:
            pinch, rot, cmd_pinch, cmd_rot = kinematics
        pos, yaw = parts[..., :3].astype(float), quat_yaw(parts[..., 3:].astype(float))
        # The plan: parts in the order they left the mat, each with the hand that was at it.
        up = pos[:, :, 2] > self.scene.rest_z + LIFTED
        if not up.any(axis=0).all():
            return None
        lift = up.argmax(axis=0)
        order = np.argsort(lift, kind="stable")
        side = np.array([np.argmin(np.linalg.norm(pinch[lift[j]] - pos[lift[j], j], axis=-1)) for j in order])
        shape = np.array([SHAPE_CODE[layout[j]["shape"]] for j in order])
        bins = np.array([BINS.index(mapping[layout[j]["color"]]) for j in order])
        P = MAX_PARTS
        pad = lambda a, fill=0: np.concatenate([a, np.full((P - n, *a.shape[1:]), fill, a.dtype)])  # noqa: E731
        tile = lambda a: np.broadcast_to(pad(a), (T, P))                                           # noqa: E731
        ppos = np.concatenate([pos[:, order], np.zeros((T, P - n, 3))], axis=1)
        pyaw = np.concatenate([yaw[:, order], np.zeros((T, P - n))], axis=1)
        present = np.broadcast_to(np.arange(P) < n, (T, P))
        before = np.maximum(np.arange(T) - 1, 0)                     # the tick before each tick
        x = self.feat.observe(pinch, rot, state[:, 14:], ppos, pyaw, tile(shape), tile(bins), tile(side), present,
                              cmd_pinch[before], action[before, 14:])
        a, b = (0, T) if span is None else (int(span[0]), int(span[1]))
        idx = np.minimum(np.arange(a, b)[:, None] + np.arange(HORIZON), T - 1)                     # (T', H)
        y = self.feat.command(cmd_pinch[idx], cmd_rot[idx], action[idx, 14:], cmd_pinch[before[a:b], None])
        return x[a:b], y.astype(np.float32)

    def episode(self, root: Path, name: str, group: list[dict], echoes: bool = True, stride: int = 1) -> list[tuple]:
        """One operator episode (a file): its demonstrations as (x, y, the operator's own?)."""
        d = np.load(root / name)
        # Every row in an episode shares its tick count.  FK used to run once per row (up to
        # 49 times for the operator plus echoes); batching the file makes the latest-16 remote
        # load substantially cheaper while producing the exact same arrays.
        rows = [r for r in group if echoes or r["source"] == "human"]
        if rows:
            idx = [int(r["row"]) for r in rows]
            state = d["state"][idx, :, :14].reshape(-1, 14)
            action = d["action"][idx, :, :14].reshape(-1, 14)
            T = d["state"].shape[1]
            pinch, rot = self.pinch(state)
            cmd_pinch, cmd_rot = self.pinch(action)
            batched = (pinch.reshape(len(rows), T, 2, 3), rot.reshape(len(rows), T, 2, 3, 3),
                       cmd_pinch.reshape(len(rows), T, 2, 3), cmd_rot.reshape(len(rows), T, 2, 3, 3))
        else:
            batched = None
        out = []
        for i, r in enumerate(rows):
            k = r["row"]
            kin = tuple(a[i] for a in batched) if batched is not None else None
            got = self.demo(d["state"][k], d["action"][k], d["parts"][k], r["layout"], r["mapping"],
                            d["span"][k], kin)
            if got is not None:
                out.append((got[0][::stride], got[1][::stride], r["source"] == "human"))
        return out

    def load(self, root: Path | str, max_episodes: int | None = None, stride: int = 1, log=None,
             echoes: bool = True, progress=None, workers: int | None = None, skip: int = 0,
             last: int | None = None, context: str = "spawn", orphans: bool = False) -> Pairs:
        """Every demonstration in a recorder folder (of its first ``max_episodes`` operator
        episodes that the operator's own robot completed). Without ``echoes``, the operator's
        own demonstrations only; ``skip`` leaves out that many of the earliest episodes, and
        ``last`` keeps only that many of the latest.
        ``progress(done, total)`` is called after every episode.
        ``workers`` processes share the episodes (default: one per core, up to 16)."""
        root = Path(root)
        by_file: dict[str, list[dict]] = {}
        for r in rows_of(root):
            by_file.setdefault(r["file"], []).append(r)
        # The operator's own run failed: nothing to compare against. A file still being written: next time.
        # ``orphans`` takes files with no operator's row too: the offline baseline stores what a
        # generator kept in files of their own, beside the demonstrations they were made from.
        names = [n for n in sorted(by_file)
                 if ((orphans and echoes) or any(r["source"] == "human" for r in by_file[n])) and (root / n).exists()]
        names = names[skip:][:max_episodes]
        if last:
            names = names[-last:]
        if not names:
            raise ValueError(f"no finished demonstrations in {root}")
        jobs = [(root, n, by_file[n], echoes, stride) for n in names]
        workers = min(workers or min(os.cpu_count() or 1, 16), len(jobs))
        xs, ys, eps, hum = [], [], [], []
        demos = humans = 0
        # ``spawn`` is the portable and conservative default.  A Linux Modal worker may opt into
        # ``fork``: it avoids importing MuJoCo and constructing the kinematics model 16 times
        # before the episode workers can start, while keeping the live app and local CLI unchanged.
        pool = ProcessPoolExecutor(workers, mp.get_context(context)) if workers > 1 and echoes else None
        try:
            results = pool.map(_episode, jobs) if pool else (self.episode(*j) for j in jobs)
            for used, (name, got) in enumerate(zip(names, results)):
                for x, y, human in got:
                    xs.append(x)
                    ys.append(y)
                    eps.append(np.full(len(x), used))
                    hum.append(np.full(len(x), human))
                    demos += 1
                    humans += human
                if log:
                    log(f"  {name}: {len(got)} demonstrations")
                if progress:
                    progress(used + 1, len(names))
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)
        if not xs:
            raise ValueError(f"no finished demonstrations in {root}")
        return Pairs(np.concatenate(xs), np.concatenate(ys), np.concatenate(eps), np.concatenate(hum), demos, int(humans))


_LOADER: Loader | None = None


def _episode(job: tuple) -> list[tuple]:
    """``Loader.episode`` in a worker process, which keeps one loader for all its episodes."""
    global _LOADER
    if _LOADER is None:
        _LOADER = Loader()
    return _LOADER.episode(*job)
