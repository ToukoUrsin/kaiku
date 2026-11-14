"""Demonstrations on disk, as the session produces them.

Every tick the recorder keeps, for each robot, what the challenge's dataset keeps: the measured
joints (``observation.state``, 28) and the command that reached the motors after the safety
filter (``action``, 28), plus where the parts were. When an episode ends, each robot that put
every part in its bin without one clipped command becomes a demonstration. The rest are counted
and dropped.

On disk, under ``root``:

    episodes.jsonl      one line per demonstration: who made it, the scene, where its frames are
    ep000012.npz        the frames of every demonstration from lead episode 12
                        state (K, T, 28), action (K, T, 28), parts (K, T, n, 7: xyz + wxyz),
                        span (K, 2): the ticks in which that robot was actually working

``kaiku.data.export`` turns a folder like this into the challenge's LeRobot layout.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from kaiku.sim.batch import HOME_ARM

FPS = 30
LEAD_IN = 6        # ticks kept before a robot first moves
LEAD_OUT = 20      # ticks kept after its last part came to rest


class Recorder:
    def __init__(self, session, root: Path | str | None, operator: str = "autopilot"):
        self.s = session
        self.root = Path(root) if root else None
        self.operator = operator
        self.enabled = self.root is not None
        self.demos = 0             # demonstrations written
        self.human = 0             # of which the operator's own
        self.frames = 0
        self.dropped = 0
        self.episodes = 0
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
            side = self.root / "episodes.jsonl"
            if side.exists():      # resume: keep counting where the folder left off
                rows = [json.loads(line) for line in side.read_text().splitlines() if line.strip()]
                self.demos = len(rows)
                self.human = sum(r["source"] == "human" for r in rows)
                self.frames = sum(r["frames"] for r in rows)
                self.episodes = 1 + max((r["episode"] for r in rows), default=-1)
        self.begin()

    def begin(self) -> None:
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._parts: list[np.ndarray] = []
        self._lead_pinch: list[np.ndarray] = []
        self._lead_rot: list[np.ndarray] = []
        self._lead_hand: list[np.ndarray] = []
        self._hands: list[np.ndarray] = []

    def add(self, obs, arm: np.ndarray, hand: np.ndarray, command=None, hands=None) -> None:
        """One tick: the observation it started from and the command that was sent."""
        if not self.enabled:
            return
        rows = np.arange(self.s.n)[:, None]
        idx = self.s.obj_idx
        self._state.append(np.concatenate([obs.arm_q, obs.hand_q], axis=1).astype(np.float32))
        self._action.append(np.concatenate([arm, hand], axis=1).astype(np.float32))
        self._parts.append(np.concatenate([obs.obj_pos[rows, idx], obs.obj_quat[rows, idx]], axis=-1).astype(np.float32))
        if command is not None:
            self._lead_pinch.append(np.asarray(command.pinch, dtype=np.float32).copy())
            self._lead_rot.append(np.asarray(command.rot, dtype=np.float32).copy())
            self._lead_hand.append(np.asarray(command.hand, dtype=np.float32).copy())
        if hands is not None:
            points = np.asarray(hands, dtype=np.float32)
            if points.shape == (2, 21, 3):
                self._hands.append(points.copy())

    def end(self, result, source: str | None = None, source_label: str | None = None) -> dict:
        """Close the episode: write the robots that made a clean demonstration. Returns counts."""
        out = {"kept": 0, "human": 0, "dropped": 0}
        if not self.enabled or not self._state:
            self.begin()
            return out
        sc = self.s.scene
        state, action, parts = np.stack(self._state, 1), np.stack(self._action, 1), np.stack(self._parts, 1)
        lead_extra = {
            "lead_pinch": np.stack(self._lead_pinch).astype(np.float32),
            "lead_rot": np.stack(self._lead_rot).astype(np.float32),
            "lead_hand": np.stack(self._lead_hand).astype(np.float32),
        } if len(self._lead_pinch) == state.shape[1] else {}
        if lead_extra:
            lead_extra["lead_command"] = np.concatenate(
                [lead_extra["lead_pinch"].reshape(state.shape[1], -1),
                 lead_extra["lead_rot"].reshape(state.shape[1], -1), lead_extra["lead_hand"]], axis=1)
        hands_extra = {"hands": np.stack(self._hands).astype(np.float32)} if len(self._hands) == state.shape[1] else {}
        self.begin()
        good = result.success & (result.clipped == 0) & ~result.aborted
        keep = np.flatnonzero(good)
        out["dropped"] = int((~good).sum())
        self.dropped += out["dropped"]
        if not len(keep):
            return out
        # The ticks a robot was working: from its first move off the home pose to the last
        # change of its command.
        moving = np.abs(action[keep, :, :14] - HOME_ARM.astype(np.float32)).max(axis=-1) > 2e-3      # (K, T)
        changing = np.abs(np.diff(action[keep], axis=1)).max(axis=-1) > 1e-4                         # (K, T-1)
        T = state.shape[1]
        first = np.where(moving.any(1), moving.argmax(1), 0)
        last = np.where(changing.any(1), T - 1 - changing[:, ::-1].argmax(1), T - 1)
        span = np.stack([np.maximum(first - LEAD_IN, 0), np.minimum(last + LEAD_OUT, T)], axis=1)
        ep = self.episodes
        name = f"ep{ep:06d}.npz"
        extra = lead_extra | hands_extra
        np.savez_compressed(self.root / name, state=state[keep], action=action[keep], parts=parts[keep], span=span,
                            **extra)
        lead = result.layouts[0]
        with open(self.root / "episodes.jsonl", "a") as f:
            for row, i in enumerate(keep):
                lay = result.layouts[i]
                who = (source or self.operator) if i == 0 else "echo"
                label = ("human" if i == 0 else "echo") if source_label is None else source_label
                rec = {
                    "demo": self.demos, "file": name, "row": row, "episode": ep, "env": int(i),
                    "source": label, "operator": who,
                    "frames": int(span[row, 1] - span[row, 0]), "fps": FPS, "success": True, "clipped": 0,
                    "lag_s": 0.0 if i == 0 else round(float(result.lag[i - 1]), 2),
                    "task": "sort_fixed", "instruction": instruction(lay.mapping), "mapping": lay.mapping,
                    "layout": [{"name": p.name, "color": sc.by_name[p.name].color, "shape": sc.by_name[p.name].shape,
                                "pos": [round(float(v), 4) for v in p.pos], "yaw": round(float(p.yaw), 4)}
                               for p in lay.placements],
                    "lead_layout": None if i == 0 else [[round(float(v), 4) for v in (*p.pos[:2], p.yaw)]
                                                        for p in lead.placements],
                    "time": round(time.time(), 1),
                }
                f.write(json.dumps(rec) + "\n")
                self.demos += 1
                self.frames += rec["frames"]
                if label == "human":
                    self.human += 1
                    out["human"] = 1
        self.episodes += 1
        out["kept"] = len(keep)
        return out


def instruction(mapping: dict[str, str]) -> str:
    """The episode's task sentence, in the challenge's wording."""
    c = [k for k in ("red", "yellow", "blue") if k in mapping]
    return (f"Put the {c[0]} objects in the {mapping[c[0]]} bin, the {c[1]} objects in the {mapping[c[1]]} bin "
            f"and the {c[2]} objects in the {mapping[c[2]]} bin.")
