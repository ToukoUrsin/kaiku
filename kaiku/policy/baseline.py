"""Offline MimicGen-style replay for the sorting task.

This is deliberately a small re-implementation of the method in MimicGen: a source
episode is split into object-centred subtasks, each hand pose is transformed from the
source object frame to a new object's frame, and the result is executed through the
same IK, safety filter and simulator as a live echo.  DexMimicGen's coordination
subtasks do not occur here: the two hands never hold one part together.

The module has no dependency on the recorder.  ``load_source`` reads one recorder row,
``segments`` finds reach/grasp and carry/drop pieces, and ``generate`` executes one
target layout.  ``generate_dataset`` is the command line entry point used for the
study.  Runs and logs belong under ``/bulk2/runs/robotteleop/baseline``; only summary
JSON is checked into the repository.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kaiku.sim.scene import Layout, Placement, SHAPE_SYMMETRY, Scene

if TYPE_CHECKING:  # importing Session at runtime loads MuJoCo.
    from kaiku.app.session import Session
    from kaiku.data.recorder import Recorder


FPS = 30
LIFTED = 0.012
DROP_Z = 0.018


def wrap(a: float | np.ndarray, period: float = 2 * np.pi):
    """Wrap angles to ``[-period/2, period/2)``."""
    return (np.asarray(a) + period / 2) % period - period / 2


def symmetry_delta(shape: str, source_yaw: float, target_yaw: float) -> float:
    """Shortest valid yaw change, modulo the shape's rotational symmetry."""
    if SHAPE_SYMMETRY[shape] == 0:
        return float(wrap(target_yaw - source_yaw))
    period = 2 * np.pi / SHAPE_SYMMETRY[shape]
    return float(wrap(target_yaw - source_yaw, period))


def transform_xy(points: np.ndarray, source_xy: np.ndarray, target_xy: np.ndarray,
                 delta_yaw: float) -> np.ndarray:
    """Move points from one planar object frame to another."""
    points = np.asarray(points, float)
    c, s = np.cos(delta_yaw), np.sin(delta_yaw)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(target_xy, float) + (points - np.asarray(source_xy, float)) @ R.T


@dataclass(frozen=True)
class Segment:
    """One object-centric piece of a demonstration."""

    part: int
    kind: str                         # reach_grasp or carry_drop
    start: int
    end: int                          # exclusive
    source_xy: tuple[float, float]
    source_yaw: float
    target_frame: str                 # object or bin


@dataclass
class SourceDemo:
    root: Path
    row: dict
    state: np.ndarray                 # (T, 28)
    action: np.ndarray                # (T, 28)
    parts: np.ndarray                 # (T, n, 7)
    span: tuple[int, int]


@dataclass
class Attempt:
    source_episode: int
    target_index: int
    distance: float
    same_shape: bool
    clean: bool
    clipped: int
    seconds: float
    ticks: int
    error: str | None = None


@dataclass
class GenerationSummary:
    attempts: list[Attempt]

    @property
    def kept(self) -> int:
        return sum(a.clean for a in self.attempts)

    @property
    def yield_rate(self) -> float:
        return self.kept / len(self.attempts) if self.attempts else 0.0

    def by_bucket(self) -> dict[str, dict[str, float | int]]:
        """Yield split into distance bands and same/other shape."""
        out: dict[str, dict[str, float | int]] = {}
        for name, pred in (
            ("near", lambda a: a.distance < 0.05),
            ("mid", lambda a: 0.05 <= a.distance < 0.10),
            ("far", lambda a: a.distance >= 0.10),
        ):
            for shape, sp in (("same_shape", True), ("other_shape", False)):
                xs = [a for a in self.attempts if pred(a) and a.same_shape == sp]
                if xs:
                    out[f"{name}_{shape}"] = {"attempts": len(xs), "clean": sum(a.clean for a in xs),
                                                "yield": sum(a.clean for a in xs) / len(xs)}
        return out


def load_source(root: Path | str, row: dict) -> SourceDemo:
    """Load one human recorder row and its arrays."""
    root = Path(root)
    d = np.load(root / row["file"])
    k = int(row["row"])
    span = tuple(int(v) for v in d["span"][k])
    return SourceDemo(root, row, d["state"][k], d["action"][k], d["parts"][k], span)


def _in_bin(scene: Scene, pos: np.ndarray, name: str) -> np.ndarray:
    lo, hi = scene.bin_inner(name)
    return np.all((pos >= lo) & (pos <= hi), axis=-1)


def segments(source: SourceDemo, scene: Scene | None = None) -> list[Segment]:
    """Split a source episode into reach/grasp and carry/drop pieces.

    The source recorder stores the measured object pose.  Lift is the first frame above
    the mat; drop is the first frame resting in the mapped bin.  A short lead-in before
    lift is retained for the reach and the final frames are retained for the drop.
    """
    scene = scene or Scene()
    n, T = source.parts.shape[1], source.parts.shape[0]
    z = source.parts[..., 2]
    active = np.array([p["name"] for p in source.row["layout"]])
    mapping = source.row.get("mapping", {"red": "left", "yellow": "center", "blue": "right"})
    result: list[Segment] = []
    previous = int(source.span[0])
    for j in range(n):
        lift = np.flatnonzero(z[:, j] > scene.rest_z + LIFTED)
        if not len(lift):
            continue
        a = max(previous, int(lift[0]) - 45)
        color = scene.by_name[str(active[j])].color
        bname = mapping[color]
        p = source.parts[:, j, :3]
        inb = _in_bin(scene, p, bname) & (z[:, j] <= scene.rest_z + DROP_Z)
        after = np.flatnonzero(inb & (np.arange(T) >= lift[0]))
        drop = int(after[0]) if len(after) else min(T, int(lift[0]) + 90)
        drop = max(drop + 1, int(lift[0]) + 2)
        item = source.row["layout"][j]
        xy = tuple(float(v) for v in item["pos"][:2])
        yaw = float(item["yaw"])
        result.append(Segment(j, "reach_grasp", a, min(drop, T), xy, yaw, "object"))
        result.append(Segment(j, "carry_drop", int(lift[0]), min(drop, T), xy, yaw, "bin"))
        previous = min(drop, T)
    return result


def target_layout(source: SourceDemo, scene: Scene, rng: np.random.Generator,
                  *, same_shape: bool | None = None, spread: float = 0.08) -> Layout:
    """Draw a table with the source colours, optionally preserving its shapes."""
    src = source.row["layout"]
    base = scene.sample_layout(rng, len(src), mapping=source.row.get("mapping"))
    by_color: dict[str, list[dict]] = {}
    for p in src:
        by_color.setdefault(scene.by_name[p["name"]].color, []).append(p)
    seen: dict[str, int] = {}
    base_by_color: dict[str, list[Placement]] = {}
    for p in base.placements:
        base_by_color.setdefault(scene.by_name[p.name].color, []).append(p)
    out: list[Placement] = []
    used_names: set[str] = set()
    for ref in src:
        color = scene.by_name[ref["name"]].color
        i = seen.get(color, 0)
        seen[color] = i + 1
        p = base_by_color[color][min(i, len(base_by_color[color]) - 1)]
        shape = scene.by_name[ref["name"]].shape if same_shape is True else scene.by_name[p.name].shape
        if same_shape is False and shape == scene.by_name[ref["name"]].shape:
            other = next(s for s in scene.objects if s.color == color and s.shape != shape)
            name = other.name
        else:
            name = next(s.name for s in scene.objects if s.color == color and s.shape == shape
                        and s.name not in used_names)
        # ``sample_layout`` already enforces spacing; only add the requested spread to
        # make the bucket a measured property rather than a hidden tuning parameter.
        pos = p.pos.copy()
        if spread:
            pos[:2] += rng.normal(0.0, spread / 2, 2)
        out.append(Placement(name, pos, float(p.yaw)))
        used_names.add(name)
    return Layout(out, dict(source.row.get("mapping", {})))


def _pose_from_action(session: "Session", arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, R, _ = session.kin.fk(np.asarray(arm, float).reshape(-1, 2, 7))
    return p + np.einsum("nsij,sj->nsi", R, session.off), R


def transformed_targets(source: SourceDemo, target: Layout, session: "Session",
                        scene: Scene | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return transformed pinch, palm rotation and fingers for one target table."""
    scene = scene or session.scene
    T = source.action.shape[0]
    source_p, source_R = _pose_from_action(session, source.action[:, :14])
    pinch = source_p.copy()
    rot = source_R.copy()
    fingers = source.action[:, 14:].astype(float).copy()
    target_by_color: dict[str, list[Placement]] = {}
    for p in target.placements:
        target_by_color.setdefault(scene.by_name[p.name].color, []).append(p)
    for seg in segments(source, scene):
        src = source.row["layout"][seg.part]
        color = scene.by_name[src["name"]].color
        # Every colour can occur twice.  Segments refer to the source ordering, so
        # preserve the same occurrence when selecting the target anchor.
        occurrence = sum(scene.by_name[q["name"]].color == color for q in source.row["layout"][:seg.part])
        dst = target_by_color[color][occurrence]
        if seg.target_frame == "object":
            delta = symmetry_delta(scene.by_name[src["name"]].shape, float(src["yaw"]), float(dst.yaw))
            for t in range(seg.start, seg.end):
                pinch[t, :, :2] = transform_xy(pinch[t, :, :2], np.asarray(src["pos"][:2]), dst.pos[:2], delta)
                c, s = np.cos(delta), np.sin(delta)
                Q = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                rot[t] = Q @ rot[t]
        else:
            # Bin frames have the same fixed pose in every table.  Blend the hand pose
            # over a few frames at the stitch, as MimicGen does, then play the source
            # carry/drop motion open loop in the bin frame.
            blend = 6
            a = max(0, seg.start - blend)
            for t in range(a, seg.start):
                u = (t - a + 1) / max(seg.start - a, 1)
                pinch[t] = (1 - u) * pinch[t] + u * source_p[t]
                for hand in range(2):
                    M = (1 - u) * rot[t, hand] + u * source_R[t, hand]
                    U, _, Vt = np.linalg.svd(M)
                    rot[t, hand] = U @ Vt
            for t in range(seg.start, seg.end):
                pinch[t] = source_p[t]
                rot[t] = source_R[t]
    return pinch, rot, fingers


def generate(source: SourceDemo, targets: list[Layout], *, seed: int = 0,
             out_root: Path | str | None = None, method: str = "mimicgen") -> tuple[GenerationSummary, list[dict]]:
    """Execute transformed segments on target layouts and optionally record clean runs.

    ``method="copy"`` is the joint-target control: it skips all object-frame transforms.
    """
    scene = Scene()
    attempts: list[Attempt] = []
    rows: list[dict] = []
    for ti, layout in enumerate(targets):
        t0 = time.perf_counter()
        from kaiku.app.session import Session
        session = Session(1, seed=seed + ti, threads=1, spread=0.0, vary_shape=False)
        session.new_episode(layout, [])
        if method == "copy":
            pinch, rot = _pose_from_action(session, source.action[:, :14])
            fingers = source.action[:, 14:].astype(float).copy()
        elif method == "mimicgen":
            pinch, rot, fingers = transformed_targets(source, layout, session, scene)
        else:
            raise ValueError(f"unknown baseline method: {method}")
        rec = None
        if out_root:
            from kaiku.data.recorder import Recorder
            rec = Recorder(session, out_root, operator=method)
        # Source span is the recorded working interval; preserving it is the offline
        # analogue of MimicGen's open-loop controller and keeps lead-in/out identical.
        for t in range(source.action.shape[0]):
            obs = session.observe()
            if source.span[0] <= t < source.span[1]:
                session.drive(pinch[t:t + 1], rot[t:t + 1], fingers[t:t + 1])
            else:
                home = session.home_command()
                session.drive(home.pinch[None], home.rot[None], home.hand[None])
            if rec is not None:
                rec.add(obs, *session.last_action)
        session.finish_lead()
        for _ in range(session.timeout_ticks + session.settle_ticks):
            if session.settled:
                break
            session.tick(session.home_command())
        res = session.result()
        clipped = int(res.clipped[0])
        clean = bool(res.success[0] and clipped == 0 and not res.aborted[0])
        src_xy = np.array([p["pos"][:2] for p in source.row["layout"]])
        dst_xy = np.array([p.pos[:2] for p in layout.placements])
        distance = float(np.mean(np.linalg.norm(dst_xy - src_xy[:len(dst_xy)], axis=1)))
        same = all(scene.by_name[a["name"]].shape == scene.by_name[b.name].shape
                   for a, b in zip(source.row["layout"], layout.placements))
        attempts.append(Attempt(int(source.row.get("episode", -1)), ti, distance, same, clean, clipped,
                                time.perf_counter() - t0, int(res.ticks)))
        if rec is not None:
            rec.end(res, source=method, source_label=method)
        session.close()
    return GenerationSummary(attempts), rows


def _rows(root: Path) -> list[dict]:
    return [json.loads(x) for x in (root / "episodes.jsonl").read_text().splitlines() if x.strip()]


def generate_dataset(source_root: Path | str, out_root: Path | str, *, scenes: int = 32,
                     tables: int = 47, seed: int = 0, same_shape: bool | None = None,
                     method: str = "mimicgen") -> GenerationSummary:
    """Generate ``scenes * tables`` attempts from disjoint human lead episodes."""
    source_root, out_root = Path(source_root), Path(out_root)
    scene = Scene()
    rows = [r for r in _rows(source_root) if r.get("source") == "human" and (source_root / r["file"]).exists()]
    # One row per operator episode; sort by episode to make disjoint sets reproducible.
    chosen = []
    seen = set()
    for r in rows:
        if r["episode"] not in seen:
            chosen.append(r)
            seen.add(r["episode"])
        if len(chosen) >= scenes:
            break
    rng = np.random.default_rng(seed)
    all_attempts: list[Attempt] = []
    for i, row in enumerate(chosen):
        src = load_source(source_root, row)
        targets = [target_layout(src, scene, rng, same_shape=same_shape) for _ in range(tables)]
        summary, _ = generate(src, targets, seed=seed + i * 1000, out_root=out_root, method=method)
        all_attempts.extend(summary.attempts)
    summary = GenerationSummary(all_attempts)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"{method}_attempts.jsonl").write_text("".join(
        json.dumps(a.__dict__, sort_keys=True) + "\n" for a in all_attempts))
    (out_root / f"{method}_summary.json").write_text(json.dumps({
        "method": "mimicgen_reimplementation" if method == "mimicgen" else method,
        "scenes": len(chosen), "tables": tables,
        "attempts": len(all_attempts), "clean": summary.kept, "yield": summary.yield_rate,
        "seconds_per_demo": float(np.mean([a.seconds for a in all_attempts])) if all_attempts else 0.0,
        "offline": True, "buckets": summary.by_bucket(),
    }, indent=2) + "\n")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--scenes", type=int, default=32)
    ap.add_argument("--tables", type=int, default=47)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same-shape", action="store_true")
    ap.add_argument("--method", choices=("mimicgen", "copy"), default="mimicgen")
    a = ap.parse_args(argv)
    s = generate_dataset(a.source, a.out, scenes=a.scenes, tables=a.tables, seed=a.seed,
                         same_shape=True if a.same_shape else None, method=a.method)
    print(json.dumps({"attempts": len(s.attempts), "clean": s.kept, "yield": s.yield_rate,
                      "buckets": s.by_bucket()}, indent=2))


if __name__ == "__main__":
    main()
