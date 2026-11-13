"""Geometry, layouts and the scoring rule of the Aaltonomous counter-top sorting task.

Everything is derived from the challenge's own ``scene.yaml`` (copied unchanged into
``assets/aaltonomous``), so a demonstration that scores here scores under the organisers'
rule. All positions in this module are in the pelvis frame (x forward, y left, z up, metres),
which is also the simulator's world frame.

The printed mat has its own frame (origin at the mat corner on the robot's right, nearest
edge; x along that edge to the robot's left, y away from the robot). Layout cards use it,
so :func:`Scene.mat_to_pelvis` is kept for reading them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

ASSETS = Path(__file__).resolve().parent / "assets" / "aaltonomous"
COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
BINS = ("left", "center", "right")
FIXED_MAPPING = {"red": "left", "yellow": "center", "blue": "right"}
OUTCOMES = ("correct", "wrong", "counter", "off")
# Rotational symmetry of each shape about its vertical axis (cylinder: continuous).
SHAPE_SYMMETRY = {"cube": 4, "cylinder": 0, "triangle": 3}


@dataclass(frozen=True)
class ObjectSpec:
    """One of the 18 printed parts; ``index`` is its row in every per-object array."""

    index: int
    name: str
    color: str
    shape: str


@dataclass
class Placement:
    """Where one object starts in an episode."""

    name: str
    pos: np.ndarray  # (3,) centre, pelvis frame
    yaw: float       # about pelvis z


@dataclass
class Layout:
    """An episode's initial scene: the active objects and where each colour belongs."""

    placements: list[Placement]
    mapping: dict[str, str] = field(default_factory=lambda: dict(FIXED_MAPPING))

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.placements]


class Scene:
    def __init__(self, path: Path | str | None = None):
        self.cfg = yaml.safe_load(Path(path or ASSETS / "scene.yaml").read_text())
        c, b, o = self.cfg["counter"], self.cfg["bins"], self.cfg["objects"]
        self.origin = np.asarray(c["mat_origin_in_pelvis"], float)
        self.counter_z = float(self.origin[2])
        self.mat_size = np.asarray(c["mat_size"], float)          # (along robot y, along robot x)
        self.counter_size = np.asarray(c["counter_size"], float)
        self.size = float(o["size"])
        self.height = self.size * float(o.get("height_ratio", 1.0))
        self.bin_outer = np.asarray(b["outer_size"], float)
        self.bin_wall = float(b["wall_thickness"])
        self.bin_floor = float(b["floor_thickness"])
        self.bin_height = float(b["wall_height"])
        self.colors = {k: np.asarray(v, float) / 255.0 for k, v in o["colors"].items()}
        self.bin_rgb = np.asarray(b["rgb"], float) / 255.0
        self.objects = [
            ObjectSpec(i, f"obj_{c_}_{s}_{k}", c_, s)
            for i, (s, c_, k) in enumerate(
                (s, c_, k) for s in o["shapes"] for c_ in o["colors"] for k in range(int(o["copies_per_type"])))
        ]
        self.by_name = {ob.name: ob for ob in self.objects}

    # ------------------------------------------------------------------ frames
    def mat_to_pelvis(self, p) -> np.ndarray:
        """(..., 2|3) mat coordinates -> (..., 3) pelvis frame (mat x -> pelvis y, mat y -> pelvis x)."""
        p = np.asarray(p, float)
        z = p[..., 2] if p.shape[-1] == 3 else np.zeros(p.shape[:-1])
        return self.origin + np.stack([p[..., 1], p[..., 0], z], axis=-1)

    @cached_property
    def counter_box(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) xy of the physical counter top; it is centred on the mat."""
        centre = self.mat_to_pelvis(self.mat_size / 2)[:2]
        half = self.counter_size[::-1] / 2
        return centre - half, centre + half

    @cached_property
    def mat_box(self) -> tuple[np.ndarray, np.ndarray]:
        lo = self.mat_to_pelvis([0.0, 0.0])[:2]
        return lo, lo + self.mat_size[::-1]

    @cached_property
    def bin_centers(self) -> dict[str, np.ndarray]:
        """Bin centre on the counter surface, pelvis frame, keyed left / center / right."""
        return {k: self.mat_to_pelvis(v) for k, v in self.cfg["bins"]["centers"].items()}

    def bin_inner(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of a bin's inner volume: inside the walls, from its floor to the rim."""
        c = self.bin_centers[name]
        half = self.bin_outer[::-1] / 2 - self.bin_wall
        lo = np.r_[c[:2] - half, self.counter_z + self.bin_floor]
        hi = np.r_[c[:2] + half, self.counter_z + self.bin_height]
        return lo, hi

    @cached_property
    def rest_z(self) -> float:
        """Height of an object's centre when it stands on the mat."""
        return self.counter_z + self.height / 2

    # ------------------------------------------------------------------ layouts
    @cached_property
    def region(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) xy of the band in which object centres are placed."""
        L = self.cfg["layout"]
        a = self.mat_to_pelvis([L["region_x"][0], L["region_y"][0]])[:2]
        b = self.mat_to_pelvis([L["region_x"][1], L["region_y"][1]])[:2]
        return np.minimum(a, b), np.maximum(a, b)

    def sample_layout(self, rng: np.random.Generator, n: int | None = None,
                      mapping: dict[str, str] | None = None) -> Layout:
        """Random episode following the challenge's rule: 3-5 objects, every colour present,
        at most two per colour, centres in the band at least ``min_spacing`` apart, yaw within
        +-45 degrees, shape drawn per object."""
        L = self.cfg["layout"]
        n = int(rng.integers(L["n_cubes"][0], L["n_cubes"][1] + 1)) if n is None else int(n)
        per = {c: 1 for c in COLORS}
        for c in rng.permutation(COLORS)[: max(0, n - 3)]:
            per[c] += 1
        colors = [c for c in COLORS for _ in range(per[c])][:n]
        lo, hi = self.region
        yaw_lo, yaw_hi = np.radians(L["yaw_range_deg"])
        for _ in range(200):
            pts: list[np.ndarray] = []
            for _ in colors:
                for _ in range(100):
                    p = rng.uniform(lo, hi)
                    if all(np.linalg.norm(p - q) >= L["min_spacing"] for q in pts):
                        pts.append(p)
                        break
                else:
                    break
            if len(pts) == len(colors):
                break
        else:
            raise RuntimeError("could not place the objects with the required spacing")
        used: set[str] = set()
        placements = []
        for color, p in zip(rng.permutation(colors), pts):
            shape = SHAPES[int(rng.integers(len(SHAPES)))]
            ob = next((o for o in self.objects if o.color == color and o.name not in used
                       and o.shape == shape), None) or next(
                o for o in self.objects if o.color == color and o.name not in used)
            used.add(ob.name)
            placements.append(Placement(ob.name, np.r_[p, self.rest_z], float(rng.uniform(yaw_lo, yaw_hi))))
        return Layout(placements, dict(mapping or FIXED_MAPPING))

    def park_pos(self, index: int) -> np.ndarray:
        """Where an object that is not part of the episode waits: under the floor slab, where
        nothing can touch it (the simulator also cancels its weight)."""
        row, col = divmod(index, 6)
        return np.array([-0.30 + 0.10 * col, -0.10 + 0.10 * row, -1.20])

    # ------------------------------------------------------------------ scoring
    def on_counter(self, pos) -> bool:
        lo, hi = self.counter_box
        drop = float(self.cfg["success"]["off_counter_drop"])
        return bool(np.all(pos[:2] >= lo) and np.all(pos[:2] <= hi) and pos[2] > self.counter_z - drop)

    def bin_of(self, pos) -> str | None:
        """Name of the bin whose inner volume contains the point, else None."""
        for name in self.bin_centers:
            lo, hi = self.bin_inner(name)
            if np.all(pos >= lo) and np.all(pos <= hi):
                return name
        return None

    def classify(self, pos, speed: float, color: str, mapping: dict[str, str]) -> str:
        """The challenge's rule for one object at episode end: ``correct`` / ``wrong`` if its
        centre rests inside a bin's inner volume, ``off`` if it left the counter, else ``counter``."""
        pos = np.asarray(pos, float)
        if not self.on_counter(pos):
            return "off"
        b = self.bin_of(pos)
        if b is None or speed > float(self.cfg["success"]["at_rest_speed"]):
            return "counter"
        return "correct" if mapping.get(color) == b else "wrong"

    def score(self, outcomes, task: str = "sort_fixed") -> int:
        table = self.cfg["scoring"][task]
        return int(sum(table[o] for o in outcomes))
