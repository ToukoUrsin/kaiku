"""Where each arm can pinch a part from above, and with its fingers pointing which way.

A top-down grasp is a place on the mat plus one angle (the finger direction). This module
tabulates, for both arms, how well the arm can hold the palm level at that place and angle,
both at grasp height and at the hover height above it. The table answers in microseconds
what would otherwise take a full inverse-kinematics solve, so the fan-out can pick, for every
follower robot and every part, the grasp among the equivalent ones that the arm reaches best.

The table is computed once from the robot model (about 20 s) and cached next to the assets.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from kaiku.sim.grasp import HAND_OPEN, NATURAL_DIR, TRIANGLE_PHI, grasp_offset, hand14
from kaiku.sim.ik import BatchArmIK
from kaiku.sim.scene import Scene

CACHE = Path(__file__).resolve().parent / "assets" / "reach.npz"
CUBE, CYLINDER, TRIANGLE = 0, 1, 2
TOL = 0.004            # m: a pose counts as reachable below this cost
HOVER = 0.085          # m above the grasp, as in the pick-and-place plan
# Finger directions tried on a cylinder (any direction pinches it), relative to the arm's natural one.
CYL_OFFSETS = np.radians([0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0, -60.0, 60.0])
CLEARANCE = 0.008      # m the open hand must keep from the parts it is not picking
WALL_CLEARANCE = 0.003 # m its knuckles must keep from the front wall of the bins


def part_radius(shape: np.ndarray, size: float) -> np.ndarray:
    """Radius of the circle around each part (shape code): what a neighbour takes up on the mat."""
    return np.where(shape == CUBE, size * np.sqrt(0.5), np.where(shape == TRIANGLE, 2.0 * size / 3.0, 0.5 * size))


def level_palm(direction: np.ndarray) -> np.ndarray:
    """Palm rotations for both hands (..., 2, 3, 3), fingers along ``direction`` (..., 2), palm down."""
    c, s = np.cos(direction), np.sin(direction)
    z = np.zeros_like(c)
    x = np.stack([c, s, z], axis=-1)
    up = np.broadcast_to([0.0, 0.0, 1.0], x.shape)
    y = np.where(np.arange(2)[:, None] == 0, up, -up)                     # left: +z is the back of the hand
    return np.stack([x, y, np.cross(x, y)], axis=-1)


class ReachMap:
    def __init__(self, scene: Scene | None = None, kin: BatchArmIK | None = None, margin: float = 0.03,
                 step_xy: float = 0.0125, step_dir: float = np.radians(7.5), cache: bool = True, twin=None):
        self.scene = scene or Scene()
        self.kin = kin or BatchArmIK()
        lo, hi = self.scene.region
        self.x = np.arange(lo[0] - margin, hi[0] + margin + 1e-9, step_xy)
        self.y = np.arange(lo[1] - margin, hi[1] + margin + 1e-9, step_xy)
        self.d = np.arange(-np.radians(105.0), np.radians(105.0) + 1e-9, step_dir)
        self.natural = np.array([NATURAL_DIR["left"], NATURAL_DIR["right"]])
        key = hashlib.sha1(b"".join(np.ascontiguousarray(a, float).tobytes() for a in (
            self.x, self.y, self.d, self.kin.link_p, self.kin.link_R, self.kin.axis, self.kin.lo, self.kin.hi,
            self.kin.palm, [self.scene.rest_z, HOVER, 3.5]))).hexdigest()
        sc = self.scene
        # The bins' front wall: a hand part lower than its rim may not reach past this x.
        self.wall_x = min(c[0] for c in sc.bin_centers.values()) - 0.5 * float(np.max(sc.bin_outer)) - 0.003
        if cache and CACHE.exists():
            z = np.load(CACHE)
            if str(z["key"]) == key:
                self.cost, self.q, self.foot, self.reach_x = z["cost"], z["q"], z["foot"], z["reach_x"]
                return
        self.cost, self.q = self._compute()
        self.foot, self.reach_x = self._footprint(twin)
        if cache:
            np.savez_compressed(CACHE, key=key, cost=self.cost.astype(np.float32), q=self.q.astype(np.float32),
                                foot=self.foot.astype(np.float32), reach_x=self.reach_x.astype(np.float32))

    def _footprint(self, twin=None) -> tuple[np.ndarray, np.ndarray]:
        """Where the open hand comes down to part height when it pinches: points (2 hands, K, 2)
        on the mat, in the hand's own frame (origin at the pinch point, x along the fingers).
        A neighbouring part under any of them would be hit. And the same for the hand parts
        that come lower than the rim of the bins (2 hands, K2, 2): none may reach the bins."""
        from kaiku.sim.twin import build_twin

        tw = twin or build_twin(self.scene)
        m = tw.model
        d = mujoco.MjData(m)
        pinch = np.array([[[0.30, 0.06, self.scene.rest_z + 0.004], [0.30, -0.06, self.scene.rest_z + 0.004]]])
        rot = level_palm(np.zeros((1, 2)))
        off = np.stack([grasp_offset("left"), grasp_offset("right")])
        q, _, _ = self.kin.solve(self.kin.rest.reshape(1, 14).copy(), pinch - np.einsum("nsij,sj->nsi", rot, off), rot,
                                 iters=120)
        d.qpos[tw.arm_qadr], d.qpos[tw.hand_qadr] = q[0], hand14(HAND_OPEN, HAND_OPEN)
        mujoco.mj_kinematics(m, d)
        top = self.scene.rest_z + 0.5 * self.scene.size + 0.012
        rim = self.scene.counter_z + self.scene.bin_height + 0.004
        g = np.linspace(-1.0, 1.0, 5)
        grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
        clouds, rims = [], []
        for s, side in enumerate(("left", "right")):
            pts, low = [], []
            for k in range(m.ngeom):
                name = m.body(m.geom_bodyid[k]).name
                if not (name.startswith(side) and "hand" in name):
                    continue
                if not (m.geom_contype[k] or m.geom_conaffinity[k]):
                    continue
                centre, half = m.geom_aabb[k, :3], m.geom_aabb[k, 3:]
                world = d.geom_xpos[k] + (centre + grid * half) @ d.geom_xmat[k].reshape(3, 3).T
                pts.append(world[world[:, 2] < top, :2] - pinch[0, s, :2])
                low.append(world[world[:, 2] < rim, :2] - pinch[0, s, :2])
            clouds.append(np.unique(np.round(np.concatenate(pts) / 0.008).astype(int), axis=0) * 0.008)
            rims.append(np.unique(np.round(np.concatenate(low) / 0.004).astype(int), axis=0) * 0.004)

        def pad(cs):
            k = max(len(c) for c in cs)
            return np.stack([np.concatenate([c, np.repeat(c[:1], k - len(c), axis=0)]) for c in cs])

        return pad(clouds), pad(rims)

    def _compute(self) -> tuple[np.ndarray, np.ndarray]:
        """Cost (2, nx, ny, nd) in metres: position error plus 2 cm per radian of palm error, the
        worse of hover and grasp height. And the arm postures found, (2 heights: grasp then
        hover, 2 arms, nx, ny, nd, 7)."""
        X, Y, D = np.meshgrid(self.x, self.y, self.d, indexing="ij")
        n = X.size
        direction = np.stack([D.ravel(), D.ravel()], axis=1)                              # (n, 2)
        rot = level_palm(direction)
        off = np.stack([grasp_offset("left"), grasp_offset("right")])
        pinch = np.stack([X.ravel(), Y.ravel(), np.full(n, self.scene.rest_z + 0.004)], axis=1)[:, None, :]
        pinch = np.broadcast_to(pinch, (n, 2, 3))
        lean = np.einsum("nsij,sj->nsi", rot, off)
        q = np.tile(self.kin.rest.reshape(1, 14), (n, 1))
        cost = np.zeros((n, 2))
        found = []
        for dz, iters in ((HOVER, 90), (0.0, 40)):
            q, ep, er = self.kin.solve(q, pinch + [0.0, 0.0, dz] - lean, rot, iters=iters)
            cost = np.maximum(cost, ep + 0.02 * er)
            found.append(q.reshape(len(self.x), len(self.y), len(self.d), 2, 7))
        posture = np.moveaxis(np.stack(found[::-1]), 4, 1)                                # (2, 2, nx, ny, nd, 7)
        return np.moveaxis(cost.reshape(len(self.x), len(self.y), len(self.d), 2), -1, 0), posture

    # ------------------------------------------------------------------ lookup
    def _cells(self, xy: np.ndarray, direction: np.ndarray):
        """The eight table cells around a query and their trilinear weights (clamped at the edges)."""
        def frac(v, grid):
            f = np.clip((v - grid[0]) / (grid[1] - grid[0]), 0.0, len(grid) - 1.000001)
            i = np.floor(f).astype(int)
            return i, f - i

        (ix, fx), (iy, fy), (id_, fd) = frac(xy[..., 0], self.x), frac(xy[..., 1], self.y), frac(direction, self.d)
        for ax, wx in ((0, 1 - fx), (1, fx)):
            for ay, wy in ((0, 1 - fy), (1, fy)):
                for ad, wd in ((0, 1 - fd), (1, fd)):
                    yield wx * wy * wd, ix + ax, iy + ay, id_ + ad

    def lookup(self, side: np.ndarray | int, xy: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Cost (m) of pinching at ``xy`` (..., 2) with fingers along ``direction`` (...), for arm
        ``side`` (0 left, 1 right; broadcastable). Trilinear; outside the table it is clamped."""
        side = np.broadcast_to(side, np.shape(direction))
        direction = (np.asarray(direction) + np.pi) % (2 * np.pi) - np.pi
        outside = np.maximum(np.abs(direction) - self.d[-1], 0.0)                         # beyond the table: unreachable
        out = 0.0
        for w, i, j, k in self._cells(xy, direction):
            out = out + w * self.cost[side, i, j, k]
        return out + outside

    def posture(self, pinch: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """An arm posture known to put each hand's pinch point at ``pinch`` (N, 2, 3) with the
        palm level and the fingers along ``direction`` (N, 2): joints (N, 2, 7), and how far that
        can be trusted (N, 2), 1 over the mat at grasp-to-hover height, falling to 0 away from it."""
        side = np.broadcast_to(np.arange(2), direction.shape)
        direction = (np.asarray(direction) + np.pi) % (2 * np.pi) - np.pi
        height = np.clip((pinch[..., 2] - self.scene.rest_z) / HOVER, 0.0, 1.0)[..., None]
        q = 0.0
        for w, i, j, k in self._cells(pinch[..., :2], direction):
            cell = (1.0 - height) * self.q[0, side, i, j, k] + height * self.q[1, side, i, j, k]
            q = q + w[..., None] * cell
        lo = np.array([self.x[0], self.y[0]])
        hi = np.array([self.x[-1], self.y[-1]])
        out_xy = np.linalg.norm(np.maximum(lo - pinch[..., :2], 0.0) + np.maximum(pinch[..., :2] - hi, 0.0), axis=-1)
        out_z = np.maximum(pinch[..., 2] - self.scene.rest_z - HOVER - 0.02, 0.0)
        out_d = np.maximum(np.abs(direction) - self.d[-1], 0.0)
        trust = np.clip(1.0 - out_xy / 0.04, 0.0, 1.0) * np.clip(1.0 - out_z / 0.06, 0.0, 1.0) \
            * np.clip(1.0 - out_d / 0.3, 0.0, 1.0)
        return q, trust

    def candidates(self, shape: np.ndarray, yaw: np.ndarray, side: np.ndarray | int) -> np.ndarray:
        """Finger directions (..., 9) that pinch each part (NaN-padded): 4 for a cube, 6 for a
        triangle, 9 spread around the arm's natural direction for a cylinder. Each is wrapped
        to within half a turn of the natural direction."""
        shape, yaw = np.asarray(shape), np.asarray(yaw, float)
        nat = np.broadcast_to(self.natural[side], yaw.shape)[..., None]
        k = np.arange(9)
        cube = np.where(k < 4, yaw[..., None] + k * np.pi / 2, np.nan)
        apex = yaw[..., None] + np.pi / 2
        tri = np.where(k < 3, apex + TRIANGLE_PHI + k * 2 * np.pi / 3,
                       np.where(k < 6, apex - TRIANGLE_PHI + (k - 3) * 2 * np.pi / 3, np.nan))
        cyl = nat + CYL_OFFSETS
        c = np.where(shape[..., None] == CUBE, cube, np.where(shape[..., None] == TRIANGLE, tri, cyl))
        return nat + (c - nat + np.pi) % (2 * np.pi) - np.pi

    def blocked(self, side: int, centre: np.ndarray, direction: np.ndarray, others: np.ndarray | None = None,
                radius: np.ndarray | None = None) -> np.ndarray:
        """Metres by which the open hand, pinching at ``centre`` (..., 2) with its fingers along
        ``direction`` (...), comes too close to the bins or to other parts at ``others``
        (..., M, 2) of ``radius`` (..., M). NaN entries are no part. Zero means clear."""
        c, s = np.cos(direction)[..., None], np.sin(direction)[..., None]
        ahead = centre[..., 0] + (c * self.reach_x[side][:, 0] - s * self.reach_x[side][:, 1]).max(axis=-1)
        hit = np.maximum(ahead - (self.wall_x - WALL_CLEARANCE), 0.0)
        if others is None or not others.shape[-2]:
            return hit
        rel = others - centre[..., None, :]
        local = np.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], axis=-1)
        gap = np.linalg.norm(local[..., None, :] - self.foot[side], axis=-1).min(axis=-1) - radius   # (..., M)
        gap = np.where(np.isfinite(gap), gap, np.inf)
        return np.maximum(hit, np.maximum(CLEARANCE - gap.min(axis=-1, initial=np.inf), 0.0))

    def best(self, shape: np.ndarray, yaw: np.ndarray, xy: np.ndarray, side: int,
             size: float | None = None, others: np.ndarray | None = None, radius: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
        """The grasp to use for each part: among the directions that pinch it, the one closest to
        the arm's natural direction that the arm reaches without its open hand coming down on a
        neighbour (``others`` (..., M, 2) with ``radius`` (..., M)) or on the bins; if there is none,
        the least bad.
        Returns (direction, cost); cost above ``TOL`` means there is no comfortable grasp."""
        from kaiku.fanout.echo import pinch_shift          # local import: echo imports this module's users

        size = self.scene.size if size is None else size
        shape, yaw = np.asarray(shape), np.asarray(yaw, float)
        cand = self.candidates(shape, yaw, side)                                          # (..., 9)
        ok = np.isfinite(cand)
        c = np.where(ok, cand, 0.0)
        centre = xy[..., None, :] + pinch_shift(shape[..., None], yaw[..., None], c, size)
        cost = self.lookup(side, centre, c)
        if others is not None and others.shape[-2]:
            hit = self.blocked(side, centre, c, others[..., None, :, :], radius[..., None, :])
        else:
            hit = self.blocked(side, centre, c)
        cost = np.where(hit > 0.0, np.maximum(cost, TOL + hit), cost)
        away = np.abs(c - np.broadcast_to(self.natural[side], yaw.shape)[..., None])
        score = np.where(ok, away + 100.0 * np.maximum(cost - TOL, 0.0), np.inf)
        j = np.argmin(score, axis=-1)[..., None]
        return np.take_along_axis(c, j, -1)[..., 0], np.take_along_axis(cost, j, -1)[..., 0]

    def comfortable(self, shape: str | int, yaw: float, xy: np.ndarray, side: int,
                    max_turn: float = np.radians(50.0), others: np.ndarray | None = None,
                    radius: np.ndarray | None = None) -> bool:
        """True if this arm can pinch the part with its fingers within ``max_turn`` of the arm's
        natural direction and clear of the ``others``. Beyond that turn the pose exists but the
        arm tracks it poorly."""
        code = {"cube": CUBE, "cylinder": CYLINDER, "triangle": TRIANGLE}.get(shape, shape)
        d, c = self.best(np.array(code), np.array(float(yaw)), np.asarray(xy, float), side, others=others,
                         radius=radius)
        return bool(c < TOL and abs(d - self.natural[side]) <= max_turn)

    def parts_ok(self, shape: np.ndarray, yaw: np.ndarray, xy: np.ndarray, sides: np.ndarray,
                 max_turn: float = np.radians(50.0)) -> np.ndarray:
        """For K layouts of n parts (``shape`` and ``yaw`` (K, n), ``xy`` (K, n, 2)): (K, n), True
        for each part that has a comfortable grasp for its arm (``sides`` (n,)) with all the
        other parts where they are."""
        n = shape.shape[1]
        rest = np.array([[j for j in range(n) if j != k] for k in range(n)], int).reshape(n, n - 1)
        others, radius = xy[:, rest], part_radius(shape, self.scene.size)[:, rest]
        ok = np.ones(shape.shape, bool)
        for s in range(2):
            d, c = self.best(shape, yaw, xy, s, others=others, radius=radius)
            ok &= (np.asarray(sides) != s) | ((c < TOL) & (np.abs(d - self.natural[s]) <= max_turn))
        return ok

    def layouts_ok(self, shape: np.ndarray, yaw: np.ndarray, xy: np.ndarray, sides: np.ndarray) -> np.ndarray:
        """(K,) True where every part of the layout is ok; see :meth:`parts_ok`."""
        return self.parts_ok(shape, yaw, xy, sides).all(axis=1)

    def layout_ok(self, layout, sides: list[int]) -> bool:
        """The same for one :class:`~kaiku.sim.scene.Layout`."""
        sc = self.scene
        code = {"cube": CUBE, "cylinder": CYLINDER, "triangle": TRIANGLE}
        if not layout.placements:
            return True
        shape = np.array([[code[sc.by_name[p.name].shape] for p in layout.placements]])
        xy = np.array([[p.pos[:2] for p in layout.placements]])
        yaw = np.array([[p.yaw for p in layout.placements]])
        return bool(self.layouts_ok(shape, yaw, xy, np.asarray(sides))[0])
